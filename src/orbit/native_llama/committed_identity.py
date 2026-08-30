"""Ownership of the committed-token identity.

The committed sequence is the exact token sequence currently resident in the
backend KV: the prompt that was prefilled plus every generated token that was
successfully decoded into it. It is the runtime's semantic claim about physical
state, and every reuse decision rests on it.

The immutable rule is strict equality:

    prompt_tokens[:len(committed)] == committed

No longest-common-prefix, no fuzzy reuse, no semantic equivalence, no
retokenized approximation. Anything short of exact identity must yield no reuse,
because a stale or overstated identity is a false cache hit -- the next turn
would skip prefill for a position that was never decoded, which corrupts output
silently rather than merely losing speed.

This collaborator owns that state and the operations over it. It deliberately
knows nothing about physical KV, the speculative pair, or decoding: the backend
owns physical truth and reports it, and this side decides only semantic
eligibility. It takes the small dependencies it actually needs (a tokenizer, an
opt-out predicate, a profile id for diagnostics) rather than the client.
"""
from __future__ import annotations

from typing import Callable, Sequence

from .kv_diag import emit_strict_append_miss


class CommittedIdentity:
    """The single owner of committed-token identity for one session.

    `tokenize` and `coder_protocol` are passed as callables rather than by
    handing over the client, so this module never imports the client and cannot
    reach back into it.
    """

    __slots__ = ("_tokens", "_tokenize", "_coder_protocol", "_session_id", "_profile_id")

    def __init__(
        self,
        *,
        tokenize: Callable[[str], list[int]],
        coder_protocol: Callable[[], bool],
        session_id: Callable[[], object] | None = None,
        profile_id: Callable[[], object] | None = None,
    ) -> None:
        self._tokens: list[int] = []
        self._tokenize = tokenize
        self._coder_protocol = coder_protocol
        self._session_id = session_id
        self._profile_id = profile_id

    @property
    def tokens(self) -> list[int]:
        """The committed sequence itself.

        Returned by reference, exactly as the attribute it replaces was read, so
        callers that mutate or alias it behave as before. Making this a copy
        would be a behaviour change, not a cleanup.
        """
        return self._tokens

    def invalidate(self) -> None:
        """Drop identity whenever backend KV may diverge.

        Strict append-only continuation is only safe while the recorded tokens
        are exactly the ones resident in KV. Every path that clears, rewrites or
        replaces that memory must call this; a stale identity would produce a
        false cache hit, which is a correctness bug rather than a slow path.
        """
        self._tokens.clear()

    def commit(self, prompt_tokens: Sequence[int], generated_tokens: Sequence[int]) -> None:
        """Record the exact sequence now resident in KV."""
        self._tokens = list(prompt_tokens) + list(generated_tokens)

    def adopt(self, tokens: Sequence[int]) -> None:
        """Record an exact resident sequence measured elsewhere.

        Used where a restore has already proven which tokens are resident, so
        there is no prompt/generated split to reconstruct.
        """
        self._tokens = list(tokens)

    def resident_prefix_len_for_mtp(self, mtp_prompt: str) -> int:
        """Length of the committed prefix an MTP completion may reuse, or 0.

        The same strict append-only identity the non-MTP path enforces, with one
        extra requirement: a STRICT PROPER prefix. Proper matters -- a claim
        equal to the whole prompt would leave no suffix to decode, so sampling
        would read logits from the previous completion (Defect A).

        Returning 0 is always safe: the backend then rebuilds cold.
        """
        committed = self.tokens
        if not committed:
            return 0
        if self._coder_protocol():
            return 0
        try:
            prompt_tokens = self._tokenize(mtp_prompt)
        except Exception:
            return 0
        n = len(committed)
        if not (0 < n < len(prompt_tokens)):
            return 0
        if prompt_tokens[:n] != committed:
            # Diagnostic only, opt-in via ORBIT_KV_DIAG, and never consulted by
            # any decision -- the refusal below is unconditional. Emitted because
            # a divergence here and "no identity yet" both surface as a claim of
            # 0, and only this distinguishes them when attributing a missed reuse.
            emit_strict_append_miss(
                committed=list(committed),
                prompt=list(prompt_tokens),
                session_id=self._session_id() if self._session_id else None,
                profile_id=self._profile_id() if self._profile_id else None,
                lifecycle="mtp-resident-claim",
            )
            return 0
        return n

    def publish_from_mtp(self, result) -> None:
        """Record what the backend proved is resident, or drop the identity.

        The verdict comes from the backend, captured at completion time.
        Identity must never outlive the physical pair it describes, so anything
        short of a canonical success drops it and the next turn rebuilds cold.
        """
        if not (
            getattr(result, "success", False)
            and getattr(result, "pair_canonical", False)
        ):
            self.invalidate()
            return
        # The identity is the backend's own record of what is physically
        # resident, not a reconstruction. `prompt + generated_tokens` would be
        # one token too long: a sampled token enters `generated` at sample time
        # but only enters the resident sequence on the following iteration. An
        # identity longer than KV is a false cache hit -- the next turn would
        # skip prefill for a position that was never decoded and land every
        # subsequent token one position early, which corrupts output silently
        # rather than merely losing reuse. Retokenizing the text is likewise not
        # equivalent. So: publish what the backend measured, or publish nothing.
        resident = tuple(getattr(result, "resident_tokens", ()) or ())
        if not resident:
            self.invalidate()
            return
        self.adopt(resident)


class AttributeBackedIdentity(CommittedIdentity):
    """Identity for a session that only carries a plain token attribute.

    Before this responsibility was extracted, identity lived in
    `session.committed_sequence_tokens` and any object exposing that attribute
    worked. Some callers build a client with such a stand-in. This keeps the
    same policy while reading and writing through that attribute, so those
    sessions behave exactly as they did -- the extraction must not impose a
    session type that was never previously required.
    """

    __slots__ = ("_session",)

    def __init__(self, session, *, tokenize, coder_protocol,
                 session_id=None, profile_id=None) -> None:
        # The client's OWN tokenizer and opt-out, not stand-ins: baseline
        # derived the resident claim with them even for these sessions, so
        # stubbing them here would silently refuse every reuse.
        super().__init__(
            tokenize=tokenize,
            coder_protocol=coder_protocol,
            session_id=session_id,
            profile_id=profile_id,
        )
        self._session = session

    @property
    def tokens(self) -> list[int]:
        return getattr(self._session, "committed_sequence_tokens", None) or []

    def invalidate(self) -> None:
        current = getattr(self._session, "committed_sequence_tokens", None)
        if current is None:
            self._session.committed_sequence_tokens = []
        else:
            # Clear in place, exactly as the pre-extraction code did, so a
            # caller holding a reference to this list still sees it emptied.
            current.clear()

    def commit(self, prompt_tokens, generated_tokens) -> None:
        self._session.committed_sequence_tokens = list(prompt_tokens) + list(generated_tokens)

    def adopt(self, tokens) -> None:
        self._session.committed_sequence_tokens = list(tokens)
