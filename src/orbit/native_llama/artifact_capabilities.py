"""What one exact GGUF is qualified to do, as distinct from what its profile supports.

Discovery answers "which model is this?" from GGUF metadata, cheaply. This
module answers a different question -- "is *these bytes* qualified for X?" --
and it has to, because two builds of the same model share every metadata field
discovery reads. The current and legacy Ornith artifacts have the same
architecture, the same `general.name`, the same chat template and the same 753
tensors; only ~20 blk.40 weights differ. Nothing short of content identity can
separate them.

So capability is keyed on the SHA-256 of the file's bytes. That costs a full
read of a 20 GiB file (measured ~46 s), which is why it is never done during
discovery and only ever on an explicit query.

Everything about ARTIFACT STATE fails closed: an unknown digest, an unverified
profile, a missing file or an unreadable one all yield "not qualified". Caller
bugs are deliberately not absorbed -- an unhashable `capability` argument, or a
non-OSError fault while hashing, propagates rather than being reported as a
policy decision. Silently answering "not qualified" for a defect would hide it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from orbit.native_llama.model_profiles import (
    NativeModelProfile,
    artifact_capability_map,
    verified_native_model_identity,
)

_CHUNK = 1024 * 1024


def artifact_sha256(path: Path) -> str:
    """SHA-256 of the file's bytes. No caching: correctness over cleverness.

    Deliberately not memoised. A cache keyed on path or on (size, mtime) can be
    defeated by an in-place replacement that preserves both, and the failure
    mode is an unqualified artifact inheriting a qualified verdict. Orbit has
    no existing primitive with invalidation strong enough to carry that risk,
    so the honest answer is to pay the read.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_artifact_capabilities(
    path: Path,
    profile: NativeModelProfile | None,
) -> frozenset[str]:
    """Capabilities this exact artifact is qualified for. Empty when unsure.

    `profile` is the already-resolved discovery result, passed in rather than
    recomputed: this module must not become a second way to identify a model.
    A profile that is absent or unverified short-circuits before any hashing,
    so the expensive path is only reached for a model Orbit already trusts.
    """
    if profile is None or not profile.verified:
        return frozenset()
    identity = verified_native_model_identity(profile.profile_id)
    if identity is None or not identity.artifact_capabilities:
        return frozenset()
    # Only now is the digest worth computing: this profile has *some* artifact
    # qualified for *something*, so the bytes decide which.
    try:
        digest = artifact_sha256(Path(path))
    except OSError:
        # Unreadable, vanished, or a directory. Not qualified, not an error --
        # the caller's normal path must keep working without capabilities.
        return frozenset()
    return frozenset(artifact_capability_map(identity).get(digest, frozenset()))


def verified_artifact_supports(
    path: Path,
    capability: str,
    profile: NativeModelProfile | None,
) -> bool:
    """Is this exact artifact qualified for `capability`?"""
    return capability in verified_artifact_capabilities(path, profile)
