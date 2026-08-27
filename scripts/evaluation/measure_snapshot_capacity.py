#!/usr/bin/env python3
"""What would a LOSSLESS completion snapshot actually cost, in exact tokens?

Zero inference. The GGUF is opened vocab-only -- no weight tensors are
allocated -- so this measures the real tokenizer without ever running the
model. Character estimates are not used anywhere: the question is how many
tokens the verifier prompt would really be, and only the tokenizer knows.

Reads the preserved corpus read-only and reconstructs, for each persisted
checkpoint, the snapshot that Orbit *would* have built had nothing been
truncated or omitted.
"""
from __future__ import annotations

import ctypes
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from orbit.native_llama.bindings import LlamaLibrary  # noqa: E402
from orbit.runtime.completion_shadow import (  # noqa: E402
    VERIFIER_A_INSTRUCTION,
    VERIFIER_B_INSTRUCTION,
    CompletionSnapshot,
)
from orbit.runtime.completion_shadow_ledger import read_ledger  # noqa: E402
from orbit.runtime.evidence import EvidenceStore  # noqa: E402

MODEL = ROOT / "models/ornith-ai--Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q4_K_M.gguf"
BUILD = ROOT / "src/orbit/native_llama/vendor/build/llama.cpp/bin"
BUDGETS = (1024, 2048, 4096, 6144, 8192, 12288)


class VocabTokenizer:
    """Exact tokenization with no weights loaded and no inference possible."""

    def __init__(self, build_bin: Path, model_path: Path) -> None:
        self._binding = LlamaLibrary(build_bin)
        lib = self._binding.lib
        lib.llama_backend_init()
        params = lib.llama_model_default_params()
        params.vocab_only = True
        params.use_mmap = True
        self._model = lib.llama_model_load_from_file(str(model_path).encode(), params)
        if not self._model:
            raise RuntimeError("vocab-only load failed")
        self._vocab = lib.llama_model_get_vocab(self._model)
        self._lib = lib

    def count(self, text: str) -> int:
        data = text.encode("utf-8")
        cap = len(data) + 64
        out = (ctypes.c_int32 * cap)()
        n = self._lib.llama_tokenize(
            self._vocab, data, len(data), out, cap, False, False
        )
        if n < 0:
            raise RuntimeError("tokenization overflow")
        return int(n)

    def close(self) -> None:
        self._lib.llama_model_free(self._model)


def full_snapshot(checkpoint: dict, store: EvidenceStore, request: str) -> CompletionSnapshot:
    """The snapshot the contract would produce with nothing dropped."""
    evidence = tuple(
        (e["evidence_id"], store.load_raw(e["evidence_id"]) or "")
        for e in checkpoint.get("snapshot_evidence", [])
    )
    return CompletionSnapshot(
        request=request,
        evidence=evidence,
        artifacts=tuple(checkpoint.get("snapshot_artifacts", [])),
        digest="",
    )


def bounded_snapshot(checkpoint: dict, request: str) -> CompletionSnapshot:
    return CompletionSnapshot(
        request=request,
        evidence=tuple(
            (e["evidence_id"], e.get("text", ""))
            for e in checkpoint.get("snapshot_evidence", [])
        ),
        artifacts=tuple(checkpoint.get("snapshot_artifacts", [])),
        digest="",
    )


def main(argv: list[str]) -> int:
    corpus = Path(argv[1])
    tok = VocabTokenizer(BUILD, MODEL)
    # The verifier prompt is instruction + snapshot; both must fit the budget.
    overhead = max(tok.count(VERIFIER_A_INSTRUCTION), tok.count(VERIFIER_B_INSTRUCTION))
    print(f"verifier instruction overhead: {overhead} tokens (larger of A/B)\n")

    rows = []
    for label in ("A", "B", "C"):
        d = corpus / "runs" / label
        ledger = read_ledger(d / "completion-shadow.jsonl")
        store = EvidenceStore(root=d / "evidence")
        store.load_index()
        for cp in ledger.checkpoints:
            request = cp.get("request", "")
            full = full_snapshot(cp, store, request)
            bounded = bounded_snapshot(cp, request)
            full_text, bounded_text = full.render(), bounded.render()
            full_tokens = tok.count(full_text)
            largest = max(
                (tok.count(text) for _i, text in full.evidence), default=0
            )
            rows.append({
                "run": label,
                "action": cp["action"],
                "records": len(full.evidence),
                "artifacts": len(full.artifacts),
                "bounded_chars": len(bounded_text),
                "full_chars": len(full_text),
                "bounded_tokens": tok.count(bounded_text),
                "full_tokens": full_tokens,
                "largest_record_tokens": largest,
                "prompt_tokens_if_lossless": full_tokens + overhead,
            })
    tok.close()

    print(f"{'run':>4}{'act':>5}{'recs':>6}{'arts':>6}{'bounded_tok':>13}{'full_tok':>10}"
          f"{'delta':>8}{'largest':>9}{'prompt':>8}  " + "".join(f"{b//1024}K".rjust(6) for b in BUDGETS))
    for r in rows:
        fits = "".join(
            ("yes" if r["prompt_tokens_if_lossless"] <= b else "-").rjust(6) for b in BUDGETS
        )
        print(f"{r['run']:>4}{r['action']:>5}{r['records']:>6}{r['artifacts']:>6}"
              f"{r['bounded_tokens']:>13}{r['full_tokens']:>10}"
              f"{r['full_tokens']-r['bounded_tokens']:>8}{r['largest_record_tokens']:>9}"
              f"{r['prompt_tokens_if_lossless']:>8}  {fits}")

    print(f"\n{'budget':>8}{'lossless / 13':>16}")
    for b in BUDGETS:
        n = sum(1 for r in rows if r["prompt_tokens_if_lossless"] <= b)
        print(f"{b:>8}{f'{n} / 13':>16}")

    if len(argv) > 2:
        Path(argv[2]).write_text(json.dumps(
            {"instruction_overhead_tokens": overhead, "budgets": list(BUDGETS), "rows": rows},
            indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
