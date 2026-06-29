#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.native_llama.chat_template import render_gemma4_route_prompt_segments
from orbit.native_llama.client import NativeClientConfig, NativeLlamaClient
from orbit.native_llama.native_names import runtime_library_filename
from orbit.native_llama.paths import DEFAULT_MODEL_ID, resolve_legacy_paths, resolve_paths
from orbit.native_llama.prefix_anchor_probe import probe_route_prefix_prefill_only
from orbit.runtime.messages import ROUTE_SYSTEM_PROMPT


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        paths = (
            resolve_legacy_paths(llama_root=args.llama_root, model=args.model, mmproj=args.mmproj)
            if args.model is not None
            else resolve_paths(
                llama_root=args.llama_root,
                model_id=args.model_id,
                mmproj=args.mmproj,
                models_dir=args.models_dir,
                hf_cache=args.hf_cache,
            )
        )
        client = NativeLlamaClient(
            paths,
            NativeClientConfig(
                context_tokens=args.ctx,
                threads=args.threads,
                threads_batch=args.threads_batch,
                batch_size=args.batch,
                ubatch_size=args.ubatch,
            ),
        )
        if not args.verbose_llama_log:
            client.set_quiet_logging()
        client.load()
        try:
            segments = render_gemma4_route_prompt_segments(
                [{"role": "system", "content": ROUTE_SYSTEM_PROMPT}],
                thinking=False,
            )
            if not segments.boundary_available:
                print(json.dumps({"probe_ok": False, "reason": "route_boundary_unavailable"}, sort_keys=True))
                return 2
            result = probe_route_prefix_prefill_only(
                lib=client.lib.lib,
                ctx=client._session.ctx_tgt,
                tokenize=client.tokenize,
                prefix_text=segments.stable_prefix_text,
                prefix_hash=segments.stable_prefix_hash,
                model_id=str(paths.model),
                template_id="gemma4-route-prefix-v1",
                tool_schema_hash=segments.stable_prefix_hash,
                capability_summary_hash=segments.stable_prefix_hash,
                runtime_policy_hash=segments.stable_prefix_hash,
                route_contract_hash=segments.stable_prefix_hash,
                backend_version="orbit-native",
                native_version=runtime_library_filename("llama"),
                tools_mode="on",
            )
            print(json.dumps(result.to_metadata(), sort_keys=True))
            return 0 if result.ok else 2
        finally:
            client.close()
    except Exception as exc:
        print(json.dumps({"probe_ok": False, "reason": type(exc).__name__}, sort_keys=True))
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="probe_native_route_prefix_prefill_only.py",
        description="Probe native route prefix prefill-only checkpoint capture. Emits metadata only.",
    )
    parser.add_argument("--llama-root", type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--mmproj", type=Path)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--hf-cache", type=Path)
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--threads-batch", type=int, default=6)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--ubatch", type=int, default=128)
    parser.add_argument("--verbose-llama-log", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
