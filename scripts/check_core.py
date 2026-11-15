#!/usr/bin/env python3
"""Cheap, source-fresh core gate; no native build, weights or external corpus."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    "tests.test_command_request",
    "tests.test_tool_contract",
    "tests.test_tool_healing",
    "tests.test_tool_completion_authority",
    "tests.test_native_server_protocol",
    "tests.test_native_model_profiles",
    "tests.test_llama_server_backend",
    "tests.test_qualification_core",
    "tests.test_qualification_fixtures",
    "tests.test_qualification_tool_intent",
    "tests.test_model_contract",
    "tests.test_research_admission",
    "tests.test_model_probe",
    "tests.test_cache_profile_policy",
    "tests.test_core_checks",
)


def run_suite() -> int:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    suite = unittest.defaultTestLoader.loadTestsFromNames(MODULES)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if result.testsRun == 0 or result.skipped:
        print("core gate requires non-empty coverage with zero skips", file=sys.stderr)
        return 1
    return 0 if result.wasSuccessful() else 1


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["--run-suite"]:
        return run_suite()
    if args:
        print("usage: python3 scripts/check_core.py", file=sys.stderr)
        return 2
    from qualify_fresh import run_fresh
    return run_fresh([str(Path(__file__).resolve()), "--run-suite"]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
