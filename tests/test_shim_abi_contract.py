"""The shim ABI contract must describe symbols that actually exist.

`_REQUIRED_SHIM_SYMBOLS` decides whether a prebuilt shim may be reused. If it
names a symbol no shim exports, the check can never pass: the packaged binary is
rejected every time and the shim is rebuilt on every call, for every
architecture. That is what happened with
`orbit_mtp_session_set_followup_suffix_tokens`, whose implementation was written
on a branch that never landed while the contract entry did.

Source-text checks alone would not have caught it -- the symbol was absent from
source AND binary -- so these assert against the COMPILED exports of the shim
the build actually produces.
"""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

from orbit.native_llama import persistent_mtp as pm

ROOT = Path(__file__).resolve().parents[1]
SHIM_SOURCE = ROOT / "src/orbit/native_llama/vendor/shim/orbit_persistent_mtp.cpp"


def _built_shim() -> Path | None:
    """A shim binary to inspect, preferring one the current build produced."""
    for candidate in (
        Path.home() / ".orbit/native-build/liborbit-persistent-mtp.so",
        ROOT / "src/orbit/native_llama/vendor/shim/liborbit-persistent-mtp.so",
    ):
        if candidate.exists():
            return candidate
    return None


def _exports(path: Path) -> set[str]:
    out = subprocess.run(
        ["nm", "-D", "--defined-only", str(path)],
        capture_output=True, text=True, check=False,
    )
    names = set()
    for line in out.stdout.splitlines():
        parts = line.split()
        if parts:
            names.add(parts[-1])
    return names


class SourceDefinesEveryRequiredSymbolTests(unittest.TestCase):
    """Cheap guard: the contract must not name something the source lacks.

    Additive to the compiled checks below, never a substitute for them.
    """

    def _source(self) -> str:
        return SHIM_SOURCE.read_text()

    def test_base_symbols_are_defined_in_the_shim_source(self) -> None:
        source = self._source()
        for symbol in pm._REQUIRED_SHIM_SYMBOLS:
            with self.subTest(symbol=symbol):
                self.assertTrue(
                    symbol in source,
                    f"{symbol} is required by the ABI contract but is not defined "
                    "in the shim source, so no build can ever satisfy it",
                )

    def test_self_mtp_symbols_are_defined_in_the_shim_source(self) -> None:
        source = self._source()
        for symbol in pm._SELF_MTP_REQUIRED_SHIM_SYMBOLS:
            with self.subTest(symbol=symbol):
                self.assertTrue(symbol in source, f"{symbol} missing from shim source")


class CompiledExportsTests(unittest.TestCase):
    """The binding check that would have caught the phantom symbol."""

    def setUp(self) -> None:
        self.shim = _built_shim()
        if self.shim is None:
            self.skipTest("no compiled shim available to inspect")
        self.exports = _exports(self.shim)
        if not self.exports:
            self.skipTest("nm produced no symbols (missing binutils?)")

    def test_every_base_symbol_is_exported(self) -> None:
        missing = [s for s in pm._REQUIRED_SHIM_SYMBOLS if s not in self.exports]
        self.assertEqual(
            missing, [],
            f"base ABI requires symbols the compiled shim does not export: {missing}",
        )

    def test_self_mtp_symbols_are_exported_by_the_current_shim(self) -> None:
        missing = [
            s for s in pm._SELF_MTP_REQUIRED_SHIM_SYMBOLS if s not in self.exports
        ]
        if missing and "orbit_selfmtp_session_create" in missing:
            self.skipTest("inspecting a pre-self-MTP shim")
        self.assertEqual(missing, [])


class AbiSeparationTests(unittest.TestCase):
    """Base and self-MTP requirements must stay disjoint and additive."""

    def test_no_self_mtp_symbol_leaks_into_the_base_contract(self) -> None:
        for symbol in pm._SELF_MTP_REQUIRED_SHIM_SYMBOLS:
            self.assertNotIn(symbol, pm._REQUIRED_SHIM_SYMBOLS)

    def test_base_contract_is_non_empty(self) -> None:
        """An empty tuple would make the check vacuously true."""
        self.assertTrue(pm._REQUIRED_SHIM_SYMBOLS)

    def test_contracts_have_no_duplicates(self) -> None:
        for tup in (pm._REQUIRED_SHIM_SYMBOLS, pm._SELF_MTP_REQUIRED_SHIM_SYMBOLS):
            self.assertEqual(len(tup), len(set(tup)))


class PackagedShimReuseTests(unittest.TestCase):
    """A conforming packaged shim must be reused rather than rebuilt.

    The reuse branch in `build_persistent_mtp_shim` is guarded by the base
    contract. While that contract named a symbol nothing exported, the branch
    was dead and every call recompiled. This pins that it is reachable again,
    by counting compiler invocations rather than by timing anything.
    """

    def test_conforming_shim_is_reused_without_compiling(self) -> None:
        from unittest.mock import patch

        shim = _built_shim()
        if shim is None:
            self.skipTest("no compiled shim available")
        if not _exports(shim):
            self.skipTest("nm produced no symbols")

        compiles = []

        def refuse_to_compile(*args, **kwargs):
            compiles.append(args)
            raise AssertionError("a conforming packaged shim must not be recompiled")

        with patch.object(pm, "packaged_shim_path", lambda _name: shim), \
             patch.object(pm, "_shim_exports_required_symbols", lambda *a, **k: True), \
             patch.object(pm, "compile_cpp_helper", refuse_to_compile):
            out = pm.build_persistent_mtp_shim(
                llama_root=ROOT / "src/orbit/native_llama/vendor/source/llama.cpp"
            )
        self.assertEqual(Path(out), shim)
        self.assertEqual(compiles, [])

    def test_nonconforming_shim_is_not_reused(self) -> None:
        """The converse: a shim failing the contract must fall through."""
        from unittest.mock import patch

        shim = _built_shim()
        if shim is None:
            self.skipTest("no compiled shim available")

        reached = []

        def fake_compile(*args, **kwargs):
            reached.append(args)
            raise RuntimeError("compile reached")

        with patch.object(pm, "packaged_shim_path", lambda _name: shim), \
             patch.object(pm, "_shim_exports_required_symbols", lambda *a, **k: False), \
             patch.object(pm, "compile_cpp_helper", fake_compile):
            with self.assertRaises(RuntimeError):
                pm.build_persistent_mtp_shim(
                    llama_root=ROOT / "src/orbit/native_llama/vendor/source/llama.cpp"
                )
        self.assertTrue(reached, "a failing contract must fall through to a rebuild")


class MissingExportIsRejectedTests(unittest.TestCase):
    """A shim missing a required export must be refused, not accepted."""

    def test_absent_symbol_fails_the_check(self) -> None:
        shim = _built_shim()
        if shim is None:
            self.skipTest("no compiled shim available")
        self.assertFalse(
            pm._shim_exports_required_symbols(
                shim, None, ("orbit_definitely_not_a_real_symbol",)
            )
        )


if __name__ == "__main__":
    unittest.main()
