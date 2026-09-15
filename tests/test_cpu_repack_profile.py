"""CPU weight-repack default for the qualified Dell + Ornith backend invocation.

The contract these pin: repack is turned OFF by default for exactly one
(machine, model) pair -- the Dell Pro 5 14 P514260 running Ornith-1.5-35B-A3B
Q4_K_M, qualified end-to-end by ORNITH-NOREPACK-* -- and for nothing else. An
explicit CLI or ORBIT_CPU_REPACK value always wins, and the new machine_model
field never changes a tuning profile's cache identity.
"""
from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.native_server.server_profile import (  # noqa: E402
    ENV_CPU_REPACK,
    HostTopology,
    QUALIFIED_NOREPACK_MACHINE,
    QUALIFIED_NOREPACK_MODEL_ID,
    detect_topology,
    profile_fingerprint,
    resolve_cpu_repack,
)

OTHER_MACHINE = "Intel(R) NUC Kit"
OTHER_MODEL = "qwen3-coder-30b-a3b-instruct-q4-k-m"


class CpuRepackResolution(unittest.TestCase):
    def test_A_qualified_dell_ornith_defaults_repack_off(self):
        value, source = resolve_cpu_repack(
            machine_model=QUALIFIED_NOREPACK_MACHINE,
            model_id=QUALIFIED_NOREPACK_MODEL_ID,
            cli=None,
            environ={},
        )
        self.assertIs(value, False)
        self.assertEqual(source, "qualified-dell-ornith")

    def test_B_other_machine_same_model_does_not_inherit(self):
        value, source = resolve_cpu_repack(
            machine_model=OTHER_MACHINE,
            model_id=QUALIFIED_NOREPACK_MODEL_ID,
            cli=None,
            environ={},
        )
        self.assertIsNone(value)
        self.assertEqual(source, "backend-default")

    def test_C_qualified_machine_other_model_does_not_inherit(self):
        value, source = resolve_cpu_repack(
            machine_model=QUALIFIED_NOREPACK_MACHINE,
            model_id=OTHER_MODEL,
            cli=None,
            environ={},
        )
        self.assertIsNone(value)
        self.assertEqual(source, "backend-default")

    def test_D_cli_overrides_qualified_default_both_directions(self):
        on, on_src = resolve_cpu_repack(
            machine_model=QUALIFIED_NOREPACK_MACHINE,
            model_id=QUALIFIED_NOREPACK_MODEL_ID,
            cli=True,
            environ={},
        )
        self.assertIs(on, True)
        self.assertEqual(on_src, "cli")
        off, off_src = resolve_cpu_repack(
            machine_model=OTHER_MACHINE,
            model_id=OTHER_MODEL,
            cli=False,
            environ={},
        )
        self.assertIs(off, False)
        self.assertEqual(off_src, "cli")

    def test_D_env_overrides_default_and_cli_beats_env(self):
        # env forces a value the qualified default would not have chosen
        forced_on, src = resolve_cpu_repack(
            machine_model=QUALIFIED_NOREPACK_MACHINE,
            model_id=QUALIFIED_NOREPACK_MODEL_ID,
            cli=None,
            environ={ENV_CPU_REPACK: "on"},
        )
        self.assertIs(forced_on, True)
        self.assertEqual(src, "env")
        # env off on an unrelated host
        forced_off, src2 = resolve_cpu_repack(
            machine_model=OTHER_MACHINE,
            model_id=OTHER_MODEL,
            cli=None,
            environ={ENV_CPU_REPACK: "0"},
        )
        self.assertIs(forced_off, False)
        self.assertEqual(src2, "env")
        # CLI beats env
        cli_wins, src3 = resolve_cpu_repack(
            machine_model=QUALIFIED_NOREPACK_MACHINE,
            model_id=QUALIFIED_NOREPACK_MODEL_ID,
            cli=False,
            environ={ENV_CPU_REPACK: "on"},
        )
        self.assertIs(cli_wins, False)
        self.assertEqual(src3, "cli")

    def test_D_unparseable_env_is_ignored_falls_through(self):
        value, source = resolve_cpu_repack(
            machine_model=QUALIFIED_NOREPACK_MACHINE,
            model_id=QUALIFIED_NOREPACK_MODEL_ID,
            cli=None,
            environ={ENV_CPU_REPACK: "maybe"},
        )
        self.assertIs(value, False)  # falls through to the qualified default
        self.assertEqual(source, "qualified-dell-ornith")

    def test_whitespace_in_machine_model_is_tolerated(self):
        value, _ = resolve_cpu_repack(
            machine_model="  Dell Pro 5 14 P514260  ",
            model_id=QUALIFIED_NOREPACK_MODEL_ID,
            cli=None,
            environ={},
        )
        self.assertIs(value, False)


class MachineModelTopology(unittest.TestCase):
    def test_detect_topology_reads_injected_machine_model(self):
        topo = detect_topology(
            meminfo_text="MemTotal: 32412036 kB\n",
            cpuinfo_text="model name : Test CPU\n",
            machine_model_text="Dell Pro 5 14 P514260\n",
        )
        self.assertEqual(topo.machine_model, "Dell Pro 5 14 P514260")

    def test_E_machine_model_does_not_change_profile_fingerprint(self):
        # Same tuning-relevant facts, different machine_model -> identical
        # fingerprint, so no cached profile is invalidated by this change.
        base = HostTopology(
            cpu_model="Intel(R) Core(TM) Ultra 7 366H",
            physical_cores=16,
            logical_cpus=16,
            total_ram_mib=31652,
        )
        with_model = HostTopology(
            cpu_model="Intel(R) Core(TM) Ultra 7 366H",
            physical_cores=16,
            logical_cpus=16,
            total_ram_mib=31652,
            machine_model="Dell Pro 5 14 P514260",
        )
        self.assertEqual(
            profile_fingerprint(base, model_sha256="m", backend_id="b9551"),
            profile_fingerprint(with_model, model_sha256="m", backend_id="b9551"),
        )


class ServerParserRepackFlag(unittest.TestCase):
    def test_server_parser_accepts_repack_and_defaults_auto(self):
        from orbit.native_server.app import build_parser

        parser = build_parser()
        self.assertEqual(parser.parse_args([]).repack, "auto")
        self.assertEqual(parser.parse_args(["--repack", "off"]).repack, "off")
        self.assertEqual(parser.parse_args(["--repack", "on"]).repack, "on")


class ClientConfigRepackField(unittest.TestCase):
    def test_native_client_config_defaults_use_extra_bufts_none(self):
        from orbit.native_llama.client import NativeClientConfig

        self.assertIsNone(NativeClientConfig().use_extra_bufts)


if __name__ == "__main__":
    unittest.main()
