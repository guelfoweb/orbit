"""Human status must not turn missing remote facts into positive claims."""
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
import unittest

from orbit.terminal.config import AppConfig
from orbit.terminal.runtime_status import (
    HostInfo, collect_runtime_status, format_status_panel,
)
from tests.test_runtime_status import _Runtime


class StatusObservabilitySemanticsTests(unittest.TestCase):
    def collect(self, props):
        backend = SimpleNamespace(
            backend_props=lambda: props, model_info=lambda: None,
            health=lambda: True,
        )
        return collect_runtime_status(
            _Runtime(), AppConfig(), backend, host_info=HostInfo(),
        )

    def test_absent_or_malformed_external_metadata_is_not_cpu_only(self):
        for props in ({}, None, [], 'unavailable', {'accel': True},
                      {'gpu_enabled': False}, {'cuda': False},
                      {'backend': 'orbit-native'}):
            with self.subTest(props=props):
                status = self.collect(props)
                self.assertEqual(status.acceleration.mode, 'unknown')
                self.assertNotIn('CPU-only', format_status_panel(status))

    def test_explicit_modes_remain_visible(self):
        for key in ('accel', 'accelerator', 'acceleration', 'gpu_backend'):
            for mode in ('CPU-only', 'CUDA', 'Vulkan'):
                with self.subTest(key=key, mode=mode):
                    status = self.collect({key: mode})
                    self.assertEqual(status.acceleration.mode, mode)
                    self.assertIn(mode, format_status_panel(status))

    def test_invalid_mode_alias_does_not_hide_a_valid_one(self):
        self.assertEqual(
            self.collect({'accel': True, 'accelerator': 'CUDA'}).acceleration.mode,
            'CUDA',
        )

    def test_zero_and_positive_offload_preserve_all_aliases(self):
        for key in ('offload_layers', 'gpu_layers', 'n_gpu_layers'):
            for count in (0, 24):
                with self.subTest(key=key, count=count):
                    status = self.collect({key: count})
                    self.assertEqual(status.acceleration.offload, f'{count} layers')
                    self.assertIn(f'{count} layers', format_status_panel(status))

    def test_known_zero_is_not_replaced_by_lower_priority_alias(self):
        status = self.collect({'offload_layers': 0, 'gpu_layers': 12, 'n_gpu_layers': 24})
        self.assertEqual(status.acceleration.offload, '0 layers')

    def test_invalid_counts_are_not_observed_layers(self):
        for value in (True, False, -1, 1.5, None, {}, []):
            with self.subTest(value=value):
                self.assertEqual(self.collect({'gpu_layers': value}).acceleration.offload, 'unknown')
        self.assertEqual(self.collect({'offload_layers': False, 'gpu_layers': 3}).acceleration.offload, '3 layers')

    def test_textual_offload_diagnostic_remains_available(self):
        self.assertEqual(self.collect({'offload_layers': 'all layers'}).acceleration.offload, 'all layers')

    def test_vram_counts_are_nonnegative_integers_not_booleans(self):
        for value in (True, False, -1, -1024, '1024', None):
            with self.subTest(value=value):
                info = self.collect({'vram_total': value, 'vram_available': value}).acceleration
                self.assertEqual(info.vram_total, 'unknown')
                self.assertEqual(info.vram_available, 'unknown')
        info = self.collect({'vram_total': 0, 'vram_available': 8 * 1024**3}).acceleration
        self.assertEqual(info.vram_total, '0 B')
        self.assertEqual(info.vram_available, '8 GB')
        self.assertEqual(self.collect({'vram_total': True, 'gpu_vram_total': 1024**3}).acceleration.vram_total, '1 GB')

    def test_multimodal_flag_reports_availability_not_initialization(self):
        for value, expected in ((True, 'available'), (False, 'unavailable'),
                                (None, 'unknown'), ('false', 'unknown')):
            with self.subTest(value=value):
                status = self.collect({'mtp_enabled': False, 'multimodal_available': value})
                self.assertEqual(status.mmproj, expected)
                self.assertIn(f'off, mmproj {expected}', format_status_panel(status))

    def test_collection_does_not_modify_metadata_or_client_state(self):
        props = {'accel': 'CUDA', 'offload_layers': 0, 'multimodal_available': True,
                 'orbit_build': {'commit': 'a' * 40}, 'cached_tokens': 123}
        saved = deepcopy(props)
        runtime = _Runtime()
        before = deepcopy(vars(runtime))
        config = AppConfig()
        config_before = asdict(config)
        backend = SimpleNamespace(backend_props=lambda: props, model_info=lambda: None, health=lambda: True)
        status = collect_runtime_status(runtime, config, backend, host_info=HostInfo())
        self.assertEqual(status.server_build.commit, 'a' * 40)
        self.assertEqual(props, saved)
        self.assertEqual(vars(runtime), before)
        self.assertEqual(asdict(config), config_before)


if __name__ == '__main__':
    unittest.main()
