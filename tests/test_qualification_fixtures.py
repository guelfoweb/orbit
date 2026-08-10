from __future__ import annotations

import json
import unittest
from pathlib import Path

from orbit.qualification.fixtures import FixtureError, load_fixture_text


def document() -> dict[str, object]:
    return {"schema_version": 1, "fixtures": [{
        "name": "simple_chat", "capability": "chat", "profiles": ["profile-a"],
        "request": {"prompt": "Reply exactly OK", "tools": False},
        "expect": {"finish_reason": "stop", "max_model_calls": 1, "exact_output": "OK"},
        "parity": {"mode": "exact"},
    }]}


class QualificationFixtureTests(unittest.TestCase):
    def test_valid_schema_and_canonical_hash(self) -> None:
        compact = load_fixture_text(json.dumps(document(), separators=(",", ":")))
        formatted = load_fixture_text(json.dumps(document(), indent=2, sort_keys=True))
        self.assertEqual(compact.content_hash, formatted.content_hash)
        self.assertEqual(compact.fixtures[0].fixture_hash, formatted.fixtures[0].fixture_hash)
        self.assertEqual(compact.fixtures[0].request.prompt, "Reply exactly OK")

    def test_duplicate_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(FixtureError, "duplicate_key"):
            load_fixture_text('{"schema_version":1,"schema_version":1,"fixtures":[]}')

    def test_unknown_keys_are_rejected_at_each_level(self) -> None:
        for path in ("root", "request"):
            value = document()
            if path == "root":
                value["future"] = True
            else:
                value["fixtures"][0]["request"]["temperature"] = 0  # type: ignore[index]
            with self.subTest(path=path), self.assertRaisesRegex(FixtureError, "unknown_key"):
                load_fixture_text(json.dumps(value))

    def test_missing_required_field_is_rejected(self) -> None:
        payload = document()
        del payload["fixtures"][0]["request"]["prompt"]  # type: ignore[index]
        with self.assertRaisesRegex(FixtureError, "missing_key"):
            load_fixture_text(json.dumps(payload))

    def test_bad_version_type_and_value_are_rejected(self) -> None:
        for value, reason in ((2, "unsupported_schema_version"), (True, "invalid_type")):
            payload = document()
            payload["schema_version"] = value
            with self.subTest(value=value), self.assertRaisesRegex(FixtureError, reason):
                load_fixture_text(json.dumps(payload))

    def test_invalid_field_types_are_rejected(self) -> None:
        payload = document()
        payload["fixtures"][0]["expect"]["max_model_calls"] = True  # type: ignore[index]
        with self.assertRaisesRegex(FixtureError, "invalid_type"):
            load_fixture_text(json.dumps(payload))

    def test_invalid_parity_duplicate_profile_and_nan_are_rejected(self) -> None:
        payload = document()
        payload["fixtures"][0]["parity"]["mode"] = "semantic"  # type: ignore[index]
        with self.assertRaisesRegex(FixtureError, "invalid_parity_mode"):
            load_fixture_text(json.dumps(payload))
        payload = document()
        payload["fixtures"][0]["profiles"] = ["profile-a", "profile-a"]  # type: ignore[index]
        with self.assertRaisesRegex(FixtureError, "duplicate_profile"):
            load_fixture_text(json.dumps(payload))
        for constant in ("NaN", "Infinity", "-Infinity"):
            raw = json.dumps(document()).replace('"max_model_calls": 1', f'"max_model_calls": {constant}')
            with self.subTest(constant=constant), self.assertRaisesRegex(FixtureError, "invalid_constant"):
                load_fixture_text(raw)

    def test_fixture_names_and_artifact_paths_are_confined(self) -> None:
        payload = document()
        payload["fixtures"][0]["name"] = "../escape"  # type: ignore[index]
        with self.assertRaisesRegex(FixtureError, "invalid_value"):
            load_fixture_text(json.dumps(payload))
        payload = json.loads(
            (Path(__file__).parents[1] / "qualification/fixtures/core-v1.json").read_text()
        )
        for invalid in ("../escape.json", "/tmp/escape.json", ".", "bad\x00name"):
            payload["fixtures"][3]["expect"]["artifact"]["path"] = invalid
            with self.subTest(path=invalid), self.assertRaisesRegex(FixtureError, "invalid_value"):
                load_fixture_text(json.dumps(payload))

    def test_capability_names_and_fixture_contracts_are_fail_closed(self) -> None:
        payload = document()
        payload["fixtures"][0]["capability"] = "chta"  # type: ignore[index]
        with self.assertRaisesRegex(FixtureError, "unsupported_capability"):
            load_fixture_text(json.dumps(payload))
        payload = document()
        payload["fixtures"][0]["request"]["tools"] = True  # type: ignore[index]
        with self.assertRaisesRegex(FixtureError, "invalid_fixture_contract"):
            load_fixture_text(json.dumps(payload))


if __name__ == "__main__":
    unittest.main()
