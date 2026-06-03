from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.core.ollama_client import OllamaClient


class ClientTests(unittest.TestCase):
    def test_chat_preserves_image_messages(self) -> None:
        client = object.__new__(OllamaClient)
        client.model = "gemma4:e2b"
        captured: dict[str, object] = {}

        def fake_chat(payload):
            captured.update(payload)
            return {"message": {"content": "ok"}}

        client._chat = fake_chat  # type: ignore[method-assign]
        response = client.chat(
            messages=[{"role": "user", "content": "describe this image", "images": ["YWJjZA=="]}],
            tools=[],
            options={"temperature": 0.0},
            think=False,
        )

        self.assertEqual(response["message"]["content"], "ok")
        self.assertEqual(captured["messages"][0]["images"], ["YWJjZA=="])

    def test_extract_context_window_prefers_num_ctx_from_parameters(self) -> None:
        value = OllamaClient._extract_context_window(
            {
                "parameters": "num_thread 6\nnum_ctx 16384",
                "model_info": {"gemma4.context_length": 131072},
            }
        )
        self.assertEqual(value, 16384)

    def test_extract_context_window_from_model_info(self) -> None:
        value = OllamaClient._extract_context_window(
            {"model_info": {"gemma4.context_length": 32768}}
        )
        self.assertEqual(value, 32768)

    def test_extract_context_window_from_modelinfo(self) -> None:
        value = OllamaClient._extract_context_window(
            {"modelinfo": {"gemma4.context_length": 32768}}
        )
        self.assertEqual(value, 32768)

    def test_extract_running_model_prefers_model_field(self) -> None:
        value = OllamaClient._extract_running_model(
            {"models": [{"model": "gemma4:26b", "name": "other"}]}
        )
        self.assertEqual(value, "gemma4:26b")

    def test_extract_running_model_falls_back_to_name(self) -> None:
        value = OllamaClient._extract_running_model(
            {"models": [{"name": "gemma4:26b"}]}
        )
        self.assertEqual(value, "gemma4:26b")

    def test_extract_capabilities(self) -> None:
        value = OllamaClient._extract_capabilities({"capabilities": ["completion", "tools", "vision"]})
        self.assertEqual(value, ("completion", "tools", "vision"))

    def test_extract_tools_supported(self) -> None:
        self.assertTrue(OllamaClient._extract_tools_supported(("completion", "tools")))
        self.assertFalse(OllamaClient._extract_tools_supported(("completion", "vision")))
        self.assertIsNone(OllamaClient._extract_tools_supported(()))
