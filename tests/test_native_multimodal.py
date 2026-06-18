from __future__ import annotations

import base64
import unittest

from orbit.native_llama.multimodal import flatten_message_content, prepare_multimodal_messages


class NativeMultimodalTests(unittest.TestCase):
    def test_prepare_multimodal_messages_replaces_image_with_marker_and_decodes_payload(self) -> None:
        raw = b"fake-image"
        prepared = prepare_multimodal_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(raw).decode('ascii')}"}},
                    ],
                }
            ],
            media_marker="<__media__>",
        )

        self.assertIsNotNone(prepared)
        assert prepared is not None
        self.assertTrue(prepared.has_image)
        self.assertFalse(prepared.has_audio)
        self.assertEqual(prepared.media_payloads, [raw])
        self.assertEqual(
            prepared.messages[0]["content"],
            [
                {"type": "text", "text": "describe"},
                {"type": "media_marker", "text": "<__media__>"},
            ],
        )

    def test_prepare_multimodal_messages_replaces_audio_with_marker_and_decodes_payload(self) -> None:
        raw = b"fake-audio"
        prepared = prepare_multimodal_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "transcribe"},
                        {"type": "input_audio", "input_audio": {"data": base64.b64encode(raw).decode("ascii"), "format": "wav"}},
                    ],
                }
            ],
            media_marker="<__media__>",
        )

        self.assertIsNotNone(prepared)
        assert prepared is not None
        self.assertFalse(prepared.has_image)
        self.assertTrue(prepared.has_audio)
        self.assertEqual(prepared.media_payloads, [raw])

    def test_flatten_message_content_keeps_marker_text(self) -> None:
        text = flatten_message_content(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "media_marker", "text": "<__media__>"},
                    {"type": "text", "text": "now"},
                ],
            }
        )

        self.assertEqual(text, "look <__media__> now")


if __name__ == "__main__":
    unittest.main()
