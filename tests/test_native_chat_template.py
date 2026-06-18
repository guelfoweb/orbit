from __future__ import annotations

import unittest

from orbit.native_llama.chat_template import render_gemma4_chat


class NativeChatTemplateTests(unittest.TestCase):
    def test_renders_system_and_user_turns_with_thinking_prompt(self) -> None:
        prompt = render_gemma4_chat(
            [
                {"role": "system", "content": "Answer normally."},
                {"role": "user", "content": "hello"},
            ],
            thinking=True,
        )

        self.assertTrue(prompt.startswith("<bos><|turn>system\n<|think|>\nAnswer normally.<turn|>\n"))
        self.assertIn("<|turn>user\nhello<turn|>\n", prompt)
        self.assertIn("<|turn>system\n<|think|>\nAnswer normally.<turn|>\n", prompt)
        self.assertTrue(prompt.endswith("<|turn>model\n"))

    def test_renders_system_and_user_turns_without_thinking_prompt_by_default(self) -> None:
        prompt = render_gemma4_chat(
            [
                {"role": "system", "content": "Answer normally."},
                {"role": "user", "content": "hello"},
            ]
        )

        self.assertTrue(prompt.startswith("<bos><|turn>system\nAnswer normally.<turn|>\n"))
        self.assertIn("<|turn>user\nhello<turn|>\n", prompt)
        self.assertTrue(prompt.endswith("<|turn>model\n<|channel>thought\n<channel|>"))
        self.assertNotIn("<|think|>", prompt)

    def test_maps_assistant_to_model_role(self) -> None:
        prompt = render_gemma4_chat([{"role": "assistant", "content": "done"}])

        self.assertIn("<|turn>model\ndone<turn|>", prompt)

    def test_renders_tool_call_arguments_in_gemma4_format(self) -> None:
        prompt = render_gemma4_chat(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "exec_shell_full_command",
                                "arguments": '{"command":"ls -F"}',
                            },
                        }
                    ],
                }
            ]
        )

        self.assertIn('<|tool_call>call:exec_shell_full_command{command:<|"|>ls -F<|"|>}<tool_call|>', prompt)

    def test_renders_tool_response(self) -> None:
        prompt = render_gemma4_chat(
            [
                {
                    "role": "tool",
                    "name": "exec_shell_full_command",
                    "content": "README.md",
                }
            ]
        )

        self.assertIn(
            '<|tool_response>response:exec_shell_full_command{value:<|"|>README.md<|"|>}<tool_response|>',
            prompt,
        )

    def test_renders_available_tool_schema_in_system_turn(self) -> None:
        prompt = render_gemma4_chat(
            [
                {"role": "system", "content": "Answer normally."},
                {"role": "user", "content": "hello"},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "exec_shell_full_command",
                        "description": "Unrestricted local shell.",
                        "parameters": {
                            "type": "object",
                            "properties": {"command": {"type": "string"}},
                            "required": ["command"],
                        },
                    },
                }
            ],
        )

        self.assertIn("<|tool>declaration:exec_shell_full_command{", prompt)
        self.assertIn('description:<|"|>Unrestricted local shell.<|"|>', prompt)
        self.assertIn('command:{type:<|"|>STRING<|"|>}', prompt)
        self.assertIn("<tool|>", prompt)

    def test_strips_thinking_channel_from_assistant_content(self) -> None:
        prompt = render_gemma4_chat(
            [
                {
                    "role": "assistant",
                    "content": "visible<|channel>thought\nhidden<channel|>",
                }
            ]
        )

        self.assertIn("visible<turn|>", prompt)
        self.assertNotIn("hidden", prompt)

    def test_rendered_prompt_can_include_thinking_off_policy_as_plain_system_text(self) -> None:
        prompt = render_gemma4_chat(
            [
                {
                    "role": "system",
                    "content": "Answer normally.\n\nThinking mode is off. Do not reveal chain-of-thought.",
                },
                {"role": "user", "content": "hello"},
            ]
        )

        self.assertIn("Thinking mode is off. Do not reveal chain-of-thought.", prompt)


if __name__ == "__main__":
    unittest.main()
