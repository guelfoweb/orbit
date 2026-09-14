"""CHAT-LOCAL-CAPABILITY-AWARENESS-1.

When a CHAT request asks for facts that Orbit can directly observe through its
existing local tools (host OS/CPU/RAM/storage/configuration), the route contract
must treat that as a locally-observable tool task -- so the model observes the
host instead of replying that it "cannot see your computer".

These tests are model-free: they pin the capability exposure and the route/tool
contract, not model output. They prove the fix is capability-driven (tool
exposure + generic prompt guidance), never a keyword/regex intent router, and
that shell safety and the ANALYSIS contract are untouched.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.runtime.messages import (
    _COMMAND_SYSTEM_TEMPLATE,
    CHAT_SYSTEM_PROMPT,
    ROUTE_SYSTEM_PROMPT,
)
from orbit.runtime.shell_guardrails import (
    exec_shell_full_definition,
    validate_tool_no_mutation_policy,
)
from orbit.runtime.system_info import system_info_definition
from orbit.runtime.tools import default_tool_names, tool_definitions
from orbit.terminal.tool_mode import allowed_tool_names_for_spec


class LocalHostCapabilityExposureTests(unittest.TestCase):
    def test_t1_tools_on_exposes_system_info_and_shell(self) -> None:
        """T1: local shell/system capability is exposed in CHAT when tools are enabled."""
        allowed = allowed_tool_names_for_spec("on")
        assert allowed is not None
        self.assertIn("system_info", allowed)
        self.assertIn("exec_shell_full_command", allowed)
        exposed = {item["function"]["name"] for item in tool_definitions(allowed)}
        self.assertIn("system_info", exposed)
        self.assertIn("exec_shell_full_command", exposed)

    def test_t2_system_info_description_identifies_local_host_observation(self) -> None:
        """T2: the system_info description clearly identifies local-host observation."""
        description = system_info_definition()["function"]["description"].lower()
        self.assertIn("local machine", description)
        for dimension in ("os", "cpu", "ram", "disk"):
            self.assertIn(dimension, description)

    def test_t2_shell_description_identifies_local_execution_and_read(self) -> None:
        """T2: the shell description identifies local execution and read capability.

        The shell is unrestricted (read/write/execute); this only pins that it names
        local execution and read capability, which covers read-only host inspection.
        """
        description = exec_shell_full_definition()["function"]["description"].lower()
        self.assertIn("local", description)
        self.assertIn("read", description)

    def test_t2_route_contract_binds_host_inspection_to_a_tool_decision(self) -> None:
        """T2: the route contract treats inspecting this host as a tool task."""
        self.assertIn("inspect this host itself", ROUTE_SYSTEM_PROMPT)
        self.assertIn("locally observable tool tasks", ROUTE_SYSTEM_PROMPT)
        self.assertIn("system_info", ROUTE_SYSTEM_PROMPT)
        self.assertIn(
            "never a direct answer or a reply that you cannot see the machine",
            ROUTE_SYSTEM_PROMPT,
        )

    def test_t2_route_contract_demonstrates_host_spec_to_system_info_mapping(self) -> None:
        """T2: an example maps a host-spec request to the system_info shape.

        Prose alone did not flip the router's strong "assistants cannot see your
        computer" prior; a request -> decision example (mirroring the existing
        "summarize README.md -> {command}" example) is the effective, capability-driven
        lever. Pin it so the demonstration cannot silently regress.
        """
        self.assertIn("Example local machine spec request:", ROUTE_SYSTEM_PROMPT)
        self.assertIn(
            '{"include_cpu":true,"include_memory":true,"include_disks":true,"include_os":true}',
            ROUTE_SYSTEM_PROMPT,
        )

    def test_t3_tools_disabled_advertises_no_local_capability(self) -> None:
        """T3: tools-disabled mode exposes no tools and makes no observation promise."""
        self.assertEqual(allowed_tool_names_for_spec("off"), ())
        # The no-tool CHAT final prompt must not claim tool/observation abilities.
        lowered = CHAT_SYSTEM_PROMPT.lower()
        self.assertNotIn("system_info", lowered)
        self.assertNotIn("shell", lowered)
        self.assertNotIn("inspect", lowered)

    def test_t4_shell_no_mutation_policy_unchanged(self) -> None:
        """T4: read-only mutation guard still blocks mutation and allows observation."""
        blocked = validate_tool_no_mutation_policy(
            "exec_shell_full_command",
            {"command": "rm -rf build"},
            user_prompt="tell me configuration about this computer",
            workdir=Path.cwd(),
        )
        self.assertIsNotNone(blocked)
        self.assertIn("mutating shell command", blocked or "")
        allowed = validate_tool_no_mutation_policy(
            "exec_shell_full_command",
            {"command": "uname -a"},
            user_prompt="tell me configuration about this computer",
            workdir=Path.cwd(),
        )
        self.assertIsNone(allowed)

    def test_t5_analysis_routing_contract_unchanged(self) -> None:
        """T5: the ANALYSIS route contract is untouched by this change."""
        self.assertIn(
            "Use ANALYSIS only when the request asks to investigate one named local artifact",
            ROUTE_SYSTEM_PROMPT,
        )
        self.assertIn('{"route":"ANALYSIS","artifact":"samples/foo.js"}', ROUTE_SYSTEM_PROMPT)
        # The clarification must not turn host questions into ANALYSIS.
        self.assertIn("locally observable tool tasks", ROUTE_SYSTEM_PROMPT)

    def test_t6_ordinary_knowledge_paths_preserved(self) -> None:
        """T6: direct-answer / CHAT / recap paths remain, so general knowledge is not forced into tools."""
        self.assertIn('return {"route":"CHAT"} only', ROUTE_SYSTEM_PROMPT)
        self.assertIn("write the answer directly and stop", ROUTE_SYSTEM_PROMPT)
        self.assertIn(
            "recap, repeat, summary, explanation, comparison, or continuation",
            ROUTE_SYSTEM_PROMPT,
        )
        # The clarification is scoped to inspecting THIS HOST, not general knowledge.
        self.assertIn("this host itself", ROUTE_SYSTEM_PROMPT)

    def test_t6_host_specs_are_not_forced_through_shell(self) -> None:
        """T6: host specs prefer system_info but do not force shell."""
        self.assertIn("system_info, or a read-only shell command", ROUTE_SYSTEM_PROMPT)

    def test_t6_host_clause_defers_to_already_known_context(self) -> None:
        """T6: the host clause does not force re-observation of state already in context.

        A host fact already present in the conversation (e.g. after system_info ran)
        stays answerable from context via the recap rule; the clause is conditioned
        on the state not already being known.
        """
        self.assertIn(
            "are locally observable tool tasks unless that state is already in the conversation",
            ROUTE_SYSTEM_PROMPT,
        )
        # The recap-from-context CHAT preference remains ahead of it.
        self.assertLess(
            ROUTE_SYSTEM_PROMPT.index("prior context is sufficient"),
            ROUTE_SYSTEM_PROMPT.index("inspect this host itself"),
        )

    def test_t7_capability_driven_not_keyword_router(self) -> None:
        """T7: the fix is tool-exposure + generic prompt guidance, not a new keyword/regex router.

        The capability is always exposed (system_info offered on every tools-on turn)
        and the routing guidance is model-facing prompt text, not a code branch keyed on
        the user's words. This does not claim chat.py is regex-free (a pre-existing
        length-only retry regex mentions "computer|system|specs"); it pins that THIS fix
        added no such router and did not hardcode the user's phrasing in code.
        """
        # Capability exposed generically, independent of wording.
        self.assertIn("system_info", default_tool_names())
        allowed = allowed_tool_names_for_spec("on")
        assert allowed is not None
        self.assertIn("system_info", allowed)
        # The guidance lives in the model-facing route template, not a code branch.
        self.assertIn("locally observable tool tasks", ROUTE_SYSTEM_PROMPT)
        self.assertIn("locally observable tool tasks", _COMMAND_SYSTEM_TEMPLATE)
        # This fix did not add a code-level router keyed on the user's phrasing.
        chat_source = (SRC / "orbit" / "runtime" / "chat.py").read_text(encoding="utf-8").lower()
        self.assertNotIn("configuration", chat_source)

    def test_t8_tool_schema_serialization_stable(self) -> None:
        """T8: the tool set / schemas are unchanged; only the route prompt was clarified."""
        exposed = {item["function"]["name"] for item in tool_definitions()}
        self.assertEqual(exposed, set(default_tool_names()))
        schema = system_info_definition()["function"]["parameters"]
        self.assertEqual(schema["additionalProperties"], False)
        self.assertEqual(schema["required"], [])
        self.assertEqual(
            set(schema["properties"]),
            {
                "include_disks",
                "include_cpu",
                "include_memory",
                "include_os",
                "include_runtime",
                "include_gpu",
                "human_readable",
            },
        )


if __name__ == "__main__":
    unittest.main()
