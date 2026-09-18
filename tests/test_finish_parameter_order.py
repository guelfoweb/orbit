"""A complete FINISH call must not disappear because status was emitted last."""
import copy
import itertools
import json
from types import SimpleNamespace
import unittest
from unittest import mock

from orbit.native_llama.client import NativeLlamaClient, _ProfileParsedOutput
from orbit.native_llama.events import NativeTimings
from orbit.native_llama.finish_parameter_order import (
    reorder_finish_parameters, exact_finish_arguments,
)
from orbit.runtime.analysis_controller import ControlError, parse_finish_call
from orbit.runtime.analysis_runtime import FINISH_TOOL_SCHEMA, PLAN_TOOL_SCHEMA


def wire(values):
    return ("<tool_call>\n<function=finish_analysis_question>\n"
            + "".join(f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in values)
            + "</function>\n</tool_call>")


VALUES = [("answer_summary", 'Exact "quoted" evidence; still unknown.\nSecond line.'),
          ("evidence_ids", '["ev_one", "ev_two"]'), ("status", "still_open")]
EXPECTED = {"answer_summary": VALUES[0][1], "evidence_ids": ["ev_one", "ev_two"], "status": "still_open"}
TOOLS = [FINISH_TOOL_SCHEMA]


def calls(arguments):
    return ({"type": "function", "function": {"name": "finish_analysis_question", "arguments": json.dumps(arguments)}},)


class FinishParameterOrderTests(unittest.TestCase):
    def test_all_parameter_orders_preserve_complete_blocks_and_values(self):
        for values in itertools.permutations(VALUES):
            with self.subTest(order=[x[0] for x in values]):
                raw = wire(values)
                result = reorder_finish_parameters(raw, TOOLS)
                if values[0][0] == "status":
                    self.assertIsNone(result)
                else:
                    ordered, expected = result
                    self.assertEqual(expected, EXPECTED)
                    self.assertEqual(ordered, wire([VALUES[2], *[x for x in values if x[0] != "status"]]))

    def test_child_dependency_is_preserved_and_remains_open(self):
        child = {"question": "Unknown?", "missing_fact": "bytes", "caused_by_evidence_id": "ev_one"}
        ordered, expected = reorder_finish_parameters(wire([("child_question", json.dumps(child)), *VALUES]), TOOLS)
        self.assertEqual(expected["child_question"], child)
        self.assertEqual(parse_finish_call(expected)["status"], "open")
        self.assertTrue(ordered.startswith("<tool_call>\n<function=finish_analysis_question>\n<parameter=status>"))

    def test_ambiguous_or_incomplete_wire_fails_closed(self):
        raw = wire(VALUES)
        invalid = [
            "prose\n" + raw, raw + "\nprose", raw + raw, raw[:-1],
            wire(VALUES[:-1]), wire([*VALUES, VALUES[0]]),
            wire([("unknown", "x"), *VALUES]),
            wire([(VALUES[0][0], "nested <parameter=status>"), *VALUES[1:]]),
            raw.replace("</function>\n", "\n</function>\n"),
            raw.replace("<function=finish_analysis_question>", "<function=execute_analysis>"),
        ]
        for item in invalid:
            with self.subTest(raw=item):
                self.assertIsNone(reorder_finish_parameters(item, TOOLS))

    def test_json_parameters_are_strict_and_not_repaired(self):
        for value in ('{"x":1,"x":2}', '{"x":NaN}', 'null', '["id",]', '"id"'):
            self.assertIsNone(reorder_finish_parameters(wire([("evidence_ids", value), VALUES[2]]), TOOLS))
        duplicate_child = '{"question":"a","question":"b","missing_fact":"x","caused_by_evidence_id":"e"}'
        self.assertIsNone(reorder_finish_parameters(wire([("child_question", duplicate_child), VALUES[2]]), TOOLS))

    def test_scope_is_exactly_one_finish_schema(self):
        for tools in (None, [], [PLAN_TOOL_SCHEMA], [FINISH_TOOL_SCHEMA, PLAN_TOOL_SCHEMA]):
            self.assertIsNone(reorder_finish_parameters(wire(VALUES), tools))
        schema = copy.deepcopy(FINISH_TOOL_SCHEMA)
        schema["function"]["parameters"]["required"] = ["missing"]
        self.assertIsNone(reorder_finish_parameters(wire(VALUES), [schema]))

    def test_reparse_must_retain_every_field_value_and_type(self):
        self.assertTrue(exact_finish_arguments(calls(EXPECTED), EXPECTED))
        for changed in ({**EXPECTED, "status": "resolved"},
                        {k:v for k,v in EXPECTED.items() if k != "evidence_ids"},
                        {**EXPECTED, "evidence_ids": ["ev_two", "ev_one"]},
                        {**EXPECTED, "extra": "x"}):
            self.assertFalse(exact_finish_arguments(calls(changed), EXPECTED))
        self.assertFalse(exact_finish_arguments(calls({"x": True}), {"x": 1}))
        self.assertFalse(exact_finish_arguments(calls(EXPECTED) * 2, EXPECTED))
        self.assertFalse(exact_finish_arguments((), EXPECTED))

    def test_completion_validation_is_not_relaxed(self):
        for values in ([('answer_summary', ''), ('status', 'resolved')],
                       [('answer_summary', 'x'), ('status', 'invented')],
                       [('evidence_ids', '[1]'), ('status', 'still_open')],
                       [('child_question', '{}'), ('answer_summary', 'x'), ('status', 'resolved')]):
            _, expected = reorder_finish_parameters(wire(values), TOOLS)
            with self.assertRaises(ControlError):
                parse_finish_call(expected)


class FinishNativeIntegrationTests(unittest.TestCase):
    def complete(self, *, cancelled=False, output_tokens=20, tools=TOOLS,
                 protocol="qwen3.6-xml", second=None, initial=None):
        client = NativeLlamaClient.__new__(NativeLlamaClient)
        client.model_profile = SimpleNamespace(tool_call_protocol=protocol)
        client._session = SimpleNamespace(continuation_ready=False)
        client._content_token_count = lambda text: 0
        raw = wire(VALUES)
        def generate(messages, **kwargs):
            kwargs["on_token"](raw)
            return NativeTimings(1774, output_tokens, 0, 1774, 1, 1, cancelled)
        client.complete_chat = generate
        empty = _ProfileParsedOutput("", "", ())
        client._parse_profile_output = mock.Mock(side_effect=[
            *([] if tools else [empty]), initial or empty,
            second or _ProfileParsedOutput("", "", calls(EXPECTED)),
        ])
        result = client._complete_profile_chat_text_once(
            [], max_tokens=2048, stop=(), tools=tools, thinking=False,
            route_prefix_anchor=False, qwen_route_prefix_anchor=False,
            qwen36_shell_tool_prefix_anchor=False, allow_mtp_experimental=None,
            final_prefix_experiment=False,
        )
        self.assertEqual(client._profile_last_raw_output, raw)
        return result, client._parse_profile_output

    def test_empty_native_parse_retries_only_lossless_ordering(self):
        result, parser = self.complete()
        self.assertEqual(result.tool_calls, calls(EXPECTED))
        self.assertEqual(parser.call_count, 2)
        self.assertEqual(parser.call_args.args[0], wire([VALUES[2], *VALUES[:2]]))
        self.assertEqual(parse_finish_call(json.loads(result.tool_calls[0]['function']['arguments']))['status'], 'open')

    def test_cancel_length_other_phases_and_protocol_never_reparse(self):
        for kwargs in ({"cancelled": True}, {"output_tokens": 2048},
                       {"tools": [PLAN_TOOL_SCHEMA]}, {"tools": []},
                       {"protocol": "other"}):
            with self.subTest(kwargs=kwargs):
                result, parser = self.complete(**kwargs)
                self.assertFalse(result.tool_calls)
                self.assertEqual(parser.call_count, 2 if kwargs.get("tools") == [] else 1)

    def test_existing_valid_parse_is_unchanged(self):
        initial = _ProfileParsedOutput("", "", calls(EXPECTED))
        result, parser = self.complete(initial=initial)
        self.assertEqual(result.tool_calls, initial.tool_calls)
        self.assertEqual(parser.call_count, 1)

    def test_changed_reparse_or_native_failure_remains_unusable(self):
        for second in (_ProfileParsedOutput("", "", calls({**EXPECTED, "status": "resolved"})),
                       _ProfileParsedOutput("extra prose", "", calls(EXPECTED)),
                       RuntimeError("invalid native parse")):
            result, parser = self.complete(second=second)
            self.assertFalse(result.tool_calls)
            self.assertEqual(parser.call_count, 2)
