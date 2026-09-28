# Tool-intent qualification gate

This Qualification Harness gate observes the normal `ChatRuntime.ask_auto`
path until its first authorized tool handoff or CHAT decision. The executor
is substituted: no shell, fetch, system inspection, artifact publication or
model-selected file operation runs. No final answer is generated. Routing,
production instructions, parsers, schemas and policy remain unchanged.

## Frozen corpus

[`tool-intent-v1.json`](../qualification/fixtures/tool-intent-v1.json) is the
byte-identical 24-case corpus from `route_intent_preservation_1/corpus.json`:
SHA256 `de9f6e5d594f24b0e235ed4a707bc7d836b5bffafd9df007c935d8144b838a9f`.
Its cases and expectations were checked against all 72 A/B/C observations in
the retained `mimo_route_contract_1/resumed_results.json`. It contains:

- 12 content requests and 12 operational requests, in Italian and English;
- code generation, explanations, email prose, explicit execution, mixed
  generation/execution, inspection, modification and history-dependent requests;
- the original 2 design, 16 held-out and 6 regression cases.

Each entry preserves `id`, `split`, `language`, `category`, `prompt`, `expected`,
`allowed_tools` and `history`. Expected mode and permitted tool classes are
test oracles, used only **after** the runtime decision. They never select tools,
modify requests or classify production user intent. Changing the oracle
requires a separately versioned corpus; the loader rejects silent edits.

## Offline replay

From a source checkout:

```bash
TMPDIR=/tmp python3 scripts/orbit_qualify_tool_intent.py \
  --replay qualification/fixtures/tool-intent-recorded-v1.json \
  --output /tmp/tool-intent.json
```

This writes JSON and a sibling Markdown table. Exit 0 means capability PASS;
exit 1 means capability FAIL. The supplied recording **expects FAIL**: it
preserves the 24 initial responses of historical contract A, including 11
requested executions that never reached an authorized handoff. This is an
offline regression test, not a new MiMo qualification or production enablement.

Each recorded call contains its raw result, ordered stream deltas, exact
tokenizer admission record, phase and request digests, source artifact path/hash
and historical metrics. Replay checks the messages, tools, temperature, cap and
phase before consuming a response. A changed production request fails closed;
the recording must not be silently rebound to it. Stored metrics are observations
of the original run, not new tokenizer measurements.

The historical recorder collected complete bounded streams before replaying
them into the production consumer. Its wall time therefore is **not** current
early-abort latency. JSON keeps `recorded_wall_seconds` separate from the current
probe's `wall_seconds`; unavailable token counts remain `null`.

## Future model qualification

Only when a live gate is explicitly authorized, use an independently managed,
idle Orbit qualification server with an already verified profile:

```bash
TMPDIR=/tmp python3 scripts/orbit_qualify_tool_intent.py \
  --base-url http://127.0.0.1:PORT --profile VERIFIED_PROFILE_ID \
  --output /tmp/tool-intent-live.json
```

The command starts no server and downloads no model. It verifies the exact
profile, records `/props`, resets the qualification server before each case,
uses greedy sampling and thinking off, and retains the production ROUTE/tool
limits and existing retries. The harness ceiling is six model calls per case;
it creates no retry loop. Use a dedicated server: its session reset must not
affect another client. Run the observer sequentially in its own process; its
temporary executor substitution is not intended for concurrent application use.

Every case gets a fresh runtime, the exact recorded conversation history and a
temporary workspace with benign `notes.txt`. Document-acquisition paths outside
the intercepted tool contract stop as DEFERRED/FAIL before reading content.
Canonical validation must be enabled. The substitute rechecks the same canonical
schema, allowed-tools and permission/policy decision before recording a handoff;
it never delegates to the real executor. Restoration also occurs on errors or
cancellation.

## Metrics and acceptance

JSON and Markdown report expected and observed mode, selected tool, false
execution, missed execution, wrong authorizable tool, invalid route/protocol,
retry count, model calls, input/output tokens and wall time. Raw requests,
responses and validation details remain in JSON.

PASS requires all 24 cases to reach a valid terminal decision, zero unwanted
authorized handoffs, zero missed operational requests and zero wrong authorized
tool classes. An invalid attempt recovered by the existing bounded repair is
recorded as `invalid_route_protocol` plus `recovered`; it does not erase the
first failure. A CHAT fallback after an invalid/truncated route is not qualified
CHAT. A runtime rejection message is not a model's content answer.

“Execution” metrics mean **would execute after authorization**: actual tool
executions are always zero. This gate checks the first handoff's class, not shell
argument semantics, generated content correctness, later actions or completion
of a mixed task. Those remain separate manual/integration qualification. A tool
with the correct name can still contain a wrong command. No LLM judge, keyword
scoring or model ranking is used.

## Deterministic checks

```bash
TMPDIR=/tmp PYTHONPATH=src python3 -m unittest tests.test_qualification_tool_intent -q
```

Positive scripted cases traverse the real runtime for all 24 intents. Negative
cases cover false/missed/wrong handoffs, invalid and incomplete completions,
policy/schema rejection, intercepted artifact publication, history isolation,
request identity and complete case coverage. Tests replace the real executor
and low-level dispatch with sentinels; reaching either fails the test.
