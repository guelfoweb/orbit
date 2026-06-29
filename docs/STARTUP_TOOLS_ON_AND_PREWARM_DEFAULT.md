# Startup Tools-On And Prewarm Defaults

## Summary

Orbit now starts in an agentic-ready default mode:

- tools are enabled by default;
- native route tools-on prefix prewarm is enabled by default at server startup;
- explicit configuration can disable either behavior.

This changes startup posture, not model policy. Orbit remains model-guided and
does not add deterministic routing, system prompt anchors, or multiple anchors.

## Tools Default

Default:

```bash
ORBIT_TOOLS=on
```

When `ORBIT_TOOLS` is unset, Orbit behaves as tools-on.

Disable tools for startup/session defaults:

```bash
ORBIT_TOOLS=off
```

CLI and interactive controls remain available:

```bash
orbit --tools off "prompt"
/tools off
/tools on
```

`/tools off` still disables tools in the current interactive session. `/tools
on` still re-enables them.

## Startup Route Prefix Prewarm Default

Default:

```bash
ORBIT_KV_PREFIX_PREWARM=startup
```

When `ORBIT_KV_PREFIX_PREWARM` is unset, native `orbit server` performs startup
prewarm for the stable route tools-on prefix, provided tools are enabled and the
route prefix-anchor is not disabled.

Disable startup prewarm only:

```bash
ORBIT_KV_PREFIX_PREWARM=off
```

Disable route prefix-anchor and startup prewarm:

```bash
ORBIT_KV_PREFIX_ANCHOR=off
```

If tools are disabled at startup with `ORBIT_TOOLS=off`, route tools-on prewarm
is skipped.

## Scope

Startup prewarm is limited to:

- native backend only;
- route tools-on prefix only;
- the existing native prefill-only hook;
- metadata-only diagnostics.

It does not implement:

- system prompt anchor;
- chat-final anchor;
- final-from-tool anchor;
- tool-call anchor;
- multiple anchors;
- prompt changes;
- routing policy changes;
- tool selection policy changes;
- final-answer policy changes;
- file/web/fetch evidence-policy changes.

Non-native compatibility backends do not use startup prewarm.

## Tradeoff

Startup prewarm does not remove the prefill cost. It moves that cost from the
first tools-on request to server startup.

Observed on the tested CPU-only Gemma 4 12B setup:

- startup prewarm cost: about 50-60 seconds;
- checkpoint size: about 238 MB;
- first tools-on cold request without prewarm: about 70 seconds;
- first tools-on request after startup prewarm: about 17 seconds.

It is like warming the oven before baking bread: the oven still needs time to
reach temperature, but the first bake no longer starts from cold.

## Failure And Fallback

If startup prewarm fails:

- the server remains usable;
- no user-facing error is required;
- the next real route call falls back to the baseline capture path;
- checkpoint state is valid only after complete prewarm success.

`ORBIT_KV_DIAG=1` reports metadata such as:

- `tools_default_enabled`
- `tools_startup_enabled`
- `prewarm_enabled`
- `prewarm_mode`
- `prewarm_attempted`
- `prewarm_succeeded`
- `prewarm_skipped_reason`
- `prewarm_failed_reason`
- `prewarm_prefix_token_count`
- `prewarm_checkpoint_size_bytes`
- `prewarm_ms`
- `restore_ready`

Diagnostics must not include raw prompt text, raw tokens, user content, tool
output, file content, or web content.

## Validation Expectations

Required validation for changes in this area:

- default tools-on path;
- explicit tools-off path;
- default startup prewarm path;
- explicit prewarm-off path;
- `ORBIT_KV_PREFIX_ANCHOR=off` kill switch;
- read/list/web/fetch regressions;
- listing followed by file read evidence;
- read-only local evidence followed by fresh web evidence;
- explicit edit remains possible;
- no system prompt anchor or multi-anchor behavior.
