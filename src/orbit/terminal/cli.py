from __future__ import annotations

import argparse
import sys
import time

from orbit import __version__
from orbit.backend.llama_server import LlamaServerBackend, LlamaServerError
from orbit.runtime import ChatRuntime
from orbit.runtime.media import load_audio, load_image
from orbit.terminal.config import add_config_arguments, load_app_config
from orbit.terminal.context_status import context_status_text
from orbit.terminal.history import PromptHistory
from orbit.terminal.prefill import MIN_PREFILL_ESTIMATE_SECONDS, estimate_prefill_tokens
from orbit.terminal.prefill_estimator import PrefillEstimator
from orbit.terminal.repl import Repl
from orbit.terminal.commands import health_text, help_text, runtime_status, set_max_tokens, tools_text
from orbit.terminal.session_selection import select_interactive_session
from orbit.terminal.status import estimate_context_status_tokens, format_turn_status
from orbit.terminal.streaming import StreamRenderer
from orbit.terminal.theme import dim
from orbit.terminal.tool_events import format_tool_call_event, format_tool_result_event
from orbit.terminal.tool_mode import allowed_tool_names_for_spec, tools_are_enabled


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orbit")
    parser.add_argument("prompt", nargs="*", help="Prompt for one-shot mode. Omit for interactive mode.")
    parser.add_argument("--image", action="append", default=[], help="Attach a local image to a one-shot prompt.")
    parser.add_argument("--audio", action="append", default=[], help="Attach a local WAV or MP3 audio file to a one-shot prompt.")
    parser.add_argument("--health", action="store_true", help="Check llama-server connectivity and model metadata.")
    add_config_arguments(parser)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_app_config(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    backend = LlamaServerBackend(base_url=config.base_url, timeout=config.timeout)
    if args.health:
        print(health_text(backend, config))
        return 0
    model_info = backend.model_info()
    context_tokens = config.context_tokens or (model_info.context_length if model_info else None)
    if args.prompt:
        runtime = ChatRuntime(
            backend=backend,
            system_prompt=None if config.no_system else config.system,
            context_tokens=context_tokens,
        )
        prompt = " ".join(args.prompt)
        command_result = _handle_one_shot_command(prompt, runtime, config, backend)
        if command_result is not None:
            print(command_result)
            return 0
        return _run_one_shot(
            runtime,
            prompt,
            image_paths=args.image,
            audio_paths=args.audio,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            workdir=config.workdir,
            tools=config.tools,
        )
    if args.image or args.audio:
        print("error: --image/--audio require a one-shot prompt", file=sys.stderr)
        return 1

    session = select_interactive_session(config.workdir)
    session_messages, session_warning = session.load_with_warning()
    if session_warning:
        print(dim(session_warning), file=sys.stderr)
    runtime = ChatRuntime(
        backend=backend,
        system_prompt=None if config.no_system else config.system,
        messages=session_messages or [],
        context_tokens=context_tokens,
    )
    history = PromptHistory.for_workdir(config.workdir)
    return Repl(runtime=runtime, backend=backend, config=config, session=session, history=history).run()


def _handle_one_shot_command(
    prompt: str,
    runtime: ChatRuntime,
    config,
    backend: LlamaServerBackend,
) -> str | None:
    command = prompt.strip()
    if not command.startswith("/"):
        return None
    if command == "/status":
        return runtime_status(runtime, config, backend, tools_mode=config.tools)
    if command in {"/status ctx", "/status context"}:
        return context_status_text(runtime.messages, context_tokens=runtime.context_tokens)
    if command == "/compact" or command == "/compact tools":
        return "error: /compact is available only in interactive mode"
    if command == "/tools":
        return tools_text(config.tools)
    if command == "/health":
        return health_text(backend, config)
    if command == "/help":
        return help_text()
    if command == "/max-tokens" or command.startswith("/max-tokens "):
        _, message = set_max_tokens(config, command.removeprefix("/max-tokens").strip())
        return message
    return f"unknown command: {command}"


def _run_one_shot(
    runtime: ChatRuntime,
    prompt: str,
    *,
    image_paths: list[str],
    audio_paths: list[str],
    temperature: float,
    max_tokens: int,
    workdir,
    tools: str,
) -> int:
    prefill_estimator = PrefillEstimator()
    prefill_tokens = estimate_prefill_tokens(runtime.messages, prompt)
    prefill_seconds = prefill_estimator.estimate_seconds(prefill_tokens)
    renderer = StreamRenderer(
        prefill_estimate_seconds=_visible_prefill_seconds(prefill_seconds),
        prefill_estimate_tokens=prefill_tokens,
    )
    started = time.monotonic()
    print()
    renderer.start()
    try:
        images = [load_image(path) for path in image_paths]
        audios = [load_audio(path) for path in audio_paths]
        if images or audios:
            result = runtime.ask(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                images=images,
                audios=audios,
                on_final_delta=renderer.write,
            )
        elif not tools_are_enabled(tools):
            result = runtime.ask_chat(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                on_final_delta=renderer.write,
            )
        else:
            result = runtime.ask_auto(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                workdir=workdir,
                allowed_tool_names=allowed_tool_names_for_spec(tools),
                on_final_delta=renderer.write,
                on_tool_call=lambda name, args: renderer.event(format_tool_call_event(name, args), restart_timer=False),
                on_tool_result=lambda name, chars, source, content: renderer.event(
                    format_tool_result_event(name, chars, source, content),
                    trailing_blank_line=True,
                ),
            )
    except KeyboardInterrupt:
        renderer.finish()
        print(dim("interrupted"), flush=True)
        return 130
    except ValueError as exc:
        renderer.finish()
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except LlamaServerError as exc:
        renderer.finish()
        print(f"error: {exc}", file=sys.stderr)
        return 1
    renderer.finish()
    prefill_estimator.update(
        prompt_tokens=result.prompt_tokens,
        prompt_tokens_per_second=result.prompt_tokens_per_second,
    )
    elapsed = time.monotonic() - started
    print("\n\n", end="", flush=True)
    print(
        dim(
            format_turn_status(
                result,
                elapsed_seconds=elapsed,
                estimated_context_tokens=estimate_context_status_tokens(runtime.messages),
                context_tokens=runtime.context_tokens,
            )
        ),
        flush=True,
    )
    return 0


def _visible_prefill_seconds(seconds: float | None) -> float | None:
    if seconds is None or seconds < MIN_PREFILL_ESTIMATE_SECONDS:
        return None
    return seconds


if __name__ == "__main__":
    raise SystemExit(main())
