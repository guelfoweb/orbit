from __future__ import annotations

import argparse
import sys

from orbit import __version__
from orbit.backend.llama_server import LlamaServerBackend, LlamaServerError
from orbit.runtime import ChatRuntime
from orbit.runtime.media import load_audio, load_image
from orbit.runtime.sessions import SessionStore
from orbit.terminal.config import add_config_arguments, load_app_config
from orbit.terminal.history import PromptHistory
from orbit.terminal.repl import Repl
from orbit.terminal.status import format_turn_status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orbit")
    parser.add_argument("prompt", nargs="*", help="Prompt for one-shot mode. Omit for interactive mode.")
    parser.add_argument("--image", action="append", default=[], help="Attach a local image to a one-shot prompt.")
    parser.add_argument("--audio", action="append", default=[], help="Attach a local WAV or MP3 audio file to a one-shot prompt.")
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

    backend = LlamaServerBackend(base_url=config.base_url, model=config.model, timeout=config.timeout)
    model_info = backend.model_info()
    context_tokens = config.context_tokens or (model_info.context_length if model_info else None)
    if args.prompt:
        runtime = ChatRuntime(
            backend=backend,
            system_prompt=None if config.no_system else config.system,
            context_tokens=context_tokens,
        )
        prompt = " ".join(args.prompt)
        return _run_one_shot(
            runtime,
            prompt,
            image_paths=args.image,
            audio_paths=args.audio,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    if args.image or args.audio:
        print("error: --image/--audio require a one-shot prompt", file=sys.stderr)
        return 1

    session = SessionStore.for_workdir(config.workdir)
    runtime = ChatRuntime(
        backend=backend,
        system_prompt=None if config.no_system else config.system,
        messages=session.load() or [],
        context_tokens=context_tokens,
    )
    history = PromptHistory.for_workdir(config.workdir)
    return Repl(runtime=runtime, backend=backend, config=config, session=session, history=history).run()


def _run_one_shot(
    runtime: ChatRuntime,
    prompt: str,
    *,
    image_paths: list[str],
    audio_paths: list[str],
    temperature: float,
    max_tokens: int,
) -> int:
    try:
        images = [load_image(path) for path in image_paths]
        audios = [load_audio(path) for path in audio_paths]
        result = runtime.ask(prompt, temperature=temperature, max_tokens=max_tokens, images=images, audios=audios)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except LlamaServerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(result.content, flush=True)
    print(format_turn_status(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
