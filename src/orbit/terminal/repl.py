from __future__ import annotations

import sys
import time
from dataclasses import dataclass

from orbit.backend.llama_server import LlamaServerBackend, LlamaServerError
from orbit.runtime import ChatRuntime
from orbit.runtime.sessions import SessionStore
from orbit.terminal.commands import help_text, reset_session, runtime_status, set_max_tokens, tools_text
from orbit.terminal.config import AppConfig
from orbit.terminal.history import PromptHistory
from orbit.terminal.status import estimate_context_status_tokens, format_memory_refresh, format_turn_status
from orbit.terminal.streaming import StreamRenderer
from orbit.terminal.tool_events import format_tool_result_event
from orbit.terminal.theme import dim


@dataclass
class Repl:
    runtime: ChatRuntime
    backend: LlamaServerBackend
    config: AppConfig
    session: SessionStore | None = None
    history: PromptHistory | None = None

    def run(self) -> int:
        if self.history:
            self.history.load()
        print("orbit interactive mode. Type /help for commands.")
        while True:
            try:
                prompt = input("> ").strip()
            except EOFError:
                self._save_history()
                print()
                return 0
            except KeyboardInterrupt:
                self._save_history()
                print()
                return 130
            if not prompt:
                continue
            if prompt.startswith("/"):
                if self._handle_command(prompt):
                    continue
                self._save_history()
                return 0
            if self.history:
                self.history.add(prompt)
                self.history.save()
            self._ask(prompt)

    def _ask(self, prompt: str) -> None:
        renderer = StreamRenderer()
        checkpoint = len(self.runtime.messages)
        print()
        started = time.monotonic()
        renderer.start()
        try:
            result = self.runtime.ask_auto(
                prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                workdir=self.config.workdir,
                on_final_delta=renderer.write,
                on_tool_call=lambda name, args: renderer.event(f"{name} {args}", restart_timer=False),
                on_tool_result=lambda name, chars, source: renderer.event(
                    format_tool_result_event(name, chars, source),
                    trailing_blank_line=True,
                ),
            )
        except KeyboardInterrupt:
            renderer.finish()
            self.runtime.restore_message_count(checkpoint)
            print(dim("interrupted"), flush=True)
            return
        except LlamaServerError as exc:
            renderer.finish()
            self.runtime.restore_message_count(checkpoint)
            print(f"error: {exc}", file=sys.stderr)
            return
        renderer.finish()
        self._save_session()
        elapsed = time.monotonic() - started
        print("\n\n", end="", flush=True)
        if self.runtime.last_memory_refresh:
            refresh = self.runtime.last_memory_refresh
            print(dim(format_memory_refresh(refresh)), flush=True)
        print(
            dim(
                format_turn_status(
                    result,
                    elapsed_seconds=elapsed,
                    estimated_context_tokens=estimate_context_status_tokens(self.runtime.messages),
                    context_tokens=self.runtime.context_tokens,
                )
            ),
            flush=True,
        )

    def _handle_command(self, command: str) -> bool:
        if command == "/exit":
            return False
        if command == "/help":
            print(help_text())
            return True
        if command == "/reset":
            print(reset_session(self.runtime, self.session))
            return True
        if command == "/health":
            print("llama-server: ok" if self.backend.health() else "llama-server: unavailable")
            return True
        if command == "/max-tokens" or command.startswith("/max-tokens "):
            value = command.removeprefix("/max-tokens").strip()
            self.config, message = set_max_tokens(self.config, value)
            print(message)
            return True
        if command == "/status":
            print(runtime_status(self.runtime, self.config, self.backend))
            return True
        if command == "/tools":
            print(tools_text(self.backend))
            return True
        print(f"unknown command: {command}", file=sys.stderr)
        return True

    def _save_session(self) -> None:
        if not self.session:
            return
        self.session.save(
            messages=self.runtime.messages,
            workdir=self.config.workdir,
            model=self.config.model,
            base_url=self.config.base_url,
        )

    def _save_history(self) -> None:
        if self.history:
            self.history.save()
