from __future__ import annotations

from dataclasses import dataclass

from .. import __version__
from .agent import AgentLoop, TurnResult
from .client import ModelMetadata, OllamaClient
from .events import EventSink, SessionAutoCompactEvent
from ..paths import ensure_orbit_home
from ..session import create_session_name, delete_sessions_for_workdir, load_session, save_session
from ..skills import DEFAULT_SKILL_REF, Skill, default_skill, resolve_skill
from ..terminal.config import AppConfig
from ..tooling.registry import ToolRegistry

MODEL_FIRST_MAX_LOOPS = 10


@dataclass
class OrbitRuntime:
    config: AppConfig
    client: OllamaClient
    registry: ToolRegistry
    agent: AgentLoop
    session_name: str
    active_skill_ref: str | None = None
    model_metadata: ModelMetadata | None = None

    @classmethod
    def from_config(cls, config: AppConfig) -> "OrbitRuntime":
        ensure_orbit_home()
        registry = ToolRegistry(workdir=config.workdir)
        client = OllamaClient(base_url=config.base_url, model=config.model, timeout=config.timeout)
        model_metadata = client.inspect_model()
        effective_max_loops = config.max_loops
        if not getattr(config, "max_loops_explicit", False):
            if _prefers_model_first_runtime(model_metadata):
                effective_max_loops = min(config.max_loops, MODEL_FIRST_MAX_LOOPS)
        active_skill = _resolve_optional_skill(config.skill_ref) or default_skill()
        agent = AgentLoop(
            client=client,
            registry=registry,
            max_loops=effective_max_loops,
            temperature=config.temperature,
            skill=active_skill,
            tools_enabled=model_metadata.tools_supported is not False,
            think_mode=config.think_mode,
            show_thinking=config.show_thinking,
            debug_timing=config.debug_timing,
        )
        agent._model_metadata = model_metadata
        session_name = config.session_name or create_session_name(config.workdir)
        runtime = cls(
            config=config,
            client=client,
            registry=registry,
            agent=agent,
            session_name=session_name,
            active_skill_ref=config.skill_ref or DEFAULT_SKILL_REF,
            model_metadata=model_metadata,
        )
        runtime._apply_initial_thinking_mode()
        runtime._restore_session()
        return runtime

    @property
    def active_model(self) -> str:
        return self.client.model or "-"

    @property
    def tools_enabled(self) -> bool:
        return self.agent.tools_enabled

    @property
    def startup_notice(self) -> str | None:
        if self.model_metadata is not None and self.model_metadata.tools_supported is False:
            return (
                f"warning: model {self.active_model} does not advertise tool support; "
                "orbit will run in chat-only mode"
            )
        return None

    @property
    def startup_summary(self) -> tuple[str, str]:
        first = f"orbit v{__version__} | workdir={self.config.workdir}"
        params = "-"
        tool_call = "no"
        think = self.effective_think_state()
        show_thinking = self.effective_show_thinking_state()
        if self.model_metadata is not None:
            if self.model_metadata.parameter_size:
                params = self.model_metadata.parameter_size
            if self.model_metadata.tools_supported is not False:
                tool_call = "yes"
        second = (
            f"model={self.active_model} | params={params} | "
            f"tool-call={tool_call} | think={think} | show-thinking={show_thinking}"
        )
        return first, second

    def supports_thinking(self) -> bool:
        return self.model_metadata is not None and "thinking" in self.model_metadata.capabilities

    def effective_think_state(self) -> str:
        if not self.supports_thinking():
            return "no"
        return "on" if self.agent.think_mode != "off" else "off"

    def effective_show_thinking_state(self) -> str:
        if self.effective_think_state() != "on":
            return "off"
        return "on" if self.agent.show_thinking else "off"

    def thinking_status_text(self) -> str:
        return (
            f"think: {self.effective_think_state()} | "
            f"show-thinking: {self.effective_show_thinking_state()}"
        )

    def run_turn(self, user_input: str, on_event: EventSink | None = None) -> TurnResult:
        pressure = self.agent.context_pressure(pending_user_input=user_input)
        if pressure.should_compact:
            changed = self.compact_session(overflow_tokens=pressure.overflow_tokens)
            if changed and on_event is not None:
                on_event(
                    SessionAutoCompactEvent(
                        level=pressure.level,
                        score=pressure.score,
                        reason=pressure.reason,
                        session_messages=pressure.session_messages,
                        estimated_prompt_tokens=pressure.estimated_prompt_tokens,
                    )
                )
        result = self.agent.run_turn(user_input, on_event=on_event)
        self.save_session()
        return result

    def save_session(self) -> None:
        save_session(self.session_name, self.agent.messages, self.active_skill_ref, self.config.workdir)

    def set_skill(self, skill_ref: str) -> None:
        skill = resolve_skill(skill_ref)
        self.agent.set_skill(skill)
        self.active_skill_ref = skill_ref
        self.save_session()

    def clear_skill(self) -> None:
        self.agent.set_skill(default_skill())
        self.active_skill_ref = DEFAULT_SKILL_REF
        self.save_session()

    def set_think_mode(self, mode: str) -> None:
        self.agent.think_mode = mode
        if mode == "on":
            if self.model_metadata is not None and "thinking" in self.model_metadata.capabilities:
                self.agent.show_thinking = True
            else:
                self.agent.think_mode = "off"
                self.agent.show_thinking = False
        elif mode == "off":
            self.agent.show_thinking = False
        elif mode == "auto":
            self._apply_initial_thinking_mode()

    def set_show_thinking(self, enabled: bool) -> None:
        self.agent.show_thinking = enabled
        if not enabled:
            self.agent.think_mode = "off"
        elif self.model_metadata is not None and "thinking" in self.model_metadata.capabilities:
            self.agent.think_mode = "on"
        else:
            self.agent.think_mode = "off"
            self.agent.show_thinking = False

    def reset_session(self) -> None:
        self.agent.reset()
        self.save_session()

    def clear_sessions_for_workdir(self) -> int:
        deleted = delete_sessions_for_workdir(self.config.workdir)
        self.agent.reset()
        self.session_name = create_session_name(self.config.workdir)
        return deleted

    def compact_session(self, *, overflow_tokens: int = 0) -> bool:
        changed = self.agent.compact(overflow_tokens=overflow_tokens)
        if changed:
            self.save_session()
        return changed

    def _restore_session(self) -> None:
        session_data = load_session(self.session_name)
        if session_data is None:
            return
        if session_data.skill_ref:
            if self.active_skill_ref not in {None, DEFAULT_SKILL_REF, session_data.skill_ref}:
                return
            if self.active_skill_ref != session_data.skill_ref:
                skill = _resolve_optional_skill(session_data.skill_ref)
                if skill is not None:
                    self.agent.set_skill(skill)
                    self.active_skill_ref = session_data.skill_ref
        self.agent.restore_messages(session_data.messages)

    def _apply_initial_thinking_mode(self) -> None:
        capabilities = self.model_metadata.capabilities if self.model_metadata is not None else ()
        supports_thinking = "thinking" in capabilities
        if not supports_thinking:
            self.agent.think_mode = "off"
            self.agent.show_thinking = False
            return
        if self.config.think_explicit:
            if self.config.think_mode == "off":
                self.agent.think_mode = "off"
                self.agent.show_thinking = False
                return
            if self.config.think_mode == "on":
                self.agent.think_mode = "on"
                self.agent.show_thinking = True
                return
        if self.config.show_thinking_explicit:
            if self.config.show_thinking:
                self.agent.think_mode = "on"
                self.agent.show_thinking = True
            else:
                self.agent.think_mode = "off"
                self.agent.show_thinking = False
            return
        if self.model_metadata is not None and _prefers_model_first_runtime(self.model_metadata):
            self.agent.think_mode = "off"
            self.agent.show_thinking = False
            return
        self.agent.think_mode = "on"
        self.agent.show_thinking = True


def _resolve_optional_skill(skill_ref: str | None) -> Skill | None:
    if not skill_ref:
        return None
    return resolve_skill(skill_ref)


def _prefers_model_first_runtime(model_metadata: ModelMetadata) -> bool:
    return model_metadata.active_model.lower().startswith("gemma4:")
