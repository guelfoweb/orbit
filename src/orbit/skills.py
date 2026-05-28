from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .paths import SKILLS_DIR, ensure_skills_dir


DEFAULT_SKILL_REF = "orbit-default"
BUILTIN_SKILLS_DIR = Path(__file__).resolve().parent / "builtins"


DEFAULT_SKILL_ROOTS = [
    SKILLS_DIR,
    Path.cwd() / "skills",
    BUILTIN_SKILLS_DIR,
]


@dataclass(frozen=True)
class Skill:
    name: str
    path: Path
    content: str


def list_skills(skill_roots: list[Path] | None = None, *, include_default: bool = False) -> list[Skill]:
    ensure_skills_dir()
    found: dict[str, Skill] = {}
    for root in skill_roots or DEFAULT_SKILL_ROOTS:
        for skill in _list_skills_from_root(root.expanduser()):
            found.setdefault(skill.name, skill)
    if not include_default:
        found.pop(DEFAULT_SKILL_REF, None)
    return [found[name] for name in sorted(found)]


def resolve_skill(reference: str, skill_roots: list[Path] | None = None) -> Skill:
    value = reference.strip()
    if not value:
        raise FileNotFoundError("empty skill reference")
    ensure_skills_dir()
    path_candidate = Path(value).expanduser()
    direct_match = _resolve_direct_path(path_candidate)
    if direct_match is not None:
        return direct_match
    for root in skill_roots or DEFAULT_SKILL_ROOTS:
        resolved = _resolve_from_root(root.expanduser(), value)
        if resolved is not None:
            return resolved
    raise FileNotFoundError(f"skill not found: {reference}")


def default_skill() -> Skill:
    return resolve_skill(DEFAULT_SKILL_REF)


def _resolve_direct_path(candidate: Path) -> Skill | None:
    if candidate.is_file() and candidate.name == "SKILL.md":
        return _load_skill(candidate.parent.name, candidate)
    if candidate.is_dir():
        skill_path = candidate / "SKILL.md"
        if skill_path.is_file():
            return _load_skill(candidate.name, skill_path)
    return None


def _resolve_from_root(root: Path, value: str) -> Skill | None:
    if not root.exists() or not root.is_dir():
        return None
    direct_dir = root / value
    if direct_dir.is_dir() and (direct_dir / "SKILL.md").is_file():
        return _load_skill(direct_dir.name, direct_dir / "SKILL.md")
    direct_file = root / value
    if direct_file.is_file() and direct_file.name == "SKILL.md":
        return _load_skill(direct_file.parent.name, direct_file)
    nested_file = root / value / "SKILL.md"
    if nested_file.is_file():
        return _load_skill(value, nested_file)
    return None


def _list_skills_from_root(root: Path) -> list[Skill]:
    if not root.exists() or not root.is_dir():
        return []
    skills: list[Skill] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        skill_path = entry / "SKILL.md"
        if skill_path.is_file():
            skills.append(_load_skill(entry.name, skill_path))
    return skills


def _load_skill(name: str, path: Path) -> Skill:
    content = path.read_text(encoding="utf-8")
    return Skill(name=name, path=path.resolve(), content=content)
