from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.skills import DEFAULT_SKILL_REF, default_skill, list_skills, resolve_skill
from orbit.core.skill_hints import extra_categories_for_skill, startup_prompt_for_skill, workspace_doc_bootstrap_actions


class SkillTests(unittest.TestCase):
    def test_list_skills_returns_sorted_unique_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root_a = Path(tmp) / "a"
            root_b = Path(tmp) / "b"
            root_a.mkdir()
            root_b.mkdir()
            for root, names in ((root_a, ["beta", "alpha"]), (root_b, ["alpha", "gamma"])):
                for name in names:
                    skill_dir = root / name
                    skill_dir.mkdir()
                    (skill_dir / "SKILL.md").write_text(name, encoding="utf-8")
            skills = list_skills(skill_roots=[root_a, root_b])
            self.assertEqual([skill.name for skill in skills], ["alpha", "beta", "gamma"])

    def test_list_skills_hides_default_skill_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ["orbit-default", "demo"]:
                skill_dir = root / name
                skill_dir.mkdir()
                (skill_dir / "SKILL.md").write_text(name, encoding="utf-8")
            skills = list_skills(skill_roots=[root])
            self.assertEqual([skill.name for skill in skills], ["demo"])
            skills_with_default = list_skills(skill_roots=[root], include_default=True)
            self.assertEqual([skill.name for skill in skills_with_default], ["demo", "orbit-default"])

    def test_resolve_skill_from_directory_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = root / "demo-skill"
            skill_dir.mkdir()
            skill_file = skill_dir / "SKILL.md"
            skill_file.write_text("# Demo\nUse this skill.\n", encoding="utf-8")
            skill = resolve_skill(str(skill_dir), skill_roots=[])
            self.assertEqual(skill.name, "demo-skill")
            self.assertEqual(skill.path, skill_file.resolve())

    def test_resolve_skill_from_named_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = root / "demo"
            skill_dir.mkdir()
            skill_file = skill_dir / "SKILL.md"
            skill_file.write_text("demo", encoding="utf-8")
            skill = resolve_skill("demo", skill_roots=[root])
            self.assertEqual(skill.name, "demo")
            self.assertEqual(skill.path, skill_file.resolve())

    def test_default_skill_is_bundled(self) -> None:
        skill = default_skill()
        self.assertEqual(skill.name, DEFAULT_SKILL_REF)
        self.assertIn("minimal default orchestration skill", skill.content)

    def test_skill_hints_enable_write_for_analysis_workflows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = root / "analysis-skill"
            skill_dir.mkdir()
            skill_file = skill_dir / "SKILL.md"
            skill_file.write_text(
                "Create or reuse a case directory.\nCreate or read `AGENTS.md` and `REPORT.md`.\n",
                encoding="utf-8",
            )
            skill = resolve_skill("analysis-skill", skill_roots=[root])
            self.assertEqual(extra_categories_for_skill(skill, "binary_or_pdf_analysis"), ("write",))
            prompt = startup_prompt_for_skill(skill, "binary_or_pdf_analysis", [{"role": "system", "content": "base"}])
            self.assertIn("Active skill startup is mandatory", prompt)

    def test_workspace_doc_bootstrap_actions_create_missing_docs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = root / "analysis-skill"
            skill_dir.mkdir()
            skill_file = skill_dir / "SKILL.md"
            skill_file.write_text(
                "Create or reuse a case directory.\nCreate or read `AGENTS.md` and `REPORT.md`.\n",
                encoding="utf-8",
            )
            skill = resolve_skill("analysis-skill", skill_roots=[root])
            workdir = root / "case"
            workdir.mkdir()
            actions = workspace_doc_bootstrap_actions(skill, workdir)
            self.assertEqual([name for name, _ in actions], ["write_file", "write_file"])
