#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.backend.llama_server import LlamaServerBackend, LlamaServerError  # noqa: E402
from orbit.runtime import messages as runtime_messages  # noqa: E402
from orbit.runtime.chat import ChatRuntime  # noqa: E402


CheckResult = tuple[bool, str]
SetupFn = Callable[[Path], None]
CheckFn = Callable[[Path, str], CheckResult]


@dataclass(frozen=True)
class Case:
    case_id: str
    title: str
    prompt: str
    fixture: str
    expected: str
    checker: str
    risk: str
    status: str
    likely_failure: str
    setup: SetupFn
    check: CheckFn
    sequence: tuple[str, ...] = ()


def write(path: Path, content: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_local(cmd: list[str], cwd: Path, *, timeout: int = 15) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)


def ok_if(condition: bool, success: str, failure: str) -> CheckResult:
    return (True, success) if condition else (False, failure)


def check_python(cwd: Path, code: str) -> CheckResult:
    result = run_local([sys.executable, "-c", code], cwd)
    if result.returncode == 0:
        return True, "behavioral Python check passed"
    return False, f"Python check failed: {result.stderr.strip() or result.stdout.strip()}"


def setup_html_title(workdir: Path) -> None:
    write(
        workdir / "index.html",
        """<!doctype html>
<html>
<head>
  <title>
    Old Portal
  </title>
</head>
<body><h1>Welcome</h1></body>
</html>
""",
    )


def check_html_title(workdir: Path, _answer: str) -> CheckResult:
    html = read(workdir / "index.html")
    match = re.search(r"<title>\s*(.*?)\s*</title>", html, flags=re.I | re.S)
    return ok_if(bool(match and match.group(1) == "Release Ready"), "title is Release Ready", "title was not updated")


def setup_css_regex(workdir: Path) -> None:
    write(
        workdir / "style.css",
        """.hero .btn/path {
  background: url("/img/a.b/c.png");
  color: #333;
  margin: 4px 8px;
}
""",
    )


def check_css_regex(workdir: Path, _answer: str) -> CheckResult:
    css = read(workdir / "style.css")
    valid_shape = css.count("{") == css.count("}") == 1
    updated = "color: #0a7cff;" in css
    preserved = 'url("/img/a.b/c.png")' in css and ".hero .btn/path" in css
    return ok_if(valid_shape and updated and preserved, "CSS is valid and target property changed", "CSS update is invalid or incomplete")


def setup_python_tiny(workdir: Path) -> None:
    write(workdir / "math_utils.py", "def add(left, right):\n    return left - right\n")


def check_python_tiny(workdir: Path, _answer: str) -> CheckResult:
    return check_python(workdir, "from math_utils import add\nassert add(2, 3) == 5\nassert add(-1, 1) == 0\n")


def setup_shell_hardening(workdir: Path) -> None:
    write(workdir / "copy_file.sh", "#!/usr/bin/env bash\ncp $1 $2\n", executable=True)


def check_shell_hardening(workdir: Path, _answer: str) -> CheckResult:
    script = read(workdir / "copy_file.sh")
    write(workdir / "source file.txt", "payload\n")
    result = run_local(["bash", "copy_file.sh", "source file.txt", "target file.txt"], workdir)
    copied = (workdir / "target file.txt").exists() and read(workdir / "target file.txt") == "payload\n"
    hardened = "set -euo pipefail" in script
    if result.returncode != 0:
        return False, f"shell script failed with spaced paths: {result.stderr.strip()}"
    return ok_if(hardened and copied, "script is hardened and handles spaced paths", "script hardening or spaced-path behavior missing")


def setup_json_config(workdir: Path) -> None:
    write(workdir / "config.json", json.dumps({"service": {"timeout": 10, "cache": False}}, indent=2) + "\n")


def check_json_config(workdir: Path, _answer: str) -> CheckResult:
    try:
        data = json.loads(read(workdir / "config.json"))
    except json.JSONDecodeError as exc:
        return False, f"invalid JSON: {exc}"
    return ok_if(
        data.get("service", {}).get("timeout") == 30 and data.get("service", {}).get("cache") is True,
        "JSON is valid and values are updated",
        "JSON values were not updated correctly",
    )


def setup_yaml_config(workdir: Path) -> None:
    write(workdir / "service.yaml", "service:\n  port: 8080\n  debug: false\n  name: orbit\n")


def check_yaml_config(workdir: Path, _answer: str) -> CheckResult:
    yaml = read(workdir / "service.yaml")
    balanced = "\t" not in yaml and "service:" in yaml
    updated = re.search(r"^\s*port:\s*9090\s*$", yaml, re.M) and re.search(r"^\s*debug:\s*true\s*$", yaml, re.M)
    return ok_if(bool(balanced and updated), "YAML shape is preserved and values changed", "YAML update invalid or incomplete")


def setup_fix_failed_test(workdir: Path) -> None:
    write(
        workdir / "string_utils.py",
        """def is_palindrome(value):
    cleaned = value.replace(" ", "")
    return cleaned == cleaned[::-1]
""",
    )
    write(
        workdir / "test_string_utils.py",
        """import unittest
from string_utils import is_palindrome


class PalindromeTest(unittest.TestCase):
    def test_ignores_case_and_spaces(self):
        self.assertTrue(is_palindrome("Never odd or even"))

    def test_rejects_non_palindrome(self):
        self.assertFalse(is_palindrome("not one"))


if __name__ == "__main__":
    unittest.main()
""",
    )


def check_fix_failed_test(workdir: Path, _answer: str) -> CheckResult:
    result = run_local([sys.executable, "-m", "unittest", "test_string_utils.py"], workdir)
    if result.returncode == 0:
        return True, "unit test passes"
    return False, f"unit test still fails: {result.stderr.strip() or result.stdout.strip()}"


def setup_rename_symbol(workdir: Path) -> None:
    write(
        workdir / "names.py",
        """def slugify(value):
    return value.strip().lower().replace(" ", "-")


def make_id(title):
    return "doc-" + slugify(title)
""",
    )


def check_rename_symbol(workdir: Path, _answer: str) -> CheckResult:
    text = read(workdir / "names.py")
    if "def normalize_slug" not in text or "normalize_slug(title)" not in text:
        return False, "definition or internal use was not renamed"
    return check_python(workdir, "from names import normalize_slug, make_id\nassert normalize_slug('Hello World') == 'hello-world'\nassert make_id('Hello World') == 'doc-hello-world'\n")


def setup_spaced_filename(workdir: Path) -> None:
    write(workdir / "notes/todo list.txt", "title: release\nstatus: draft\n")


def check_spaced_filename(workdir: Path, _answer: str) -> CheckResult:
    text = read(workdir / "notes/todo list.txt")
    return ok_if("status: done" in text, "file with spaces was updated", "spaced filename was not updated")


def setup_recoverable_command_error(workdir: Path) -> None:
    write(workdir / "routes.txt", "GET /old/api -> upstream\n")


def check_recoverable_command_error(workdir: Path, _answer: str) -> CheckResult:
    text = read(workdir / "routes.txt")
    return ok_if("/new/api" in text and "/old/api" not in text, "slash-heavy value replaced", "slash-heavy replacement failed")


def setup_noop_mutation(workdir: Path) -> None:
    write(workdir / "settings.ini", "[service]\nenabled = false\nmode = safe\n")


def check_noop_mutation(workdir: Path, _answer: str) -> CheckResult:
    text = read(workdir / "settings.ini")
    return ok_if("enabled = true" in text or "enabled=true" in text, "silent no-op was avoided or repaired", "setting was not changed")


def setup_metadata_trap(workdir: Path) -> None:
    write(
        workdir / "calc.py",
        """def divide(left, right):
    if right == 0:
        return 0
    return left / right
""",
    )
    write(
        workdir / "test_calc.py",
        """import unittest
from calc import divide


class DivideTest(unittest.TestCase):
    def test_zero_division_raises(self):
        with self.assertRaises(ZeroDivisionError):
            divide(1, 0)


if __name__ == "__main__":
    unittest.main()
""",
    )


def check_metadata_trap(workdir: Path, _answer: str) -> CheckResult:
    result = run_local([sys.executable, "-m", "unittest", "test_calc.py"], workdir)
    if result.returncode == 0:
        return True, "test passes after content-based inspection"
    return False, "task did not move beyond metadata/listing to a passing fix"


def setup_long_command_pressure(workdir: Path) -> None:
    filler = "\n".join(f"# filler {i}" for i in range(120))
    write(
        workdir / "parser.py",
        f"""{filler}

def parse_port(value):
    return value
""",
    )


def check_long_command_pressure(workdir: Path, _answer: str) -> CheckResult:
    return check_python(workdir, "from parser import parse_port\nassert parse_port('8080') == 8080\nassert parse_port(9090) == 9090\n")


def setup_read_only_review(workdir: Path) -> None:
    write(
        workdir / "vulnerable_service.py",
        """import subprocess


def run_report(name):
    return subprocess.check_output("cat reports/" + name, shell=True)
""",
    )


def check_read_only_review(workdir: Path, answer: str) -> CheckResult:
    unchanged = sha256(workdir / "vulnerable_service.py") == sha256(workdir / ".original_vulnerable_service.py")
    lower = answer.lower()
    useful = "shell" in lower and ("injection" in lower or "command" in lower) and "run_report" in answer
    return ok_if(unchanged and useful, "review found the issue without modifying the file", "review modified file or missed command injection")


def setup_read_only_review_with_hash(workdir: Path) -> None:
    setup_read_only_review(workdir)
    shutil.copy2(workdir / "vulnerable_service.py", workdir / ".original_vulnerable_service.py")


def setup_ambiguous(workdir: Path) -> None:
    write(workdir / "service.py", "def normalize_name(value):\n    return value\n")


def check_ambiguous(workdir: Path, _answer: str) -> CheckResult:
    return check_python(workdir, "from service import normalize_name\nassert normalize_name('  Alice  ') == 'alice'\n")


CASES: tuple[Case, ...] = (
    Case(
        "html_multiline_title",
        "HTML multiline title",
        'Change the title in index.html to "Release Ready".',
        "index.html contains a multiline <title> block.",
        "The final title text is exactly Release Ready.",
        "Extracts the <title> value from the final file.",
        "robust local HTML patch generation",
        "blocker release",
        "quoting/sed fragile",
        setup_html_title,
        check_html_title,
    ),
    Case(
        "css_regex_sensitive",
        "CSS with regex-sensitive characters",
        "Update style.css so the .hero .btn/path color is #0a7cff. Preserve the URL and selector.",
        "style.css contains braces, slash, dot, URL, and spaces.",
        "CSS remains structurally valid and the property is updated.",
        "Checks balanced braces, preserved selector/URL, and final color.",
        "fragile regex or broad rewrite",
        "blocker release",
        "command generation fragile",
        setup_css_regex,
        check_css_regex,
    ),
    Case(
        "python_tiny_function",
        "Python tiny function",
        "Fix math_utils.py so add(left, right) returns the correct sum.",
        "math_utils.py has a tiny incorrect function.",
        "Behavioral assertions for addition pass.",
        "Imports the function and runs assertions.",
        "small Python local patch",
        "blocker release",
        "insufficient modification",
        setup_python_tiny,
        check_python_tiny,
    ),
    Case(
        "shell_script_hardening",
        "Shell script hardening",
        "Harden copy_file.sh: add strict shell safety and make it work with source and target paths containing spaces.",
        "copy_file.sh copies two unquoted positional arguments.",
        "Script uses strict mode and copies files whose names contain spaces.",
        "Runs the script with spaced paths and checks set -euo pipefail.",
        "patch completeness for shell scripts",
        "blocker release",
        "patch incompleta",
        setup_shell_hardening,
        check_shell_hardening,
    ),
    Case(
        "json_config_update",
        "Config JSON",
        "In config.json set service.timeout to 30 and service.cache to true.",
        "config.json has nested service settings.",
        "JSON stays valid and both values are updated.",
        "Parses JSON and checks values.",
        "config mutation validity",
        "blocker release",
        "discovery insufficiente",
        setup_json_config,
        check_json_config,
    ),
    Case(
        "yaml_config_update",
        "Config YAML",
        "In service.yaml, inside the existing service block, change port to 9090 and debug to true.",
        "service.yaml has simple nested YAML.",
        "YAML shape remains usable and both values are updated.",
        "Checks indentation shape and final values without requiring PyYAML.",
        "config mutation validity",
        "blocker release",
        "quoting/sed fragile",
        setup_yaml_config,
        check_yaml_config,
    ),
    Case(
        "fix_failed_test",
        "Fix failed test",
        "Run the tests, inspect the failure, then fix the code so the tests pass.",
        "A unittest file documents a failing palindrome behavior.",
        "The unittest suite passes.",
        "Runs python -m unittest on the fixture.",
        "content evidence recovery and completion",
        "blocker release",
        "discovery insufficiente",
        setup_fix_failed_test,
        check_fix_failed_test,
    ),
    Case(
        "rename_symbol",
        "Rename symbol",
        "Rename function slugify to normalize_slug in names.py and update all local uses.",
        "names.py defines slugify and uses it in make_id.",
        "Definition and use are coherent under the new name.",
        "Imports normalize_slug and make_id and runs behavior checks.",
        "symbol rename consistency",
        "blocker release",
        "insufficient modification",
        setup_rename_symbol,
        check_rename_symbol,
    ),
    Case(
        "file_with_spaces",
        "File with spaces in name",
        'In "notes/todo list.txt", change status from draft to done.',
        "A target file path contains spaces.",
        "The intended file is updated.",
        "Reads the exact spaced filename.",
        "path quoting",
        "blocker release",
        "quoting/sed fragile",
        setup_spaced_filename,
        check_spaced_filename,
    ),
    Case(
        "recoverable_command_error",
        "Command error recoverable",
        "In routes.txt replace /old/api with /new/api.",
        "routes.txt contains slash-heavy values that often break sed delimiters.",
        "The path value is replaced.",
        "Checks final file content.",
        "Shell Repair Loop for quoting/regex errors",
        "blocker release",
        "quoting/sed fragile",
        setup_recoverable_command_error,
        check_recoverable_command_error,
    ),
    Case(
        "noop_mutation",
        "No-op mutation",
        "In settings.ini enable the service by changing enabled to true.",
        "settings.ini has whitespace around the equals sign.",
        "The setting is actually changed.",
        "Checks final setting, not command stdout.",
        "Mutation Verification for silent no-op",
        "blocker release",
        "discovery insufficiente",
        setup_noop_mutation,
        check_noop_mutation,
    ),
    Case(
        "metadata_trap",
        "Metadata trap",
        "The tests describe the bug. Do not stop at listing files; inspect content and fix the code.",
        "A test file requires divide(1, 0) to raise ZeroDivisionError.",
        "The tests pass after real content inspection.",
        "Runs python -m unittest.",
        "metadata-only discovery trap",
        "blocker release",
        "discovery insufficiente",
        setup_metadata_trap,
        check_metadata_trap,
    ),
    Case(
        "long_command_pressure",
        "Long command pressure",
        "Fix parser.py so parse_port returns an integer for numeric strings and leaves existing integer values as integers. Avoid rewriting the whole file.",
        "parser.py has many filler lines and one small broken function.",
        "Behavioral parse_port assertions pass.",
        "Imports parse_port and runs assertions.",
        "Minimal Patch Guard under command length pressure",
        "blocker release",
        "heredoc lungo",
        setup_long_command_pressure,
        check_long_command_pressure,
    ),
    Case(
        "read_only_review",
        "Read-only review",
        "Analyze vulnerable_service.py and report any security issue. Do not modify files.",
        "A Python file contains shell=True command construction.",
        "The answer identifies the issue and the file is unchanged.",
        "Hashes the file and checks answer content.",
        "read-only analysis must not mutate",
        "blocker release",
        "insufficient discovery",
        setup_read_only_review_with_hash,
        check_read_only_review,
    ),
    Case(
        "ambiguous_suggest_then_fix",
        "Prompt ambiguo",
        "Suggest fixes for service.py but do not modify files.",
        "service.py returns an unnormalized name.",
        "Suggest-only keeps file unchanged; follow-up fix modifies behavior.",
        "Runs two prompts and then imports the function.",
        "ambiguous mutation boundary",
        "blocker release",
        "other",
        setup_ambiguous,
        check_ambiguous,
        sequence=("Now fix it.",),
    ),
)


def run_prompt(runtime: ChatRuntime, prompt: str, *, workdir: Path, max_tokens: int, timeout_events: list[str]) -> tuple[str, list[str]]:
    commands: list[str] = []

    def on_tool_call(name: str, args: str) -> None:
        commands.append(f"{name} {args}")

    def on_tool_result(name: str, size: int, _content: str, _source: str) -> None:
        timeout_events.append(f"{name}:{size}")

    result = runtime.ask_auto(
        prompt,
        temperature=0.0,
        max_tokens=max_tokens,
        workdir=workdir,
        max_loops=12,
        allowed_tool_names=("exec_shell_full_command",),
        on_tool_call=on_tool_call,
        on_tool_result=on_tool_result,
    )
    return result.content, commands


def run_case(case: Case, *, base_url: str, timeout: int, max_tokens: int, keep_failed: bool) -> dict[str, object]:
    started = time.monotonic()
    tmp = tempfile.TemporaryDirectory(prefix=f"orbit-release-{case.case_id}-")
    workdir = Path(tmp.name)
    case.setup(workdir)
    events: list[str] = []
    commands: list[str] = []
    answer = ""
    failed_dir: str | None = None
    try:
        backend = LlamaServerBackend(base_url=base_url, timeout=timeout)
        runtime = ChatRuntime(backend=backend, system_prompt=runtime_messages.ROUTE_SYSTEM_PROMPT)
        answer, first_commands = run_prompt(runtime, case.prompt, workdir=workdir, max_tokens=max_tokens, timeout_events=events)
        commands.extend(first_commands)
        if case.case_id == "ambiguous_suggest_then_fix":
            unchanged_after_suggest = read(workdir / "service.py") == "def normalize_name(value):\n    return value\n"
            if not unchanged_after_suggest:
                ok = False
                note = "suggest-only prompt modified the file"
            else:
                follow_answer, follow_commands = run_prompt(runtime, case.sequence[0], workdir=workdir, max_tokens=max_tokens, timeout_events=events)
                answer = f"{answer}\n\n{follow_answer}"
                commands.extend(follow_commands)
                ok, note = case.check(workdir, answer)
        else:
            ok, note = case.check(workdir, answer)
    except Exception as exc:  # noqa: BLE001 - release suite must report all failures uniformly.
        ok = False
        note = f"{type(exc).__name__}: {exc}"
    elapsed = time.monotonic() - started
    if keep_failed and not ok:
        failed_dir = str(Path(tempfile.mkdtemp(prefix=f"orbit-release-failed-{case.case_id}-")))
        shutil.copytree(workdir, failed_dir, dirs_exist_ok=True)
    tmp.cleanup()
    return {
        "id": case.case_id,
        "title": case.title,
        "pass": ok,
        "note": note,
        "likely_failure": None if ok else case.likely_failure,
        "wall_seconds": round(elapsed, 1),
        "commands": commands,
        "tool_results": events,
        "kept_dir": failed_dir,
    }


def health_check(base_url: str, timeout: int) -> None:
    backend = LlamaServerBackend(base_url=base_url, timeout=timeout)
    if not backend.health():
        raise SystemExit(f"error: backend is not healthy at {base_url}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Orbit release confidence tests against a local backend.")
    parser.add_argument("--base-url", default=os.environ.get("ORBIT_BASE_URL", "http://127.0.0.1:18080"))
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("ORBIT_TEST_TIMEOUT", "300")))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("ORBIT_TEST_MAX_TOKENS", "320")))
    parser.add_argument("--only", action="append", default=[], help="Run only a case id. Can be repeated.")
    parser.add_argument("--keep-failed", action="store_true", help="Copy failed fixtures to /tmp for inspection.")
    parser.add_argument("--json-out", default="/tmp/orbit-release-confidence.json")
    parser.add_argument("--list", action="store_true", help="List cases and exit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = [case for case in CASES if not args.only or case.case_id in set(args.only)]
    if args.list:
        for case in CASES:
            print(f"{case.case_id}\t{case.title}")
        return 0
    if not selected:
        print("error: no matching cases", file=sys.stderr)
        return 2
    try:
        health_check(args.base_url, args.timeout)
    except (LlamaServerError, SystemExit) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    results: list[dict[str, object]] = []
    for index, case in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] {case.case_id}: {case.title}", flush=True)
        result = run_case(case, base_url=args.base_url, timeout=args.timeout, max_tokens=args.max_tokens, keep_failed=args.keep_failed)
        results.append(result)
        status = "PASS" if result["pass"] else "FAIL"
        print(f"  {status} {result['wall_seconds']}s - {result['note']}", flush=True)
        if not result["pass"]:
            print(f"  cause: {result['likely_failure']}", flush=True)
            if result.get("kept_dir"):
                print(f"  kept: {result['kept_dir']}", flush=True)

    passed = sum(1 for result in results if result["pass"])
    payload = {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "results": results,
    }
    Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"summary: {passed}/{len(results)} passed")
    print(f"json: {args.json_out}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
