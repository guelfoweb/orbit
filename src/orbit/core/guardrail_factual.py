from __future__ import annotations

import base64
import binascii
import re
from typing import Any, Callable

from .text_utils import prefers_english_output, word_tokens


TITLE_ONLY_HINTS = ("title only", "page title only", "solo il titolo", "only the title")
FETCH_URL_HINTS = (
    "fetch this url",
    "open this url",
    "check this url",
    "check this site",
    "check the site",
    "read this page",
    "read this site",
    "open this link",
    "summarize this site",
    "summarize this page",
    "what this page says",
    "what this site says",
    "what is written here",
    "what's written here",
    "cosa c'è scritto qui",
    "cosa ce scritto qui",
    "che dice questa pagina",
    "che dice questo sito",
    "dammi un riassunto di questo link",
    "che c'è su questa pagina",
    "che cè su questa pagina",
    "mi leggi questo sito",
    "controlla",
    "apri",
    "leggimi",
    "leggi",
    "riassumi",
    "mostra",
    "riporta",
    "dimmi",
    "url",
    "sito",
    "site",
    "pagina",
    "page",
)
ONE_RESULT_HINTS = ("one result only", "one result", "un solo risultato", "single result")
VERSION_QUERY_HINTS = ("version", "versione", "__version__", "release")
PROJECT_METADATA_CANDIDATES = ("pyproject.toml", "package.json", "Cargo.toml", "README.md")
URL_RE = re.compile(r"https?://[^\s)>\]\"']+")


def seed_current_factual_tool(
    *,
    user_input: str,
    route: Any,
    registry: Any,
    messages: list[dict[str, Any]],
    metrics: Any,
    policy_state: Any,
    on_event: Any,
    has_recent_tool_result: Callable[[list[dict[str, Any]], str], bool],
    run_guardrail_tool: Callable[..., dict[str, Any]],
) -> None:
    if route.intent != "current_factual_lookup":
        return
    lowered = user_input.lower()
    explicit_url = _extract_explicit_url(user_input)
    if _looks_like_time_query(lowered):
        return
    wants_url_fetch = any(hint in lowered for hint in TITLE_ONLY_HINTS) or (
        explicit_url is not None and any(hint in lowered for hint in FETCH_URL_HINTS)
    )
    if wants_url_fetch:
        if explicit_url is None or has_recent_tool_result(messages, "fetch_url"):
            return
        run_guardrail_tool(
            name="fetch_url",
            arguments={"url": explicit_url, "max_links": 0},
            route=route,
            registry=registry,
            messages=messages,
            metrics=metrics,
            policy_state=policy_state,
            on_event=on_event,
            emit_route=True,
        )


def apply_deterministic_bounded_command(
    *,
    user_input: str,
    route: Any,
    registry: Any,
    messages: list[dict[str, Any]],
    metrics: Any,
    policy_state: Any,
    on_event: Any,
    run_guardrail_tool: Callable[..., dict[str, Any]],
) -> str | None:
    if route.intent != "bounded_command":
        return None
    lowered = user_input.lower()
    math_result = _deterministic_math_result(lowered)
    if math_result is not None:
        return math_result
    base64_result = _deterministic_base64_result(user_input)
    if base64_result is not None:
        return base64_result
    if _looks_like_system_info_request(lowered):
        return _system_info_result(
            user_input=user_input,
            route=route,
            registry=registry,
            messages=messages,
            metrics=metrics,
            policy_state=policy_state,
            on_event=on_event,
            run_guardrail_tool=run_guardrail_tool,
        )
    if "pwd" not in lowered:
        return None
    result = run_guardrail_tool(
        name="bash",
        arguments={"command": "pwd"},
        route=route,
        registry=registry,
        messages=messages,
        metrics=metrics,
        policy_state=policy_state,
        on_event=on_event,
        emit_route=True,
    )
    stdout = result.get("stdout")
    if result.get("ok") and isinstance(stdout, str) and stdout.strip():
        return stdout.strip()
    return None


def local_codebase_metadata_result(
    *,
    intent: str | None,
    user_input: str,
    messages: list[dict[str, Any]],
    codebase_inspection_intent: str,
    successful_read_results_in_current_turn: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    normalize_relative_path: Callable[[str], str],
) -> str | None:
    if intent != codebase_inspection_intent:
        return None
    lowered = user_input.lower()
    if not any(hint in lowered for hint in VERSION_QUERY_HINTS):
        return None
    for result in reversed(successful_read_results_in_current_turn(messages)):
        path = result.get("path")
        content = result.get("content")
        if not isinstance(path, str) or not isinstance(content, str):
            continue
        version = _extract_project_version(path, content, normalize_relative_path=normalize_relative_path)
        if version is not None:
            return f"Project version: {version}"
    return None


def local_tooling_concept_result(user_input: str) -> str | None:
    lowered = user_input.lower()
    if _deterministic_math_result(lowered) is not None:
        return None
    token_set = set(word_tokens(lowered))
    english = prefers_english_output(user_input)
    if _looks_like_email_send_request(lowered):
        if english:
            return (
                "A dedicated email or notification tool is not available in this setup. "
                "I should say that clearly, ask for an available alternative, or prepare the message text without claiming it was sent."
            )
        return (
            "In questo setup una tool dedicata per email o notifiche non e` disponibile. "
            "Dovrei dirlo chiaramente, proporre un'alternativa disponibile oppure preparare il testo del messaggio senza fingere di averlo inviato."
        )
    if "log" in token_set and ("critical" in token_set or "critici" in token_set or "errori" in token_set or "errors" in token_set):
        if english:
            return "- Inspect the file format and size first.\n- Read bounded chunks instead of the whole file at once.\n- Search for error, fatal, panic, or exception patterns.\n- Rank findings by severity and recency.\n- Return the critical lines, impact, and the next checks to run."
        return "- Controllare prima formato e dimensione del file.\n- Leggere chunk bounded invece dell'intero file in una volta.\n- Cercare pattern come error, fatal, panic o exception.\n- Ordinare i finding per gravità e recenza.\n- Restituire righe critiche, impatto e prossimi controlli utili."
    if not _looks_like_tooling_concept_prompt(lowered):
        return None
    if "simulate" in token_set or "simulare" in lowered or "conversation" in token_set or "conversazione" in lowered:
        if english:
            return "Example flow:\\n1. Search for external context with a web tool.\\n2. Read the local file that anchors the answer.\\n3. Run a bounded calculation if numbers are involved.\\n4. Merge the evidence into one grounded final reply."
        return "Flusso di esempio:\\n1. Cercare il contesto esterno con una tool web.\\n2. Leggere il file locale che ancora la risposta.\\n3. Eseguire un calcolo bounded se ci sono numeri.\\n4. Unire l'evidenza in una risposta finale grounded."
    if "tool call" in lowered or "tool calls" in lowered or "tool calling" in lowered or "strument" in lowered:
        if ("cosa sono" in lowered or "what are" in lowered or "explain" in lowered or "spieg" in lowered) and "decid" not in lowered:
            if english:
                return "Tool calls are structured requests that let an LLM ask the runtime to execute a real tool, then use the returned result in its next step."
            return "Le tool calls sono richieste strutturate con cui un LLM chiede al runtime di eseguire una tool reale e poi usa il risultato nel passo successivo."
    if ("email" in lowered or "message" in lowered or "messaggio" in lowered) and ("tool" in lowered or "strument" in lowered):
        if english:
            return (
                "- Check that an email or messaging tool actually exists.\n"
                "- Prepare structured arguments such as recipient, subject, channel, and body.\n"
                "- Ask for confirmation if sending the email or message is external or irreversible.\n"
                "- Call the tool and report success or failure from the real result."
            )
        return (
            "- Verificare che esista davvero una tool per email o messaggi.\n"
            "- Preparare argomenti strutturati come destinatario, oggetto, canale e corpo.\n"
            "- Chiedere conferma se l'invio dell'email o del messaggio è esterno o irreversibile.\n"
            "- Chiamare la tool e riportare successo o errore dal risultato reale."
        )
    if ("shell" in token_set or "comando" in token_set or "command" in token_set) and ("before" in token_set or "prima" in token_set or "step" in token_set or "steps" in token_set):
        if english:
            return "- Confirm the user's goal.\n- Check whether the command is necessary.\n- Validate targets and workdir boundaries.\n- Look for destructive side effects.\n- Prefer a preview or read-only command first."
        return "- Confermare l'obiettivo dell'utente.\n- Verificare se il comando è davvero necessario.\n- Validare target e confini della workdir.\n- Controllare eventuali side effect distruttivi.\n- Preferire prima un comando di preview o sola lettura."
    if ("decid" in lowered or "use" in token_set or "usare" in token_set) and ("text" in token_set or "testo" in token_set or "tool" in token_set or "strument" in lowered):
        if english:
            return "- Answer with text when the reply depends only on internal reasoning.\n- Use a tool when fresh data, filesystem access, shell execution, or external evidence is required.\n- Prefer the smallest safe tool that can resolve the request.\n- If tool evidence is enough, finalize from the tool result instead of guessing."
        return "- Rispondere solo con testo quando basta il ragionamento interno.\n- Usare una tool quando servono dati aggiornati, accesso al filesystem, esecuzione shell o evidenza esterna.\n- Preferire la tool più piccola e sicura che risolve la richiesta.\n- Se l'evidenza della tool basta, finalizzare dal risultato reale invece di indovinare."
    if ("delete" in token_set or "cancellare" in lowered or "rimuovere" in lowered) and ("directory" in token_set or "files" in token_set or "file" in token_set):
        if english:
            return "- Refuse to execute immediately.\n- Ask for explicit confirmation and the exact target.\n- Inspect the directory first.\n- Prefer a preview of what would be deleted.\n- Only then run the smallest bounded deletion path."
        return "- Non eseguire subito.\n- Chiedere conferma esplicita e target preciso.\n- Ispezionare prima la directory.\n- Preferire una preview di ciò che verrebbe cancellato.\n- Solo dopo usare il percorso di cancellazione bounded più piccolo."
    if "run_shell" in lowered and ("safe" in token_set or "sicura" in lowered or "sicuro" in lowered or "confirm" in token_set):
        if english:
            return "- Validate the exact command and arguments.\n- Check whether paths stay inside the allowed workspace.\n- Look for destructive verbs or shell operators.\n- Prefer a harmless inspection command first.\n- Ask for confirmation if the command can modify state."
        return "- Validare comando e argomenti esatti.\n- Controllare che i path restino nella workspace consentita.\n- Cercare verbi distruttivi o operatori shell.\n- Preferire prima un comando innocuo di ispezione.\n- Chiedere conferma se il comando modifica lo stato."
    if "format_disk" in lowered or ("non esiste" in lowered and "tool" in lowered) or ("does not exist" in lowered and "tool" in lowered):
        if english:
            return "I should not invent or execute a missing tool. I should say that the tool is unavailable, explain the limitation, and offer a safe alternative that actually exists."
        return "Non dovrei inventare o eseguire una tool mancante. Dovrei dire che la tool non è disponibile, spiegare il limite e proporre un'alternativa sicura che esiste davvero."
    if "error" in token_set or "errore" in token_set:
        if english:
            return "I should report the real tool error clearly, say which step failed, avoid pretending the action succeeded, and propose the next best recovery step or fallback."
        return "Dovrei riportare chiaramente l'errore reale della tool, indicare quale passo è fallito, evitare di fingere un successo e proporre il miglior passo successivo o fallback."
    return None


def local_current_factual_result(
    *,
    intent: str | None,
    user_input: str,
    messages: list[dict[str, Any]],
    latest_search_web_result_in_current_turn: Callable[[list[dict[str, Any]]], dict[str, Any] | None],
    latest_fetch_url_result_in_current_turn: Callable[[list[dict[str, Any]]], dict[str, Any] | None],
) -> str | None:
    if intent != "current_factual_lookup":
        return None
    lowered = user_input.lower()
    if any(hint in lowered for hint in ONE_RESULT_HINTS):
        search = latest_search_web_result_in_current_turn(messages)
        if search is not None:
            results = search.get("results")
            if isinstance(results, list):
                for item in results:
                    if not isinstance(item, dict):
                        continue
                    title = item.get("title")
                    url = item.get("url")
                    if isinstance(title, str) and title.strip() and isinstance(url, str) and url.strip():
                        return f"{title.strip()} - {url.strip()}"
    if any(hint in lowered for hint in TITLE_ONLY_HINTS):
        fetched = latest_fetch_url_result_in_current_turn(messages)
        if fetched is not None:
            title = fetched.get("title")
            if isinstance(title, str) and title.strip():
                return title.strip()
    return None

def _system_info_result(
    *,
    user_input: str,
    route: Any,
    registry: Any,
    messages: list[dict[str, Any]],
    metrics: Any,
    policy_state: Any,
    on_event: Any,
    run_guardrail_tool: Callable[..., dict[str, Any]],
) -> str | None:
    commands = [
        ("os_release", "cat /etc/os-release"),
        ("uname", "uname -srm"),
        ("lscpu", "lscpu"),
        ("meminfo", "grep MemTotal /proc/meminfo"),
    ]
    outputs: dict[str, dict[str, Any]] = {}
    for key, command in commands:
        outputs[key] = run_guardrail_tool(
            name="bash",
            arguments={"command": command},
            route=route,
            registry=registry,
            messages=messages,
            metrics=metrics,
            policy_state=policy_state,
            on_event=on_event,
            emit_route=(key == "os_release"),
        )
    os_name = _extract_os_name(outputs.get("os_release", {}))
    kernel = _extract_single_line_stdout(outputs.get("uname", {}))
    cpu_model, architecture, cpu_count = _extract_lscpu_info(outputs.get("lscpu", {}))
    memory = _extract_memtotal(outputs.get("meminfo", {}))
    english = prefers_english_output(user_input)
    lines: list[str] = []
    if english:
        if os_name:
            lines.append(f"- OS: {os_name}")
        if kernel:
            lines.append(f"- Kernel: {kernel}")
        if cpu_model:
            cpu_line = f"- CPU: {cpu_model}"
            details: list[str] = []
            if architecture:
                details.append(architecture)
            if cpu_count:
                details.append(f"{cpu_count} logical CPUs")
            if details:
                cpu_line += f" ({', '.join(details)})"
            lines.append(cpu_line)
        if memory:
            lines.append(f"- Memory: {memory}")
    else:
        if os_name:
            lines.append(f"- Sistema operativo: {os_name}")
        if kernel:
            lines.append(f"- Kernel: {kernel}")
        if cpu_model:
            cpu_line = f"- CPU: {cpu_model}"
            details = []
            if architecture:
                details.append(architecture)
            if cpu_count:
                details.append(f"{cpu_count} CPU logiche")
            if details:
                cpu_line += f" ({', '.join(details)})"
            lines.append(cpu_line)
        if memory:
            lines.append(f"- Memoria: {memory}")
    return "\n".join(lines) if lines else None


def _extract_project_version(
    path: str,
    content: str,
    *,
    normalize_relative_path: Callable[[str], str],
) -> str | None:
    normalized = normalize_relative_path(path).lower()
    if normalized == "pyproject.toml":
        match = re.search(r'(?m)^version\s*=\s*["\\\']([^"\\\']+)["\\\']', content)
        if match is not None:
            return match.group(1).strip()
    if normalized == "package.json":
        match = re.search(r'"version"\s*:\s*"([^"]+)"', content)
        if match is not None:
            return match.group(1).strip()
    if normalized == "cargo.toml":
        match = re.search(r'(?m)^version\s*=\s*["\\\']([^"\\\']+)["\\\']', content)
        if match is not None:
            return match.group(1).strip()
    if normalized.endswith("readme.md"):
        match = re.search(r'(?im)^version:\s*`?([^`\\n]+)`?', content)
        if match is not None:
            return match.group(1).strip()
    return None


def _looks_like_system_info_request(lowered: str) -> bool:
    token_set = set(word_tokens(lowered))
    machine_tokens = {"pc", "computer", "machine", "macchina", "laptop", "notebook", "portatile", "system", "sistema"}
    info_tokens = {
        "config", "configuration", "configurazione", "spec", "specs", "hardware", "info",
        "information", "informazioni", "caratteristiche", "setup",
    }
    own_tokens = {"this", "my", "questo", "questa", "mio", "mia"}
    return bool(token_set & machine_tokens) and bool(token_set & (info_tokens | own_tokens))


def _looks_like_email_send_request(lowered: str) -> bool:
    token_set = set(word_tokens(lowered))
    has_action = bool(token_set & {"send", "inviare", "invia", "notifica", "notification", "email", "mail"})
    has_external_target = "@" in lowered or "subject" in token_set or "oggetto" in token_set
    return has_action and has_external_target


def _looks_like_tooling_concept_prompt(lowered: str) -> bool:
    token_set = set(word_tokens(lowered))
    return bool(token_set & {"tool", "tools", "strumento", "strumenti", "agent", "agente", "llm", "shell", "error", "errore"}) or "run_shell" in lowered or "format_disk" in lowered


def _deterministic_math_result(lowered: str) -> str | None:
    token_set = set(word_tokens(lowered))
    if not (token_set & {"calculate", "calcolare", "compute", "calcola"}):
        return None
    numbers = [int(value) for value in re.findall(r"\d+", lowered)]
    if len(numbers) < 2:
        return None
    if not any(symbol in lowered for symbol in ("÷", "/", "divided")):
        return None
    left, right = numbers[:2]
    if right == 0:
        return None
    quotient = left / right
    if any(term in lowered for term in ("multiply", "moltiplic", "moltiplica")):
        multiplier = numbers[2] if len(numbers) >= 3 else right
        result = quotient * multiplier
        return f"{result:.1f}" if result % 1 else str(int(result))
    return f"{quotient:.6f}".rstrip("0").rstrip(".")


def _deterministic_base64_result(user_input: str) -> str | None:
    lowered = user_input.lower()
    if "base64" not in lowered:
        return None
    wants_decode = any(token in lowered for token in ("decode", "decodifica"))
    wants_encode = any(token in lowered for token in ("encode", "codifica"))
    if wants_decode == wants_encode:
        return None
    value = _extract_quoted_value(user_input)
    if value is None:
        return None
    if wants_decode:
        try:
            decoded = base64.b64decode(value.encode("ascii"), validate=True)
            text = decoded.decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, UnicodeEncodeError):
            return "The provided string is not valid UTF-8 base64 text."
        return f"The decoded string is `{text}`."
    encoded = base64.b64encode(value.encode("utf-8")).decode("ascii")
    return f"The base64 encoded string is `{encoded}`."


def _extract_quoted_value(text: str) -> str | None:
    match = re.search(r'"([^"]*)"', text)
    if match is not None:
        return match.group(1)
    match = re.search(r"'([^']*)'", text)
    if match is not None:
        return match.group(1)
    return None


def _extract_single_line_stdout(result: dict[str, Any]) -> str | None:
    stdout = result.get("stdout")
    if not isinstance(stdout, str) or not stdout.strip():
        return None
    return stdout.strip().splitlines()[0].strip()


def _extract_os_name(result: dict[str, Any]) -> str | None:
    stdout = result.get("stdout")
    if not isinstance(stdout, str) or not stdout.strip():
        return None
    match = re.search(r"(?m)^PRETTY_NAME=(?P<value>.+)$", stdout)
    if match is None:
        match = re.search(r"(?m)^NAME=(?P<value>.+)$", stdout)
    if match is None:
        return None
    return match.group("value").strip().strip('"').strip("'")


def _extract_lscpu_info(result: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    stdout = result.get("stdout")
    if not isinstance(stdout, str) or not stdout.strip():
        return None, None, None
    cpu_model = None
    architecture = None
    cpu_count = None
    for line in stdout.splitlines():
        if ":" not in line:
            continue
        key, value = [part.strip() for part in line.split(":", 1)]
        lowered_key = key.lower()
        if lowered_key == "model name" and value:
            cpu_model = value
        elif lowered_key == "architecture" and value:
            architecture = value
        elif lowered_key == "cpu(s)" and value and cpu_count is None:
            cpu_count = value
    return cpu_model, architecture, cpu_count


def _extract_memtotal(result: dict[str, Any]) -> str | None:
    stdout = result.get("stdout")
    if not isinstance(stdout, str) or not stdout.strip():
        return None
    match = re.search(r"MemTotal:\s*(\d+)\s*kB", stdout, re.IGNORECASE)
    if match is None:
        return None
    kib = int(match.group(1))
    gib = kib / (1024 * 1024)
    return f"{gib:.1f} GiB"


def _extract_explicit_url(user_input: str) -> str | None:
    match = URL_RE.search(user_input)
    if match is None:
        return None
    return match.group(0).strip()


def _looks_like_time_query(lowered: str) -> bool:
    token_set = set(word_tokens(lowered))
    time_tokens = {"time", "hour", "hours", "ora", "ore", "orario"}
    query_tokens = {"what", "current", "now", "adesso", "quanto", "che"}
    return bool(token_set & time_tokens) and bool(token_set & query_tokens)


def _is_search_verb(token: str) -> bool:
    return token.startswith("cerc") or token in {"search", "lookup", "look", "trova"}
