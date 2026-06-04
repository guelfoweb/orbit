from __future__ import annotations

from typing import Callable

from ..text_utils import word_tokens


DIRECTORY_TERMS = (
    "directory",
    "directories",
    "folder",
    "folders",
    "subdirectory",
    "subdirectories",
    "workdir",
    "workspace",
    "workspace root",
    "root",
    "working directory",
    "cartella",
    "cartelle",
    "sottocartella",
    "sottocartelle",
)
DIRECTORY_ONLY_TERMS = (
    "directories",
    "folder",
    "folders",
    "subdirectory",
    "subdirectories",
    "cartella",
    "cartelle",
    "sottocartella",
    "sottocartelle",
)
DIRECTORY_DISCOVERY_HINTS = (
    *DIRECTORY_TERMS,
    "quali file",
    "quali sono i file",
)
DIRECTORY_CONTENT_TERMS = (
    "are there",
    "contain",
    "contains",
    "content",
    "contents",
    "inside",
    "inside the workspace",
    "workspace root",
    "root",
    "files",
    "show",
    "list",
    "what are",
    "which are",
    "mostra",
    "contiene",
    "ci sono",
    "elenca",
    "quali",
    "quali sono",
    "there are",
)
DIRECTORY_CONTENT_HINTS = DIRECTORY_CONTENT_TERMS
DIRECTORY_METADATA_HINTS = (
    "newest",
    "latest",
    "oldest",
    "modified",
    "modification",
    "mtime",
    "size",
    "permissions",
    "permission",
    "exist",
    "exists",
    "recente",
    "modificato",
    "modifica",
    "dimensione",
    "dimensioni",
    "permessi",
    "esiste",
)
DIRECTORY_RECURSIVE_TERMS = (
    "directory structure",
    "full tree",
    "recursively",
    "recursive",
    "subtree",
    "tree",
    "tutto",
    "tutti i file",
    "tutte le cartelle",
    "ricorsiv",
    "struttura",
    "albero",
)
DIRECTORY_FOLLOWUP_QUESTION_TERMS = (
    "are there",
    "there are",
    "there is",
    "what are",
    "which are",
    "ci sono",
    "quali sono",
    "quali",
)


def local_assistant_identity_result(
    user_input: str,
    *,
    prefers_english_output: Callable[[str], bool],
) -> str | None:
    token_set = set(word_tokens(user_input))
    if not token_set:
        return None
    assistant_refs = {"tu", "ti", "your", "you", "sei", "are"}
    creator_tokens = {"creato", "creata", "created", "creator", "origine", "origin", "fatto", "built"}
    italian_identity_query = {"chi", "sei"}
    english_identity_query = {"who", "are", "you"}
    if token_set & assistant_refs and token_set & creator_tokens:
        if prefers_english_output(user_input):
            return "I am orbit, the local assistant for this CLI. My behavior is defined by the orbit project in this workspace."
        return "Sono orbit, l'assistente locale di questa CLI. Il mio comportamento e` definito dal progetto orbit in questa workspace."
    if italian_identity_query <= token_set or english_identity_query <= token_set:
        if prefers_english_output(user_input):
            return "I am orbit, the local assistant for this CLI and workspace."
        return "Sono orbit, l'assistente locale di questa CLI e workspace."
    return None


def local_pure_chitchat_result(user_input: str) -> str | None:
    token_set = set(word_tokens(user_input))
    if not token_set:
        return None
    italian_greeting_tokens = {"ciao", "buongiorno", "buonasera"}
    english_greeting_tokens = {"hello", "hi", "hey"}
    greeting_tokens = italian_greeting_tokens | english_greeting_tokens
    italian_thanks_tokens = {"grazie"}
    english_thanks_tokens = {"thanks", "thank"}
    thanks_tokens = italian_thanks_tokens | english_thanks_tokens
    if token_set <= greeting_tokens:
        if token_set & english_greeting_tokens:
            return "Hello! How can I help?"
        return "Ciao! Come posso aiutarti?"
    if token_set <= thanks_tokens or (token_set & thanks_tokens and len(token_set) <= 4):
        if token_set & english_thanks_tokens:
            return "You're welcome."
        return "Prego."
    if len(token_set) == 1:
        english_match = _matches_typos(token_set, english_greeting_tokens)
        italian_match = _matches_typos(token_set, italian_greeting_tokens)
        if english_match and not italian_match:
            return "Hello! How can I help?"
        if italian_match and not english_match:
            return "Ciao! Come posso aiutarti?"
        if english_match or italian_match:
            if english_match:
                return "Hello! How can I help?"
            return "Ciao! Come posso aiutarti?"
    return None


def assistant_identity_system_prompt(
    user_input: str,
    *,
    prefers_english_output: Callable[[str], bool],
) -> str | None:
    token_set = set(word_tokens(user_input))
    if not token_set:
        return None
    assistant_refs = {"tu", "ti", "your", "you", "sei", "are", "te", "yourself"}
    intro_tokens = {"presentati", "descriviti", "describe", "introduce", "presentation", "description"}
    if not (token_set & assistant_refs or token_set & {"ciao", "hello", "hi", "hey"}):
        return None
    if not token_set & intro_tokens:
        return None
    if prefers_english_output(user_input):
        return (
            "Describe yourself briefly as orbit, the local assistant for this CLI and workspace. "
            "Use one natural sentence in English. "
            "Do not mention the current time, hidden placeholders, vendors, training details, or capabilities that were not asked for."
        )
    return (
        "Descriviti brevemente come orbit, l'assistente locale di questa CLI e workspace. "
        "Usa una sola frase naturale in italiano. "
        "Non menzionare ora corrente, placeholder nascosti, vendor, dettagli di training o capacita` non richieste."
    )


def local_directory_listing_result(
    *,
    intent: str | None,
    text_document_intent: str,
    user_input: str,
    messages: list[dict[str, object]],
    recent_listed_paths: Callable[[list[dict[str, object]], int], list[str]],
    recent_listed_directory_paths: Callable[[list[dict[str, object]], int], list[str]],
    prefers_english_output: Callable[[str], bool],
) -> str | None:
    if not needs_directory_discovery(user_input):
        return None
    if intent != text_document_intent:
        if not _directory_listing_followup_question(user_input):
            return None
        if not recent_listed_paths(messages, 1):
            return None
    if _directory_listing_prefers_tree(user_input):
        listed = recent_listed_paths(messages, 80)
        if not listed:
            return None
        directories = set(recent_listed_directory_paths(messages, 80))
        return _format_tree_listing(listed, directories)
    if directory_listing_wants_recursive(user_input) and not _directory_listing_prefers_directories(user_input):
        listed = recent_listed_paths(messages, 80)
        if not listed:
            return None
        if _directory_listing_prefers_mixed_entries(user_input):
            return ", ".join(listed)
        files_only = [path for path in listed if "." in path.rsplit("/", 1)[-1]]
        if files_only:
            return ", ".join(files_only)
        return ", ".join(listed)
    if _directory_listing_prefers_directories(user_input):
        listed = recent_listed_directory_paths(messages, 8)
        if not listed:
            return _empty_directory_listing_message(user_input, directories_only=True, prefers_english_output=prefers_english_output)
        return ", ".join(listed)
    if _directory_listing_prefers_mixed_entries(user_input):
        listed = recent_listed_paths(messages, 12)
        if not listed:
            return None
        if _directory_listing_prefers_separated_entries(user_input):
            directories = set(recent_listed_directory_paths(messages, 12))
            files = [path for path in listed if path not in directories]
            directory_list = [path for path in listed if path in directories]
            return _format_separated_listing(
                files=files,
                directories=directory_list,
                english=prefers_english_output(user_input),
            )
        return ", ".join(listed)
    listed = recent_listed_paths(messages, 5)
    if not listed:
        return None
    files_only = [path for path in listed if "." in path.rsplit("/", 1)[-1]]
    if files_only:
        return ", ".join(files_only[:5])
    return ", ".join(listed)


def needs_directory_discovery(user_input: str) -> bool:
    lowered = user_input.strip().lower()
    if not lowered:
        return False
    if not any(hint in lowered for hint in DIRECTORY_DISCOVERY_HINTS):
        return False
    if not any(hint in lowered for hint in DIRECTORY_CONTENT_HINTS) and not _short_directory_question(lowered):
        return False
    if any(hint in lowered for hint in DIRECTORY_METADATA_HINTS):
        return False
    if any(token in lowered for token in (".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", "/", "\\")):
        return False
    return True


def directory_listing_target_from_recent_listing(
    user_input: str,
    messages: list[dict[str, object]],
    recent_listed_directory_paths: Callable[[list[dict[str, object]], int], list[str]],
) -> str | None:
    lowered = user_input.strip().lower()
    if not lowered:
        return None
    if not any(term in lowered for term in DIRECTORY_CONTENT_TERMS):
        return None
    for path in recent_listed_directory_paths(messages, 80):
        name = path.rstrip("/").rsplit("/", 1)[-1]
        if not name:
            continue
        candidates = (
            f"{path.lower()} directory",
            f"{path.lower()} folder",
            f"{name.lower()} directory",
            f"{name.lower()} folder",
            f"directory {path.lower()}",
            f"folder {path.lower()}",
            f"cartella {path.lower()}",
            f"cartella {name.lower()}",
        )
        if any(candidate in lowered for candidate in candidates):
            return path
    return None


def directory_listing_wants_recursive(user_input: str) -> bool:
    lowered = user_input.strip().lower()
    return any(term in lowered for term in DIRECTORY_RECURSIVE_TERMS)


def _directory_listing_prefers_directories(user_input: str) -> bool:
    lowered = user_input.strip().lower()
    if (
        "files and directories" in lowered
        or "files from directories" in lowered
        or "separating files" in lowered
        or "separate files" in lowered
        or "file e cartelle" in lowered
        or "file e directory" in lowered
    ):
        return False
    explicit_directory_only_terms = ("non file", "not files", "instead of files", "rather than files")
    if any(term in lowered for term in explicit_directory_only_terms):
        return True
    return any(term in lowered for term in DIRECTORY_ONLY_TERMS)


def _directory_listing_prefers_mixed_entries(user_input: str) -> bool:
    lowered = user_input.strip().lower()
    if "files and directories" in lowered or "file e cartelle" in lowered or "file e directory" in lowered:
        return True
    mixed_terms = ("cosa contiene", "what does", "contains", "contain", "contents", "content", "inside", "contiene")
    return any(term in lowered for term in mixed_terms) and not _directory_listing_prefers_directories(user_input)


def _directory_listing_prefers_separated_entries(user_input: str) -> bool:
    lowered = user_input.strip().lower()
    separated_terms = (
        "separating files from directories",
        "separate files from directories",
        "separating files and directories",
        "separate files and directories",
        "separati",
        "separa file",
    )
    return any(term in lowered for term in separated_terms)


def _format_separated_listing(*, files: list[str], directories: list[str], english: bool) -> str:
    file_title = "Files" if english else "File"
    directory_title = "Directories" if english else "Directory"
    file_value = ", ".join(files) if files else "none"
    directory_value = ", ".join(directories) if directories else "none"
    return f"{file_title}: {file_value}\n{directory_title}: {directory_value}"


def _directory_listing_prefers_tree(user_input: str) -> bool:
    lowered = user_input.strip().lower()
    tree_terms = ("tree", "directory structure", "full structure", "struttura", "albero")
    return any(term in lowered for term in tree_terms)


def _directory_listing_followup_question(user_input: str) -> bool:
    lowered = user_input.strip().lower()
    return any(term in lowered for term in DIRECTORY_ONLY_TERMS) and (
        any(term in lowered for term in DIRECTORY_FOLLOWUP_QUESTION_TERMS) or _short_directory_question(lowered)
    )


def _short_directory_question(lowered_input: str) -> bool:
    tokens = set(word_tokens(lowered_input))
    if len(tokens) > 4:
        return False
    question_tokens = {"what", "which", "quali"}
    directory_tokens = {
        "directories",
        "folders",
        "subdirectories",
        "cartelle",
        "sottocartelle",
    }
    return bool(tokens & question_tokens) and bool(tokens & directory_tokens)


def _format_tree_listing(paths: list[str], directories: set[str]) -> str:
    lines = ["."]
    for path in sorted(set(paths)):
        parts = [part for part in path.split("/") if part]
        if not parts:
            continue
        indent = "  " * (len(parts) - 1)
        suffix = "/" if path in directories else ""
        lines.append(f"{indent}- {parts[-1]}{suffix}")
    return "\n".join(lines)


def _empty_directory_listing_message(
    user_input: str,
    *,
    directories_only: bool,
    prefers_english_output: Callable[[str], bool],
) -> str:
    if not directories_only:
        return "No matching entries found."
    if prefers_english_output(user_input):
        return "There are no subdirectories in the current working directory."
    return "Non ci sono sottocartelle nella directory di lavoro corrente."


def _matches_typos(token_set: set[str], candidates: set[str]) -> bool:
    for token in token_set:
        if any(_edit_distance(token, candidate) <= 1 for candidate in candidates):
            return True
    return False


def _edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if _is_adjacent_transposition(left, right):
        return 1
    if abs(len(left) - len(right)) > 1:
        return 2
    previous = list(range(len(right) + 1))
    for i, lch in enumerate(left, start=1):
        current = [i]
        row_min = i
        for j, rch in enumerate(right, start=1):
            cost = 0 if lch == rch else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
            row_min = min(row_min, current[-1])
        if row_min > 1:
            return 2
        previous = current
    return previous[-1]


def _is_adjacent_transposition(left: str, right: str) -> bool:
    if len(left) != len(right) or len(left) < 2:
        return False
    mismatches = [index for index, (lch, rch) in enumerate(zip(left, right)) if lch != rch]
    if len(mismatches) != 2:
        return False
    i, j = mismatches
    return j == i + 1 and left[i] == right[j] and left[j] == right[i]
