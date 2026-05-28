from __future__ import annotations

from typing import Callable

from .text_utils import word_tokens


DIRECTORY_DISCOVERY_HINTS = (
    "directory",
    "folder",
    "folders",
    "directories",
    "workdir",
    "workspace",
    "working directory",
    "cartella",
    "cartelle",
    "quali file",
    "quali sono i file",
)
DIRECTORY_CONTENT_HINTS = (
    "contain",
    "contains",
    "content",
    "contents",
    "inside",
    "files",
    "show",
    "list",
    "mostra",
    "contiene",
    "ci sono",
    "elenca",
)
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
    if intent != text_document_intent:
        return None
    if not needs_directory_discovery(user_input):
        return None
    if _directory_listing_prefers_directories(user_input):
        listed = recent_listed_directory_paths(messages, 8)
        if not listed:
            return _empty_directory_listing_message(user_input, directories_only=True, prefers_english_output=prefers_english_output)
        return ", ".join(listed)
    if _directory_listing_prefers_mixed_entries(user_input):
        listed = recent_listed_paths(messages, 12)
        if not listed:
            return None
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
    if not any(hint in lowered for hint in DIRECTORY_CONTENT_HINTS):
        return False
    if any(hint in lowered for hint in DIRECTORY_METADATA_HINTS):
        return False
    if any(token in lowered for token in (".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", "/", "\\")):
        return False
    return True


def _directory_listing_prefers_directories(user_input: str) -> bool:
    lowered = user_input.strip().lower()
    if "files and directories" in lowered or "file e cartelle" in lowered or "file e directory" in lowered:
        return False
    directory_terms = ("directories", "folder", "folders", "cartella", "cartelle")
    explicit_directory_only_terms = ("non file", "not files", "instead of files", "rather than files")
    if any(term in lowered for term in explicit_directory_only_terms):
        return True
    return any(term in lowered for term in directory_terms)


def _directory_listing_prefers_mixed_entries(user_input: str) -> bool:
    lowered = user_input.strip().lower()
    if "files and directories" in lowered or "file e cartelle" in lowered or "file e directory" in lowered:
        return True
    mixed_terms = ("cosa contiene", "what does", "contains", "contain", "contents", "content", "inside", "contiene")
    return any(term in lowered for term in mixed_terms) and not _directory_listing_prefers_directories(user_input)


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
