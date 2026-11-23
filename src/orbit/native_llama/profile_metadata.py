"""Read profile identity without loading any native library into the process."""
from pathlib import Path

from .gguf_split import read_gguf_header
from .model_profiles import PROFILE_METADATA_KEYS, detect_native_model_profile


def inspect_gguf_profile(path: Path):
    keys = PROFILE_METADATA_KEYS | {"tokenizer.chat_template"}
    values = read_gguf_header(path, keys=frozenset(keys)).kv
    metadata = {key: (str(value).lower() if isinstance(value, bool) else str(value))
                for key, value in values.items() if key in PROFILE_METADATA_KEYS}
    return detect_native_model_profile(metadata, str(values.get("tokenizer.chat_template", "")))
