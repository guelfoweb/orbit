"""Split GGUF sets (`<base>-00001-of-00003.gguf`): naming, header reading and
completeness validation.

llama.cpp opens the FIRST shard and resolves its siblings itself, so Orbit
treats shard 1 as the model and the set as one artifact: every shard must be
present, non-empty, a real GGUF, and agree on `split.count`, its own
`split.no` and `split.tensors.count`. Nothing here touches the network or the
native backend; it reads file headers only.
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian
GGUF_SUPPORTED_VERSIONS = (2, 3)
_SPLIT_NAME = re.compile(r"^(?P<base>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})(?P<ext>\.gguf)$", re.IGNORECASE)
# Metadata written by gguf-split into every shard.
SPLIT_COUNT_KEY = "split.count"
SPLIT_NO_KEY = "split.no"
SPLIT_TENSORS_KEY = "split.tensors.count"
_SPLIT_KEYS = frozenset({SPLIT_COUNT_KEY, SPLIT_NO_KEY, SPLIT_TENSORS_KEY})
_SCALAR_FORMATS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_HEADER_CAP = 64 * 1024 * 1024  # a GGUF header larger than this is not a model we serve
_MAX_ARRAY_DEPTH = 4  # gguf-split/convert never nest deeper; a crafted file must not recurse us


@dataclass(frozen=True)
class SplitName:
    base: str
    index: int  # 1-based, as written in the filename
    count: int
    width: int  # zero-padding width, preserved when deriving siblings
    ext: str

    def sibling(self, index: int) -> str:
        return f"{self.base}-{index:0{self.width}d}-of-{self.count:0{self.width}d}{self.ext}"

    @property
    def first(self) -> str:
        return self.sibling(1)

    def siblings(self) -> tuple[str, ...]:
        return tuple(self.sibling(i) for i in range(1, self.count + 1))


def parse_split_name(name: str) -> SplitName | None:
    """The split components of a filename, None for an ordinary single GGUF.

    Raises ValueError for a name that LOOKS split but cannot be a valid shard
    (index 0, index past the count, count 0), so a malformed name is never
    silently treated as a single file or expanded into nonsense.
    """
    match = _SPLIT_NAME.match(Path(name).name)
    if match is None:
        return None
    index, count = int(match.group("index")), int(match.group("count"))
    if count < 1 or index < 1 or index > count:
        raise ValueError(f"malformed split GGUF name: {name} (shard {index} of {count})")
    return SplitName(
        base=match.group("base"), index=index, count=count, width=len(match.group("count")), ext=match.group("ext")
    )


def split_sibling_names(name: str) -> tuple[str, ...]:
    """Every shard filename of the set `name` belongs to (just `name` when it is
    an ordinary GGUF), first shard first."""
    split = parse_split_name(name)
    if split is None:
        return (Path(name).name,)
    return split.siblings()


def is_secondary_shard(name: str) -> bool:
    try:
        split = parse_split_name(name)
    except ValueError:
        return False
    return split is not None and split.index != 1


@dataclass(frozen=True)
class GGUFHeader:
    version: int
    tensor_count: int
    kv: dict


def read_gguf_header(path: Path, *, keys: frozenset[str] | None = None) -> GGUFHeader:
    """Read the GGUF metadata header. Raises ValueError for anything that is not
    a well-formed GGUF (bad magic, unsupported version, truncated header).

    `keys` limits which key/value pairs are kept (arrays are always skipped
    cheaply); the whole header is still walked because the split keys sit at
    its end.
    """
    try:
        with path.open("rb") as handle:
            reader = _Reader(handle)
            magic, version = reader.unpack("<II")
            if magic != GGUF_MAGIC:
                raise ValueError(f"{path.name}: not a GGUF file (bad magic)")
            if version not in GGUF_SUPPORTED_VERSIONS:
                raise ValueError(f"{path.name}: unsupported GGUF version {version}")
            (tensor_count,) = reader.unpack("<Q")
            (kv_count,) = reader.unpack("<Q")
            if kv_count > 1_000_000:
                raise ValueError(f"{path.name}: implausible metadata count {kv_count}")
            kv: dict = {}
            for _ in range(kv_count):
                key = reader.string()
                (kind,) = reader.unpack("<I")
                value = reader.value(kind, keep=keys is None or key in keys)
                if keys is None or key in keys:
                    kv[key] = value
            return GGUFHeader(version=version, tensor_count=tensor_count, kv=kv)
    except _HeaderError as exc:
        raise ValueError(f"{path.name}: {exc}") from None
    except (struct.error, EOFError, OverflowError, MemoryError, RecursionError, TypeError) as exc:
        # Every way a crafted or damaged header can go wrong is one answer:
        # "not a header we accept", never a crash of the caller.
        raise ValueError(f"{path.name}: truncated or malformed GGUF header ({exc.__class__.__name__})") from None


class _HeaderError(ValueError):
    """A structural problem found while reading; reported with the file name."""


class _Reader:
    def __init__(self, handle) -> None:
        self.handle = handle
        self.consumed = 0

    def read(self, size: int) -> bytes:
        self.consumed += size
        if self.consumed > _HEADER_CAP:
            raise _HeaderError("GGUF header exceeds the supported size")
        data = self.handle.read(size)
        if len(data) != size:
            raise EOFError("unexpected end of file")
        return data

    def unpack(self, fmt: str):
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))

    def string(self) -> str:
        (length,) = self.unpack("<Q")
        # Metadata strings we merely skip over must not be able to fail the
        # parse; a token table with a non-UTF-8 entry is still a valid GGUF.
        return self.read(length).decode("utf-8", errors="replace")

    def value(self, kind: int, *, keep: bool, depth: int = 0):
        if kind == 8:
            return self.string()
        if kind == 9:
            if depth >= _MAX_ARRAY_DEPTH:
                raise _HeaderError("GGUF metadata arrays nested too deeply")
            (item_kind,) = self.unpack("<I")
            (length,) = self.unpack("<Q")
            if item_kind in _SCALAR_FORMATS and not keep:
                self.read(struct.calcsize(_SCALAR_FORMATS[item_kind]) * length)
                return None
            if length > _HEADER_CAP:
                raise _HeaderError("GGUF metadata array is implausibly long")
            items = [self.value(item_kind, keep=keep, depth=depth + 1) for _ in range(length)]
            return items if keep else None
        fmt = _SCALAR_FORMATS.get(kind)
        if fmt is None:
            raise _HeaderError(f"unknown GGUF value type {kind}")
        return self.unpack(fmt)[0]


@dataclass(frozen=True)
class SplitValidation:
    first: Path
    count: int
    shards: tuple[Path, ...]
    problems: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.problems


def validate_split_set(first: Path) -> SplitValidation:
    """Check the whole set a first shard belongs to. Ordinary single GGUFs
    validate trivially as a one-file set (their contract is unchanged)."""
    first = Path(first)
    try:
        split = parse_split_name(first.name)
    except ValueError as exc:
        return SplitValidation(first, 0, (first,), (str(exc),))
    if split is None:
        # An ordinary single GGUF keeps its historical contract (existence is
        # checked by the callers as before); only split sets are validated here.
        return SplitValidation(first, 1, (first,), ())
    if split.index != 1:
        first = first.with_name(split.first)
    shards = tuple(first.with_name(name) for name in split.siblings())
    problems: list[str] = []
    first_tensors: int | None = None
    for position, shard in enumerate(shards):
        file_problems = _file_problems(shard)
        if file_problems:
            problems.extend(file_problems)
            continue
        try:
            header = read_gguf_header(shard, keys=_SPLIT_KEYS)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        for key in (SPLIT_COUNT_KEY, SPLIT_NO_KEY, SPLIT_TENSORS_KEY):
            if key in header.kv and _split_int(header.kv[key]) is None:
                problems.append(f"{shard.name}: {key} is not an integer")
        count = _split_int(header.kv.get(SPLIT_COUNT_KEY))
        number = _split_int(header.kv.get(SPLIT_NO_KEY))
        tensors = _split_int(header.kv.get(SPLIT_TENSORS_KEY))
        if count != split.count:
            problems.append(f"{shard.name}: split.count is {count}, filename says {split.count}")
        if number != position:
            problems.append(f"{shard.name}: split.no is {number}, expected {position}")
        if tensors is not None:
            if first_tensors is None:
                first_tensors = tensors
            elif tensors != first_tensors:
                problems.append(f"{shard.name}: split.tensors.count is {tensors}, "
                                f"disagrees with the first shard ({first_tensors})")
    return SplitValidation(first, split.count, shards, tuple(problems))


def _split_int(value) -> int | None:
    """The split.* keys are integers; anything else (a crafted array, a string)
    reads as "not there" and is reported as a mismatch, never a crash."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def split_problems(path: Path) -> tuple[str, ...]:
    """The completeness problems of the set `path` belongs to; () when complete."""
    return validate_split_set(Path(path)).problems


def _file_problems(path: Path) -> list[str]:
    if not path.is_file():
        return [f"{path.name}: missing"]
    try:
        if path.stat().st_size == 0:
            return [f"{path.name}: empty file"]
    except OSError as exc:
        return [f"{path.name}: unreadable ({exc.strerror or exc})"]
    return []
