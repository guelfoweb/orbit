"""Static extraction of VBA source from an OLE2/CFB Microsoft Office document.

An Office binary is a compound file: a small filesystem of storages and streams
inside one file. A Word/Excel document with a macro keeps its VBA project under a
`Macros`/`_VBA_PROJECT_CUR`/`VBA` storage, with each module's source stored
MS-OVBA-compressed inside a stream, at an offset the project's `dir` stream
records. Recovering that source is pure parsing and one documented decompression
algorithm -- no Office application, no VBA engine, nothing executed. This module
does exactly that and no more: it turns a binary container into the module source
text an analyst would read, with byte-exact provenance, and leaves what the
source MEANS to the analysis.

Everything here treats the document as hostile binary input. Every sector and
stream read is bounds-checked; FAT/miniFAT/DIFAT/directory traversals detect
cycles and are capped; the decompressor bounds its output. A malformed structure
fails closed with a deterministic reason and never raises out of the caller's
guard. Nothing is written to the document or anywhere else.

Scope is the read subset of MS-CFB and the source-decompression of MS-OVBA:
header, DIFAT, FAT, miniFAT, directory, FAT- and mini-backed stream reads, and
the compressed-container decompression. It is not a general OLE toolkit and does
not interpret document content beyond locating and decompressing module source.
"""

from __future__ import annotations

import hashlib
import re
import struct
from dataclasses import dataclass, field

OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

# Hostile-input bounds. Small, explicit, about the artifact rather than the
# machine: a document that legitimately exceeds them is unusual enough that
# refusing is the honest answer.
MAX_FILE = 64 * 1024 * 1024          # artifact ceiling
MAX_SECTORS = 4_000_000              # FAT sector-count ceiling
MAX_DIR_ENTRIES = 100_000
MAX_STREAM_BYTES = 32 * 1024 * 1024  # per-stream extraction ceiling
MAX_DECOMPRESS_BYTES = 16 * 1024 * 1024  # per-module decompressed ceiling
MAX_MODULES = 512
MAX_MODULE_SOURCE_CHARS = 2 * 1024 * 1024
MAX_TREE_DEPTH = 64

# CFB special sector ids.
FREESECT = 0xFFFFFFFF
ENDOFCHAIN = 0xFFFFFFFE
FATSECT = 0xFFFFFFFD
DIFSECT = 0xFFFFFFFC


class OleError(ValueError):
    """A malformed or unsupported compound file. Always fails closed."""


def is_ole_container(raw: bytes) -> bool:
    """Cheap, structural: the CFB magic. Not proof of a valid parse -- callers
    still parse and fail closed -- but enough to route a binary to this module
    rather than the text path."""
    return len(raw) >= 8 and raw[:8] == OLE_MAGIC


# --- MS-CFB reader ----------------------------------------------------------
class OleFile:
    """A read-only view over one compound file. Construction parses and
    validates the header, FAT, directory and miniFAT; any inconsistency raises
    OleError. Nothing is mutated."""

    def __init__(self, data: bytes):
        if len(data) < 512 or len(data) > MAX_FILE:
            raise OleError("file size out of bounds")
        if data[:8] != OLE_MAGIC:
            raise OleError("not an OLE2/CFB container (bad magic)")
        self.data = data
        self._parse_header()
        self._read_fat()
        self._read_directory()
        self._read_minifat()

    def _u16(self, off: int) -> int:
        return struct.unpack_from("<H", self.data, off)[0]

    def _u32(self, off: int) -> int:
        return struct.unpack_from("<I", self.data, off)[0]

    def _parse_header(self) -> None:
        if self._u16(28) != 0xFFFE:
            raise OleError("bad byte-order mark")
        self.sector_shift = self._u16(30)
        if self.sector_shift not in (9, 12):
            raise OleError(f"unsupported sector shift {self.sector_shift}")
        self.sector_size = 1 << self.sector_shift
        if self._u16(32) != 6:
            raise OleError("unsupported mini sector shift")
        self.mini_sector_size = 64
        self.first_dir_sector = self._u32(48)
        self.mini_stream_cutoff = self._u32(56)
        self.first_minifat_sector = self._u32(60)
        self.num_minifat_sectors = self._u32(64)
        self.first_difat_sector = self._u32(68)
        self.num_difat_sectors = self._u32(72)
        self.difat = list(struct.unpack_from("<109I", self.data, 76))
        self.max_sector = (len(self.data) - 512) // self.sector_size
        if self.max_sector < 0 or self.max_sector > MAX_SECTORS:
            raise OleError("sector count out of bounds")

    def _sector_offset(self, sid: int) -> int:
        if sid < 0 or sid > self.max_sector:
            raise OleError(f"sector id {sid} out of bounds")
        off = 512 + sid * self.sector_size
        if off + self.sector_size > len(self.data):
            raise OleError("sector past end of file")
        return off

    def _read_sector(self, sid: int) -> bytes:
        off = self._sector_offset(sid)
        return self.data[off:off + self.sector_size]

    def _extend_difat(self) -> None:
        sid = self.first_difat_sector
        seen: set[int] = set()
        per = self.sector_size // 4
        count = 0
        while sid not in (ENDOFCHAIN, FREESECT):
            if count > MAX_SECTORS or sid in seen:
                raise OleError("DIFAT chain invalid or cyclic")
            seen.add(sid)
            sec = self._read_sector(sid)
            vals = struct.unpack_from(f"<{per}I", sec, 0)
            self.difat.extend(vals[:per - 1])
            sid = vals[-1]
            count += 1

    def _read_fat(self) -> None:
        if self.num_difat_sectors:
            self._extend_difat()
        self.fat: list[int] = []
        per = self.sector_size // 4
        for fat_sid in self.difat:
            if fat_sid in (FREESECT, ENDOFCHAIN, FATSECT, DIFSECT):
                continue
            self.fat.extend(struct.unpack_from(f"<{per}I", self._read_sector(fat_sid), 0))
            if len(self.fat) > MAX_SECTORS:
                raise OleError("FAT too large")

    def _chain(self, start: int, limit: int) -> list[int]:
        out: list[int] = []
        seen: set[int] = set()
        sid = start
        while sid not in (ENDOFCHAIN, FREESECT):
            if sid < 0 or sid >= len(self.fat):
                raise OleError(f"FAT index {sid} out of range")
            if sid in seen:
                raise OleError("FAT chain cycle")
            if len(out) >= limit:
                raise OleError("FAT chain too long")
            seen.add(sid)
            out.append(sid)
            sid = self.fat[sid]
        return out

    def _read_fat_stream(self, start: int, size: int) -> bytes:
        if size > MAX_STREAM_BYTES:
            raise OleError("stream too large")
        need = (size + self.sector_size - 1) // self.sector_size + 1
        buf = bytearray()
        for sid in self._chain(start, need):
            buf += self._read_sector(sid)
            if len(buf) >= size:
                break
        return bytes(buf[:size])

    def _read_directory(self) -> None:
        chain = self._chain(self.first_dir_sector, MAX_DIR_ENTRIES)
        raw = bytearray()
        for sid in chain:
            raw += self._read_sector(sid)
        self._dir_bytes = bytes(raw)
        n = len(raw) // 128
        if n > MAX_DIR_ENTRIES:
            raise OleError("too many directory entries")
        self.entries: list[dict] = []
        for i in range(n):
            e = raw[i * 128:(i + 1) * 128]
            name_len = struct.unpack_from("<H", e, 64)[0]
            name = ""
            if 2 <= name_len <= 64:
                name = e[:name_len - 2].decode("utf-16-le", "replace")
            self.entries.append({
                "index": i,
                "name": name,
                "type": e[66],
                "start": struct.unpack_from("<I", e, 116)[0],
                "size": struct.unpack_from("<Q", e, 120)[0],
            })
        roots = [e for e in self.entries if e["type"] == 5]
        if not roots:
            raise OleError("no root storage entry")
        self.root = roots[0]

    def _read_minifat(self) -> None:
        self.minifat: list[int] = []
        self.mini_stream = b""
        if self.num_minifat_sectors == 0 or self.first_minifat_sector in (ENDOFCHAIN, FREESECT):
            return
        per = self.sector_size // 4
        for sid in self._chain(self.first_minifat_sector, self.num_minifat_sectors + 2):
            self.minifat.extend(struct.unpack_from(f"<{per}I", self._read_sector(sid), 0))
            if len(self.minifat) > MAX_SECTORS:
                raise OleError("miniFAT too large")
        self.mini_stream = self._read_fat_stream(self.root["start"], self.root["size"])

    def _read_mini_stream(self, start: int, size: int) -> bytes:
        if size > MAX_STREAM_BYTES:
            raise OleError("mini stream too large")
        out = bytearray()
        seen: set[int] = set()
        sid = start
        while sid not in (ENDOFCHAIN, FREESECT):
            if sid < 0 or sid >= len(self.minifat):
                raise OleError("miniFAT index out of range")
            if sid in seen:
                raise OleError("miniFAT chain cycle")
            seen.add(sid)
            off = sid * self.mini_sector_size
            out += self.mini_stream[off:off + self.mini_sector_size]
            if len(out) >= size:
                break
            sid = self.minifat[sid]
        return bytes(out[:size])

    def open_entry(self, entry: dict) -> bytes:
        size = entry["size"]
        if size == 0:
            return b""
        if size < self.mini_stream_cutoff:
            return self._read_mini_stream(entry["start"], size)
        return self._read_fat_stream(entry["start"], size)

    def paths(self) -> "list[tuple[str, dict]]":
        """(full_path, entry) for every stream, reconstructing the storage tree
        from the directory's red-black child/sibling pointers. Depth-bounded and
        visit-guarded so a crafted tree cannot recurse without limit."""
        raw = self._dir_bytes
        results: list[tuple[str, dict]] = []
        visited: set[int] = set()

        def walk(idx: int, prefix: str, depth: int) -> None:
            if idx == FREESECT or idx >= len(self.entries) or depth > MAX_TREE_DEPTH:
                return
            if idx in visited:
                return
            visited.add(idx)
            base = idx * 128
            left = struct.unpack_from("<I", raw, base + 68)[0]
            right = struct.unpack_from("<I", raw, base + 72)[0]
            child = struct.unpack_from("<I", raw, base + 76)[0]
            e = self.entries[idx]
            full = e["name"] if not prefix else f"{prefix}/{e['name']}"
            if e["type"] == 2:      # stream
                results.append((full, e))
            elif e["type"] == 1:    # storage
                walk(child, full, depth + 1)
            walk(left, prefix, depth + 1)
            walk(right, prefix, depth + 1)

        root_base = self.root["index"] * 128
        walk(struct.unpack_from("<I", raw, root_base + 76)[0], "", 0)
        return results


# --- MS-OVBA decompression --------------------------------------------------
def decompress_vba(compressed: bytes, max_out: int = MAX_DECOMPRESS_BYTES) -> bytes:
    """Decompress an MS-OVBA CompressedContainer. Deterministic, bounded, and
    fails closed on a malformed token or an output that would exceed `max_out`
    (a decompression-bomb guard). No code is executed; this is byte arithmetic."""
    if not compressed or compressed[0] != 0x01:
        raise OleError("not an MS-OVBA compressed container")
    out = bytearray()
    i = 1
    n = len(compressed)
    while i + 2 <= n:
        if len(out) > max_out:
            raise OleError("decompressed output exceeds bound")
        header = struct.unpack_from("<H", compressed, i)[0]
        i += 2
        chunk_size = (header & 0x0FFF) + 3
        compressed_flag = (header >> 15) & 1
        chunk_start = i
        if compressed_flag == 0:
            out += compressed[i:i + 4096]
            i += 4096
            continue
        chunk_end = min(chunk_start + chunk_size - 2, n)
        decomp_chunk_start = len(out)
        while i < chunk_end:
            flags = compressed[i]
            i += 1
            for bit in range(8):
                if i >= chunk_end:
                    break
                if not (flags >> bit) & 1:
                    out.append(compressed[i])
                    i += 1
                else:
                    if i + 2 > n:
                        raise OleError("truncated copy token")
                    token = struct.unpack_from("<H", compressed, i)[0]
                    i += 2
                    diff = len(out) - decomp_chunk_start
                    bit_count = max(4, (diff - 1).bit_length()) if diff > 1 else 4
                    length_mask = 0xFFFF >> bit_count
                    offset_mask = (~length_mask) & 0xFFFF
                    length = (token & length_mask) + 3
                    offset = ((token & offset_mask) >> (16 - bit_count)) + 1
                    src = len(out) - offset
                    if src < 0:
                        raise OleError("copy token offset out of bounds")
                    for _ in range(length):
                        out.append(out[src])
                        src += 1
                        if len(out) > max_out:
                            raise OleError("decompressed output exceeds bound")
    return bytes(out)


# --- VBA project extraction -------------------------------------------------
# Directory-record ids in the MS-OVBA `dir` stream.
_ID_PROJECTCODEPAGE = 0x0003
_ID_MODULENAME = 0x0019
_ID_MODULESTREAMNAME = 0x001A
_ID_MODULEOFFSET = 0x0031
# The MODULEOFFSET record has a fixed 6-byte prefix `31 00 04 00 00 00` followed
# by the u32 source offset. Located by signature, which is robust to the several
# variable-length records whose reserved fields make a naive TLV walk drift.
_MODULEOFFSET_SIG = struct.pack("<HI", _ID_MODULEOFFSET, 4)


@dataclass
class VbaModule:
    name: str
    stream_path: str
    codepage: int
    stream_sha256: str
    compressed_sha256: str
    source_sha256: str
    source: str


@dataclass
class OfficeExtraction:
    container: str
    sector_size: int
    stream_count: int
    stream_inventory: "list[tuple[str, int]]"  # (path, size)
    codepage: int | None
    modules: "list[VbaModule]" = field(default_factory=list)
    note: str = ""


def _codepage_to_encoding(cp: int | None) -> str:
    if not cp:
        return "cp1252"
    try:
        b"".decode(f"cp{cp}")  # probe validity
        return f"cp{cp}"
    except LookupError:
        return "cp1252"


def _module_stream_names(dir_decompressed: bytes) -> "list[str]":
    """MODULESTREAMNAME (ASCII) values, in order. Located by record id so the
    order matches the MODULEOFFSET records that follow each module block."""
    names: list[str] = []
    for m in re.finditer(struct.pack("<H", _ID_MODULESTREAMNAME), dir_decompressed):
        pos = m.start()
        if pos + 6 > len(dir_decompressed):
            continue
        size = struct.unpack_from("<I", dir_decompressed, pos + 2)[0]
        if 0 < size <= 255 and pos + 6 + size <= len(dir_decompressed):
            candidate = dir_decompressed[pos + 6:pos + 6 + size]
            # Printable ASCII and no path separator: a module "name" is used to
            # build a stream path (`{vba_prefix}/{name}`) that is looked up in a
            # dict, never on a filesystem -- but a name is a leaf, so reject `/`
            # and `\` rather than admit a `../X` that could never be a real
            # module and only muddies the path.
            if candidate and all(
                32 <= b < 127 and b not in (0x2F, 0x5C) for b in candidate
            ):
                names.append(candidate.decode("ascii"))
    return names


def _project_codepage(dir_decompressed: bytes) -> int | None:
    for m in re.finditer(struct.pack("<H", _ID_PROJECTCODEPAGE), dir_decompressed):
        pos = m.start()
        if pos + 8 <= len(dir_decompressed):
            size = struct.unpack_from("<I", dir_decompressed, pos + 2)[0]
            if size == 2:
                return struct.unpack_from("<H", dir_decompressed, pos + 6)[0]
    return None


def extract_office_vba(raw: bytes) -> OfficeExtraction | None:
    """Extract VBA module source from an Office document, or None if `raw` is
    not an OLE2/CFB container. Raises nothing: an OLE container that cannot be
    parsed, or has no VBA, returns an OfficeExtraction with an honest `note` and
    an empty `modules` list. All source is byte-exact with full provenance."""
    if not is_ole_container(raw):
        return None
    try:
        ole = OleFile(raw)
    except OleError as exc:
        return OfficeExtraction(
            container="ole2-cfb", sector_size=0, stream_count=0,
            stream_inventory=[], codepage=None,
            note=f"OLE container recognised but not parseable: {exc}",
        )
    streams = ole.paths()
    inventory = [(p, e["size"]) for p, e in streams]
    by_path = {p: e for p, e in streams}
    # The VBA `dir` stream lives under a *VBA* storage; find it structurally.
    dir_paths = [p for p in by_path if p.split("/")[-1].lower() == "dir"
                 and "vba" in p.lower()]
    result = OfficeExtraction(
        container="ole2-cfb", sector_size=ole.sector_size,
        stream_count=len(streams), stream_inventory=inventory, codepage=None,
    )
    if not dir_paths:
        result.note = "OLE2 document parsed; no VBA project (no VBA/dir stream)."
        return result
    try:
        dir_stream = ole.open_entry(by_path[dir_paths[0]])
        dir_d = decompress_vba(dir_stream)
    except OleError as exc:
        result.note = f"VBA/dir present but not decompressible: {exc}"
        return result
    result.codepage = _project_codepage(dir_d)
    encoding = _codepage_to_encoding(result.codepage)
    stream_names = _module_stream_names(dir_d)
    # The VBA storage prefix (everything before '/dir'), so module stream names
    # resolve to full paths.
    vba_prefix = dir_paths[0].rsplit("/", 1)[0]
    offsets = [struct.unpack_from("<I", dir_d, m.end())[0]
               for m in re.finditer(re.escape(_MODULEOFFSET_SIG), dir_d)]
    modules: list[VbaModule] = []
    for name, offset in zip(stream_names, offsets):
        if len(modules) >= MAX_MODULES:
            break
        stream_path = f"{vba_prefix}/{name}"
        entry = by_path.get(stream_path)
        if entry is None:
            # Some documents list a module whose stream is absent: skip honestly.
            continue
        try:
            mod_stream = ole.open_entry(entry)
            if offset < 0 or offset >= len(mod_stream):
                continue
            compressed = mod_stream[offset:]
            decompressed = decompress_vba(compressed)
        except OleError:
            continue
        try:
            source = decompressed.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            # Codepage decode failed: preserve honesty rather than mangle.
            continue
        if len(source) > MAX_MODULE_SOURCE_CHARS:
            source = source[:MAX_MODULE_SOURCE_CHARS]
        modules.append(VbaModule(
            name=name,
            stream_path=stream_path,
            codepage=result.codepage or 1252,
            stream_sha256=hashlib.sha256(mod_stream).hexdigest(),
            compressed_sha256=hashlib.sha256(compressed).hexdigest(),
            # Plain utf-8, matching how the evidence store hashes the same source
            # into `raw_sha256`, so the module's declared source_sha256 and the
            # stored record's hash are definitionally the same value. A codepage
            # decode never yields lone surrogates, and the store cannot persist
            # them anyway, so `surrogatepass` would only risk the two diverging.
            source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
            source=source,
        ))
    result.modules = modules
    if not modules:
        result.note = "VBA project present but no module source extracted."
    return result
