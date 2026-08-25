"""Built native libraries must resolve their siblings relative to themselves.

A runtime family built into one checkout used to name its own build directory
by absolute path in `DT_RUNPATH`. Copying that checkout copied the path with
it, so a second checkout on the same machine loaded its own `libllama` while
resolving `libggml` back into the first one. Two independent copies of
llama/ggml then shared an address space and one allocator freed what the other
had allocated, which surfaced as an abort during exit handlers rather than as
anything the tests could see.

These tests read the ELF dynamic section of the artifacts actually on disk.
Asserting on the build command instead would pass while the linker did
something else, which is the failure this is meant to catch.
"""

from __future__ import annotations

import struct
import unittest
from pathlib import Path

from orbit.native_llama.build_support import ORIGIN_RUNPATH, origin_relative_runpath
from orbit.native_llama.paths import (
    DEFAULT_VENDOR_BUILD_BIN,
    DEFAULT_VENDOR_LIB_DIR,
    DEFAULT_VENDOR_SHIM_DIR,
)


DT_NEEDED = 1
DT_RPATH = 15
DT_RUNPATH = 29
DT_STRTAB = 5
DT_SONAME = 14


def read_dynamic_entries(path: Path) -> dict[int, list[str]]:
    """`{d_tag: [string values]}` from an ELF64 little-endian shared object.

    Written against the file rather than a tool so the assertion cannot pass
    because `readelf` was absent or its output format changed.
    """
    data = path.read_bytes()
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise AssertionError(f"not an ELF64 little-endian object: {path}")

    e_phoff = struct.unpack_from("<Q", data, 0x20)[0]
    e_phentsize = struct.unpack_from("<H", data, 0x36)[0]
    e_phnum = struct.unpack_from("<H", data, 0x38)[0]

    dynamic: tuple[int, int] | None = None
    segments: list[tuple[int, int, int]] = []
    for index in range(e_phnum):
        offset = e_phoff + index * e_phentsize
        p_type = struct.unpack_from("<I", data, offset)[0]
        p_offset = struct.unpack_from("<Q", data, offset + 0x08)[0]
        p_vaddr = struct.unpack_from("<Q", data, offset + 0x10)[0]
        p_filesz = struct.unpack_from("<Q", data, offset + 0x20)[0]
        if p_type == 2:  # PT_DYNAMIC
            dynamic = (p_offset, p_filesz)
        elif p_type == 1:  # PT_LOAD
            segments.append((p_vaddr, p_offset, p_filesz))
    if dynamic is None:
        raise AssertionError(f"no PT_DYNAMIC segment in {path}")

    def to_offset(vaddr: int) -> int:
        for seg_vaddr, seg_offset, seg_size in segments:
            if seg_vaddr <= vaddr < seg_vaddr + seg_size:
                return seg_offset + (vaddr - seg_vaddr)
        raise AssertionError(f"vaddr {vaddr:#x} outside any PT_LOAD in {path}")

    raw: list[tuple[int, int]] = []
    offset, size = dynamic
    for position in range(offset, offset + size, 16):
        d_tag, d_val = struct.unpack_from("<Qq", data, position)
        if d_tag == 0:  # DT_NULL
            break
        raw.append((d_tag, d_val))

    strtab = next((value for tag, value in raw if tag == DT_STRTAB), None)
    if strtab is None:
        raise AssertionError(f"no DT_STRTAB in {path}")
    strtab_offset = to_offset(strtab)

    def string_at(index: int) -> str:
        end = data.index(b"\0", strtab_offset + index)
        return data[strtab_offset + index : end].decode("utf-8", "replace")

    entries: dict[int, list[str]] = {}
    for tag, value in raw:
        if tag in (DT_NEEDED, DT_RPATH, DT_RUNPATH, DT_SONAME):
            entries.setdefault(tag, []).append(string_at(value))
    return entries


def is_elf(path: Path) -> bool:
    """Whether `path` starts with the ELF magic."""
    with path.open("rb") as handle:
        return handle.read(4) == b"\x7fELF"


def elf_objects(directory: Path) -> list[Path]:
    """Real, non-symlink ELF objects in `directory`.

    Selected by ELF magic rather than by name: these directories also hold
    `.so.identity.json` sidecars, `.cpp` sources and plain executables, and
    matching by name would either skip the whole test on the first sidecar --
    leaving everything after it unchecked while still reporting success -- or
    miss the shim executables, which carry a RUNPATH just as the libraries do.
    """
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and not path.is_symlink() and is_elf(path)
    )


def built_libraries() -> list[Path]:
    """Every built native object Orbit loads, across all three directories.

    `vendor/shim` is included deliberately: it is a production load path, and
    it holds the only artifacts whose search entry is not a bare `$ORIGIN` but
    a hop to the build directory. Leaving it out would pass while precisely
    the harder value was wrong.
    """
    return [
        path
        for directory in (
            DEFAULT_VENDOR_BUILD_BIN,
            DEFAULT_VENDOR_LIB_DIR,
            DEFAULT_VENDOR_SHIM_DIR,
        )
        for path in elf_objects(directory)
    ]


NATIVE_PREFIXES = ("libggml", "libllama", "libmtmd", "liborbit")


class RunpathIsolationTests(unittest.TestCase):
    """What the linker actually wrote, not what the build meant to ask for."""

    def setUp(self) -> None:
        self.libraries = built_libraries()
        if not self.libraries:
            self.skipTest("native runtime is not built")

    def test_no_library_names_an_absolute_search_directory(self) -> None:
        """An absolute entry is what survives a copy and finds the other tree.

        Detected generically: any absolute path, not a particular checkout, so
        this keeps working on a machine whose build root is somewhere else.
        """
        for path in self.libraries:
            entries = read_dynamic_entries(path)
            for tag, label in ((DT_RUNPATH, "DT_RUNPATH"), (DT_RPATH, "DT_RPATH")):
                for value in entries.get(tag, []):
                    for element in value.split(":"):
                        if not element:
                            continue
                        self.assertFalse(
                            element.startswith("/"),
                            f"{path.name} {label} names an absolute directory: {element!r}",
                        )

    def test_libraries_with_native_dependencies_search_their_own_directory(self) -> None:
        """`$ORIGIN` is what makes a copied family resolve within itself."""
        checked = 0
        for path in self.libraries:
            entries = read_dynamic_entries(path)
            needs_family = any(
                needed.startswith(NATIVE_PREFIXES) for needed in entries.get(DT_NEEDED, [])
            )
            if not needs_family:
                continue
            checked += 1
            search = entries.get(DT_RUNPATH, []) + entries.get(DT_RPATH, [])
            elements = [part for value in search for part in value.split(":") if part]
            self.assertTrue(
                any(part.startswith(ORIGIN_RUNPATH) for part in elements),
                f"{path.name} depends on the family but does not search {ORIGIN_RUNPATH}: {elements}",
            )
        self.assertGreater(checked, 0, "no native library declared a family dependency")

    def test_family_dependencies_are_resolvable_along_the_search_path(self) -> None:
        """Every sibling an object names is findable where it says to look.

        `$ORIGIN` only helps if the directory it resolves to is complete; a
        missing soname would send the loader to the system search path, which
        is where a foreign family could be picked up again. Resolved through
        each object's own entry rather than assuming its directory, because
        the shims deliberately point at the build directory instead of at
        themselves.
        """
        for path in self.libraries:
            entries = read_dynamic_entries(path)
            needed_family = [
                needed
                for needed in entries.get(DT_NEEDED, [])
                if needed.startswith(NATIVE_PREFIXES)
            ]
            if not needed_family:
                continue
            search = entries.get(DT_RUNPATH, []) + entries.get(DT_RPATH, [])
            directories = [
                Path(part.replace(ORIGIN_RUNPATH, str(path.parent)))
                for value in search
                for part in value.split(":")
                if part
            ]
            self.assertTrue(directories, f"{path.name} declares no search path")
            for needed in needed_family:
                self.assertTrue(
                    any((directory / needed).exists() for directory in directories),
                    f"{path.name} needs {needed}, not found in {directories}",
                )


class PackagedRuntimeTests(unittest.TestCase):
    """The packaged runtime must answer its own dependencies.

    `vendor/lib` used to hold only the bare `libX.so` linker names while every
    dependent asks the loader for the SONAME `libX.so.0`. Those requests left
    the directory, and with an absolute RUNPATH they landed in `vendor/build`
    -- or, from a copied checkout, in a different checkout entirely. Once the
    RUNPATH is relative the same gap stops resolving at all, so the directory
    has to be complete rather than merely relative.
    """

    def setUp(self) -> None:
        if not DEFAULT_VENDOR_LIB_DIR.is_dir():
            self.skipTest("packaged runtime is not built")
        self.libraries = elf_objects(DEFAULT_VENDOR_LIB_DIR)
        if not self.libraries:
            self.skipTest("packaged runtime is not built")

    def test_every_native_dependency_is_present_in_the_packaged_directory(self) -> None:
        for path in self.libraries:
            entries = read_dynamic_entries(path)
            for needed in entries.get(DT_NEEDED, []):
                if not needed.startswith(NATIVE_PREFIXES):
                    continue
                self.assertTrue(
                    (DEFAULT_VENDOR_LIB_DIR / needed).exists(),
                    f"{path.name} needs {needed}, missing from the packaged runtime",
                )

    def test_packaged_libraries_do_not_name_an_absolute_search_directory(self) -> None:
        for path in self.libraries:
            entries = read_dynamic_entries(path)
            for tag in (DT_RUNPATH, DT_RPATH):
                for value in entries.get(tag, []):
                    for element in value.split(":"):
                        if element:
                            self.assertFalse(
                                element.startswith("/"),
                                f"{path.name} searches an absolute directory: {element!r}",
                            )


class RunpathBuildSettingTests(unittest.TestCase):
    """The build must ask for a relative entry, literally."""

    def test_artifacts_inside_the_package_get_a_relative_entry(self) -> None:
        """A hop between sibling directories survives being copied."""
        self.assertEqual(
            origin_relative_runpath(DEFAULT_VENDOR_BUILD_BIN, DEFAULT_VENDOR_BUILD_BIN),
            ORIGIN_RUNPATH,
        )
        for directory in (DEFAULT_VENDOR_SHIM_DIR, DEFAULT_VENDOR_LIB_DIR):
            with self.subTest(directory=directory.name):
                entry = origin_relative_runpath(directory, DEFAULT_VENDOR_BUILD_BIN)
                self.assertTrue(entry.startswith(ORIGIN_RUNPATH))
                self.assertFalse(entry.startswith("/"))

    def test_an_artifact_built_outside_the_package_does_not_climb_out(self) -> None:
        """A relative entry would only describe where things sat at build time.

        Nothing outside the package travels with it, so `..`-climbing back
        into the checkout is not relocatable -- it just looks like it is.
        """
        entry = origin_relative_runpath(Path("/tmp/elsewhere/bin"), DEFAULT_VENDOR_BUILD_BIN)

        self.assertNotIn("..", entry)
        self.assertFalse(entry.startswith(ORIGIN_RUNPATH))

    def test_origin_token_is_not_expanded_by_python(self) -> None:
        """The loader needs the token; an expanded value would be a real path."""
        self.assertEqual(ORIGIN_RUNPATH, "$ORIGIN")
        self.assertFalse(ORIGIN_RUNPATH.startswith("/"))
