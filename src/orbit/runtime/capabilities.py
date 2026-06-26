from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Callable


DOCUMENT_TOOL_SPECS: tuple[tuple[str, str, str], ...] = (
    ("pdftotext", "document", "extract text from PDFs"),
    ("pandoc", "document", "convert supported document formats to text"),
    ("python3", "python", "run Python helpers for text extraction or inspection"),
    ("python", "python", "run Python helpers for text extraction or inspection"),
    ("file", "system", "identify file types and encodings"),
    ("unzip", "archive", "inspect zipped document containers such as docx/odt"),
    ("libreoffice", "document", "convert office documents in headless mode"),
    ("soffice", "document", "convert office documents in headless mode"),
    ("antiword", "document", "extract text from legacy .doc files"),
)

@dataclass(frozen=True)
class LocalCapability:
    name: str
    available: bool
    path: str | None
    category: str
    purpose: str


@dataclass(frozen=True)
class LocalCapabilities:
    items: tuple[LocalCapability, ...]

    def by_name(self, name: str) -> LocalCapability | None:
        for item in self.items:
            if item.name == name:
                return item
        return None

    def format_prompt_summary(self) -> str:
        available = ", ".join(item.name for item in self.items if item.available) or "none"
        unavailable = ", ".join(item.name for item in self.items if not item.available) or "none"
        return (
            f"Local tools available: {available}.\n"
            f"Unavailable: {unavailable}.\n"
            "Use only tools that are available or verify availability before use. "
            "Do not assume pdftotext, pandoc, libreoffice, soffice, antiword, unzip, file, python3, or python exist unless listed as available."
        )

    def format_tools_status(self) -> str:
        lines = ["Local capabilities", "------------------"]
        for item in self.items:
            status = "available" if item.available else "unavailable"
            path = item.path if item.path else "-"
            lines.append(f"{item.name:<12} {status:<11} {item.category:<8} {path:<30} {item.purpose}")
        return "\n".join(lines)


def discover_local_capabilities(which: Callable[[str], str | None] = shutil.which) -> LocalCapabilities:
    items: list[LocalCapability] = []
    for name, category, purpose in DOCUMENT_TOOL_SPECS:
        path = which(name)
        items.append(
            LocalCapability(
                name=name,
                available=path is not None,
                path=path,
                category=category,
                purpose=purpose,
            )
        )
    return LocalCapabilities(tuple(items))
