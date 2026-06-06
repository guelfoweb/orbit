from __future__ import annotations


DIM = "\033[2m"
RESET = "\033[0m"


def dim(text: str) -> str:
    return f"{DIM}{text}{RESET}"
