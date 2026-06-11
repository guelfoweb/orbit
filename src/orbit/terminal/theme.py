from __future__ import annotations


DIM = "\033[2m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def dim(text: str) -> str:
    return f"{DIM}{text}{RESET}"


def yellow_dim(text: str) -> str:
    return f"{DIM}{YELLOW}{text}{RESET}"
