from __future__ import annotations


DIM = "\033[2m"
CYAN = "\033[36m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def accent(text: str) -> str:
    return f"{CYAN}{text}{RESET}"


def dim(text: str) -> str:
    return f"{DIM}{text}{RESET}"


def yellow_dim(text: str) -> str:
    return f"{DIM}{YELLOW}{text}{RESET}"


def danger(text: str) -> str:
    return f"{RED}{text}{RESET}"
