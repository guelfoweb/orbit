"""Explicit co-located native families; one family is loaded per process."""
from dataclasses import dataclass
from pathlib import Path

from .build_support import PACKAGE_NATIVE_ROOT


@dataclass(frozen=True)
class NativeFamily:
    name: str
    root: Path
    abi_profile: str


def native_family(name: str = "legacy") -> NativeFamily:
    if name == "legacy":
        return NativeFamily(name, PACKAGE_NATIVE_ROOT, "legacy")
    if name == "mainline-support":
        return NativeFamily(name, PACKAGE_NATIVE_ROOT / "families" / "mainline_support",
                            "mainline-b12611")
    raise ValueError(f"unknown native backend family: {name}")
