"""Strict declaration parsers shared by maintained catalog checks."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import re

def unique(rows: list[dict]) -> None:
    duplicates = [key for key, count in Counter(row["id"] for row in rows).items() if count != 1]
    if duplicates:
        raise RuntimeError(f"duplicate retained identifiers: {duplicates}")


def obligations(text: str, prefix: str, count: int) -> list[dict]:
    found = []
    for line in text.splitlines():
        match = re.match(r"\| (" + prefix + r"-\d{3})(?:\s|\|)", line)
        if match:
            found.append({"id": match[1], "sourceRecord": line})
    unique(found)
    if {row["id"] for row in found} != {f"{prefix}-{i:03d}" for i in range(1, count + 1)}:
        raise RuntimeError(f"{prefix} register coverage drifted")
    return found


def explicit_yaml_scalars_text(source: str, label: str) -> dict[str, str]:
    stack: list[tuple[int, str]] = []
    values: dict[str, str] = {}
    for number, line in enumerate(source.splitlines(), start=1):
        match = re.fullmatch(r"( *)([A-Za-z_][A-Za-z_0-9]*):(?:[ \t]*(.*))?", line)
        if match is None:
            continue
        indent, key, raw = len(match[1]), match[2], (match[3] or "").strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        dotted = ".".join([part for _, part in stack] + [key])
        if raw:
            if dotted in values:
                raise RuntimeError(f"duplicate explicit YAML scalar {dotted} in {label}:{number}")
            values[dotted] = raw
        else:
            stack.append((indent, key))
    return values


def explicit_yaml_scalars(path: Path) -> dict[str, str]:
    return explicit_yaml_scalars_text(path.read_text(), str(path))


def selector_tokens(observed: set[str], mode: str) -> tuple[set[str], set[str]]:
    if mode != "product_set_v1":
        return observed, set()
    tokens: set[str] = set()
    malformed: set[str] = set()
    for raw in observed:
        if not re.fullmatch(r"\[(?:[a-z0-9_]+(?:,\s*[a-z0-9_]+)*)?\]", raw):
            malformed.add(raw)
            continue
        tokens.update(part.strip() for part in raw[1:-1].split(",") if part.strip())
    return tokens, malformed
