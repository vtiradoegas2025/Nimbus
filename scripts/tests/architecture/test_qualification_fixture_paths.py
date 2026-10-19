#!/usr/bin/env python3
"""Reject references to the removed legacy test-fixture tree."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_ROOTS = (
    ROOT / "scripts" / "qualification",
    ROOT / "scripts" / "tests",
)
FORBIDDEN = (
    'ROOT / "configs" / "test"',
    "ROOT / 'configs' / 'test'",
    "--config=configs/test/",
    'CONFIG="configs/test/',
    "CONFIG='configs/test/",
)


def main() -> int:
    stale: list[str] = []
    for root in SCRIPT_ROOTS:
        for path in sorted(root.rglob("*")):
            if (
                path.resolve() == Path(__file__).resolve()
                or path.suffix not in {".py", ".sh"}
            ):
                continue
            text = path.read_text(encoding="utf-8")
            if any(value in text for value in FORBIDDEN):
                stale.append(path.relative_to(ROOT).as_posix())

    if stale:
        print(
            "[qualification-fixtures] FAIL: legacy configs/test references: "
            + ", ".join(stale)
        )
        return 1

    print("[qualification-fixtures] PASS: script fixtures use maintained paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
