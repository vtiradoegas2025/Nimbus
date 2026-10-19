#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests/fixtures/configs/qualification"
DEFAULT_BINARY = ROOT / "bin/nimbus_sim"
QUALIFIED_MODE = "    mode: qualified"
EXPLORATORY_MODE = "    mode: exploratory"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file():
        raise RuntimeError(f"missing simulator: {binary}")

    yaml_files = sorted(FIXTURES.glob("*.yaml"))
    fixture_lines = {
        path: path.read_text(encoding="utf-8").splitlines() for path in yaml_files
    }
    missing_mode = sorted(
        path
        for path, lines in fixture_lines.items()
        if QUALIFIED_MODE not in lines and EXPLORATORY_MODE not in lines
    )
    conflicting_mode = sorted(
        path
        for path, lines in fixture_lines.items()
        if QUALIFIED_MODE in lines and EXPLORATORY_MODE in lines
    )
    if missing_mode or conflicting_mode:
        details = []
        if missing_mode:
            details.append(
                "missing execution mode: "
                + ", ".join(path.name for path in missing_mode)
            )
        if conflicting_mode:
            details.append(
                "conflicting execution modes: "
                + ", ".join(path.name for path in conflicting_mode)
            )
        raise RuntimeError("; ".join(details))

    qualified = [path for path, lines in fixture_lines.items() if QUALIFIED_MODE in lines]
    if not qualified:
        raise RuntimeError("no maintained qualified fixtures were discovered")

    stale_maturity_flags = sorted(
        path for path in qualified
        if any("acknowledge_maturity_limits:" in line for line in fixture_lines[path])
    )
    if stale_maturity_flags:
        raise RuntimeError(
            "Ready fixtures still require the obsolete maturity acknowledgment: "
            + ", ".join(path.name for path in stale_maturity_flags)
        )

    qualified_set = set(qualified)
    mislabeled = sorted(path for path in FIXTURES.glob("*_ready.yaml") if path not in qualified_set)
    if mislabeled:
        names = ", ".join(path.name for path in mislabeled)
        raise RuntimeError(f"Ready-named fixtures are not qualified: {names}")

    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="nimbus_ready_admission_") as temporary:
        temporary_root = Path(temporary)
        for path in qualified:
            result = subprocess.run(
                [str(binary), "--validate-only", "--config", str(path)],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if result.returncode != 0 or "[VALIDATE] PASS" not in result.stdout:
                failures.append(f"{path.name}:\n{result.stdout}")
            elif "[SCIENTIFIC INVALIDITY]" in result.stdout or "Science profile rejected" in result.stdout:
                failures.append(f"{path.name}: reported scientific invalidity:\n{result.stdout}")

            source = path.read_text(encoding="utf-8")
            explicit_false = source.replace(
                "    mode: qualified\n",
                "    mode: qualified\n    acknowledge_maturity_limits: false\n",
                1,
            )
            negative = temporary_root / path.name
            negative.write_text(explicit_false, encoding="utf-8")
            rejected = subprocess.run(
                [str(binary), "--validate-only", "--config", str(negative)],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if rejected.returncode == 0 or "explicit false" not in rejected.stdout:
                failures.append(
                    f"{path.name}: explicit false maturity acknowledgment did not fail closed:\n"
                    f"{rejected.stdout}"
                )

    if failures:
        raise RuntimeError(
            f"{len(failures)} of {len(qualified)} qualified fixtures failed admission:\n"
            + "\n".join(failures)
        )

    print(
        f"[ready-admission-matrix] PASS: {len(qualified)} maintained qualified "
        "fixtures passed ordinary no-maturity-flag admission and explicit-false rejection"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
