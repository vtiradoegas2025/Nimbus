#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ROOT / "configs"
DEFAULT_BINARY = ROOT / "bin/nimbus_sim"
QUALIFIED_MODE = "    mode: qualified"
EXPLORATORY_MODE = "    mode: exploratory"
RESOLUTIONS_M = (1000, 500, 450, 400, 350, 300, 250, 200, 150, 100, 50)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file():
        raise RuntimeError(f"missing simulator: {binary}")

    presets = sorted(
        path
        for resolution_m in RESOLUTIONS_M
        for path in (CONFIGS / f"{resolution_m}meterResolution").rglob("*.yaml")
    )
    if len(presets) != 440:
        raise RuntimeError(f"public preset catalog is incomplete: found {len(presets)} files")

    failures: list[str] = []
    for path in presets:
        lines = path.read_text(encoding="utf-8").splitlines()
        qualified = QUALIFIED_MODE in lines
        exploratory = EXPLORATORY_MODE in lines
        if qualified == exploratory:
            failures.append(f"{path.relative_to(ROOT)}: missing or conflicting execution mode")
            continue
        if any("acknowledge_maturity_limits:" in line for line in lines):
            failures.append(
                f"{path.relative_to(ROOT)}: obsolete maturity acknowledgment remains in public selection"
            )
            continue

        result = subprocess.run(
            [str(binary), "--validate-only", "--config", str(path)],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if result.returncode != 0 or "[VALIDATE] PASS" not in result.stdout:
            failures.append(f"{path.relative_to(ROOT)}:\n{result.stdout}")
        elif "[SCIENTIFIC INVALIDITY]" in result.stdout or "Science profile rejected" in result.stdout:
            failures.append(
                f"{path.relative_to(ROOT)}: reported scientific invalidity:\n{result.stdout}"
            )

    if failures:
        raise RuntimeError(
            f"{len(failures)} of {len(presets)} public presets failed admission:\n"
            + "\n".join(failures)
        )

    print(
        f"[public-preset-admission] PASS: {len(presets)} shipped presets "
        "declare execution mode and pass ordinary validate-only admission"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
