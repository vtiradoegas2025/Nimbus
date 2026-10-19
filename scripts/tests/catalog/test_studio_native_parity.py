#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import json
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.qualification.catalog_inventory import maintained_cases, inventory_counts
UI = ROOT / "ui"
DEFAULT_BINARY = ROOT / "bin/nimbus_sim"
QUALIFICATION = ROOT / "tests/fixtures/configs/qualification"
RESOLUTIONS_M = (1000, 500, 450, 400, 350, 300, 250, 200, 150, 100, 50)


def run(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, cwd=cwd, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file():
        raise RuntimeError(f"missing simulator: {binary}")

    qualified = sorted(
        path for path in QUALIFICATION.glob("*.yaml")
        if "    mode: qualified" in path.read_text(encoding="utf-8").splitlines()
    )
    public = sorted(
        path
        for resolution_m in RESOLUTIONS_M
        for path in (ROOT / "configs" / f"{resolution_m}meterResolution").rglob("*.yaml")
    )
    if len(qualified) != inventory_counts()["qualifiedFixtures"] or len(public) != 440:
        raise RuntimeError(
            f"catalog parity inventory drifted: qualified={len(qualified)}, public={len(public)}"
        )
    configurations = qualified + public

    native_failures: list[str] = []
    for path in configurations:
        result = run(
            [str(binary), "--validate-only", "--config", str(path)], ROOT
        )
        if result.returncode != 0 or "[VALIDATE] PASS" not in result.stdout:
            native_failures.append(f"{path.relative_to(ROOT)}:\n{result.stdout}")
    if native_failures:
        raise RuntimeError(
            f"native validation rejected {len(native_failures)} configurations:\n"
            + "\n".join(native_failures)
        )

    with tempfile.TemporaryDirectory(prefix="nimbus-studio-native-parity-") as temporary:
        temporary_root = Path(temporary)
        manifest = temporary_root / "configs.json"
        batch = temporary_root / "preflight-batch.json"
        manifest.write_text(
            json.dumps([str(path.resolve()) for path in configurations]),
            encoding="utf-8",
        )
        generated = run(
            [
                "node", "--experimental-strip-types",
                "--import", "../tests/ui/register-ts-resolve.mjs",
                "../tests/ui/config/generate_native_parity_batch.ts",
                str(binary), str(manifest), str(batch),
            ],
            UI,
        )
        if generated.returncode != 0:
            raise RuntimeError(f"Studio request generation failed:\n{generated.stdout}")
        parsed = json.loads(batch.read_text(encoding="utf-8"))
        if not isinstance(parsed, list) or len(parsed) != len(configurations):
            raise RuntimeError("Studio request batch is empty or incomplete")

        environment = os.environ.copy()
        environment["NIMBUS_PREFLIGHT_BATCH"] = str(batch)
        environment["NIMBUS_PREFLIGHT_BINARY"] = str(binary)
        rust = run(
            [
                "cargo", "test", "external_catalog_request_batch_matches_native",
                "--lib", "--", "--ignored", "--nocapture",
            ],
            UI / "src-tauri",
            environment,
        )
        if rust.returncode != 0:
            raise RuntimeError(f"Rust exact-catalog preflight failed:\n{rust.stdout}")

    print(
        "[catalog-studio-native-parity] PASS: TypeScript and Rust preflight "
        f"matched native admission for {len(qualified)} qualified fixtures and "
        f"{len(public)} shipped presets"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
