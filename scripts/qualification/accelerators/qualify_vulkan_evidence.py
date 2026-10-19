#!/usr/bin/env python3
"""Run and seal the complete physical Vulkan qualification umbrella."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINARY = ROOT / "bin" / "nimbus_sim"
DEFAULT_MATRIX = (
    ROOT / "reports" / "qualification" /
    "vulkan-combination-matrix-current.json"
)
DEFAULT_REPORT = (
    ROOT / "reports" / "qualification" /
    "vulkan-qualification-current.json"
)
DEFAULT_LOG = (
    ROOT / "reports" / "qualification" /
    "vulkan-qualification-current.log"
)
REQUIRED_MARKERS = (
    "SPIR-V verification passed: 52 sources/artifacts",
    "Kernel manifest verification passed: 50 compute operations",
    "[gpu-residency-equivalence] RESULT: PASS",
    "[vulkan-strict] RESULT: PASS",
    "[vulkan-report] RESULT: PASS",
    "[vulkan-restart] RESULT: PASS duration=12s checkpoint=6s",
    "[vulkan-restart] RESULT: PASS duration=140s checkpoint=70s",
    "owner_identity_checks=25500 mixed_physics=yes",
    "[vulkan-combinations] RESULT: PASS cases=168 physical=90 rejected=78",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str:
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            + result.stderr[-5000:]
        )
    return result.stdout.strip()


def command_bytes(command: list[str]) -> bytes:
    result = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            + result.stderr.decode("utf-8", errors="replace")[-5000:]
        )
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument(
        "--make-arg",
        action="append",
        default=[],
        help="additional make assignment passed before qualify-vulkan",
    )
    args = parser.parse_args()

    binary = args.binary.resolve()
    matrix_path = args.matrix.resolve()
    report_path = args.report.resolve()
    log_path = args.log.resolve()
    for required in (binary, ROOT / "Makefile"):
        if not required.is_file():
            parser.error(f"required input not found: {required}")

    command = ["make", *args.make_arg, "qualify-vulkan"]
    started = time.monotonic()
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.monotonic() - started
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"Vulkan umbrella failed ({result.returncode}); log={log_path}\n"
            + result.stdout[-8000:]
        )

    missing = [marker for marker in REQUIRED_MARKERS if marker not in result.stdout]
    if missing:
        raise RuntimeError(
            f"Vulkan umbrella omitted required PASS markers: {missing}; "
            f"log={log_path}"
        )
    if not matrix_path.is_file():
        raise RuntimeError(f"combination matrix was not produced: {matrix_path}")
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if (
        matrix.get("status") != "pass"
        or matrix.get("case_count") != 168
        or matrix.get("physical_pass_count") != 90
        or matrix.get("expected_preflight_rejection_count") != 78
    ):
        raise RuntimeError("combination matrix summary does not match the gate")
    binary_hash = sha256_file(binary)
    if matrix.get("provenance", {}).get("binary_sha256") != binary_hash:
        raise RuntimeError(
            "combination matrix was not produced by the qualified binary"
        )

    catalog = command_bytes([str(binary), "--scheme-catalog=json"])
    catalog_hash = hashlib.sha256(catalog).hexdigest()
    if matrix.get("provenance", {}).get("scheme_catalog_sha256") != catalog_hash:
        raise RuntimeError(
            "combination matrix compiled-catalog hash does not match"
        )
    source_revision = command_output(["git", "rev-parse", "HEAD"])
    source_status = command_output([
        "git",
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        ".",
        ":(exclude)reports/**",
    ])
    source_worktree_clean = not bool(source_status)

    report = {
        "schema": "nimbus.vulkan-qualification-evidence.v1",
        "status": "pass",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_wall_seconds": elapsed,
        "command": command,
        "source_revision": source_revision,
        "source_worktree_clean": source_worktree_clean,
        "source_change_count": len(source_status.splitlines()),
        "cleanliness_exclusions": ["reports/**"],
        "release_record_eligible": source_worktree_clean,
        "physical_device": matrix["physical_device"],
        "gate_markers": list(REQUIRED_MARKERS),
        "artifacts": {
            "binary": str(binary.relative_to(ROOT)),
            "binary_sha256": binary_hash,
            "qualification_log": str(log_path.relative_to(ROOT)),
            "qualification_log_sha256": sha256_file(log_path),
            "combination_matrix": str(matrix_path.relative_to(ROOT)),
            "combination_matrix_sha256": sha256_file(matrix_path),
            "scheme_catalog_sha256": catalog_hash,
            "shader_manifest_sha256": sha256_file(
                ROOT / "vulkan" / "shaders" / "manifest.json"
            ),
            "qualification_script_sha256": sha256_file(
                Path(__file__).resolve()
            ),
        },
        "combination_summary": {
            "case_count": matrix["case_count"],
            "physical_pass_count": matrix["physical_pass_count"],
            "expected_preflight_rejection_count":
                matrix["expected_preflight_rejection_count"],
            "guard_modes": matrix["guard_modes"],
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "[vulkan-evidence] RESULT: PASS "
        f"device={matrix['physical_device']['name']} "
        f"cases={matrix['case_count']} elapsed_s={elapsed:.3f} "
        f"release_record_eligible={str(source_worktree_clean).lower()}"
    )
    print(f"[vulkan-evidence] report={report_path}")
    print(f"[vulkan-evidence] log={log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
