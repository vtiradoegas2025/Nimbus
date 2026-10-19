#!/usr/bin/env python3
"""Qualify root/child publication and independent predictive-ZFP histories."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[3]
BINARY = ROOT / "bin" / "nimbus_sim"
VALIDATOR = ROOT / "bin" / "field_validator"
CONFIG = ROOT / "tests" / "fixtures" / "configs" / "test" / "cart_cgrid_stationary_nest.yaml"
CORE_FIELDS = {"p", "qr", "qv", "rho", "theta", "u", "v", "w"}


def fail(message: str) -> None:
    raise RuntimeError(message)


def run(command: list[str]) -> str:
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        fail(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stdout}"
        )
    return result.stdout


def load_manifest(step: Path) -> dict[str, object]:
    path = step / "manifest.json"
    if not path.is_file():
        fail(f"missing completion manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        fail(f"{label}: expected {expected!r}, got {actual!r}")


def compression_roles(manifest: dict[str, object]) -> set[tuple[str, tuple[int, ...]]]:
    published = set(manifest["published_fields"])
    roles: set[tuple[str, tuple[int, ...]]] = set()
    for field in manifest["fields"]:
        if field["field_id"] not in published:
            continue
        compression = field.get("compression")
        if compression is None:
            fail(f"published field {field['field_id']} has no ZFP contract")
        roles.add(
            (
                compression["frame_role"],
                tuple(compression["parent_step_indices"]),
            )
        )
    return roles


def require_payload_inventory(step: Path, manifest: dict[str, object]) -> None:
    published = set(manifest["published_fields"])
    require_equal(published, CORE_FIELDS, f"{step} published field set")
    payloads = {path.stem for path in step.glob("*.zfp3d")}
    require_equal(payloads, CORE_FIELDS, f"{step} ZFP payload set")


def main() -> int:
    for required in (BINARY, VALIDATOR, CONFIG):
        if not required.is_file():
            fail(f"required qualification input is missing: {required}")

    with tempfile.TemporaryDirectory(prefix="nimbus_stationary_nested_output_") as temporary:
        run_dir = Path(temporary) / "run"
        run(
            [
                str(BINARY),
                "--config",
                str(CONFIG),
                "--headless",
                "--compute-backend",
                "cpu",
                "--threads",
                "1",
                "--duration",
                "2",
                "--write-every=1",
                f"--outdir={run_dir}",
                "--log-profile=quiet",
            ]
        )

        contract = json.loads((run_dir / "nimbus_output.json").read_text(encoding="utf-8"))
        require_equal(contract["publication_state"], "ready", "run publication state")
        require_equal(
            contract["data_products"],
            ["history", "nested_child_history"],
            "run data products",
        )

        root_steps = [run_dir / f"step_{index:06d}" for index in range(3)]
        child_root = run_dir / "nests" / "child_0000"
        child_steps = [child_root / f"step_{index:06d}" for index in (1, 2)]
        if (child_root / "step_000000").exists():
            fail("child history must not publish before the first causal parent interval")

        root_manifests = [load_manifest(step) for step in root_steps]
        child_manifests = [load_manifest(step) for step in child_steps]
        for step, manifest in zip(root_steps, root_manifests):
            require_equal(manifest["data_product"]["kind"], "history", f"{step} product")
            require_equal(manifest["grid"]["dimensions"], {"x": 16, "y": 16, "z": 24}, f"{step} shape")
            require_payload_inventory(step, manifest)
        for step, manifest in zip(child_steps, child_manifests):
            require_equal(
                manifest["data_product"]["kind"],
                "nested_child_history",
                f"{step} product",
            )
            require_equal(manifest["grid"]["dimensions"], {"x": 10, "y": 10, "z": 24}, f"{step} shape")
            require_equal(manifest["grid"]["spacing_m"], {"x": 500, "y": 500, "z": 500}, f"{step} spacing")
            require_equal(
                manifest["grid"]["physical_bounds_m"],
                {"lower": [6000, 6000, 0], "upper": [10000, 10000, 12000]},
                f"{step} Earth footprint",
            )
            require_payload_inventory(step, manifest)

        require_equal(compression_roles(root_manifests[0]), {("keyframe", ())}, "root t0 compression")
        require_equal(compression_roles(root_manifests[1]), {("simple_delta", (0,))}, "root t1 compression")
        require_equal(
            compression_roles(root_manifests[2]),
            {("predictive_delta", (1, 0))},
            "root t2 compression",
        )
        require_equal(compression_roles(child_manifests[0]), {("keyframe", ())}, "child t1 compression")
        require_equal(compression_roles(child_manifests[1]), {("simple_delta", (1,))}, "child t2 compression")

        for history_root in (run_dir, child_root):
            output = run(
                [
                    str(VALIDATOR),
                    "--input",
                    str(history_root),
                    "--mode",
                    "report",
                    "--scope",
                    "required",
                ]
            )
            if "nonfinite=0 bounds=0" not in output or "failed=no" not in output:
                fail(f"decoded history validation did not close for {history_root}:\n{output}")

    print(
        "[stationary-nested-output] PASS: root and child manifests, Earth geometry, "
        "independent predictive-ZFP ancestry, and decoded finite/bounds validation"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
