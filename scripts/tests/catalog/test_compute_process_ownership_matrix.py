#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINARY = ROOT / "bin/nimbus_sim"
QUALIFICATION = ROOT / "tests/fixtures/configs/qualification"

SCHEMES = {
    "dynamics": (
        "klemp_wilhelmson_split_cartesian_agrid",
        "klemp_wilhelmson_split_cartesian_cgrid",
    ),
    "advection": ("tvd", "weno5", "none"),
    "diffusion": ("explicit", "implicit"),
    "time_stepping": ("rk3", "rk4", "split_explicit"),
    "microphysics": ("kessler", "lin", "thompson", "milbrandt", "none"),
    "radiation": ("simple_grey", "rrtmg", "none"),
    "boundary_layer": ("slab", "ysu", "mynn", "none"),
    "turbulence": ("smagorinsky", "tke", "none"),
    "chaos": ("none", "initial_conditions", "boundary_layer", "full_stochastic"),
    "terrain": ("none", "bell", "schar"),
}

SELECTION_PATHS = {
    "dynamics": ("dynamics.scheme", None),
    "advection": ("advection.scheme", None),
    "diffusion": ("diffusion.scheme", "explicit"),
    "time_stepping": ("time.integrator", "split_explicit"),
    "microphysics": ("microphysics.scheme", "none"),
    "radiation": ("radiation.scheme", "none"),
    "boundary_layer": ("boundary_layer.scheme", "none"),
    "turbulence": ("turbulence.scheme", "none"),
    "chaos": ("extensions.chaos.scheme", "none"),
    "terrain": ("terrain.scheme", "none"),
}

GPU_DEVICE = {
    ("dynamics", "klemp_wilhelmson_split_cartesian_agrid"),
    ("dynamics", "klemp_wilhelmson_split_cartesian_cgrid"),
    ("advection", "weno5"),
    ("diffusion", "explicit"),
    ("diffusion", "implicit"),
    ("time_stepping", "split_explicit"),
    ("microphysics", "kessler"),
    ("microphysics", "lin"),
    ("microphysics", "thompson"),
    ("radiation", "rrtmg"),
    ("boundary_layer", "mynn"),
    ("turbulence", "smagorinsky"),
    ("turbulence", "tke"),
    ("terrain", "bell"),
    ("terrain", "schar"),
}

GPU_HOST = {
    ("advection", "tvd"),
    ("radiation", "simple_grey"),
    ("boundary_layer", "slab"),
    ("boundary_layer", "ysu"),
    ("chaos", "initial_conditions"),
    ("chaos", "boundary_layer"),
    ("chaos", "full_stochastic"),
}

GPU_UNSUPPORTED = {
    ("time_stepping", "rk3"),
    ("time_stepping", "rk4"),
    ("microphysics", "milbrandt"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_json(binary: Path, option: str) -> tuple[dict[str, object], bytes]:
    result = subprocess.run(
        [str(binary), option], cwd=ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{option} failed:\n" + result.stderr.decode("utf-8", errors="replace")
        )
    return json.loads(result.stdout), result.stdout


def expected_process_owner(module: str, scheme: str, backend: str) -> str:
    if scheme == "none":
        return "inactive"
    if backend == "cpu":
        return "host"
    identity = (module, scheme)
    if identity in GPU_DEVICE:
        return "device"
    if identity in GPU_HOST:
        return "host"
    if identity in GPU_UNSUPPORTED:
        return "unsupported"
    raise RuntimeError(f"independent process-ownership oracle omits {identity}")


def validate_process_matrix(matrix: dict[str, object]) -> dict[tuple[str, str, str], str]:
    if matrix.get("schema") != "nimbus.compute-process-ownership-matrix.v1":
        raise RuntimeError("compute process ownership schema drifted")
    if matrix.get("version") != 1 or matrix.get("scope") != "descriptor_backed_solver_processes":
        raise RuntimeError("compute process ownership identity drifted")
    records = matrix.get("records")
    if not isinstance(records, list):
        raise RuntimeError("compute process ownership records are missing")

    expected_identities = {
        (module, scheme, backend)
        for module, schemes in SCHEMES.items()
        for scheme in schemes
        for backend in ("cpu", "gpu")
    }
    owners: dict[tuple[str, str, str], str] = {}
    supported = 0
    unsupported = 0
    for index, record in enumerate(records):
        if not isinstance(record, dict) or record.get("index") != index:
            raise RuntimeError("compute process ownership indices are not contiguous")
        identity = (record.get("module"), record.get("scheme"), record.get("backend"))
        if identity in owners:
            raise RuntimeError(f"duplicate compute process ownership identity {identity}")
        if not all(isinstance(value, str) for value in identity):
            raise RuntimeError(f"malformed compute process ownership identity at {index}")
        typed_identity = (identity[0], identity[1], identity[2])
        expected = expected_process_owner(*typed_identity)
        if record.get("owner") != expected:
            raise RuntimeError(
                f"process ownership differs from the independent oracle for {typed_identity}: "
                f"expected={expected}, actual={record.get('owner')}"
            )
        valid = expected != "unsupported"
        if record.get("valid") != valid:
            raise RuntimeError(f"process ownership validity differs for {typed_identity}")
        owners[typed_identity] = expected
        if valid:
            supported += 1
        else:
            unsupported += 1
    if set(owners) != expected_identities:
        raise RuntimeError(
            "compute process ownership axes drifted: "
            f"missing={len(expected_identities - set(owners))}, "
            f"extra={len(set(owners) - expected_identities)}"
        )
    summary = {"evaluated": 64, "supported": 61, "unsupported": 3}
    if matrix.get("summary") != summary or supported != 61 or unsupported != 3:
        raise RuntimeError(f"compute process ownership summary drifted: {matrix.get('summary')}")
    return owners


def flatten_yaml_scalars(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    stack: list[tuple[int, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        match = re.match(r"^(\s*)([A-Za-z0-9_]+):(?:\s*(.*))?$", line)
        if match is None:
            continue
        indentation = len(match.group(1))
        key = match.group(2)
        value = match.group(3).strip()
        while stack and stack[-1][0] >= indentation:
            stack.pop()
        full_path = ".".join([entry[1] for entry in stack] + [key])
        if value:
            values[full_path] = value.strip("\"'")
        else:
            stack.append((indentation, key))
    return values


def boolean(values: dict[str, str], path: str, default: bool | None = None) -> bool:
    value = values.get(path)
    if value is None and default is not None:
        return default
    if value not in ("true", "false"):
        raise RuntimeError(f"{path} must be explicitly true or false")
    return value == "true"


def state_lookup(matrix: dict[str, object]) -> dict[tuple[object, ...], dict[str, object]]:
    records = matrix.get("records")
    if not isinstance(records, list) or matrix.get("schema") != "nimbus.compute-ownership-matrix.v1":
        raise RuntimeError("state-commit ownership matrix is missing or malformed")
    result: dict[tuple[object, ...], dict[str, object]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("apply"), dict):
            raise RuntimeError("state-commit ownership record is malformed")
        apply = record["apply"]
        key = (
            record.get("backend"), record.get("requireQualified"), record.get("dynamics"),
            record.get("microphysics"), record.get("turbulence"), apply.get("microphysics"),
            apply.get("slow"), apply.get("sgs"),
        )
        if key in result:
            raise RuntimeError("state-commit ownership matrix contains duplicate identities")
        result[key] = record
    if len(result) != 960:
        raise RuntimeError(f"state-commit ownership matrix expected 960 records, got {len(result)}")
    return result


def accelerator_fixtures(
    process_owners: dict[tuple[str, str, str], str],
    state_records: dict[tuple[object, ...], dict[str, object]],
) -> list[dict[str, object]]:
    fixtures: list[dict[str, object]] = []
    for path in sorted(QUALIFICATION.glob("*.yaml")):
        values = flatten_yaml_scalars(path)
        backend = values.get("extensions.compute.backend")
        if values.get("extensions.execution.mode") != "qualified" or backend not in ("vulkan", "metal"):
            continue
        if boolean(values, "extensions.compute.allow_fallback"):
            raise RuntimeError(f"{path.name} permits accelerator fallback")

        selections: dict[str, str] = {}
        selected_owners: dict[str, str] = {}
        for module, (selection_path, default) in SELECTION_PATHS.items():
            scheme = values.get(selection_path, default)
            if scheme is None:
                raise RuntimeError(f"{path.name} omits required {selection_path}")
            if scheme not in SCHEMES[module]:
                raise RuntimeError(f"{path.name} selects unknown {module} scheme {scheme}")
            owner = process_owners[(module, scheme, "gpu")]
            if owner == "unsupported":
                raise RuntimeError(f"{path.name} selects unsupported GPU process {module}.{scheme}")
            selections[module] = scheme
            selected_owners[module] = owner

        apply = {
            "microphysics": boolean(values, "extensions.compute.gpu_apply_microphysics", True),
            "slow": boolean(values, "extensions.compute.gpu_apply_slow", False),
            "sgs": boolean(values, "extensions.compute.gpu_apply_sgs", False),
        }
        state_key = (
            "gpu", True, selections["dynamics"], selections["microphysics"], selections["turbulence"],
            apply["microphysics"], apply["slow"], apply["sgs"],
        )
        state = state_records.get(state_key)
        if state is None or state.get("valid") is not True or not isinstance(state.get("owners"), dict):
            raise RuntimeError(f"{path.name} lacks valid state-commit ownership")
        fixtures.append({
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "backend": backend,
            "fallbackAllowed": False,
            "selections": selections,
            "processOwners": selected_owners,
            "stateCommitOwners": state["owners"],
        })

    backend_counts = {
        backend: sum(fixture["backend"] == backend for fixture in fixtures)
        for backend in ("vulkan", "metal")
    }
    if len(fixtures) != 46 or backend_counts != {"vulkan": 25, "metal": 21}:
        raise RuntimeError(
            f"maintained accelerator fixture inventory drifted: total={len(fixtures)}, "
            f"backends={backend_counts}"
        )
    return fixtures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file():
        raise RuntimeError(f"missing simulator: {binary}")
    binary_sha256 = sha256_file(binary)
    process_matrix, process_bytes = command_json(binary, "--compute-process-ownership-matrix=json")
    state_matrix, state_bytes = command_json(binary, "--compute-ownership-matrix=json")
    owners = validate_process_matrix(process_matrix)
    fixtures = accelerator_fixtures(owners, state_lookup(state_matrix))
    if sha256_file(binary) != binary_sha256:
        raise RuntimeError("simulator changed while backend process evidence was generated")

    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        evidence = {
            "schema": "nimbus.compute-process-ownership-evidence.v1",
            "version": 1,
            "binary": {
                "path": str(binary.relative_to(ROOT)),
                "sha256": binary_sha256,
            },
            "processMatrixSha256": hashlib.sha256(process_bytes).hexdigest(),
            "stateCommitMatrixSha256": hashlib.sha256(state_bytes).hexdigest(),
            "processMatrix": process_matrix,
            "maintainedAcceleratorFixtures": fixtures,
            "summary": {
                "processRecords": 64,
                "supportedProcessRecords": 61,
                "unsupportedProcessRecords": 3,
                "acceleratorFixtures": 46,
                "vulkanFixtures": 25,
                "metalFixtures": 21,
            },
        }
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(output)

    print(
        "[compute-process-ownership] PASS: 64 descriptor-backed process records "
        "(61 supported, 3 unsupported); 46 maintained accelerator fixtures "
        f"bound (25 Vulkan, 21 Metal); binary_sha256={binary_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
