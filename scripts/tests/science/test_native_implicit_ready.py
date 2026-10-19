#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
from contextlib import nullcontext

from test_native_mapped_terrain_ready import execute, field_payload, require_ledgers, require_success, run

ROOT = Path(__file__).resolve().parents[3]
FIELDS = ("u", "v", "w", "theta", "rho", "p", "qv")
LEDGERS = ("budget.csv", "momentum_budget.csv", "energy_budget.csv", "moisture_budget.csv",
           "thermodynamic_source_budget.csv", "radiation_budget.csv")


def provenance(log: str, backend: str, steps: int) -> dict[str, int]:
    for marker in (f"[COMPUTE] backend={backend}", "fallback=off", "Execution mode: qualified"):
        if marker not in log:
            raise RuntimeError(f"native implicit diffusion omitted {marker}")
    line = next((line for line in log.splitlines() if f"[COMPUTE DISPATCH] backend={backend}," in line), "")
    kernels = {name: int(count) for name, count in re.findall(r"([^,{} ]+\.comp):(\d+)", line)}
    required = ("diffusion/implicit_diffusion_stencil.comp", "diffusion/implicit_diffusion_vector.comp")
    if kernels.get(required[0], 0) < 5 * steps or kernels.get(required[1], 0) < 10 * steps:
        raise RuntimeError(f"implicit diffusion omitted a native solver stage: {kernels}")
    if kernels.get("diffusion/diffusion.comp", 0):
        raise RuntimeError("implicit diffusion substituted the explicit device operator")
    if kernels.get("dynamics/slow_apply.comp", 0) != 2 * steps:
        raise RuntimeError("implicit profile omitted a native Heun slow commit")
    for name in ("radiation/rrtmg_longwave_transfer_batch.comp", "radiation/rrtmg_shortwave_transfer_batch.comp"):
        if kernels.get(name, 0) == 0:
            raise RuntimeError("native implicit diffusion/RRTMG lost its radiation owner")
    return kernels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--backend", choices=("metal", "vulkan"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    config = ROOT / f"tests/fixtures/configs/qualification/cgrid_implicit_rrtmg_{args.backend}_ready.yaml"
    source = config.read_text()
    identity = hashlib.sha256(binary.read_bytes()).hexdigest()
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        if any(args.output.iterdir()):
            raise RuntimeError("native implicit diffusion evidence directory must be empty")
    context = nullcontext(args.output.resolve()) if args.output else tempfile.TemporaryDirectory(prefix="nimbus-native-implicit-")
    with context as directory:
        root = Path(directory)
        preflight = run([str(binary), "--validate-only", "--config", str(config)])
        require_success(preflight, "native implicit diffusion preflight")
        mutations = {
            "fallback": ("    allow_fallback: false", "    allow_fallback: true"),
            "host-slow": ("    gpu_apply_slow: true", "    gpu_apply_slow: false"),
            "euler": ("    slow_rk2: true", "    slow_rk2: false"),
            "coefficient": ("  K_h: 20", "  K_h: 40"),
            "spacing": ("  dx: 1000", "  dx: 900"),
        }
        for name, (old, new) in mutations.items():
            changed = source.replace(old, new, 1)
            if changed == source:
                raise RuntimeError(f"ineffective negative fixture: {name}")
            path = root / f"invalid-{name}.yaml"
            path.write_text(changed)
            for arguments in (("--validate-only",), ("--headless", "--duration=0", f"--outdir={root / name}")):
                result = run([str(binary), "--config", str(path), *arguments])
                if result.returncode == 0 or not any(word in result.stdout for word in
                        ("diffusion.scheme=implicit", "qualified Metal execution", "Science profile rejected", "strict nimbus.v1 load failed")):
                    raise RuntimeError(f"native implicit diffusion neighbor {name} did not reject before execution:\n{result.stdout}")
        one, four, resumed, cpu = (root / name for name in ("one", "four", "resumed", "cpu"))
        logs = [execute(binary, config, one, 1, "native implicit diffusion one worker"),
                execute(binary, config, four, 4, "native implicit diffusion four workers")]
        counts = [provenance(result.stdout, args.backend, 10) for result in logs]
        if counts[0] != counts[1]:
            raise RuntimeError("implicit diffusion device ownership changed with worker count")
        for name, result in zip(("one", "four"), logs):
            (root / f"{name}.log").write_text(result.stdout)
        cpu_config = ROOT / "tests/fixtures/configs/qualification/cgrid_implicit_rrtmg_ready.yaml"
        execute(binary, cpu_config, cpu, 1, "implicit diffusion CPU reference")
        parity = {}
        for second in (1, 2):
            for field in FIELDS:
                shape, _, expected = field_payload(cpu, second, field)
                native_shape, raw, actual = field_payload(one, second, field)
                _, threaded_raw, _ = field_payload(four, second, field)
                if shape != (12, 12, 16) or native_shape != shape or raw != threaded_raw:
                    raise RuntimeError(f"implicit diffusion {field} shape or bitwise threading mismatch")
                if any(not math.isfinite(value) for values in (expected, actual) for value in values):
                    raise RuntimeError(f"implicit diffusion {field} became non-finite")
                steps = 5 * second
                if field in ("u", "v", "w"):
                    limit = steps * 5e-5
                else:
                    exponent = math.frexp(max(abs(value) for value in expected))[1]
                    limit = steps * 4 * math.ldexp(1.0, exponent - 24)
                error = max(abs(a-b) for a,b in zip(actual,expected))
                parity[f"{second}s/{field}"] = {"max_absolute_error": error, "absolute_limit": limit}
        (root / "parity.json").write_text(json.dumps(parity, indent=2)+"\n")
        failed = {field: data for field, data in parity.items() if data["max_absolute_error"] > data["absolute_limit"]}
        if failed:
            raise RuntimeError(f"native implicit diffusion CPU parity failed: {failed}")
        for field in ("theta", "u"):
            if field_payload(one, 0, field)[1] == field_payload(one, 2, field)[1]:
                raise RuntimeError(f"native implicit diffusion left {field} inert")
        for output in (one, four, cpu):
            if min(field_payload(output, 2, "qv")[2]) < 0:
                raise RuntimeError("implicit diffusion produced negative water vapor")
        with (one / "thermodynamic_source_budget.csv").open() as stream:
            ledger = list(csv.DictReader(stream))[-1]
        if (int(ledger["diffusion_momentum_work_step_count"]) != 10
                or int(ledger["diffusion_moist_energy_step_count"]) != 10
                or float(ledger["diffusion_cumulative_committed_momentum_work_j"]) == 0.0
                or float(ledger["diffusion_moist_energy_cumulative_dry_internal_change_j"]) == 0.0):
            raise RuntimeError("implicit diffusion omitted active momentum or thermal state commits")
        checks = require_ledgers(one)
        if require_ledgers(four) != checks:
            raise RuntimeError("implicit diffusion owner identity coverage changed with worker count")
        for name in LEDGERS:
            if (one/name).read_bytes() != (four/name).read_bytes():
                raise RuntimeError(f"implicit diffusion threading changed {name}")
        for step in (5, 10):
            name = f"checkpoints/checkpoint_step_{step:09d}.nimbus"
            if (one/name).read_bytes() != (four/name).read_bytes():
                raise RuntimeError("implicit diffusion checkpoint differs across workers")
        checkpoint = one / "checkpoints/checkpoint_step_000000005.nimbus"
        result = run([str(binary), "--headless", "--config", str(config), f"--resume={checkpoint}",
                      f"--outdir={resumed}", "--threads=4", "--log-profile=normal"])
        require_success(result, "native implicit diffusion restart")
        provenance(result.stdout, args.backend, 5)
        require_ledgers(resumed)
        for field in FIELDS:
            if field_payload(one, 2, field)[1] != field_payload(resumed, 1, field)[1]:
                raise RuntimeError(f"native implicit diffusion restart changed {field}")
        for name in LEDGERS:
            continuous = (one/name).read_text().splitlines()
            restart = (resumed/name).read_text().splitlines()
            if not restart or restart[0] != continuous[0] or len(restart) < 2 or continuous[-(len(restart)-1):] != restart[1:]:
                raise RuntimeError(f"native implicit diffusion restart changed {name}")
        if hashlib.sha256(binary.read_bytes()).hexdigest() != identity:
            raise RuntimeError("simulator changed during native implicit diffusion validation")
        (root / "result.json").write_text(json.dumps({
            "status": "passed", "backend": args.backend, "binary_sha256": identity,
            "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "parity": parity, "kernels": counts[0], "owner_identity_checks": checks,
            "threading": "bitwise", "restart": "bitwise",
        }, indent=2)+"\n")
    print(f"[native-implicit-ready] PASS backend={args.backend}: native backward-Euler solve, RRTMG, CPU parity, conservation, threading and restart")


if __name__ == "__main__":
    main()
