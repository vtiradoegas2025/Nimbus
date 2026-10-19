#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile

import test_lin_rrtmg_accelerator_ready as native
import test_thompson_cpu_ready as thompson
from test_native_mapped_terrain_ready import require_ledgers

ROOT = native.ROOT


def require_dispatch(log: str, resumed: bool = False, implicit: bool = False, weno: bool = False) -> None:
    lines = [line for line in log.splitlines() if "[COMPUTE DISPATCH]" in line]
    if not lines:
        raise RuntimeError("native Thompson omitted dispatch evidence")
    steps = 10 if resumed else 20
    expected = {f"microphysics/thompson_{name}.comp": steps for name in (
        "cloud_adjustment", "warm_rain", "deposition", "freezing", "completion", "cloud_phase")}
    expected["microphysics/thompson_moment_sedimentation.comp"] = 5 * steps
    for name in ("longwave", "shortwave"):
        expected[f"radiation/rrtmg_{name}_transfer_batch.comp"] = 4 if resumed else 8
    for kernel, count in expected.items():
        match = re.search(re.escape(kernel) + r":(\d+)", lines[-1])
        if match is None or int(match.group(1)) != count:
            raise RuntimeError(f"expected {count} dispatches of {kernel}: {lines[-1]}")

    if weno:
        counts = {name: int(value) for name, value in re.findall(r"([^,{} ]+\.comp):(\d+)", lines[-1])}
        if counts.get("advection/advect_conservative_update_cartesian_cgrid.comp", 0) < 8 * steps:
            raise RuntimeError("WENO/Thompson omitted primary or private-moment transport")
        if counts.get("advection/advect_conservative_mass_flux_cartesian_cgrid.comp", 0):
            raise RuntimeError("WENO/Thompson replaced retained acoustic mass flux")
        if "backend=vulkan" in lines[-1] and counts.get("advection/advect_conservative_scalar_flux_cartesian_cgrid.comp", 0) < 8 * steps:
            raise RuntimeError("Vulkan WENO/Thompson omitted scalar reconstruction")
        if counts.get("dynamics/slow_apply.comp", 0) or any(name.startswith("turbulence/") and count for name, count in counts.items()):
            raise RuntimeError("WENO/Thompson changed its declared slow or turbulence owner")

    if implicit:
        counts = {name: int(value) for name, value in re.findall(r"([^,{} ]+\.comp):(\d+)", lines[-1])}
        if counts.get("diffusion/implicit_diffusion_stencil.comp", 0) < 5 * steps or counts.get("diffusion/implicit_diffusion_vector.comp", 0) < 10 * steps:
            raise RuntimeError("Thompson coupling omitted native implicit solver stages")
        if counts.get("diffusion/diffusion.comp", 0) or counts.get("dynamics/slow_apply.comp", 0):
            raise RuntimeError("Thompson coupling changed its declared diffusion or slow-commit ownership")


def require_parity(reference: Path, candidate: Path) -> dict:
    water_fields = ("qv", "qc", "qr", "qi", "qs", "qg", "qh")
    def water_at(path):
        fields = [thompson.npy_float_values(thompson.final_frame(path) / f"{field}.npy") for field in water_fields]
        if len({len(field) for field in fields}) != 1 or any(not math.isfinite(v) or v < 0 for field in fields for v in field):
            raise RuntimeError("invalid coupled Thompson water state")
        return [math.fsum(values) for values in zip(*fields)]
    expected_water, actual_water = water_at(reference), water_at(candidate)
    if len(expected_water) != len(actual_water):
        raise RuntimeError("Thompson water layout differs")
    water_scale = max(map(abs, expected_water))
    water_ulp = math.ldexp(1.0, math.frexp(water_scale)[1] - 24) if water_scale else 0
    total_limit = max(8e-9, 20 * water_ulp)
    total_error = max(abs(a-b) for a,b in zip(expected_water, actual_water))
    if total_error > total_limit:
        raise RuntimeError(f"Thompson total-water parity failed: {total_error} > {total_limit}")
    values = {"total_water": {"maxAbsoluteError": total_error, "absoluteLimit": total_limit},
              "water_partition_reference_scale": water_scale}
    for field in thompson.STATE_FIELDS:
        expected = thompson.npy_float_values(thompson.final_frame(reference) / f"{field}.npy")
        actual = thompson.npy_float_values(thompson.final_frame(candidate) / f"{field}.npy")
        if len(expected) != len(actual) or any(not math.isfinite(x) for row in (expected, actual) for x in row):
            raise RuntimeError(f"invalid parity field {field}")
        scale = max(abs(x) for x in expected)
        limit = 80 * math.ldexp(1.0, math.frexp(scale)[1] - 24) if scale > 0 else 0
        if field in ("u", "v", "w"):
            limit = 1e-3
        if field in water_fields:
            limit = max(8e-9, 80 * water_ulp)
        error = max(abs(a-b) for a, b in zip(actual, expected))
        values[field] = {"maxAbsoluteError": error, "absoluteLimit": limit}
        if error > limit:
            raise RuntimeError(f"Thompson CPU parity failed for {field}: {error} > {limit}")
    return values


def require_column_coupling(output: Path) -> int:
    checks = require_ledgers(output)
    with (output / "thermodynamic_source_budget.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not any(abs(float(row["pbl_committed_momentum_work_j"])) > 0 for row in rows):
        raise RuntimeError("WENO/Thompson/YSU has no committed column momentum exchange")
    return checks


def main(*, implicit: bool = False) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=native.DEFAULT_BINARY)
    parser.add_argument("--backend", choices=("metal", "vulkan", "all"), default="all")
    parser.add_argument("--implicit", action="store_true", default=implicit)
    parser.add_argument("--weno", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    backends = ("metal", "vulkan") if args.backend == "all" else (args.backend,)
    native.SCOPE_ERROR = "microphysics.scheme=thompson requires a maintained CPU profile or the native C-grid Thompson reference"
    identity = hashlib.sha256(binary.read_bytes()).hexdigest()
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        if any(args.output.iterdir()):
            raise RuntimeError("Thompson evidence output must be empty")
    context = nullcontext(args.output.resolve()) if args.output else tempfile.TemporaryDirectory(prefix="nimbus-thompson-native-ready-")
    with context as temporary:
        root = Path(temporary)
        results = {}
        owner_checks = {}
        for backend in backends:
            family = "thompson_implicit" if args.implicit else "thompson"
            if args.weno:
                family += "_weno"
            config = ROOT / f"tests/fixtures/configs/qualification/cgrid_{family}_{backend}_ready.yaml"
            source = config.read_text()
            result = native.run(binary, backend, ["--validate-only", "--config", str(config)])
            native.require_success(result, f"{backend} preflight")
            if "[VALIDATE] PASS" not in result.stdout:
                raise RuntimeError("native Thompson preflight omitted PASS")
            for old, new, label in (
                ("allow_fallback: false", "allow_fallback: true", "fallback"),
                ("gpu_apply_microphysics: true", "gpu_apply_microphysics: false", "ownership"),
                ("water_friendly_aerosol_surface_m3: 350000000", "water_friendly_aerosol_surface_m3: 340000000", "aerosol"),
            ):
                native.require_rejection(binary, backend, source, old, new, root, label)
            if args.implicit:
                for old, new, label in (("K_h: 20", "K_h: 40", "diffusion-coefficient"),
                                        ("apply_to: scalars_and_momentum", "apply_to: scalars", "diffusion-target"),
                                        ("gpu_apply_slow: false", "gpu_apply_slow: true", "slow-ownership")):
                    native.require_rejection(binary, backend, source, old, new, root, label)
            if args.weno:
                for old, new, label in (("slow_rk2: true", "slow_rk2: false", "euler"),
                                        ("gpu_apply_slow: false", "gpu_apply_slow: true", "slow-ownership"),
                                        ("clear_sky_only: true", "clear_sky_only: false", "radiation-policy"),
                                        ("  scheme: ysu", "  scheme: slab", "column-owner")):
                    native.require_rejection(binary, backend, source, old, new, root, label)
            outputs = {}
            for workers in (1, 4):
                output = root / f"{backend}-{workers}"
                result = native.run(binary, backend, [
                    "--headless", "--config", str(config), f"--outdir={output}",
                    "--write-every=4", "--checkpoint-every=2", f"--threads={workers}",
                    "--log-profile=normal",
                ], workers)
                native.require_success(result, f"{backend} {workers}-worker evolution")
                for marker in (f"[COMPUTE] backend={backend}", "fallback=off"):
                    if marker not in result.stdout:
                        raise RuntimeError(f"missing {marker}")
                require_dispatch(result.stdout, implicit=args.implicit, weno=args.weno)
                (root / f"{backend}-{workers}.log").write_text(result.stdout)
                thompson.require_process_activity(output)
                if args.weno:
                    for marker in ("Execution mode: qualified", "Initialized boundary layer scheme: ysu"):
                        if marker not in result.stdout:
                            raise RuntimeError(f"WENO/Thompson omitted {marker}")
                    owner_checks[f"{backend}-{workers}"] = require_column_coupling(output)
                outputs[workers] = output
            thompson.require_state_identity(outputs[1], outputs[4], f"{backend} threading")
            thompson.require_ledger_identity(outputs[1], outputs[4], f"{backend} threading")
            for step in (10, 20):
                name = f"checkpoints/checkpoint_step_{step:09d}.nimbus"
                if (outputs[1]/name).read_bytes() != (outputs[4]/name).read_bytes():
                    raise RuntimeError(f"{backend} private checkpoint differs across workers")
            if args.implicit or args.weno:
                cpu_config = ROOT / f"tests/fixtures/configs/qualification/cgrid_{family}_cpu_ready.yaml"
                if args.weno:
                    thompson.require_scope_rejection(binary, cpu_config.read_text(), "slow_rk2: true", "slow_rk2: false", root, f"{backend}-cpu-euler")
                    cpu_source = cpu_config.read_text()
                    begin = cpu_source.index("microphysics:\n")
                    end = cpu_source.index("turbulence:\n", begin)
                    invalid = root / f"{backend}-cpu-wrong-scheme.yaml"
                    invalid.write_text(cpu_source[:begin] + "microphysics:\n  scheme: none\n\n" + cpu_source[end:])
                    for arguments in (("--validate-only",), ("--headless", "--duration=0", f"--outdir={root / (backend + '-wrong-scheme')}")):
                        result = native.run(binary, "cpu", ["--config", str(invalid), *arguments], 1)
                        if result.returncode == 0 or "advection.scheme=weno5" not in result.stdout:
                            raise RuntimeError(f"non-Thompson selector inherited WENO admission:\n{result.stdout}")
                cpu_outputs = {}
                for workers in (1, 4):
                    output = root / f"{backend}-cpu-{workers}"
                    result = native.run(binary, "cpu", ["--headless", "--config", str(cpu_config), f"--outdir={output}",
                        "--write-every=4", "--checkpoint-every=2", f"--threads={workers}", "--log-profile=normal"], workers)
                    native.require_success(result, "Thompson CPU reference")
                    if args.weno:
                        thompson.require_process_activity(output)
                        owner_checks[f"{backend}-cpu-{workers}"] = require_column_coupling(output)
                    cpu_outputs[workers] = output
                thompson.require_state_identity(cpu_outputs[1], cpu_outputs[4], "CPU threading")
                thompson.require_ledger_identity(cpu_outputs[1], cpu_outputs[4], "CPU threading")
                for step in (10, 20):
                    name = f"checkpoints/checkpoint_step_{step:09d}.nimbus"
                    if (cpu_outputs[1]/name).read_bytes() != (cpu_outputs[4]/name).read_bytes():
                        raise RuntimeError("CPU Thompson private checkpoint differs across workers")
                results[backend] = require_parity(cpu_outputs[1], outputs[1])
                if args.weno:
                    tvd_family = "thompson_implicit" if args.implicit else "thompson"
                    tvd_output = root / f"{backend}-cpu-tvd"
                    result = native.run(binary, "cpu", ["--headless", "--config",
                        str(ROOT / f"tests/fixtures/configs/qualification/cgrid_{tvd_family}_cpu_ready.yaml"),
                        f"--outdir={tvd_output}", "--write-every=4", "--checkpoint-every=0", "--threads=1"], 1)
                    native.require_success(result, "Thompson TVD transport control")
                    if all((thompson.final_frame(tvd_output)/f"{field}.npy").read_bytes() ==
                           (thompson.final_frame(cpu_outputs[1])/f"{field}.npy").read_bytes()
                           for field in thompson.STATE_FIELDS):
                        raise RuntimeError("WENO has no resolved effect against the matched TVD control")
                if args.implicit:
                    control = root / f"{backend}-without-implicit"
                    result = native.run(binary, backend, ["--headless", "--config",
                        str(ROOT / f"tests/fixtures/configs/qualification/cgrid_thompson{'_weno' if args.weno else ''}_{backend}_ready.yaml"),
                        f"--outdir={control}", "--write-every=4", "--checkpoint-every=0", "--threads=1"])
                    native.require_success(result, "Thompson no-implicit control")
                    if all((thompson.final_frame(control)/f"{f}.npy").read_bytes() == (thompson.final_frame(outputs[1])/f"{f}.npy").read_bytes() for f in ("theta", "u", "qv")):
                        raise RuntimeError("implicit diffusion has no resolved coupled effect")
                    for output in (*outputs.values(), *cpu_outputs.values()):
                        with (output / "thermodynamic_source_budget.csv").open() as stream:
                            ledger = list(csv.DictReader(stream))[-1]
                        if float(ledger["diffusion_cumulative_committed_momentum_work_j"]) == 0:
                            raise RuntimeError("Thompson implicit coupling omitted momentum work")
                cpu_resumed = root / f"{backend}-cpu-resumed"
                result = native.run(binary, "cpu", ["--headless", "--config", str(cpu_config),
                    f"--resume={cpu_outputs[4] / 'checkpoints/checkpoint_step_000000010.nimbus'}",
                    "--duration=4", "--write-every=4", "--checkpoint-every=0", f"--outdir={cpu_resumed}", "--threads=4"], 4)
                native.require_success(result, "CPU Thompson restart")
                thompson.require_state_identity(cpu_outputs[4], cpu_resumed, "CPU restart")
                thompson.require_restart_suffix(cpu_outputs[4], cpu_resumed)
                if args.weno:
                    owner_checks[f"{backend}-cpu-resumed"] = require_column_coupling(cpu_resumed)
            checkpoint = outputs[4] / "checkpoints/checkpoint_step_000000010.nimbus"
            resumed = root / f"{backend}-resumed"
            result = native.run(binary, backend, [
                "--headless", "--config", str(config), f"--resume={checkpoint}",
                "--duration=4", "--write-every=4", "--checkpoint-every=0",
                f"--outdir={resumed}", "--threads=4", "--log-profile=normal",
            ], 4)
            native.require_success(result, f"{backend} restart")
            if "[CHECKPOINT] resumed step=10, time_s=2" not in result.stdout:
                raise RuntimeError("native Thompson restart omitted restore marker")
            require_dispatch(result.stdout, resumed=True, implicit=args.implicit, weno=args.weno)
            (root / f"{backend}-resumed.log").write_text(result.stdout)
            thompson.require_state_identity(outputs[4], resumed, f"{backend} restart")
            thompson.require_restart_suffix(outputs[4], resumed)
            if args.weno:
                owner_checks[f"{backend}-resumed"] = require_column_coupling(resumed)
        if hashlib.sha256(binary.read_bytes()).hexdigest() != identity:
            raise RuntimeError("simulator changed during Thompson readiness checks")
        (root / "result.json").write_text(json.dumps({"status": "passed", "binarySha256": identity,
            "implicit": args.implicit, "weno": args.weno, "ownerIdentityChecks": owner_checks, "parity": results, "backends": list(backends),
            "scriptSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "configSha256": {b: hashlib.sha256((ROOT / f"tests/fixtures/configs/qualification/cgrid_{family}_{b}_ready.yaml").read_bytes()).hexdigest() for b in (*backends, "cpu")}}, indent=2)+"\n")
    print("[thompson-accelerator-ready] PASS: public native admission, seven process owners, "
          "water closure, frozen hydrometeors, exact 1/4-worker identity and restart; negative scope gates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
