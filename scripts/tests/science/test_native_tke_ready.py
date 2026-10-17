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
FIELDS = ("u", "v", "w", "theta", "rho", "p", "qv", "tke")
LEDGERS = ("budget.csv", "momentum_budget.csv", "energy_budget.csv", "moisture_budget.csv",
           "thermodynamic_source_budget.csv", "radiation_budget.csv")


def provenance(log: str, backend: str, steps: int, thompson: bool = False, weno: bool = False) -> dict[str, int]:
    for marker in (f"[COMPUTE] backend={backend}", "fallback=off", "Execution mode: qualified"):
        if marker not in log:
            raise RuntimeError(f"native TKE omitted {marker}")
    line = next((line for line in log.splitlines() if f"[COMPUTE DISPATCH] backend={backend}," in line), "")
    kernels = {name: int(count) for name, count in re.findall(r"([^,{} ]+\.comp):(\d+)", line)}
    for stage in ("pressure", "momentum"):
        if kernels.get(f"acoustic/acoustic_{stage}_cartesian_cgrid.comp") != 6 * steps:
            raise RuntimeError(f"native TKE omitted acoustic {stage} updates")
    required = ("turbulence/tke_closure.comp", "turbulence/tke_tendency.comp",
                "turbulence/smagorinsky_tendencies.comp", "turbulence/sgs_apply.comp")
    if any(kernels.get(name, 0) < steps for name in required):
        raise RuntimeError(f"native TKE omitted a closure, transport or commit stage: {kernels}")
    if len({kernels[name] for name in required[:3]}) != 1:
        raise RuntimeError("native TKE closure and transport dispatch counts disagree")
    if kernels.get("turbulence/smagorinsky_coefficients.comp", 0):
        raise RuntimeError("native TKE substituted Smagorinsky coefficients")
    if kernels.get("dynamics/slow_apply.comp", 0) != 2 * steps:
        raise RuntimeError("native TKE omitted a device Heun predictor or slow commit")
    for name in ("radiation/rrtmg_longwave_transfer_batch.comp", "radiation/rrtmg_shortwave_transfer_batch.comp"):
        if kernels.get(name, 0) == 0:
            raise RuntimeError("native TKE/RRTMG lost its radiation owner")
    if thompson:
        for name in ("cloud_adjustment", "warm_rain", "deposition", "freezing", "completion", "cloud_phase", "moment_sedimentation"):
            expected = steps * (5 if name == "moment_sedimentation" else 1)
            if kernels.get(f"microphysics/thompson_{name}.comp", 0) != expected:
                raise RuntimeError(f"Thompson/TKE omitted {name}: {kernels}")
        if kernels.get("microphysics/microphysics_apply.comp", 0):
            raise RuntimeError("Thompson/TKE changed its guarded microphysics commit owner")
    if weno:
        if kernels.get("advection/advect_conservative_update_cartesian_cgrid.comp", 0) < (8 if thompson else 3) * steps:
            raise RuntimeError("native WENO omitted a primary or private scalar owner")
        if kernels.get("advection/advect_conservative_mass_flux_cartesian_cgrid.comp", 0):
            raise RuntimeError("native WENO replaced retained acoustic mass flux")
        if backend == "vulkan" and kernels.get("advection/advect_conservative_scalar_flux_cartesian_cgrid.comp", 0) < (8 if thompson else 3) * steps:
            raise RuntimeError("Vulkan WENO omitted scalar reconstruction")
    if weno and not thompson:
        if kernels.get("advection/advect_conservative_update_cartesian_cgrid.comp", 0) != 3 * steps:
            raise RuntimeError("unseeded hydrometeors acquired an unexpected transport owner")
        if any(name.startswith("microphysics/") and count for name, count in kernels.items()):
            raise RuntimeError("microphysics dispatched in a profile without a microphysics scheme")
    return kernels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--backend", choices=("metal", "vulkan"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--implicit", action="store_true", help="exercise coupled native implicit diffusion")
    parser.add_argument("--thompson", action="store_true", help="include coupled native Thompson microphysics")
    parser.add_argument("--weno", action="store_true", help="native WENO transport, including private moments when selected")
    args = parser.parse_args()
    binary = args.binary.resolve()
    family = "tke_implicit_rrtmg" if args.implicit else "tke_rrtmg"
    if args.thompson:
        family = "thompson_tke_implicit" if args.implicit else "thompson_tke"
    if args.weno:
        family += "_weno"
    fields = FIELDS + (("qc", "qr", "qi", "qs", "qg") if args.thompson else ())
    config = ROOT / f"tests/fixtures/configs/qualification/cgrid_{family}_{args.backend}_ready.yaml"
    source = config.read_text()
    identity = hashlib.sha256(binary.read_bytes()).hexdigest()
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        if any(args.output.iterdir()):
            raise RuntimeError("native TKE evidence directory must be empty")
    context = nullcontext(args.output.resolve()) if args.output else tempfile.TemporaryDirectory(prefix="nimbus-native-tke-")
    with context as directory:
        root = Path(directory)
        preflight = run([str(binary), "--validate-only", "--config", str(config)])
        require_success(preflight, "native TKE preflight")
        mutations = {
            "fallback": ("    allow_fallback: false", "    allow_fallback: true"),
            "host-sgs": ("    gpu_apply_sgs: true", "    gpu_apply_sgs: false"),
            "host-slow": ("    gpu_apply_slow: true", "    gpu_apply_slow: false"),
            "euler": ("    slow_rk2: true", "    slow_rk2: false"),
            "neutral": ("  stability_correction: tke_based", "  stability_correction: none"),
            "spacing": ("  dx: 1000", "  dx: 900"),
        }
        if args.implicit:
            mutations.update({
                "diffusion-coefficient": ("  K_h: 20", "  K_h: 40"),
                "diffusion-target": ("  apply_to: scalars_and_momentum", "  apply_to: scalars"),
            })
        if args.thompson:
            mutations.update({
                "host-microphysics": ("gpu_apply_microphysics: true", "gpu_apply_microphysics: false"),
                "aerosol": ("water_friendly_aerosol_surface_m3: 350000000", "water_friendly_aerosol_surface_m3: 340000000"),
                "cloudy-radiation": ("clear_sky_only: true", "clear_sky_only: false"),
                "cloud-seed": ("cloud_water_kgkg: 0.001", "cloud_water_kgkg: 0.002"),
            })
        if args.weno and not args.thompson:
            mutations.update({
                "cloud-seed": ("cloud_water_kgkg: 0", "cloud_water_kgkg: 0.001"),
                "bubble-amplitude": ("dtheta_k: 2", "dtheta_k: 3"),
                "no-bubble": ("type: bubble", "type: none"),
                "microphysics-owner": ("gpu_apply_microphysics: false", "gpu_apply_microphysics: true"),
                "radiation-policy": ("clear_sky_only: false", "clear_sky_only: true"),
            })
        for name, (old, new) in mutations.items():
            changed = source.replace(old, new, 1)
            if changed == source:
                raise RuntimeError(f"ineffective negative fixture: {name}")
            path = root / f"invalid-{name}.yaml"
            path.write_text(changed)
            for arguments in (("--validate-only",), ("--headless", "--duration=0", f"--outdir={root / name}")):
                result = run([str(binary), "--config", str(path), *arguments])
                if result.returncode == 0 or not any(word in result.stdout for word in
                        ("turbulence.scheme=tke", "microphysics.scheme=thompson", "qualified Metal execution", "Science profile rejected", "strict nimbus.v1 load failed")):
                    raise RuntimeError(f"native TKE neighbor {name} did not reject before execution:\n{result.stdout}")
        one, four, resumed, cpu = (root / name for name in ("one", "four", "resumed", "cpu"))
        logs = [execute(binary, config, one, 1, "native TKE one worker"),
                execute(binary, config, four, 4, "native TKE four workers")]
        counts = [provenance(result.stdout, args.backend, 10, args.thompson, args.weno) for result in logs]
        if counts[0] != counts[1]:
            raise RuntimeError("TKE device ownership changed with worker count")
        for name, result in zip(("one", "four"), logs):
            (root / f"{name}.log").write_text(result.stdout)
        if args.implicit:
            from test_native_implicit_ready import provenance as implicit_provenance
            for result in logs:
                implicit_provenance(result.stdout, args.backend, 10)
        cpu_name = "cgrid_tke_implicit_rrtmg_cpu_ready.yaml" if args.implicit else "cgrid_tke_rrtmg_ready.yaml"
        if args.thompson or args.weno:
            cpu_name = f"cgrid_{family}_cpu_ready.yaml"
        cpu_config = ROOT / "tests/fixtures/configs/qualification" / cpu_name
        if args.weno:
            cpu_source = cpu_config.read_text()
            changed = cpu_source.replace("slow_rk2: true", "slow_rk2: false", 1)
            if changed == cpu_source and "  solver:" not in cpu_source:
                changed = cpu_source.replace("extensions:\n", "extensions:\n  solver:\n    slow_rk2: false\n", 1)
            if changed == cpu_source:
                raise RuntimeError("ineffective CPU WENO/TKE Euler neighbor")
            invalid = root / "invalid-cpu-euler.yaml"
            invalid.write_text(changed)
            for arguments in (("--validate-only",), ("--headless", "--duration=0", f"--outdir={root / 'cpu-euler'}")):
                result = run([str(binary), "--config", str(invalid), *arguments])
                if result.returncode == 0 or not any(word in result.stdout for word in
                        ("Science profile rejected", "turbulence.scheme=tke", "qualified", "strict nimbus.v1 load failed")):
                    raise RuntimeError(f"CPU WENO/TKE admitted the Euler neighbor:\n{result.stdout}")
        execute(binary, cpu_config, cpu, 1, "TKE CPU reference")
        if args.implicit:
            control = root / "without-implicit"
            control_family = "thompson_tke" if args.thompson else "tke_rrtmg"
            if args.weno:
                control_family += "_weno"
            execute(binary, ROOT / f"tests/fixtures/configs/qualification/cgrid_{control_family}_{args.backend}_ready.yaml",
                    control, 1, "TKE control without implicit diffusion")
            if all(field_payload(control, 2, f)[1] == field_payload(one, 2, f)[1] for f in ("u", "theta", "tke")):
                raise RuntimeError("coupled implicit diffusion has no resolved effect")
            for output in (one, four, cpu):
                with (output / "thermodynamic_source_budget.csv").open() as stream:
                    ledger = list(csv.DictReader(stream))[-1]
                if float(ledger["diffusion_cumulative_committed_momentum_work_j"]) == 0:
                    raise RuntimeError("coupled implicit diffusion has no committed momentum work")
        if args.implicit or args.thompson or args.weno:
            cpu_four, cpu_resumed = root / "cpu-four", root / "cpu-resumed"
            execute(binary, cpu_config, cpu_four, 4, "coupled CPU four workers")
            cpu_checkpoint = cpu_four / "checkpoints/checkpoint_step_000000005.nimbus"
            result = run([str(binary), "--headless", "--config", str(cpu_config), f"--resume={cpu_checkpoint}",
                          f"--outdir={cpu_resumed}", "--threads=4", "--log-profile=normal"])
            require_success(result, "coupled CPU restart")
            for field in fields:
                expected = field_payload(cpu, 2, field)[1]
                if expected != field_payload(cpu_four, 2, field)[1] or expected != field_payload(cpu_resumed, 1, field)[1]:
                    raise RuntimeError(f"coupled CPU threading/restart changed {field}")
            for step in (5, 10):
                name = f"checkpoints/checkpoint_step_{step:09d}.nimbus"
                if (cpu/name).read_bytes() != (cpu_four/name).read_bytes():
                    raise RuntimeError("coupled CPU threading changed checkpoint")
            for name in LEDGERS:
                if (cpu/name).read_bytes() != (cpu_four/name).read_bytes():
                    raise RuntimeError(f"coupled CPU threading changed {name}")
                continuous = (cpu/name).read_text().splitlines()
                resumed_lines = (cpu_resumed/name).read_text().splitlines()
                if len(resumed_lines) < 2 or resumed_lines[0] != continuous[0] or continuous[-(len(resumed_lines)-1):] != resumed_lines[1:]:
                    raise RuntimeError(f"coupled CPU restart changed {name}")
        parity = {}
        for second in (1, 2):
            for field in fields:
                shape, _, expected = field_payload(cpu, second, field)
                native_shape, raw, actual = field_payload(one, second, field)
                _, threaded_raw, _ = field_payload(four, second, field)
                if shape != (12, 12, 16) or native_shape != shape or raw != threaded_raw:
                    raise RuntimeError(f"TKE {field} shape or bitwise threading mismatch")
                if any(not math.isfinite(value) for values in (expected, actual) for value in values):
                    raise RuntimeError(f"TKE {field} became non-finite")
                steps = 5 * second
                if field in ("u", "v", "w"):
                    limit = steps * 5e-5
                else:
                    scale = max(abs(value) for value in expected)
                    exponent = math.frexp(scale)[1]
                    limit = steps * 4 * math.ldexp(1.0, exponent - 24) if scale else 0
                    if args.weno and field == "theta":
                        limit /= 4
                    if args.weno and field == "tke":
                        limit *= (steps + 1) / 2
                        if not args.thompson:
                            # Retained-density rounding perturbs the near-neutral stability-length transition.
                            # Matched-input transition tests constrain the closure independently.
                            limit = max(limit, 1e-6 + 1e-4 * scale)
                    if field.startswith("q"):
                        limit = max(limit, second * 2e-9)
                error = max(abs(a-b) for a,b in zip(actual,expected))
                parity[f"{second}s/{field}"] = {"max_absolute_error": error, "absolute_limit": limit}
        (root / "parity.json").write_text(json.dumps(parity, indent=2)+"\n")
        failed = {field: data for field, data in parity.items() if data["max_absolute_error"] > data["absolute_limit"]}
        if failed:
            raise RuntimeError(f"native TKE CPU parity failed: {failed}")
        for field in ("tke", "theta", "u"):
            if field_payload(one, 0, field)[1] == field_payload(one, 2, field)[1]:
                raise RuntimeError(f"native TKE left {field} inert")
        for output in (one, four, cpu):
            if min(field_payload(output, 2, "tke")[2]) < 0:
                raise RuntimeError("native TKE became negative")
        if args.thompson:
            for output in (one, four, cpu):
                with (output / "thermodynamic_source_budget.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                if not rows or not any(abs(float(r["microphysics_qv_requested_kg"])) > 0 for r in rows):
                    raise RuntimeError("Thompson/TKE exercised no vapor exchange")
                if max(field_payload(output, 0, "qc")[2]) <= 0:
                    raise RuntimeError("Thompson/TKE initialized no cloud")
                for r in rows:
                    if int(r["microphysics_conservation_active"]) != 1 or abs(float(r["microphysics_conservation_residual_kg"])) > 1e-6 * abs(float(r["microphysics_baseline_water_kg"])):
                        raise RuntimeError("Thompson/TKE failed its water closure")
        checks = require_ledgers(one)
        if require_ledgers(four) != checks:
            raise RuntimeError("TKE owner identity coverage changed with worker count")
        for name in LEDGERS:
            if (one/name).read_bytes() != (four/name).read_bytes():
                raise RuntimeError(f"TKE threading changed {name}")
        for step in (5, 10):
            name = f"checkpoints/checkpoint_step_{step:09d}.nimbus"
            if (one/name).read_bytes() != (four/name).read_bytes():
                raise RuntimeError("TKE checkpoint differs across workers")
        checkpoint = one / "checkpoints/checkpoint_step_000000005.nimbus"
        result = run([str(binary), "--headless", "--config", str(config), f"--resume={checkpoint}",
                      f"--outdir={resumed}", "--threads=4", "--log-profile=normal"])
        require_success(result, "native TKE restart")
        provenance(result.stdout, args.backend, 5, args.thompson, args.weno)
        if args.implicit:
            implicit_provenance(result.stdout, args.backend, 5)
        require_ledgers(resumed)
        for field in fields:
            if field_payload(one, 2, field)[1] != field_payload(resumed, 1, field)[1]:
                raise RuntimeError(f"native TKE restart changed {field}")
        for name in LEDGERS:
            continuous = (one/name).read_text().splitlines()
            restart = (resumed/name).read_text().splitlines()
            if not restart or restart[0] != continuous[0] or len(restart) < 2 or continuous[-(len(restart)-1):] != restart[1:]:
                raise RuntimeError(f"native TKE restart changed {name}")
        if hashlib.sha256(binary.read_bytes()).hexdigest() != identity:
            raise RuntimeError("simulator changed during native TKE validation")
        (root / "result.json").write_text(json.dumps({
            "status": "passed", "backend": args.backend, "binary_sha256": identity,
            "implicit_diffusion": args.implicit, "thompson": args.thompson, "weno": args.weno,
            "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "parity": parity, "kernels": counts[0], "owner_identity_checks": checks,
            "threading": "bitwise", "restart": "bitwise",
        }, indent=2)+"\n")
    print(f"[native-tke-ready] PASS backend={args.backend} implicit={args.implicit} thompson={args.thompson} weno={args.weno}: native closure, transport, commit, RRTMG, CPU parity, conservation, threading and restart")


if __name__ == "__main__":
    main()
