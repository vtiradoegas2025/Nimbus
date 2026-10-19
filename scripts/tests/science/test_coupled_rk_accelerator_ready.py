#!/usr/bin/env python3
"""Bounded public native RK coupling, direct dispatch, conservation and restart gate."""
from __future__ import annotations

import argparse
import csv
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile

import test_coupled_rk_ready as coupled
import test_lin_rrtmg_accelerator_ready as runtime
from npy_support import read_float32_npy

ROOT = runtime.ROOT
FIXTURES = ROOT / "tests/fixtures/configs/qualification"


def setting(source: str, section: str, key: str, value: str) -> str:
    block = re.search(rf"(?m)^{re.escape(section)}:\n(?:[ \t].*\n|\n)*", source)
    if block is None:
        raise RuntimeError(f"missing {section} fixture section")
    text = block.group()
    changed, count = re.subn(rf"(?m)^  {re.escape(key)}:.*$", f"  {key}: {value}", text)
    if count == 0:
        changed = text.replace(f"{section}:\n", f"{section}:\n  {key}: {value}\n", 1)
    return source[:block.start()] + changed + source[block.end():]


def native_source(source: str, backend: str) -> str:
    source, count = re.subn(r"(?m)^    backend: (cpu|metal|vulkan)$", f"    backend: {backend}", source)
    if count != 1:
        raise RuntimeError("fixture lacks one compute backend")
    for flag in ("gpu_apply_slow", "gpu_apply_microphysics", "gpu_apply_sgs"):
        source, count = re.subn(rf"(?m)^    {flag}: (true|false)$", f"    {flag}: {'false' if backend == 'cpu' else 'true'}", source)
        if count == 0:
            source = source.replace(f"    backend: {backend}\n", f"    backend: {backend}\n    {flag}: {'false' if backend == 'cpu' else 'true'}\n", 1)
    return source


def cases():
    core = coupled.CONFIGS["rk3"].read_text()
    variants = {f"core-{scheme}": setting(core, "advection", "scheme", scheme) for scheme in ("none", "tvd", "weno5")}
    variants["kessler"] = setting(core, "microphysics", "scheme", "kessler")
    for name, filename in (("lin", "cgrid_lin_rrtmg_ready.yaml"),
                           ("thompson", "cgrid_thompson_cpu_ready.yaml"),
                           ("milbrandt", "cgrid_milbrandt_cpu_ready.yaml"),
                           ("tke", "cgrid_tke_rrtmg_ready.yaml"),
                           ("mynn", "cgrid_mynn_tke_top_ready.yaml")):
        variants[name] = (FIXTURES / filename).read_text()
    variants["thompson-implicit"] = setting(setting(setting(variants["thompson"], "diffusion", "scheme", "implicit"), "diffusion", "K_h", "100"), "diffusion", "K_v", "25")
    variants["smagorinsky"] = setting(setting(setting(variants["kessler"], "turbulence", "scheme", "smagorinsky"), "diffusion", "K_h", "100"), "diffusion", "K_v", "25")
    slab = setting(setting(variants["smagorinsky"], "surface", "scheme", "slab"), "surface", "apply_fluxes", "true")
    slab = setting(setting(slab, "boundary_layer", "scheme", "slab"), "radiation", "scheme", "simple_grey")
    slab = setting(slab, "surface", "tsk_k", "300")
    for key, value in (("solar_forcing", "idealized_diurnal"), ("latitude_deg", "35"),
                       ("declination_deg", "15"), ("local_solar_time_at_start_h", "0")):
        slab = setting(slab, "radiation", key, value)
    variants["slab"] = setting(slab, "turbulence", "mode", "2d")
    variants["milbrandt-no-hail"] = variants["milbrandt"].replace("hail_processes: true", "hail_processes: false")
    for integrator in ("ssprk3", "rk4"):
        for name, source in variants.items():
            source = setting(source, "time", "integrator", integrator)
            source = setting(source, "output", "preset", "dynamics_full")
            source = setting(source, "time", "run_time", "2")
            source = setting(source, "time", "dt_large", "0.05")
            source = setting(source, "time", "adaptive_dt", "false")
            source = setting(source, "time", "acoustic_substeps", "1")
            source = source.replace("  start_h: 4\n", "  start_h: 0\n")
            yield f"{integrator}-{name}", source
            collocated = setting(source, "grid", "staggering", "collocated")
            collocated = setting(collocated, "dynamics", "scheme", "klemp_wilhelmson_split_cartesian_agrid")
            yield f"{integrator}-agrid-{name}", collocated


def payloads(output: Path) -> dict[str, bytes]:
    frame = runtime.final_frame(output)
    result = {}
    for field in coupled.FIELDS:
        shape, raw, values = read_float32_npy(frame / f"{field}.npy")
        if not values or any(not math.isfinite(v) for v in values):
            raise RuntimeError(f"invalid {field} in {frame}")
        if field not in ("u", "v", "w") and min(values) < 0:
            raise RuntimeError(f"negative {field} in {frame}")
        if field in ("rho", "p", "theta") and min(values) <= 0:
            raise RuntimeError(f"nonpositive {field} in {frame}")
        result[field] = raw
    return result


def parity(reference: Path, candidate: Path) -> dict:
    expected_fields = {}
    actual_fields = {}
    for field in coupled.FIELDS:
        shape, _, expected = read_float32_npy(runtime.final_frame(reference) / f"{field}.npy")
        actual_shape, _, actual = read_float32_npy(runtime.final_frame(candidate) / f"{field}.npy")
        if shape != actual_shape or any(not math.isfinite(x) for values in (expected, actual) for x in values):
            raise RuntimeError(f"invalid parity data for {field}")
        expected_fields[field] = expected
        actual_fields[field] = actual
    water = ("qv", "qc", "qr", "qi", "qs", "qg", "qh")
    expected_water = [sum(values) for values in zip(*(expected_fields[f] for f in water))]
    actual_water = [sum(values) for values in zip(*(actual_fields[f] for f in water))]
    water_scale = max(map(abs, expected_water))
    water_error = max(abs(a-b) for a,b in zip(expected_water, actual_water))
    total_limit = 8e-9 if water_scale else 0
    partition_limit = max(8e-9, 80 * math.ldexp(1., math.frexp(water_scale)[1] - 24)) if water_scale else 0
    if water_error > total_limit:
        raise RuntimeError(f"total-water parity: {water_error} > {total_limit} in {candidate}")
    result = {"totalWater": {"maxAbsoluteError": water_error, "absoluteLimit": total_limit},
              "waterPartitionReferenceScale": water_scale}
    for field in coupled.FIELDS:
        expected, actual = expected_fields[field], actual_fields[field]
        scale = max(map(abs, expected))
        tolerance = 160 * math.ldexp(1., math.frexp(scale)[1] - 24) if scale else 0
        if field in ("u", "v", "w"):
            tolerance = 1e-3
        elif field in water:
            tolerance = partition_limit
        error = max(abs(a-b) for a,b in zip(expected, actual))
        if error > tolerance:
            raise RuntimeError(f"{field} parity: {error} > {tolerance} in {candidate}")
        result[field] = {"maxAbsoluteError": error, "absoluteLimit": tolerance}
    return result


def require_native_ownership(log: str, key: str) -> None:
    dynamics = "dynamics/cartesian_tendencies.comp" if "-agrid-" in key else "dynamics/cartesian_cgrid_slow_tendencies.comp"
    kernels = [dynamics, "time_stepping/rk_tendency.comp", "time_stepping/rk_combination.comp"]
    if "slab" in key:
        kernels += ["boundary_layer/slab_column.comp", "radiation/simple_grey.comp"]
    if "thompson" in key:
        kernels += [f"microphysics/thompson_{name}.comp" for name in (
            "cloud_adjustment", "warm_rain", "deposition", "freezing", "completion", "cloud_phase", "moment_sedimentation")]
    elif "milbrandt" in key:
        kernels += [f"microphysics/milbrandt_{name}.comp" for name in (
            "warm_rain", "cloud_adjustment", "melting", "cold", "sedimentation")]
    elif any(name in key for name in ("kessler", "smagorinsky", "slab")):
        kernels += ["microphysics/kessler_pointwise.comp", "microphysics/kessler_sedimentation.comp"]
    elif "-lin" in key:
        kernels += ["microphysics/lin_pointwise.comp", "microphysics/lin_sedimentation.comp"]
    if "-agrid-" in key and any(name in key for name in ("-tke", "smagorinsky", "slab")):
        kernels += ["turbulence/collocated_sgs.comp", "turbulence/sgs_apply.comp"]
    else:
        if "-tke" in key:
            kernels += ["turbulence/tke_closure.comp", "turbulence/tke_tendency.comp", "turbulence/sgs_apply.comp"]
        if "smagorinsky" in key or "slab" in key:
            kernels += ["turbulence/smagorinsky_tendencies.comp", "turbulence/sgs_apply.comp"]
    if "mynn" in key:
        kernels += ["boundary_layer/mynn_level25_stability.comp"]
    if "implicit" in key:
        kernels += ["diffusion/implicit_diffusion_stencil.comp", "diffusion/implicit_diffusion_vector.comp"]
    if any(name in key for name in ("-lin", "thompson", "milbrandt", "-tke", "mynn")):
        kernels += ["radiation/rrtmg_longwave_transfer_batch.comp", "radiation/rrtmg_shortwave_transfer_batch.comp"]
    for kernel in kernels:
        if not re.search(re.escape(kernel) + r":[1-9][0-9]*", log):
            raise RuntimeError(f"{key} omitted real dispatches for {kernel}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=runtime.DEFAULT_BINARY)
    parser.add_argument("--backend", choices=("all", "metal", "vulkan"), default="all")
    parser.add_argument("--case", action="append", help="restrict the bounded case list")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    identity = hashlib.sha256(binary.read_bytes()).hexdigest()
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        if any(args.output.iterdir()):
            raise RuntimeError("evidence directory must be empty")
    context = nullcontext(args.output.resolve()) if args.output else tempfile.TemporaryDirectory(prefix="nimbus-native-rk-")
    backends = ("cpu", "metal", "vulkan") if args.backend == "all" else ("cpu", args.backend)
    with context as temporary:
        root = Path(temporary)
        results = {}
        selected = [(key, source) for key, source in cases() if not args.case or key in args.case]
        if not selected or (args.case and set(args.case) - {key for key,_ in selected}):
            raise RuntimeError("unknown or empty case selection")
        for key, source in selected:
            outputs = {}
            configs = {}
            for backend in backends:
                config = root / f"{key}-{backend}.yaml"
                config.write_text(native_source(source, backend))
                configs[backend] = config
                preflight = runtime.run(binary, backend, ["--validate-only", "--config", str(config)])
                (root / f"{key}-{backend}-preflight.log").write_text(preflight.stdout)
                runtime.require_success(preflight, f"{key}-{backend} preflight")
                if "[VALIDATE] PASS" not in preflight.stdout:
                    raise RuntimeError("public preflight omitted PASS")
                if backend != "cpu" and key.endswith("core-weno5"):
                    for label, invalid_source in (
                        ("fallback", config.read_text().replace("allow_fallback: false", "allow_fallback: true", 1)),
                        ("owner", config.read_text().replace("gpu_apply_slow: true", "gpu_apply_slow: false", 1)),
                        ("terrain", setting(config.read_text(), "terrain", "scheme", "bell")),
                        ("lid", config.read_text().replace("  top: rigid\n", "  top: open\n", 1)),
                    ):
                        if invalid_source == config.read_text():
                            raise RuntimeError(f"{label} rejection case did not change")
                        invalid = root / f"{key}-{backend}-invalid-{label}.yaml"
                        invalid.write_text(invalid_source)
                        rejected = runtime.run(binary, backend, ["--validate-only", "--config", str(invalid)])
                        (root / f"{key}-{backend}-invalid-{label}.log").write_text(rejected.stdout)
                        if rejected.returncode == 0 or not any(marker in rejected.stdout for marker in (
                            "qualified coupled RK integration requires", "qualified Cartesian A-grid execution is limited",
                            "qualified mapped terrain is limited", "qualified Metal execution is limited",
                            "advection.scheme=weno5 is qualified only")):
                            raise RuntimeError(f"{label} did not reject at public science admission: {rejected.stdout}")
                for workers in (1, 4):
                    output = root / f"{key}-{backend}-{workers}"
                    result = runtime.run(binary, backend, ["--headless", "--config", str(config), f"--outdir={output}",
                        "--write-every=1", "--checkpoint-every=1", f"--threads={workers}", "--log-profile=normal"], workers)
                    (root / f"{key}-{backend}-{workers}.log").write_text(result.stdout)
                    runtime.require_success(result, f"{key}-{backend}-{workers}")
                    coupled.require_complete_unsplit_ledger(output, key, require_every_step_active=False)
                    with (output / "budget.csv").open() as stream:
                        mass_rows = list(csv.DictReader(stream))
                    if not mass_rows or any(not math.isfinite(float(row["rel_dry_mass"])) or
                                            abs(float(row["rel_dry_mass"])) > 1e-6 for row in mass_rows):
                        raise RuntimeError(f"{key}-{backend} dry-mass conservation failed")
                    if key.endswith("core-none"):
                        for field in coupled.FIELDS[5:]:
                            _, initial, _ = read_float32_npy(output / "step_000000" / f"{field}.npy")
                            _, final, _ = read_float32_npy(runtime.final_frame(output) / f"{field}.npy")
                            if initial != final:
                                raise RuntimeError(f"no-advection RK changed {field}")
                    outputs[backend, workers] = output
                    payloads(output)
                    if backend != "cpu":
                        require_native_ownership(result.stdout, key)
                if payloads(outputs[backend, 1]) != payloads(outputs[backend, 4]):
                    raise RuntimeError(f"{key}-{backend} threading mismatch")
                runtime.require_ledger_identity(outputs[backend, 1], outputs[backend, 4], key)
                checkpoints = sorted((outputs[backend, 1] / "checkpoints").glob("*.nimbus"))
                if len(checkpoints) < 2:
                    raise RuntimeError("missing midpoint and final checkpoints")
                for checkpoint in checkpoints:
                    other = outputs[backend, 4] / "checkpoints" / checkpoint.name
                    if checkpoint.read_bytes() != other.read_bytes():
                        raise RuntimeError(f"{key}-{backend} private state threading mismatch")
                resumed = root / f"{key}-{backend}-resumed"
                result = runtime.run(binary, backend, ["--headless", "--config", str(config), f"--resume={checkpoints[0]}",
                    f"--outdir={resumed}", "--duration=2", "--write-every=1", "--checkpoint-every=0", "--threads=4", "--log-profile=normal"], 4)
                (root / f"{key}-{backend}-resumed.log").write_text(result.stdout)
                runtime.require_success(result, f"{key}-{backend} restart")
                if payloads(resumed) != payloads(outputs[backend, 4]):
                    raise RuntimeError(f"{key}-{backend} restart mismatch")
                runtime.require_restart_suffix(outputs[backend, 4], resumed)
                results[f"{key}-{backend}"] = {"threading": "exact", "restart": "exact", "ledger": "complete"}
                if backend != "cpu":
                    results[f"{key}-{backend}"]["cpuParity"] = parity(outputs["cpu", 1], outputs[backend, 1])
                print(f"passed {key}-{backend}", flush=True)
            if "thompson" in key or "milbrandt" in key:
                for backend in backends[1:]:
                    for first, second in (("cpu", backend), (backend, "cpu")):
                        checkpoint = sorted((outputs[first, 1] / "checkpoints").glob("*.nimbus"))[0]
                        output = root / f"{key}-{first}-to-{second}"
                        result = runtime.run(binary, second, ["--headless", "--config", str(configs[second]), f"--resume={checkpoint}",
                            f"--outdir={output}", "--duration=2", "--write-every=1", "--checkpoint-every=0", "--threads=1", "--log-profile=normal"])
                        (root / f"{key}-{first}-to-{second}.log").write_text(result.stdout)
                        runtime.require_success(result, "cross-backend private continuation")
                        payloads(output)
                        results[f"{key}-{first}-to-{second}"] = parity(outputs[first, 1], output)
        if hashlib.sha256(binary.read_bytes()).hexdigest() != identity:
            raise RuntimeError("binary changed during the bounded gate")
        (root / "summary.json").write_text(json.dumps({"binarySha256": identity, "results": results}, indent=2)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
