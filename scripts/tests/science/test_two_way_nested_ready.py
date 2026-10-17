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
import subprocess
import tempfile

from npy_support import read_float32_npy
from test_earth_fixed_terrain_motion_ready import read_npy


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "tests/fixtures/configs/qualification/stationary_nested_cpu_two_way_ready.yaml"
DEFAULT_BINARY = ROOT / "bin/nimbus_sim"
FIELDS = ("rho", "theta", "p", "u", "v", "w")
CURRENCIES = ("dry_mass", "rho_theta", "x_momentum", "y_momentum", "z_momentum")
CLOSURE = re.compile(
    r"\[NESTING TWO-WAY CLOSURE\] currency=(\w+).*?mismatch=(\S+)"
    r".*?reflux=(\S+).*?dynamics_storage=(\S+).*?normalized_residual=(\S+)"
)


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )


def require_success(result: subprocess.CompletedProcess[str], label: str) -> None:
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed:\n{result.stdout}")
    for marker in ("[SCIENTIFIC INVALIDITY]", "Science profile rejected"):
        if marker in result.stdout:
            raise RuntimeError(f"{label} reported {marker}:\n{result.stdout}")


def require_rejection(result: subprocess.CompletedProcess[str], label: str) -> None:
    if result.returncode == 0:
        raise RuntimeError(f"{label} unexpectedly succeeded:\n{result.stdout}")
    if "domain_motion.nesting" not in result.stdout and "microphysics.scheme" not in result.stdout:
        raise RuntimeError(f"{label} omitted its science scope error:\n{result.stdout}")


def field_payload(output: Path, second: int, field: str) -> bytes:
    path = output / f"step_{second:06d}" / f"{field}.npy"
    if not path.is_file():
        raise RuntimeError(f"missing output field: {path}")
    shape, raw, values = read_float32_npy(path)
    if shape != (16, 16, 24):
        raise RuntimeError(f"{field} has unexpected shape {shape}")
    if any(not math.isfinite(value) or (field.startswith("q") and value < 0) for value in values):
        raise RuntimeError(f"{field} contains invalid values")
    return raw


def child_field_payload(output: Path, second: int, field: str, child_size: int = 10) -> bytes:
    path = output / "nests/child_0000" / f"step_{second:06d}" / f"{field}.npy"
    if not path.is_file():
        raise RuntimeError(f"missing child output field: {path}")
    shape, raw, values = read_float32_npy(path)
    if shape != (child_size, child_size, 24):
        raise RuntimeError(f"child {field} has unexpected shape {shape}")
    if any(not math.isfinite(value) or (field.startswith("q") and value < 0) for value in values):
        raise RuntimeError(f"child {field} contains invalid values")
    return raw


def require_closure(
    output: str, label: str
) -> dict[str, list[tuple[float, float, float, float]]]:
    observed: dict[str, list[tuple[float, float, float, float]]] = {}
    for currency, mismatch, reflux, storage, normalized in CLOSURE.findall(output):
        values = tuple(float(value) for value in (mismatch, reflux, storage, normalized))
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"{label} published non-finite {currency} closure")
        if values[3] > 1.0:
            raise RuntimeError(f"{label} exceeded {currency} closure tolerance: {values[3]}")
        observed.setdefault(currency, []).append(values)
    if set(observed) != set(CURRENCIES):
        raise RuntimeError(f"{label} omitted currencies: {set(CURRENCIES) - set(observed)}")
    if not any(abs(row[0]) > 0.0 and abs(row[1]) > 0.0
               for rows in observed.values() for row in rows):
        raise RuntimeError(f"{label} did not exercise nonzero child feedback")
    if not any(abs(row[2]) > 0.0 for row in observed["z_momentum"]):
        raise RuntimeError(f"{label} did not retain FP32 momentum storage provenance")
    return observed


WATER = re.compile(r"\[NESTING TWO-WAY WATER\] field=(\w+), reflux=([^,\s]+), redistributed=([^,\s]+), redistributed_cells=([^,\s]+), physics_source=([^,\s]+), interface_source=([^,\s]+), residual=([^,\s]+), tolerance=([^,\s]+), normalized_residual=([^,\s]+)")


def require_water_closure(log: str, active_cloud: bool = False) -> dict:
    result = {}
    for name, *raw in WATER.findall(log):
        values = tuple(map(float, raw))
        if any(not math.isfinite(value) for value in values) or values[-2] < 0 or values[-1] > 1 or values[1] < 0 or values[2] < 0:
            raise RuntimeError(f"invalid nested {name} closure: {values}")
        result.setdefault(name, []).append(values)
    if set(result) != {"qv", "qc", "qr", "qi", "qs", "qg", "qh"}:
        raise RuntimeError("moist nesting omitted a water owner")
    if active_cloud and not (any(row[3] < -1 for row in result["qv"]) and any(row[3] > 1 for row in result["qc"])):
        raise RuntimeError("moist feedback did not exercise condensation and latent-heating sources")
    return result


PRIVATE = re.compile(r"\[NESTING TWO-WAY PRIVATE\] field=([^,]+), reflux=([^,]+), physics_source=([^,]+), interface_source=([^,]+), transport_storage=([^,]+), residual=([^,]+), tolerance=([^\s]+)")


def require_private_closure(log: str, scheme: str, require_activity: bool = True) -> dict:
    moments = ("Nc", "Nr", "Ni", "Nwfa", "Nifa") if scheme == "thompson" else ("Nc", "Nr", "Ni", "Ns", "Ng", "Nh")
    expected = {f"{scheme}.{name}_kg-1" for name in moments}
    result = {}
    for name, *raw in PRIVATE.findall(log):
        values = tuple(map(float, raw))
        if any(not math.isfinite(v) for v in values) or values[-1] < 0 or abs(values[-2]) > values[-1]:
            raise RuntimeError(f"invalid private carrier closure: {name}: {values}")
        result.setdefault(name, []).append(values)
    if set(result) != expected or len({len(rows) for rows in result.values()}) != 1:
        raise RuntimeError("private feedback omitted or renamed a moment owner")
    reactions = {}
    for name, raw in re.findall(
            r"\[NESTING TWO-WAY PRIVATE\] field=([^,]+),[^\n]* phase_exchange=([^\s]+)", log):
        value = float(raw)
        if not math.isfinite(value):
            raise RuntimeError("private reaction exchange is not finite")
        reactions.setdefault(name, []).append(value)
    if reactions:
        if set(reactions) != expected or any(len(reactions[name]) != len(result[name]) for name in expected):
            raise RuntimeError("private reaction exchange omitted a carrier")
        for row in zip(*(reactions[name] for name in sorted(expected))):
            exchanges = dict(zip(sorted(expected), row))
            cloud = exchanges.get("thompson.Nc_kg-1", 0)
            aerosol = exchanges.get("thompson.Nwfa_kg-1", 0)
            if cloud < 0 or aerosol != -cloud or any(value != 0 for name, value in exchanges.items()
                    if name not in ("thompson.Nc_kg-1", "thompson.Nwfa_kg-1")):
                raise RuntimeError("private source replay does not balance its cloud/aerosol pair")
    if require_activity and (not any(abs(row[0]) > 0 for rows in result.values() for row in rows)
                             or not any(abs(row[1]) > 0 for rows in result.values() for row in rows)):
        raise RuntimeError("private feedback did not exercise reflux and physics sources")
    return result


def require_feedback_output_budget(output: Path, frame: int, names: tuple[str, ...]) -> None:
    with (output / "energy_budget.csv").open() as stream:
        rows = [row for row in csv.DictReader(stream) if row["stage"] == "nested_feedback"]
    if not rows:
        raise RuntimeError("parent feedback omitted its diagnostic transition")
    shape, _, density = read_float32_npy(output / f"step_{frame:06d}/rho.npy")
    nx, ny, nz = shape
    water = [read_float32_npy(output / f"step_{frame:06d}/{name}.npy")[2] for name in names]
    manifest = json.loads((output / f"step_{frame:06d}/manifest.json").read_text())
    geometry = manifest.get("grid", {}).get("vertical_geometry", {})
    spacing = manifest["grid"]["spacing_m"]
    area = spacing["x"] * spacing["y"]
    zface = None
    if geometry.get("storage") == "versioned_npy_geometry":
        _, zface = read_npy((output / f"step_{frame:06d}" / geometry["interface_path"]).resolve())
    def volume(i: int, j: int, k: int) -> float:
        if zface is None:
            return area * spacing["z"]
        offset = (i * ny + j) * (nz + 1) + k
        return area * (zface[offset + 1] - zface[offset])
    expected = math.fsum(float(density[(i * ny + j) * nz + k]) * volume(i, j, k) *
                         math.fsum(float(values[(i * ny + j) * nz + k]) for values in water)
                         for i in range(1, nx - 1) for j in range(1, ny - 1) for k in range(nz))
    actual = float(rows[-1]["moist_v2_after_total_water_mass_kg"])
    if abs(actual-expected) > 128*math.ulp(max(1.0,abs(expected))):
        raise RuntimeError(f"feedback budget/output water mismatch: {actual} vs {expected}")


def ordinary_run(
    binary: Path, output: Path, threads: int, config: Path = CONFIG
) -> dict[str, list[tuple[float, float, float, float]]]:
    result = run([
        str(binary), "--headless", "--config", str(config),
        f"--outdir={output}", "--write-every=1", "--checkpoint-every=1",
        f"--threads={threads}", "--log-profile=normal",
    ])
    output.with_suffix(".log").write_text(result.stdout)
    require_success(result, f"two-way nested {threads}-thread run")
    backend = re.search(r"(?m)^    backend: (\w+)", config.read_text())[1]
    for marker in (
        "Execution mode: qualified", "stationary two-way child",
        f"[COMPUTE] backend={backend}", "fallback=off",
        "moist initialization suppresses" if any(f"scheme: {scheme}" in config.read_text() for scheme in ("kessler", "lin", "thompson", "milbrandt")) else "exact dry initialization suppresses",
    ):
        if marker not in result.stdout:
            raise RuntimeError(f"two-way nested run omitted {marker!r}")
    return require_closure(result.stdout, f"two-way nested {threads}-thread run")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    selected_config = args.config.resolve()
    scheme = re.search(r"microphysics:\n  scheme: (\w+)", selected_config.read_text())[1]
    lin = scheme == "lin"
    private = scheme in ("thompson", "milbrandt")
    moist = scheme != "none"
    active_cloud = any(name in selected_config.read_text() for name in ("simple_grey_condensing_profile.txt", "lin_cloudy_profile.txt", "milbrandt_mixed_profile.txt"))
    seeded_cloud = private and "cloud_water_kgkg: 0.001" in selected_config.read_text()
    if seeded_cloud:
        active_cloud = False
    water_fields = ("qv", "qc", "qr", "qi", "qs", "qg", "qh") if lin or private else ("qv", "qc", "qr")
    fields = FIELDS + (water_fields if moist else ())
    config_text=selected_config.read_text()
    duration=int(float(re.search(r"(?m)^  run_time: ([^\n]+)",config_text)[1]))
    dt=float(re.search(r"(?m)^  dt_large: ([^\n]+)",config_text)[1])
    child_size=int(re.search(r"(?m)^    child_cells_x: (\d+)",config_text)[1])+2
    midpoint=duration//2
    midpoint_step=round(midpoint/dt)

    for path in (binary, selected_config):
        if not path.is_file():
            raise RuntimeError(f"required Ready-state input is missing: {path}")

    preflight = run([str(binary), "--validate-only", "--config", str(selected_config)])
    require_success(preflight, "two-way nested qualified preflight")
    if "[VALIDATE] PASS" not in preflight.stdout:
        raise RuntimeError("two-way nested preflight omitted PASS")

    source = selected_config.read_text(encoding="utf-8")
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        if any(args.output.iterdir()):
            raise RuntimeError("two-way evidence directory must be empty")
    context = nullcontext(args.output.resolve()) if args.output else tempfile.TemporaryDirectory(prefix="nimbus-two-way-nested-ready-")
    with context as temporary:
        root = Path(temporary)
        backend = re.search(r"(?m)^    backend: (\w+)", source)[1]
        weno = "scheme: weno5" in source
        mutations = [
            ("fallback", "    allow_fallback: false\n", "    allow_fallback: true\n"),
            ("diffusion", "  K_h: 0\n", "  K_h: 1\n"),
            ("motion", "  mode: stationary\n", "  mode: constant_velocity\n"),
        ]
        mutations.append(("advection", f"  scheme: {'weno5' if weno else 'tvd'}\n", "  scheme: none\n"))
        if moist:
            scheme_line = f"  scheme: {scheme}\n"
            mutations.append(("microphysics", scheme_line,
                              scheme_line + "  shared:\n    cloud_water_threshold_kgkg: 0.002\n"))
        if private:
            native = backend != "cpu"
            mutations.append(("private-ownership", f"    gpu_apply_microphysics: {str(native).lower()}\n",
                              f"    gpu_apply_microphysics: {str(not native).lower()}\n"))
            if scheme == "thompson":
                mutations.append(("aerosol", "water_friendly_aerosol_surface_m3: 350000000", "water_friendly_aerosol_surface_m3: 340000000"))
                mutations.extend([
                    ("unseeded", "cloud_water_kgkg: 0.001", "cloud_water_kgkg: 0"),
                    ("bubble-amplitude", "dtheta_k: 2", "dtheta_k: 2.5"),
                    ("ice-aerosol", "ice_friendly_aerosol_surface_m3: 2000000", "ice_friendly_aerosol_surface_m3: 2100000"),
                ])
                ratio = re.search(r"(?m)^    parent_ratio_x: (\d+)\n    parent_ratio_y: (\d+)\n    parent_time_step_ratio: (\d+)\n", source)
                if ratio is None:
                    raise RuntimeError("private fixture omitted its refinement controls")
                mutations.append(("refinement-ratio", ratio[0],
                                  "    parent_ratio_x: 4\n    parent_ratio_y: 4\n    parent_time_step_ratio: 4\n"))
        if private and "    type: vortex_seed" in source:
            mutations = [item for item in mutations if item[0] not in ("unseeded", "bubble-amplitude")]
            mutations.append(("vortex-amplitude", "v_max_ms: 1", "v_max_ms: 1.5"))
        for label, old, new in mutations:
            mutated = source.replace(old, new, 1)
            if mutated == source:
                raise RuntimeError(f"{label} mutation did not alter the fixture")
            config = root / f"invalid-{label}.yaml"
            config.write_text(mutated, encoding="utf-8")
            require_rejection(
                run([str(binary), "--validate-only", "--config", str(config)]),
                f"two-way nested {label} validation guard",
            )
            rejected_output = root / f"invalid-{label}-output"
            require_rejection(
                run([str(binary), "--headless", "--duration=0", "--write-every=0",
                     f"--outdir={rejected_output}", "--config", str(config)]),
                f"two-way nested {label} startup guard",
            )
            if (rejected_output / "nimbus_output.json").exists():
                raise RuntimeError(f"{label} rejection published output")

        serial = root / "serial"
        parallel = root / "parallel"
        resumed = root / "resumed"
        serial_closure = ordinary_run(binary, serial, 1, selected_config)
        parallel_closure = ordinary_run(binary, parallel, 4, selected_config)
        if serial_closure != parallel_closure:
            raise RuntimeError("two-way closure ledgers differ between workers")
        for output in (serial, parallel):
            for path in output.rglob("u.npy"):
                _, values = read_npy(path)
                if min(values) < 2.0 or max(values) > 6.0:
                    raise RuntimeError(f"bounded weak-forcing nesting wind amplified: {path}")
        evolved = False
        for field in fields:
            initial = field_payload(parallel, 0, field)
            serial_final = field_payload(serial, duration, field)
            parallel_final = field_payload(parallel, duration, field)
            if serial_final != parallel_final:
                raise RuntimeError(f"{field} differs between one and four workers")
            if child_field_payload(serial, duration, field, child_size) != child_field_payload(
                parallel, duration, field, child_size
            ):
                raise RuntimeError(
                    f"child {field} differs between one and four workers"
                )
            evolved = evolved or initial != parallel_final
        if not evolved:
            raise RuntimeError("qualified two-way nested profile did not evolve")
        if lin:
            for location in (serial, serial / "nests/child_0000"):
                for species, threshold in (("qi", 1e-8), ("qg", 1e-12)):
                    _, _, values = read_float32_npy(location / f"step_{duration:06d}/{species}.npy")
                    if max(values) <= threshold:
                        raise RuntimeError(f"Lin feedback did not exercise {species} at {location}")

        checkpoint = parallel / f"checkpoints/checkpoint_step_{midpoint_step:09d}.nimbus"
        if not checkpoint.is_file() or not Path(f"{checkpoint}.json").is_file():
            raise RuntimeError("two-way nested run omitted its synchronized midpoint checkpoint")
        continuation = run([
            str(binary), "--headless", "--config", str(selected_config),
            f"--resume={checkpoint}", f"--duration={duration}", "--write-every=1",
            "--checkpoint-every=1", f"--outdir={resumed}", "--threads=4",
            "--log-profile=normal",
        ])
        (root / "resumed.log").write_text(continuation.stdout)
        require_success(continuation, "two-way nested checkpoint continuation")
        if f"[CHECKPOINT] resumed step={midpoint_step}, time_s={midpoint}" not in continuation.stdout:
            raise RuntimeError("two-way nested continuation did not restore the synchronized midpoint")
        resumed_closure = require_closure(
            continuation.stdout, "two-way nested checkpoint continuation"
        )
        for currency in CURRENCIES:
            resumed_rows = resumed_closure[currency]
            if resumed_rows != parallel_closure[currency][-len(resumed_rows):]:
                raise RuntimeError(
                    f"{currency} closure ledger changed across restart"
                )
        for field in fields:
            if field_payload(parallel, duration, field) != field_payload(resumed, duration-midpoint, field):
                raise RuntimeError(f"{field} restart continuation is not bitwise exact")
            if child_field_payload(parallel, duration, field, child_size) != child_field_payload(
                resumed, duration-midpoint, field, child_size
            ):
                raise RuntimeError(
                    f"child {field} restart continuation is not bitwise exact"
                )

        private_closure = {}
        if private:
            private_closure = require_private_closure((root / "serial.log").read_text(), scheme)
            if private_closure != require_private_closure((root / "parallel.log").read_text(), scheme):
                raise RuntimeError("private ledgers changed across worker counts")
            for name, rows in require_private_closure(continuation.stdout, scheme, False).items():
                if rows != private_closure[name][-len(rows):]:
                    raise RuntimeError(f"private {name} source/reflux history changed across restart")
        reaction_replay = []
        if (private and scheme == "thompson" and "    type: vortex_seed" in source
                and "  scheme: schar" in source and "    parent_ratio_x: 2" in source):
            reaction_replay = [float(value) for value in re.findall(
                r"\[NESTING TWO-WAY PRIVATE\] field=thompson.Nc_kg-1,[^\n]*phase_exchange=([^\s]+)",
                (root / "serial.log").read_text())]
            if not any(value > 0 for value in reaction_replay):
                raise RuntimeError("unseeded Thompson reference did not exercise source-consistent particle feedback")
        water = {}
        if moist:
            water = require_water_closure((root / "serial.log").read_text(), active_cloud)
            if seeded_cloud and not (any(abs(row[3]) > 1 for row in water["qv"]) and any(abs(row[3]) > 1 for row in water["qc"])):
                raise RuntimeError("seeded private physics did not exchange cloud and vapor mass")
            parallel_water = require_water_closure((root / "parallel.log").read_text(), active_cloud)
            resumed_water = require_water_closure(continuation.stdout)
            if water != parallel_water:
                raise RuntimeError("water ledgers changed with worker count")
            for name, values in resumed_water.items():
                if values != water[name][-len(values):]:
                    raise RuntimeError(f"{name} water ledger changed across restart")
            require_feedback_output_budget(serial,duration,water_fields)
            require_feedback_output_budget(parallel,duration,water_fields)
            require_feedback_output_budget(resumed,duration-midpoint,water_fields)
        for checkpoint_path in (serial / "checkpoints").glob("*.nimbus"):
            if checkpoint_path.read_bytes() != (parallel / "checkpoints" / checkpoint_path.name).read_bytes():
                raise RuntimeError("checkpoint state changed with worker count")
        final_checkpoint=f"checkpoint_step_{round(duration/dt):09d}.nimbus"
        if (parallel / "checkpoints" / final_checkpoint).read_bytes() != (resumed / "checkpoints" / final_checkpoint).read_bytes():
            raise RuntimeError("completed nested checkpoint is not restart-exact")
        (root / "summary.json").write_text(json.dumps({
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "configuration_sha256": hashlib.sha256(selected_config.read_bytes()).hexdigest(),
            "backend": backend, "advection": "weno5" if weno else "tvd",
            "moist": moist, "seeded_cloud": seeded_cloud, "private_closure": private_closure, "active_condensation": active_cloud, "lin_ice_activity": lin, "water_closure": water,
            "dry_currency_closure": serial_closure, "threading": "exact", "restart": "exact", "negative_admission": True,
            "reaction_replay": reaction_replay,
        },indent=2)+"\n")

    print(
        "[two-way-nested-ready] PASS: maintained stationary feedback "
        "evolved with nonzero local reflux, every selected currency closed, "
        "1/4 workers and restart matched bitwise, and near profiles failed closed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
