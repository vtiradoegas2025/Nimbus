#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from pathlib import Path
import subprocess
import tempfile

from npy_support import read_float32_npy


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "tests/fixtures/configs/qualification/agrid_full_stochastic_ready.yaml"
DEFAULT_BINARY = ROOT / "bin/nimbus_sim"
FIELDS = (
    "u", "v", "w", "rho", "p", "theta", "qv", "tracer",
    "qc", "qr", "qi", "qs", "qg", "qh", "tke",
)
LEDGERS = (
    "budget.csv", "momentum_budget.csv", "energy_budget.csv",
    "radiation_budget.csv", "moisture_budget.csv",
    "thermodynamic_source_budget.csv",
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


def field_payload(output: Path, second: int, field: str) -> bytes:
    path = output / f"step_{second:06d}" / f"{field}.npy"
    shape, raw, values = read_float32_npy(path)
    if shape != (18, 18, 18) or any(not math.isfinite(value) for value in values):
        raise RuntimeError(f"invalid A-grid chaos output {path}")
    if field in ("rho", "p", "theta") and any(value <= 0.0 for value in values):
        raise RuntimeError(f"non-positive A-grid chaos output {path}")
    if field in ("qv", "tracer", "qc", "qr", "qi", "qs", "qg", "qh", "tke") and any(
        value < 0.0 for value in values
    ):
        raise RuntimeError(f"negative A-grid chaos output {path}")
    return raw


def render_chaos(
    source: str,
    scheme: str,
    tendencies: tuple[str, ...] = (),
    member_id: int = 4,
) -> str:
    begin = source.index("  chaos:\n")
    end = source.index("  checkpoint:\n", begin)
    lines = [
        "  chaos:",
        f"    scheme: {scheme}",
        "    seed: 918273",
        f"    member_id: {member_id}",
        "    native_generation_required: false",
    ]
    if scheme in ("initial_conditions", "full_stochastic"):
        lines.append("    apply_to_ic: u,v,w,theta")
    if tendencies:
        lines.append("    apply_to_tendencies: " + ",".join(tendencies))
    return source[:begin] + "\n".join(lines) + "\n" + source[end:]


def render_coriolis(source: str, enabled: bool, mode: str) -> str:
    begin = source.index("coriolis:\n")
    end = source.index("\nmicrophysics:\n", begin)
    block = (
        "coriolis:\n"
        f"  enabled: {'true' if enabled else 'false'}\n"
        f"  mode: {mode}\n"
        "  latitude_deg: 35\n"
    )
    return source[:begin] + block + source[end + 1:]


def execute(
    binary: Path,
    config: Path,
    output: Path,
    threads: int,
    quiet: bool = True,
) -> str:
    result = run([
        str(binary), "--headless", "--config", str(config),
        f"--outdir={output}", "--write-every=1", f"--threads={threads}",
        "--log-profile=" + ("quiet" if quiet else "normal"),
    ])
    require_success(result, f"A-grid chaos run ({config.stem}, {threads} workers)")
    return result.stdout


def require_distinct(
    lhs: dict[str, bytes],
    rhs: dict[str, bytes],
    label: str,
    candidates: tuple[str, ...] = FIELDS,
) -> None:
    if all(lhs[field] == rhs[field] for field in candidates):
        raise RuntimeError(f"{label} was inert across {candidates}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    args = parser.parse_args()
    binary = args.binary.resolve()
    for path in (binary, CONFIG):
        if not path.is_file():
            raise RuntimeError(f"required A-grid chaos input is missing: {path}")

    source = CONFIG.read_text(encoding="utf-8")
    profiles = {
        "none": render_chaos(source, "none"),
        "initial_conditions": render_chaos(source, "initial_conditions"),
        "boundary_layer": render_chaos(source, "boundary_layer", ("pbl",)),
        "full_microphysics": render_chaos(
            source, "full_stochastic", ("microphysics",)
        ),
        "full_pbl": render_chaos(source, "full_stochastic", ("pbl",)),
        "full_diffusion": render_chaos(
            source, "full_stochastic", ("diffusion",)
        ),
        "full_all": source,
        "coriolis_disabled": render_coriolis(source, False, "f_plane"),
        "coriolis_f_plane": render_coriolis(source, True, "f_plane"),
    }

    with tempfile.TemporaryDirectory(prefix="nimbus-agrid-chaos-") as temporary:
        root = Path(temporary)
        finals: dict[str, dict[str, bytes]] = {}
        logs: dict[str, str] = {}
        configs: dict[str, Path] = {}
        for name, rendered in profiles.items():
            config = root / f"{name}.yaml"
            config.write_text(rendered, encoding="utf-8")
            configs[name] = config
            preflight = run([str(binary), "--validate-only", "--config", str(config)])
            require_success(preflight, f"A-grid chaos {name} preflight")
            if "[VALIDATE] PASS" not in preflight.stdout:
                raise RuntimeError(f"A-grid chaos {name} preflight omitted PASS")
            output = root / name
            logs[name] = execute(binary, config, output, 1, name != "full_all")
            finals[name] = {
                field: field_payload(output, 2, field) for field in FIELDS
            }

        require_distinct(
            finals["initial_conditions"], finals["none"],
            "A-grid initial-condition chaos", ("u", "v", "w", "theta", "rho", "p"),
        )
        require_distinct(
            finals["boundary_layer"], finals["none"],
            "A-grid boundary-layer chaos", ("u", "v", "theta", "qv", "tke"),
        )
        require_distinct(
            finals["full_microphysics"], finals["initial_conditions"],
            "A-grid full-stochastic microphysics control",
            ("theta", "qv", "qc", "qr", "qi", "qs", "qg", "qh"),
        )
        require_distinct(
            finals["full_pbl"], finals["initial_conditions"],
            "A-grid full-stochastic PBL control",
            ("u", "v", "theta", "qv", "tke"),
        )
        require_distinct(
            finals["full_diffusion"], finals["initial_conditions"],
            "A-grid full-stochastic SGS control",
            ("u", "v", "w", "theta", "qv"),
        )
        for isolated in ("full_microphysics", "full_pbl", "full_diffusion"):
            require_distinct(
                finals["full_all"], finals[isolated],
                f"A-grid all-process stochastic composition versus {isolated}",
            )
        require_distinct(
            finals["coriolis_f_plane"], finals["coriolis_disabled"],
            "A-grid f-plane Coriolis", ("u", "v"),
        )
        require_distinct(
            finals["full_all"], finals["coriolis_f_plane"],
            "A-grid beta-plane Coriolis", ("u", "v"),
        )

        full_log = logs["full_all"]
        for marker in (
            "Execution mode: qualified",
            "Dynamics: klemp_wilhelmson_split_cartesian_agrid",
            "Numerics-Advection: weno5",
            "Numerics-Diffusion: implicit",
            "Numerics-TimeStepping: rk4",
            "Initialized microphysics scheme: thompson",
            "Initialized boundary layer scheme: mynn",
            "Initialized turbulence scheme: smagorinsky",
            "Initialized radiation scheme: rrtmg",
            "Initialized chaos scheme: full_stochastic",
            "Coriolis f(y=0)",
            "[COMPUTE] backend=cpu",
            "fallback=off",
        ):
            if marker not in full_log:
                raise RuntimeError(f"A-grid full stochastic run omitted {marker!r}")

        parallel = root / "full-all-parallel"
        parallel_log = execute(binary, CONFIG, parallel, 4, False)
        for marker in ("Initialized chaos scheme: full_stochastic", "threads=4"):
            if marker not in parallel_log:
                raise RuntimeError(f"parallel A-grid chaos run omitted {marker!r}")
        for field in FIELDS:
            if field_payload(parallel, 2, field) != finals["full_all"][field]:
                raise RuntimeError(f"A-grid full-stochastic {field} differs across workers")
        serial = root / "full_all"
        for ledger in LEDGERS:
            if (serial / ledger).read_bytes() != (parallel / ledger).read_bytes():
                raise RuntimeError(f"A-grid full-stochastic {ledger} differs across workers")
        serial_checkpoint = serial / "checkpoints/checkpoint_step_000000010.nimbus"
        checkpoint = parallel / "checkpoints/checkpoint_step_000000010.nimbus"
        if serial_checkpoint.read_bytes() != checkpoint.read_bytes():
            raise RuntimeError("A-grid full-stochastic checkpoint differs across workers")

        resumed = root / "resumed"
        continuation = run([
            str(binary), "--headless", "--config", str(CONFIG),
            f"--resume={checkpoint}", "--duration=2", "--write-every=1",
            "--checkpoint-every=0", f"--outdir={resumed}", "--threads=4",
            "--log-profile=normal",
        ])
        require_success(continuation, "A-grid full-stochastic continuation")
        for field in FIELDS:
            if field_payload(resumed, 1, field) != finals["full_all"][field]:
                raise RuntimeError(f"A-grid full-stochastic {field} restart mismatch")
        for ledger in LEDGERS:
            continuous = (parallel / ledger).read_text().splitlines()
            suffix = (resumed / ledger).read_text().splitlines()[1:]
            if suffix != continuous[-len(suffix):]:
                raise RuntimeError(f"A-grid full-stochastic {ledger} restart mismatch")

        member_config = root / "member-five.yaml"
        member_config.write_text(
            render_chaos(source, "full_stochastic", ("microphysics", "pbl", "diffusion"), 5),
            encoding="utf-8",
        )
        member_output = root / "member-five"
        execute(binary, member_config, member_output, 1)
        member_fields = {
            field: field_payload(member_output, 2, field) for field in FIELDS
        }
        require_distinct(
            member_fields, finals["full_all"], "A-grid stochastic member identity"
        )

        invalid_documents = {
            "native": source.replace(
                "    native_generation_required: false\n",
                "    native_generation_required: true\n", 1,
            ),
            "backend": source.replace("    backend: cpu\n", "    backend: metal\n", 1),
            "terrain": source.replace(
                "terrain:\n  scheme: none\n", "terrain:\n  scheme: bell\n", 1
            ),
            "target": source.replace(
                "    apply_to_tendencies: microphysics,pbl,diffusion\n",
                "    apply_to_tendencies: microphysics,unknown\n", 1,
            ),
            "latitude": source.replace(
                "  latitude_deg: 35\n", "  latitude_deg: 95\n", 1
            ),
        }
        for label, document in invalid_documents.items():
            if document == source:
                raise RuntimeError(f"A-grid chaos {label} mutation drifted")
            invalid = root / f"invalid-{label}.yaml"
            invalid.write_text(document, encoding="utf-8")
            rejected = run([str(binary), "--validate-only", "--config", str(invalid)])
            if rejected.returncode == 0:
                raise RuntimeError(f"A-grid chaos {label} neighbor was admitted")

    print(
        "[agrid-chaos-ready] PASS: initial-condition, PBL, each "
        "full-stochastic process, and f/beta-plane Coriolis changed the "
        "maximal collocated stack; the "
        "all-process profile matched 1/4-worker fields, checkpoints and "
        "ledgers, resumed exactly, preserved member diversity, and rejected "
        "unsupported neighbors"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
