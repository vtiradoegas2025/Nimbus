#!/usr/bin/env python3
"""Enumerate a fixed flat dry C-grid boundary family and probe bounded admission."""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import product
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.qualification.materialize_retained_fixtures import identity
from scripts.tests.catalog.qualify_catalog_bounded_runs import write_json_atomic


BASE = ROOT / "tests/fixtures/reference/catalog/generated_metal_canonical_v1.yaml"
LATERAL = ("rigid", "rigid_slip", "radiative", "zero_gradient")
BOTTOM = ("rigid_slip", "rigid_noslip")
TOP = ("rigid", "rigid_with_sponge", "open")
BACKENDS = (("cpu", "cpu"), ("vulkan", "gpu"), ("metal", "metal"))
FACES = ("west", "east", "south", "north")
REPRESENTATIVES = (
    (("rigid",) * 4, "rigid_slip", "rigid"),
    (("rigid_slip",) * 4, "rigid_noslip", "open"),
    (("radiative",) * 4, "rigid_slip", "rigid_with_sponge"),
    (("zero_gradient",) * 4, "rigid_noslip", "open"),
    (("rigid", "rigid_slip", "radiative", "zero_gradient"), "rigid_slip", "rigid"),
)
SOURCE_OWNERS = (
    "src/init/config/v1_loader.cpp",
    "src/init/config/config_v1.cpp",
    "src/core/infra/coordinate_system.cpp",
    "src/core/runtime/runtime_config.cpp",
    "src/core/runtime/nimbus_sim.cpp",
    "src/boundary_conditions/boundary_conditions_cartesian_cgrid.cpp",
    "src/numerics/advection/advection_cartesian.cpp",
    "src/dynamics/native_cgrid_boundary.cpp",
    "include/boundary_conditions/runtime_state.hpp",
    "include/core/runtime/runtime_config.hpp",
)
EVIDENCE_CLASSES = {
    "lateral_impermeable": {
        "owners": ["src/boundary_conditions/boundary_conditions_cartesian_cgrid.cpp",
                   "src/numerics/advection/advection_cartesian.cpp"],
        "required": ["normal_face_flux_zero", "tangential_rigid_vs_slip_reference",
                     "boundary_mass_and_momentum_closure", "native_cpu_vulkan_metal_parity"]},
    "lateral_radiative_history": {
        "owners": ["include/boundary_conditions/runtime_state.hpp",
                   "src/boundary_conditions/boundary_conditions_cartesian_cgrid.cpp",
                   "src/dynamics/native_cgrid_boundary.cpp"],
        "required": ["kw78_outgoing_and_incoming_reference", "history_restart_continuation",
                     "failure_atomicity_of_wave_state", "thread_and_backend_parity"]},
    "lateral_zero_gradient": {
        "owners": ["src/boundary_conditions/boundary_conditions_cartesian_cgrid.cpp",
                   "src/numerics/advection/advection_cartesian.cpp"],
        "required": ["face_extrapolation_reference", "signed_boundary_flux_closure",
                     "thread_and_backend_parity"]},
    "bottom_free_slip": {
        "owners": ["src/boundary_conditions/boundary_conditions_cartesian_cgrid.cpp"],
        "required": ["prognostic_near_surface_shear", "zero_vertical_momentum_flux",
                     "thread_and_backend_parity"]},
    "bottom_no_slip": {
        "owners": ["src/boundary_conditions/boundary_conditions_cartesian_cgrid.cpp"],
        "required": ["ground_tangential_velocity_pinning", "surface_momentum_budget",
                     "thread_and_backend_parity"]},
    "top_open_flux": {
        "owners": ["src/boundary_conditions/boundary_conditions_cartesian_cgrid.cpp",
                   "src/numerics/advection/advection_cartesian.cpp"],
        "required": ["vertical_outflow_reference", "top_mass_scalar_energy_flux_closure",
                     "thread_and_backend_parity"]},
    "top_impermeable_flux": {
        "owners": ["src/boundary_conditions/boundary_conditions_cartesian_cgrid.cpp",
                   "src/numerics/advection/advection_cartesian.cpp"],
        "required": ["zero_top_face_flux", "thermal_and_momentum_closure",
                     "thread_and_backend_parity"]},
    "separate_top_sponge_inactive": {
        "owners": ["src/core/runtime/runtime_config.cpp",
                   "src/boundary_conditions/boundary_conditions_cartesian_cgrid.cpp"],
        "required": ["damping_disabled_preflight", "same_top_face_closure_as_rigid",
                     "distinct_public_selector_admission"]},
    "mixed_lateral_corner": {
        "owners": ["src/boundary_conditions/boundary_conditions_cartesian_cgrid.cpp",
                   "src/numerics/advection/advection_cartesian.cpp"],
        "required": ["corner_order_reference", "face_attribution_without_double_counting",
                     "thread_and_backend_parity"]},
}


def family_key(laterals: tuple[str, ...], bottom: str, top: str) -> str:
    values = {**dict(zip(FACES, laterals)), "bottom": bottom, "top": top}
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def family_id(laterals: tuple[str, ...], bottom: str, top: str) -> str:
    return "boundary:" + hashlib.sha256(family_key(laterals, bottom, top).encode()).hexdigest()[:24]


def physical_id(laterals: tuple[str, ...], bottom: str, top: str) -> str:
    closure = "rigid" if top == "rigid_with_sponge" else top
    return "boundary-closure:" + hashlib.sha256(
        family_key(laterals, bottom, closure).encode()).hexdigest()[:24]


def evidence_classes(laterals: tuple[str, ...], bottom: str, top: str) -> list[str]:
    classes = []
    if any(value in {"rigid", "rigid_slip"} for value in laterals):
        classes.append("lateral_impermeable")
    if "radiative" in laterals:
        classes.append("lateral_radiative_history")
    if "zero_gradient" in laterals:
        classes.append("lateral_zero_gradient")
    classes.append("bottom_free_slip" if bottom == "rigid_slip" else "bottom_no_slip")
    classes.append("top_open_flux" if top == "open" else "top_impermeable_flux")
    if top == "rigid_with_sponge":
        classes.append("separate_top_sponge_inactive")
    if len(set(laterals)) > 1:
        classes.append("mixed_lateral_corner")
    return classes


def enumerate_rows() -> list[dict]:
    rows = []
    for laterals in product(LATERAL, repeat=4):
        for bottom, top in product(BOTTOM, TOP):
            rows.append({"id": family_id(laterals, bottom, top),
                         "lateral": dict(zip(FACES, laterals)), "bottom": bottom, "top": top,
                         "physicalBoundaryClosureId": physical_id(laterals, bottom, top),
                         "topClosure": "rigid" if top == "rigid_with_sponge" else top,
                         "topSpongeState": "selector_requests_sponge_but_separate_damping_disabled"
                         if top == "rigid_with_sponge" else "not_selected",
                         "requiredEvidenceClasses": evidence_classes(laterals, bottom, top),
                         "structuralLoaderCompatibility": "source_reviewed_nonperiodic_c_grid",
                         "ordinaryAdmission": "not_checked_by_enumeration",
                         "boundedEvidence": "not_checked_by_enumeration"})
    if len(rows) != 1536 or len({row["id"] for row in rows}) != len(rows) or \
            len({row["physicalBoundaryClosureId"] for row in rows}) != 1024:
        raise RuntimeError("flat boundary categorical or physical-closure accounting changed")
    return rows


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"boundary source token is missing or ambiguous: {old!r}")
    return source.replace(old, new, 1)


def variant(source: str, laterals: tuple[str, ...], bottom: str, top: str,
            backend: str) -> str:
    if len(laterals) != 4 or backend not in {entry[1] for entry in BACKENDS}:
        raise RuntimeError("boundary variant selector is invalid")
    output = source
    for face, value in zip(FACES, laterals):
        output = replace_once(output, f"    {face}: rigid\n", f"    {face}: {value}\n")
    output = replace_once(output, "  bottom: rigid_slip\n", f"  bottom: {bottom}\n")
    output = replace_once(output, "  top: rigid\n", f"  top: {top}\n")
    output = replace_once(output, "    backend: metal\n", f"    backend: {backend}\n")
    return output


def run_command(command: list[str]) -> dict:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                            timeout=60, check=False,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    output = result.stdout + result.stderr
    return {"exitCode": result.returncode,
            "stdoutSha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
            "stderrSha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
            "outputExcerpt": output[-1600:]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--native", type=Path, default=ROOT / "bin/nimbus_native_admission_batch")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("use a new output directory")
    output.mkdir(parents=True)
    binary = args.binary.resolve(strict=True)
    native = args.native.resolve(strict=True)
    inputs = {"binary": identity(binary), "native": identity(native),
              "base": identity(BASE), "enumerator": identity(Path(__file__))}
    inputs.update({path: identity(ROOT / path) for path in SOURCE_OWNERS})
    source = BASE.read_text()
    if "  scheme: none\n" not in source or "  staggering: c_grid\n" not in source or \
            "  mode: qualified\n" not in source or "  format: npy_3d\n" not in source or \
            "  enabled: false\n" not in source:
        raise RuntimeError("canonical flat dry boundary seed changed structural class")
    rows = enumerate_rows()
    reps = []
    sample_dir = output / "representatives"
    sample_dir.mkdir()
    for laterals, bottom, top in REPRESENTATIVES:
        identifier = family_id(laterals, bottom, top)
        for label, backend in BACKENDS:
            path = sample_dir / f"{identifier.removeprefix('boundary:')}_{label}.yaml"
            path.write_text(variant(source, laterals, bottom, top, backend))
            canonical = subprocess.run([str(native), "--canonical-config", str(path)],
                                       cwd=ROOT, text=True, capture_output=True, timeout=60,
                                       check=False)
            if canonical.returncode != 0:
                raise RuntimeError(f"source-compatible boundary representative rejected by strict loader: {path}: {canonical.stderr[-1000:]}")
            admission = run_command([str(binary), "--config", str(path), "--validate-only"])
            lines = [line for line in admission["outputExcerpt"].splitlines()
                     if line.startswith("[VALIDATE]")]
            admitted = admission["exitCode"] == 0 and any(line.startswith("[VALIDATE] PASS:") for line in lines)
            reps.append({"familyId": identifier, "backend": label,
                         "selector": backend, "config": identity(path),
                         "canonicalYamlSha256": hashlib.sha256(canonical.stdout.encode()).hexdigest(),
                         "strictLoader": "accepted", "ordinaryQualifiedAdmission":
                         "admitted" if admitted else "rejected", "validation": admission})
    exclusions = [
        {"selector": "lateral.open", "classification": "alias",
         "reason": "C-grid strict loader resolves legacy open to radiative before serialization",
         "owner": "src/init/config/v1_loader.cpp"},
        {"selector": "single_periodic_face", "classification": "incompatible",
         "reason": "strict loader requires periodic opposing faces as a pair",
         "owner": "src/init/config/v1_loader.cpp"},
        {"selector": "paired_periodic_faces", "classification": "different_retained_family",
         "reason": "valid periodic domain topology is not a nonperiodic-face closure",
         "owner": "src/init/config/v1_loader.cpp"},
        {"selector": "collocated_radiative", "classification": "incompatible",
         "reason": "radiative lateral closure requires C-grid staggering",
         "owner": "src/init/config/v1_loader.cpp"},
        {"selector": "top.rigid_with_sponge_with_damping_off", "classification": "same_physical_boundary_closure_not_public_alias",
         "reason": "C-grid top face is rigid; Rayleigh damping has a separate disabled owner in this fixed family",
         "owner": "src/boundary_conditions/boundary_conditions_cartesian_cgrid.cpp"},
    ]
    baseline = variant(source, ("radiative", "rigid", "rigid", "rigid"),
                       "rigid_slip", "rigid", "cpu")
    negatives = (("lateral_open_alias", replace_once(baseline, "    west: radiative\n", "    west: open\n")),
                 ("single_periodic_face", replace_once(baseline, "    west: radiative\n", "    west: periodic\n")),
                 ("paired_periodic_faces", replace_once(replace_once(baseline, "    west: radiative\n", "    west: periodic\n"),
                                                        "    east: rigid\n", "    east: periodic\n")),
                 ("collocated_radiative", replace_once(baseline, "  staggering: c_grid\n", "  staggering: collocated\n")))
    probes = []
    canonical_reference = None
    for name, config in (("radiative_reference", baseline), *negatives):
        path = sample_dir / (name + ".yaml")
        path.write_text(config)
        result = subprocess.run([str(native), "--canonical-config", str(path)],
                                cwd=ROOT, text=True, capture_output=True, timeout=60,
                                check=False)
        if name == "radiative_reference":
            canonical_reference = result.stdout
        probes.append({"name": name, "config": identity(path),
                       "strictLoader": "accepted" if result.returncode == 0 else "rejected",
                       "canonicalYamlSha256": hashlib.sha256(result.stdout.encode()).hexdigest()
                       if result.returncode == 0 else None,
                       "errorExcerpt": result.stderr[-1000:] if result.returncode else None})
    probe_by_name = {row["name"]: row for row in probes}
    if probe_by_name["radiative_reference"]["strictLoader"] != "accepted" or \
            probe_by_name["lateral_open_alias"]["canonicalYamlSha256"] != \
            hashlib.sha256(canonical_reference.encode()).hexdigest() or \
            probe_by_name["single_periodic_face"]["strictLoader"] != "rejected" or \
            probe_by_name["paired_periodic_faces"]["strictLoader"] != "accepted" or \
            probe_by_name["collocated_radiative"]["strictLoader"] != "rejected":
        raise RuntimeError("boundary alias/incompatibility probe disagreed with source rule")
    for name, entry in inputs.items():
        if identity(ROOT / entry["path"]) != entry:
            raise RuntimeError(f"boundary family input changed during enumeration: {name}")
    report = {"schema": "nimbus.flat-dry-cgrid-boundary-family.v1", "status": "complete",
              "claimBoundary": "Complete fixed-profile nonperiodic boundary selector enumeration and 15 admission representatives; not full-product combinations, all-profile admission, bounded execution, backend parity or Ready.",
              "inputs": inputs, "fixedScope": {
                  "grid": "Cartesian C-grid 18x18x18, 500/500/250 m",
                  "solver": "dry TVD, split-explicit 0.1 s, six acoustic substeps",
                  "motion": "stationary, no nesting/tracking", "terrain": "none",
                  "damping": "disabled, including separate top sponge",
                  "output": "minimal NPY", "execution": "qualified"},
              "axes": {"lateralPerFace": LATERAL, "bottom": BOTTOM, "top": TOP},
              "categoricalProfiles": len(rows),
              "physicalBoundaryClosureClasses": len({row["physicalBoundaryClosureId"] for row in rows}),
              "representativeAdmissionCounts": dict(sorted(Counter(row["ordinaryQualifiedAdmission"]
                                                              for row in reps).items())),
              "sourceRuleExclusions": exclusions,
              "stateOwnerEvidenceClasses": EVIDENCE_CLASSES,
              "sourceRuleProbes": probes, "representatives": reps, "rows": rows,
              "I1Complete": False, "wholeBackendReady": False}
    write_json_atomic(output / "summary.json", report)
    print(json.dumps({"status": "complete", "profiles": len(rows),
                      "physicalClosures": report["physicalBoundaryClosureClasses"],
                      "representatives": report["representativeAdmissionCounts"]}))


if __name__ == "__main__":
    main()
