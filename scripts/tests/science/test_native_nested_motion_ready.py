#!/usr/bin/env python3
"""Public native moving-child and parent-recycling feedback evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from npy_support import read_float32_npy
from test_native_two_way_nested_ready import compare_private

ROOT = Path(__file__).resolve().parents[3]
LIMITS = {"rho": .001, "theta": .05, "p": 200, "qv": 1e-5, "qc": 1e-5, "qr": 1e-5,
          "qi": 1e-5, "qs": 1e-5, "qg": 1e-5, "qh": 1e-5,
          "u": .1, "v": .1, "w": .05}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(reference, actual, exact=False):
    result = {}
    for level in (Path(""),Path("nests/child_0000")):
        source = reference / level / "step_000001"
        target = actual / level / "step_000001"
        for key in ("grid_origin_t0_earth_m", "frame_velocity_ms"):
            a = json.loads((source/"manifest.json").read_text())["grid"]["horizontal_frame"]
            b = json.loads((target/"manifest.json").read_text())["grid"]["horizontal_frame"]
            if a.get(key) != b.get(key):
                raise RuntimeError(f"frame parity changed {key}: {a} vs {b}")
        for field,limit in LIMITS.items():
            a = source/f"{field}.npy"
            b = target/f"{field}.npy"
            if not a.exists():
                continue
            shape,raw,expected = read_float32_npy(a)
            other_shape,other_raw,values = read_float32_npy(b)
            if shape != other_shape or any(not math.isfinite(v) for v in values):
                raise RuntimeError(f"invalid native field: {b}")
            if field.startswith("q") and min(values) < 0:
                raise RuntimeError(f"negative water: {b}")
            difference = max(abs(x-y) for x,y in zip(expected,values))
            if (exact and raw != other_raw) or difference > limit:
                raise RuntimeError(f"native motion parity failed: {b}: {difference} > {0 if exact else limit}")
            result[f"{level}/{field}"] = {"max_absolute_error":difference,"absolute_limit":0 if exact else limit}
    return result


def motion_signature(output, moving):
    if moving:
        result=[]
        for p in sorted((output/"storm_track").glob("track_step_*.json")):
            event=json.loads(p.read_text())
            if event["state_mutation"]:
                authority=event["movement_authority"]
                result.append((event["simulation_time_s"],authority["committed_shift_cells"],
                               authority["post_shift_child_grid_origin_t0_earth_m"]))
        return result
    return [(event["simulation_time_s"],event["shift_cells"]) for event in
            (json.loads(p.read_text()) for p in sorted((output/"domain_recycling_swath").glob("transaction_*/event.json")))]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--binary",type=Path,default=ROOT/"bin/nimbus_sim")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--advection",choices=("weno","tvd"),default="weno")
    parser.add_argument("--family",choices=("all","moving","recycling"),default="all")
    parser.add_argument("--terrain", choices=("flat", "bell2", "bell3", "schar2", "schar3"), default="flat")
    parser.add_argument("--microphysics", choices=("kessler", "lin", "thompson", "milbrandt", "milbrandt_no_hail"), default="kessler")
    parser.add_argument("--restart-coverage", choices=("full", "synchronized"), default="full")
    parser.add_argument("--private-probe", type=Path, default=ROOT / "bin/nimbus_private_checkpoint_probe")
    args=parser.parse_args()
    private = args.microphysics in ("thompson", "milbrandt", "milbrandt_no_hail")
    if args.terrain != "flat" and args.family != "moving":
        parser.error("mapped terrain currently requires --family moving")
    if args.terrain == "flat" and args.microphysics == "lin":
        parser.error("flat Lin motion profiles are not yet admitted")
    destination=args.output.resolve()
    destination.mkdir(parents=True,exist_ok=True)
    if any(destination.iterdir()):
        raise RuntimeError("native motion evidence directory must be empty")
    families=("moving","recycling") if args.family=="all" else (args.family,)
    binary=args.binary.resolve()
    identity=sha(binary)
    probe = args.private_probe.resolve() if private else None
    probe_identity = sha(probe) if private else None
    results={}
    configurations={}
    with tempfile.TemporaryDirectory(prefix=".native-feedback-motion-",dir=binary.parent) as temporary:
        snapshot=Path(temporary)/"nimbus_sim"
        shutil.copy2(binary,snapshot)
        if sha(snapshot) != identity:
            raise RuntimeError("native motion executable copy differs")
        probe_snapshot = Path(temporary) / "private_probe"
        if private:
            shutil.copy2(probe, probe_snapshot)
            if sha(probe_snapshot) != probe_identity:
                raise RuntimeError("private motion probe copy differs")
        for family in families:
            moving=family=="moving"
            prefix="moving_nested_two_way" if moving else "nested_grid_recycling_two_way"
            if private:
                prefix = (f"moving_nested_{args.microphysics}_two_way" if moving
                          else f"nested_grid_recycling_{args.microphysics}_two_way")
            if args.terrain != "flat":
                prefix = f"moving_mapped_nested_{args.microphysics}_{args.terrain}"
            script="test_moving_two_way_nested_ready.py" if moving else "test_nested_grid_recycling_ready.py"
            final="serial" if moving else "movement-t1"
            for backend in ("cpu","metal","vulkan","vulkan-staged"):
                label=f"{family}-{backend}"
                config=ROOT/"tests/fixtures/configs/qualification"/f"{prefix}_{args.advection}_{backend.removesuffix('-staged')}_ready.yaml"
                configuration_hash=sha(config)
                configurations[str(config.relative_to(ROOT))]=configuration_hash
                environment=os.environ.copy()
                environment.pop("NIMBUS_VULKAN_MEMORY_POLICY",None)
                environment["NIMBUS_GPU_KERNEL_TIMING"]="1"
                if backend=="vulkan-staged":
                    environment["NIMBUS_VULKAN_MEMORY_POLICY"]="staged"
                command=[sys.executable,str(ROOT/"scripts/tests/science"/script),"--binary",str(snapshot),
                         "--config",str(config),"--output",str(destination/label)]
                if moving:
                    command += ["--restart-coverage", args.restart_coverage]
                result=subprocess.run(command,cwd=ROOT,env=environment,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                (destination/f"{label}.log").write_text(result.stdout)
                if result.returncode:
                    raise RuntimeError(f"{label} failed:\n{result.stdout[-9000:]}")
                evidence=json.loads((destination/label/"summary.json").read_text())
                if evidence["binary_sha256"]!=identity or evidence["configuration_sha256"]!=configuration_hash:
                    raise RuntimeError("native motion evidence identity changed")
                if backend!="cpu":
                    log=(destination/label/f"{final}.log").read_text()
                    counts={name:int(value) for name,value in re.findall(r"([^,{} ]+\.comp):(\d+)",log)}
                    transport = (
                        "advection/advect_conservative_update_cartesian_cgrid.comp",
                        "chaos/earth_control_evolution.comp", "turbulence/sgs_apply.comp"
                    ) if args.terrain == "flat" else (
                        "terrain/mapped_scalar_update.comp", "terrain/mapped_slow_tendencies.comp",
                        "terrain/mapped_fast_momentum.comp"
                    )
                    for kernel in ("dynamics/conservative_momentum.comp", "dynamics/conservative_reduction.comp", *transport):
                        if counts.get(kernel,0)<=0:
                            raise RuntimeError(f"{label} omitted required native operator {kernel}")
                    if private:
                        stages = (("cloud_adjustment", "warm_rain", "freezing", "deposition", "completion", "cloud_phase", "moment_sedimentation")
                                  if args.microphysics == "thompson" else ("cloud_adjustment", "warm_rain", "cold", "melting", "sedimentation"))
                        scheme = "thompson" if args.microphysics == "thompson" else "milbrandt"
                        for stage in stages:
                            kernel = f"microphysics/{scheme}_{stage}.comp"
                            if counts.get(kernel, 0) <= 0:
                                raise RuntimeError(f"{label} omitted native private process {kernel}")
                    reference=destination/f"{family}-cpu"/final
                    output=destination/label/final
                    if motion_signature(reference,moving)!=motion_signature(output,moving):
                        raise RuntimeError(f"{label} movement authority differs from CPU")
                    evidence["parity"]=compare(reference,output)
                    evidence["dispatches"]=counts
                    if private:
                        step, seconds = (240, 120) if moving else (130, 65)
                        ratio = 3 if args.terrain.endswith("3") else 2
                        checkpoint_dirs = None if moving else (
                            destination / f"{family}-cpu/checkpoints-t1", destination / label / "checkpoints-t1")
                        evidence["private_parity"] = compare_private(
                            probe_snapshot, reference, output, seconds, .5, ratio,
                            checkpoint_samples=[(1, step)], checkpoint_directories=checkpoint_dirs)
                    if backend=="vulkan-staged":
                        if "memory plan=staged_forced" not in log:
                            raise RuntimeError("native motion did not exercise forced staging")
                        compare(destination/f"{family}-vulkan"/final,output,True)
                        if private:
                            checkpoint_dirs = None if moving else (
                                destination / f"{family}-vulkan/checkpoints-t1", destination / label / "checkpoints-t1")
                            compare_private(probe_snapshot, destination / f"{family}-vulkan" / final, output,
                                            seconds, .5, ratio, True, checkpoint_samples=[(1, step)],
                                            checkpoint_directories=checkpoint_dirs)
                results[label]=evidence
                print(f"passed {label}: relocation, closure, parity, threading and restart",flush=True)
        if sha(snapshot) != identity or (private and sha(probe_snapshot) != probe_identity):
            raise RuntimeError("executed motion snapshots changed")
    if any(sha(ROOT/name)!=value for name,value in configurations.items()):
        raise RuntimeError("native motion qualification inputs changed")
    (destination/"summary.json").write_text(json.dumps({"status":"Ready","advection":args.advection,
        "terrain":args.terrain,"microphysics":args.microphysics,
        "restart_coverage":args.restart_coverage,
        "binary_sha256":identity,"private_probe_sha256":probe_identity,
        "configuration_sha256":configurations,"results":results,
        "campaign_validated":False,"externally_reviewed":False},indent=2)+"\n")


if __name__=="__main__":
    main()
