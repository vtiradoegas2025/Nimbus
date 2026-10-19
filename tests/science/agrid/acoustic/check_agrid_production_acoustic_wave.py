#!/usr/bin/env python3
"""Measure a production-integrator A-grid acoustic-wave timestep sweep."""
from __future__ import annotations
import argparse, cmath, json, math
import sys
from pathlib import Path
from typing import Sequence

COMMON_TOOLS = Path(__file__).resolve().parents[2] / "common"
if str(COMMON_TOOLS) not in sys.path:
    sys.path.insert(0, str(COMMON_TOOLS))

from check_hydrostatic_hold import (QualificationError, STEP_DIR_PATTERN, load_fields,
    parse_frame, physical_indices, read_json_object, require, valid_generation, validate_summary)

GAMMA = 1.4

def wrap(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))

def frames(run: Path):
    marker = read_json_object(run / "nimbus_output.json")
    generation = marker.get("publication_generation")
    require(valid_generation(generation), f"{run}: invalid generation")
    found = sorted((parse_frame(p, generation) for p in run.iterdir()
                    if p.is_dir() and STEP_DIR_PATTERN.fullmatch(p.name)), key=lambda f: f.time_s)
    require(len(found) == 2, f"{run}: expected two frames")
    validate_summary(run)
    return found[0], load_fields(found[0]), found[1], load_fields(found[1])

def metrics(run: Path) -> dict[str, float]:
    f0, a, f1, b = frames(run)
    n = f0.valid_upper[0] - f0.valid_lower[0] + 1
    length = n * f0.dx_m
    duration = f1.time_s - f0.time_s
    ratios: list[complex] = []
    energy0 = energy1 = mass0 = mass1 = 0.0
    for j in range(f0.valid_lower[1], f0.valid_upper[1] + 1):
      for k in range(f0.valid_lower[2], f0.valid_upper[2] + 1):
        p0bar = math.fsum(a["p"].at(i,j,k) for i in range(f0.valid_lower[0], f0.valid_upper[0]+1))/n
        r0bar = math.fsum(a["rho"].at(i,j,k) for i in range(f0.valid_lower[0], f0.valid_upper[0]+1))/n
        p1bar = math.fsum(b["p"].at(i,j,k) for i in range(f0.valid_lower[0], f0.valid_upper[0]+1))/n
        u0bar = math.fsum(a["u"].at(i,j,k) for i in range(f0.valid_lower[0], f0.valid_upper[0]+1))/n
        u1bar = math.fsum(b["u"].at(i,j,k) for i in range(f0.valid_lower[0], f0.valid_upper[0]+1))/n
        c0 = c1 = 0j
        sound = math.sqrt(GAMMA*p0bar/r0bar)
        for m,i in enumerate(range(f0.valid_lower[0], f0.valid_upper[0]+1)):
          basis = cmath.exp(-2j*math.pi*(m+0.5)/n)
          pp0=a["p"].at(i,j,k)-p0bar; pp1=b["p"].at(i,j,k)-p1bar
          up0=a["u"].at(i,j,k)-u0bar; up1=b["u"].at(i,j,k)-u1bar
          c0 += pp0*basis; c1 += pp1*basis
          energy0 += .5*r0bar*up0*up0 + .5*pp0*pp0/(r0bar*sound*sound)
          energy1 += .5*r0bar*up1*up1 + .5*pp1*pp1/(r0bar*sound*sound)
          mass0 += a["rho"].at(i,j,k); mass1 += b["rho"].at(i,j,k)
        require(abs(c0)>0, "zero initial acoustic mode")
        ratios.append(c1/c0)
    ratio=sum(ratios)/len(ratios)
    mean_sound=math.fsum(math.sqrt(GAMMA*(math.fsum(a["p"].at(i,j,k) for i in range(f0.valid_lower[0],f0.valid_upper[0]+1))/n)/(math.fsum(a["rho"].at(i,j,k) for i in range(f0.valid_lower[0],f0.valid_upper[0]+1))/n)) for j in range(f0.valid_lower[1],f0.valid_upper[1]+1) for k in range(f0.valid_lower[2],f0.valid_upper[2]+1))/len(ratios)
    expected=-2*math.pi*mean_sound*duration/length
    return {"duration_s":duration,"physical_nx":n,"dx_m":f0.dx_m,"phase_rad":cmath.phase(ratio),"expected_phase_rad":wrap(expected),
      "phase_error_rad":abs(wrap(cmath.phase(ratio)-expected)),"amplitude_ratio":abs(ratio),
      "acoustic_energy_relative_drift":abs(energy1-energy0)/energy0,
      "dry_mass_relative_drift":abs(mass1-mass0)/mass0}

def qualify(runs: dict[str,Path]):
    out={name:metrics(path) for name,path in runs.items()}
    for name,m in out.items():
      require(m["phase_error_rad"]<0.03, f"{name}: phase error too large")
      require(0.98<m["amplitude_ratio"]<1.03, f"{name}: amplitude ratio out of range")
      require(m["acoustic_energy_relative_drift"]<0.01, f"{name}: acoustic energy drift too large")
      require(m["dry_mass_relative_drift"]<2e-6, f"{name}: mass drift too large")
    coarse, medium, fine = (out["1.2"]["phase_error_rad"], out["0.6"]["phase_error_rad"], out["0.3"]["phase_error_rad"])
    require(coarse/medium>1.1 and medium/fine>1.1, "phase error does not improve consistently under timestep refinement")
    return {"status":"PASS","timesteps_s":[1.2,0.6,0.3],"phase_error_refinement_ratios":[coarse/medium,medium/fine],"runs":out}

def main(argv: Sequence[str]|None=None)->int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("run_1_2",type=Path); p.add_argument("run_0_6",type=Path); p.add_argument("run_0_3",type=Path)
    a=p.parse_args(argv)
    try: result=qualify({"1.2":a.run_1_2,"0.6":a.run_0_6,"0.3":a.run_0_3})
    except QualificationError as e: p.exit(1,f"[agrid-production-acoustic] FAIL: {e}\n")
    print(json.dumps(result,indent=2,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
