#!/usr/bin/env python3
"""Bounded strict native WENO5 transport readiness."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile

from test_native_mapped_terrain_ready import ROOT, execute, field_payload, require_ledgers, require_success, run

FIELDS = ("p", "rho", "theta", "u", "v", "w", "qv")
LEDGERS = ("budget.csv", "momentum_budget.csv", "energy_budget.csv", "moisture_budget.csv",
           "thermodynamic_source_budget.csv", "radiation_budget.csv")


def dispatches(text, backend, steps):
    for marker in (f"[COMPUTE] backend={backend}", "fallback=off", "Execution mode: qualified"):
        if marker not in text:
            raise RuntimeError(f"missing native WENO marker: {marker}")
    line = next((line for line in text.splitlines() if "[COMPUTE DISPATCH]" in line), "")
    counts = {name: int(n) for name, n in re.findall(r"([^,{} ]+\.comp):(\d+)", line)}
    expected = {"advection/advect_conservative_update_cartesian_cgrid.comp": 3*steps}
    if backend == "vulkan":
        expected["advection/advect_conservative_scalar_flux_cartesian_cgrid.comp"] = 3*steps
        expected["advection/advect_pd_limiter_cartesian_cgrid.comp"] = 2*steps
    elif counts.get("advection/advect_pd_limiter_cartesian_cgrid.comp", 0):
        raise RuntimeError("Metal reported a separate dispatch for its fused donor limiter")
    for name, count in expected.items():
        if counts.get(name, 0) != count:
            raise RuntimeError(f"WENO owner {name}: expected {count}, observed {counts}")
    if counts.get("advection/advect_conservative_mass_flux_cartesian_cgrid.comp", 0):
        raise RuntimeError("WENO recomputed rather than consumed retained acoustic mass fluxes")
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--backend", choices=("metal", "vulkan"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    identity = hashlib.sha256(binary.read_bytes()).hexdigest()
    config = ROOT / f"tests/fixtures/configs/qualification/cgrid_weno5_rrtmg_{args.backend}_ready.yaml"
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        if any(args.output.iterdir()):
            raise RuntimeError("WENO evidence output must be empty")
    context = nullcontext(args.output.resolve()) if args.output else tempfile.TemporaryDirectory(prefix="nimbus-native-weno-")
    with context as directory:
        root = Path(directory)
        require_success(run([str(binary), "--validate-only", "--config", str(config)]), "WENO preflight")
        source = config.read_text()
        for label, old, new in (("fallback", "allow_fallback: false", "allow_fallback: true"),
                                ("shape", "nx: 12", "nx: 13"),
                                ("slow-owner", "gpu_apply_slow: false", "gpu_apply_slow: true")):
            changed = source.replace(old, new, 1)
            if changed == source:
                raise RuntimeError(f"missing mutation anchor: {label}")
            invalid = root / f"invalid-{label}.yaml"
            invalid.write_text(changed)
            for options in (("--validate-only",), ("--headless", "--duration=0", f"--outdir={root/label}")):
                result = run([str(binary), "--config", str(invalid), *options])
                if result.returncode == 0:
                    raise RuntimeError(f"invalid WENO profile was admitted: {label}")
        one, four, cpu, resumed = (root/name for name in ("one", "four", "cpu", "resumed"))
        counts = []
        for workers, output in ((1, one), (4, four)):
            result = execute(binary, config, output, workers, f"native WENO {workers} workers")
            counts.append(dispatches(result.stdout, args.backend, 10))
            (root/f"{workers}-workers.log").write_text(result.stdout)
            require_ledgers(output)
        if counts[0] != counts[1]:
            raise RuntimeError("threading changed WENO operation counts")
        cpu_config = ROOT / "tests/fixtures/configs/qualification/cgrid_weno5_rrtmg_ready.yaml"
        execute(binary, cpu_config, cpu, 1, "CPU WENO reference")
        parity = {}
        for second in (1, 2):
            for field in FIELDS:
                _, _, expected = field_payload(cpu, second, field)
                _, raw, actual = field_payload(one, second, field)
                if raw != field_payload(four, second, field)[1] or any(not math.isfinite(x) for values in (expected, actual) for x in values):
                    raise RuntimeError(f"WENO {field} threading or finiteness failed")
                scale = max(abs(x) for x in expected)
                limit = second*5*4*math.ldexp(1, math.frexp(scale)[1]-24) if scale else 0
                if field in ("u", "v", "w"):
                    limit = second*5*5e-5
                error = max(abs(a-b) for a, b in zip(actual, expected))
                parity[f"{second}s/{field}"] = {"maxAbsoluteError":error,"absoluteLimit":limit}
        (root/'parity.json').write_text(json.dumps(parity,indent=2)+'\n')
        if any(r['maxAbsoluteError'] > r['absoluteLimit'] for r in parity.values()):
            raise RuntimeError(f"native WENO parity failed: {parity}")
        for name in LEDGERS:
            if (one/name).read_bytes() != (four/name).read_bytes():
                raise RuntimeError(f"threading changed WENO {name}")
        for step in (5,10):
            name=f"checkpoints/checkpoint_step_{step:09d}.nimbus"
            if (one/name).read_bytes() != (four/name).read_bytes():
                raise RuntimeError("threading changed WENO checkpoint")
        checkpoint=four/'checkpoints/checkpoint_step_000000005.nimbus'
        result=run([str(binary),'--headless','--config',str(config),f'--resume={checkpoint}',f'--outdir={resumed}','--threads=4','--log-profile=normal'])
        require_success(result,'native WENO restart')
        dispatches(result.stdout,args.backend,5)
        require_ledgers(resumed)
        for field in FIELDS:
            if field_payload(one,2,field)[1] != field_payload(resumed,1,field)[1]:
                raise RuntimeError(f'WENO restart changed {field}')
        for name in LEDGERS:
            reference=(one/name).read_text().splitlines();restart=(resumed/name).read_text().splitlines()
            if len(restart)<2 or restart[0]!=reference[0] or reference[-(len(restart)-1):]!=restart[1:]:
                raise RuntimeError(f'WENO restart changed {name}')
        if hashlib.sha256(binary.read_bytes()).hexdigest()!=identity:
            raise RuntimeError('WENO simulator changed during checks')
        (root/'result.json').write_text(json.dumps({'status':'passed','binarySha256':identity,'backend':args.backend,'parity':parity,'kernels':counts[0]},indent=2)+'\n')
    print(f'[native-weno-ready] PASS {args.backend}: native transport, conservation, CPU parity, exact threading/checkpoints/restart')


if __name__=='__main__':
    main()
