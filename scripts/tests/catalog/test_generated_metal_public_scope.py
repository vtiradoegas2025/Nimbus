#!/usr/bin/env python3
"""Check exact public admission for the generated canonical Metal profile."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "tests/fixtures/reference/catalog/generated_metal_canonical_v1.yaml"
NETCDF = ROOT / "tests/fixtures/reference/catalog/generated_metal_netcdf_v1.yaml"
ZFP_CANDIDATE = ROOT / "tests/fixtures/reference/catalog/generated_metal_zfp_candidate_v1.yaml"
GENERATOR = ROOT / "scripts/qualification/generate_metal_scope.py"
OUTPUT_GENERATOR = ROOT / "scripts/qualification/generate_metal_output_scope.py"
OUTPUT_GENERATOR_V2 = ROOT / "scripts/qualification/generate_metal_output_scope_v2.py"
ZFP_TYPED_GENERATOR = ROOT / "scripts/qualification/generate_metal_zfp_typed_scope.py"
ERROR = "qualified Metal execution is limited to exact public"


def validate(binary: Path, config: Path) -> tuple[int, str]:
    result = subprocess.run([str(binary), "--validate-only", "--config", str(config)],
                            cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False, timeout=60)
    return result.returncode, result.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    args = parser.parse_args()
    subprocess.run(["python3", str(GENERATOR), "--verify"], cwd=ROOT, check=True)
    subprocess.run(["python3", str(OUTPUT_GENERATOR), "--verify"], cwd=ROOT, check=True)
    subprocess.run(["python3", str(ZFP_TYPED_GENERATOR), "--verify"], cwd=ROOT, check=True)
    subprocess.run(["python3", str(OUTPUT_GENERATOR_V2), "--verify"], cwd=ROOT, check=True)
    binary = args.binary.resolve()
    for path in (CONFIG, NETCDF):
        code, output = validate(binary, path)
        if code != 0 or "[VALIDATE] PASS:" not in output or "[VALIDATE WARNING]" in output:
            raise RuntimeError("canonical generated Metal profile failed ordinary public validation:\n" + output)
    source = CONFIG.read_text()
    mutations = {
        "spacing": ("  dx: 500\n", "  dx: 501\n"),
        "timestep": ("  dt_large: 0.1\n", "  dt_large: 0.101\n"),
        "fallback": ("    allow_fallback: false\n", "    allow_fallback: true\n"),
        "radiation_cadence": ("  cadence_s: 300\n", "  cadence_s: 301\n"),
    }
    with tempfile.TemporaryDirectory(prefix="nimbus-generated-metal-scope-") as temporary:
        for name, (old, new) in mutations.items():
            if source.count(old) != 1:
                raise RuntimeError(f"{name} mutation anchor is not unique")
            config = Path(temporary) / f"{name}.yaml"
            config.write_text(source.replace(old, new, 1))
            code, output = validate(binary, config)
            if code == 0 or ERROR not in output:
                raise RuntimeError(f"{name} neighbor was not rejected by the exact scope:\n{output}")
        netcdf = NETCDF.read_text()
        for name, setting in (("deflate", "  netcdf_deflate_level: 9\n"),
                              ("shuffle", "  netcdf_shuffle: false\n"),
                              ("chunk", "  netcdf_target_chunk_kib: 2048\n")):
            if netcdf.count("  format: netcdf\n") != 1:
                raise RuntimeError("netCDF format anchor is not unique")
            config = Path(temporary) / f"netcdf-{name}.yaml"
            config.write_text(netcdf.replace("  format: netcdf\n",
                                             "  format: netcdf\n" + setting, 1))
            code, output = validate(binary, config)
            if code == 0 or ERROR not in output:
                raise RuntimeError(f"netCDF {name} neighbor escaped the exact scope:\n{output}")
        zfp = ZFP_CANDIDATE.read_text()
        if zfp.count("    mode: exploratory\n") != 1:
            raise RuntimeError("qualified ZFP negative-scope anchor is not unique")
        config = Path(temporary) / "zfp-qualified.yaml"
        config.write_text(zfp.replace("    mode: exploratory\n", "    mode: qualified\n", 1))
        code, output = validate(binary, config)
        if code != 0 or "[VALIDATE] PASS:" not in output or "[VALIDATE WARNING]" in output:
            raise RuntimeError("exact ZFP candidate failed ordinary qualified Metal admission:\n" + output)
        qualified_zfp = config.read_text()
        zfp_neighbors = {
            "tier": ("  format: zfp_high\n", "  format: zfp_medium\n"),
            "async": ("  async_io: true\n", "  async_io: false\n"),
            "tolerance": ("    u: 1.0e-4\n", "    u: 5.0e-5\n"),
            "keyframe": ("  zfp_keyframe_interval: 4\n", "  zfp_keyframe_interval: 3\n"),
            "retention": ("  keep_last_frames: 3\n", "  keep_last_frames: 4\n"),
            "field_execution": ("  zfp_field_execution: serial\n",
                                "  zfp_field_execution: field_parallel\n"),
            "file_ceiling": ("  maximum_retained_history_files: 256\n",
                             "  maximum_retained_history_files: 512\n"),
        }
        for name, (old, new) in zfp_neighbors.items():
            if qualified_zfp.count(old) != 1:
                raise RuntimeError(f"ZFP {name} exact-scope anchor is not unique")
            neighbor = Path(temporary) / f"zfp-{name}.yaml"
            neighbor.write_text(qualified_zfp.replace(old, new, 1))
            code, output = validate(binary, neighbor)
            if code == 0 or ERROR not in output:
                raise RuntimeError(f"ZFP {name} neighbor escaped the exact Metal scope:\n{output}")
        if zfp.count("  zfp_keyframe_interval: 4\n") != 1:
            raise RuntimeError("ZFP predictive keyframe interval anchor is not unique")
        invalid_predictive = Path(temporary) / "zfp-invalid-predictive.yaml"
        invalid_predictive.write_text(qualified_zfp.replace("  zfp_keyframe_interval: 4\n",
                                                         "  zfp_keyframe_interval: 2\n", 1))
        code, output = validate(binary, invalid_predictive)
        if code == 0 or "predictive delta requires a keyframe interval of at least three" not in output:
            raise RuntimeError("non-operative predictive ZFP selection escaped preflight:\n" + output)
    print("[generated-metal-public-scope] PASS: NPY/netCDF/ZFP exact public admission, fourteen exact-scope rejections, and predictive ZFP preflight rejection")


if __name__ == "__main__":
    main()
