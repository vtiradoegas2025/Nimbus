#!/usr/bin/env python3
"""Generate conservative momentum Metal arithmetic from the canonical GLSL."""
import argparse
from generate_tke_msl import ROOT, translate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = ROOT / "metal/include/compute/metal/shaders/dynamics/conservative_reduction_msl.hpp"
    source = translate("conservative_reduction", "dynamics").replace(
        "generate_tke_msl.py", "generate_conservative_reduction_msl.py"
    ).replace("using vec2 = float2;", "using vec2 = float2;\nusing ivec3 = int3;")
    if args.check:
        if not path.exists() or path.read_text() != source:
            raise SystemExit("conservative reduction Metal source is stale")
    else:
        path.write_text(source)

if __name__ == "__main__":
    main()
