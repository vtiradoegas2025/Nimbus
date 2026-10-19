#!/usr/bin/env python3
"""Generate Metal wrappers around the canonical implicit-diffusion GLSL."""
import argparse
from generate_tke_msl import ROOT, translate


def main():
    parser = argparse.ArgumentParser()
    names = ("implicit_diffusion_stencil", "implicit_diffusion_vector")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--kernel", choices=names)
    args = parser.parse_args()
    for name in ((args.kernel,) if args.kernel else names):
        path = ROOT / f"metal/include/compute/metal/shaders/diffusion/{name}_msl.hpp"
        expected = translate(name, "diffusion")
        if args.check:
            if not path.is_file() or path.read_text() != expected:
                raise RuntimeError(f"generated implicit Metal source drifted: {path}")
        else:
            path.write_text(expected)


if __name__ == "__main__":
    main()
