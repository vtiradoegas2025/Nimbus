#!/usr/bin/env python3
"""Share the canonical FP32 WENO equations with the fused Metal transport kernel."""
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'vulkan/shaders/compute/advection/advect_conservative_scalar_flux_cartesian_cgrid.comp'
DESTINATION = ROOT / 'metal/include/compute/metal/shaders/advection/weno_common_msl.hpp'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    source = SOURCE.read_text()
    functions = source[source.index('float weno_difference('):source.index('float minmod3(')].strip()
    functions = functions.replace('float weno_difference(', 'inline float weno_difference(')
    functions = functions.replace('float weno5_face_offset(', 'inline float weno5_offset(')
    functions = re.sub(r'(?<![\w.])(\d*\.\d+(?:e[+-]?\d+)?|\d+e[+-]?\d+)(?![\w.])', r'\1f', functions)
    expected = '#pragma once\n\n// Generated from the canonical scalar-flux shader by generate_weno_common_msl.py.\n'
    expected += '#define NIMBUS_WENO_COMMON_MSL ' + chr(92) + '\n'
    expected += (' ' + chr(92) + '\n').join(json.dumps(line + '\n') for line in functions.splitlines()) + '\n' 
    if args.check:
        if not DESTINATION.is_file() or DESTINATION.read_text() != expected:
            raise RuntimeError('fused Metal WENO equations differ from the canonical shader')
    else:
        DESTINATION.write_text(expected)


if __name__ == '__main__':
    main()
