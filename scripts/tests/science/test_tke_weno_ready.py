#!/usr/bin/env python3
"""Run the native WENO/TKE family without microphysics."""
import sys
from test_tke_family_ready import main

if __name__ == "__main__":
    sys.argv.append("--weno")
    main()
