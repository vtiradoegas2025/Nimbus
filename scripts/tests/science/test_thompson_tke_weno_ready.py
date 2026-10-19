#!/usr/bin/env python3
"""Run the maintained native WENO/Thompson/TKE restart family."""
import sys
from test_thompson_tke_ready import main

if __name__ == "__main__":
    sys.argv.append("--weno")
    main()
