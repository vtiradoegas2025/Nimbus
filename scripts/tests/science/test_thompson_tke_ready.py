#!/usr/bin/env python3
"""Run the maintained Thompson/TKE family, including implicit diffusion."""
from test_tke_family_ready import main as run_family


def main():
    run_family(thompson=True)


if __name__ == "__main__":
    main()
