# Scientific test drivers

These reusable drivers exercise grid, acoustic, boundary, hydrostatic and coupled
physics contracts. They are maintained test code; generated runs and evidence
belong outside this directory. Use `python3 tests/science/run.py list` to inspect
available checks and each driver's `--help` or the corresponding Make target to
identify inputs, duration and hardware requirements.

Bounded parser tests are part of `make test-qualification-tools`. Longer drivers
must be selected deliberately with the required profile admitted. A passing
bounded test is not completion of a long campaign or independent science review.
