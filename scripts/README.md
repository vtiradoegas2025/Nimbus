# Repository scripts

The maintained repository scripts are grouped by purpose:

- `benchmarks/` is split into `cpu/`, `gpu/`, `domain/`, and `storage/`.
- `qualification/` is split into `accelerators/`, `checkpointing/`,
  `domain_motion/`, `nested_domains/`, and `output/`.
- `tests/` is split into `architecture/`, `public_api/`, and `gpu/`.
- `dev/` includes `build/`, `configs/`, `docs/`, and `shaders/`.

Import-dependent script families remain together. In particular, the nested
domain reference, runtime, comparison, and evidence scripts intentionally form
one workflow under `qualification/nested_domains/`.

Use `python3 scripts/run.py list` to discover commands. Run a command by group
and stem, for example:

```sh
python3 scripts/run.py benchmark cpu/cpu_scaling -- --help
python3 scripts/run.py qualify output/netcdf_history -- --help
python3 scripts/run.py test static --compiler clang++
python3 scripts/run.py dev shaders/verify_spirv -- --help
```

Unique stems such as `cpu_scaling` continue to work without the folder prefix.

`setup.sh` and `run_with_vulkan_env.sh` remain at the top level because other
automation treats them as stable environment entry points.

`dependencies/fetch_thompson_tables.py` installs checksum-pinned runtime inputs
from the dedicated data release, or from a verified offline archive. Its manifest
must agree with the existing scientific identities in the runtime headers.
`dev/shaders/build_spirv_tools.sh` builds the matching pinned GLSL compiler and
SPIR-V tools used for explicit shader reproducibility checks.

## Configuration reference

The [namelist reference](../docs/NAMELIST.md) records configuration names,
defaults, enum aliases, conditional rules and ordinary admission behavior.
The source, typed configuration and runtime remain authoritative. After a
configuration change, review affected entries with the maintainer, who keeps
the editable inventory and generator locally. `make check-docs` verifies the
reviewed published bytes and pinned configuration-source identities; changing
a hash alone is not a documentation review.

The [mathematics](../docs/MATHEMATICS.pdf),
[architecture](../docs/ARCHITECTURE.pdf) and [Studio](../docs/STUDIO.pdf) manuals
are published PDFs. Their editable sources and PDF build tooling are maintained
locally by the maintainer and are not distributed in the public checkout.

## Volume source inspection

With NumPy installed, extract an unmodified sample line from a saved float32 or
float64 NPY volume:

```sh
python3 scripts/science/audit_volume_line.py /path/to/step_000023/theta.npy \
  --axis-order xyz --start 240 256 150 --axis x --length 64 \
  --output theta-line.json
```

Multiple input files produce one time-history report. Array order is explicit;
the tool checks declared manifest axes, dimensions, publication state and valid
index bounds when present. It memory-maps each source and copies at most 4096
samples. Existing reports are not overwritten. Missing/nonfinite samples remain
gaps, and whole-line difference metrics are omitted for such lines.

Inspect a bounded native plane and its source-space frequency content:

```sh
python3 scripts/science/audit_volume_plane.py /path/to/step_000023/theta.npy \
  --axis-order xyz --start 240 240 150 --axes xy --shape 64 64 \
  --output theta-plane.json
```

Plane axes accept either order of XY, XZ or YZ. Each plane axis contains
8–256 unmodified samples. The report retains original values and fingerprints,
neighbor differences, and raw and affine-detrended Hann spectra. Frequencies
are cycles per saved cell; high-frequency energy is a diagnostic rather than
an artifact classifier or scientific acceptance threshold. Spectra are omitted
when any sample is nonfinite; missing values are not filled or interpolated.

The report records original values, selected-line and manifest fingerprints,
source metadata, range and neighbor differences. These are index-space
descriptions, not physical derivatives or a scientific pass/fail test. A constant
model-level index can cross different physical heights in terrain-following
data. Use these diagnostics to inspect the supplied data; they do not by themselves establish scientific accuracy.
