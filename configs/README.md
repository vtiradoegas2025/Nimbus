# Choosing and running a configuration

The public catalog is organized by horizontal resolution, microphysics and
physics profile. Each YAML file is a complete starting configuration. These
presets declare exploratory execution and validation settings; they are not
blanket scientific qualification claims. The executable catalog and ordinary
runtime admission determine which stricter combinations are supported.

The [complete namelist reference](../docs/NAMELIST.md) lists the strict-v1 YAML
settings, defaults, units, aliases and restrictions, including unfinished and
guarded capabilities. Nimbus uses YAML rather than a CM1 Fortran namelist;
matching terminology does not imply identical physics or defaults.

## Resolution and cost

| Folder | Horizontal grid | Approximate horizontal domain | Grid cells (`nx × ny × nz`) |
|---|---:|---:|---:|
| `1000meterResolution/` | 1000 m | 40.0 km square | `40 × 40 × 64` |
| `500meterResolution/` | 500 m | 40.0 km square | `80 × 80 × 64` |
| `450meterResolution/` | 450 m | 40.5 km square | `90 × 90 × 64` |
| `400meterResolution/` | 400 m | 40.0 km square | `100 × 100 × 64` |
| `350meterResolution/` | 350 m | 40.6 km square | `116 × 116 × 64` |
| `300meterResolution/` | 300 m | 40.2 km square | `134 × 134 × 64` |
| `250meterResolution/` | 250 m | 40.0 km square | `160 × 160 × 64` |
| `200meterResolution/` | 200 m | 40.0 km square | `200 × 200 × 64` |
| `150meterResolution/` | 150 m | 40.2 km square | `268 × 268 × 64` |
| `100meterResolution/` | 100 m | 40.0 km square | `400 × 400 × 64` |
| `50meterResolution/` | 50 m | 40.0 km square | `800 × 800 × 64` |


Start with a coarse grid and a short run. Halving horizontal spacing increases
horizontal cell count by about four, and the timestep may also need to decrease.
The memory and runtime increase is therefore larger than the change in the
folder name suggests. Terrain, nested grids, output cadence, selected physics
and backend memory policy add their own costs.

## Physics choices

Each resolution has `dry`, `kessler`, `lin`, `thompson` and `milbrandt` folders.
The corresponding microphysics selections are none, warm-rain Kessler, mixed-phase
Lin, Nimbus's Thompson implementation, and ordinary double-moment Milbrandt–Yau.
Thompson needs the [runtime tables](../docs/BUILDING.md#provision-scientific-inputs). Public triple-moment
Milbrandt remains guarded; it is not enabled by choosing the Milbrandt folder.

| Profile | Surface/PBL | Turbulence | Radiation |
| --- | --- | --- | --- |
| `baseline.yaml` | Slab surface and slab PBL | Smagorinsky 2-D | None |
| `radiation.yaml` | Slab surface and slab PBL | Smagorinsky 2-D | Simple grey |
| `ysu.yaml` | Slab surface and YSU | Smagorinsky 2-D | None |
| `ysu_radiation.yaml` | Slab surface and YSU | Smagorinsky 2-D | Simple grey |
| `les.yaml` | No surface/PBL scheme | Smagorinsky 3-D | None |
| `les_radiation.yaml` | No surface/PBL scheme | Smagorinsky 3-D | Simple grey |
| `tke.yaml` | No surface/PBL scheme | Prognostic TKE 3-D | None |
| `tke_radiation.yaml` | No surface/PBL scheme | Prognostic TKE 3-D | Simple grey |

A more elaborate scheme is not automatically more accurate for an arbitrary
grid or application. The [mathematical guide](../docs/MATHEMATICS.pdf) explains
what the closures assume and how their sources interact.

## Validate before running

From the repository root:

```sh
./bin/nimbus_sim --validate-only --config=configs/1000meterResolution/kessler/baseline.yaml
./bin/nimbus_sim --headless --config=configs/1000meterResolution/kessler/baseline.yaml --duration=10 --outdir=runs/quickstart
```

Read warnings as well as the exit status. Validation checks configuration and
admission, not the accuracy of a future storm. The actual simulator selected in
Studio or on a remote machine may have different compiled capabilities. Use
its `--version` and `--scheme-catalog=json` outputs.

Studio discovers public configuration files under `configs/`. Save personal
experiments in a project directory or the ignored `configs/custom/` folder.
Keep relative sounding paths consistent with the working directory, and retain
the exact source configuration with the output. The
[sounding examples](../tests/fixtures/soundings/cm1) use the CM1 input format.

## Maintaining the catalog

Some serialized headers retain historical schema-reference text because exact
configuration identities include those bytes. The
[typed configuration](../include/init/config/config_v1.hpp) and executable
catalog define the current schema and admitted behavior.

The repeated resolution/profile files are generated from one maintained
recipe. Edit that recipe when changing the common catalog rather than allowing
individual copies to drift:

```sh
python3 scripts/dev/configs/generate_resolution_catalog.py --check
make check-public-configs
make test-public-config-admission
```

Run the generator without `--check` only when intentionally updating its
managed presets. It does not own personal experiments. Numerical/admission
fixtures live under [tests/fixtures/configs](../tests/fixtures/configs), where
special boundary, restart and failure cases are kept separate from public
starting configurations.
