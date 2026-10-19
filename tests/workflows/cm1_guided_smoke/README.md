# Maintained CM1 guided-smoke workflow

This directory is the small, deterministic fixture for the critical Studio
workflow: inspect a CM1-style case, review its conventional `input_sounding`,
stage a source-bound 30-second Nimbus run, verify durable output/provenance,
and reopen the CM1/Nimbus pair for comparison.

The fixture is intentionally synthetic and carries no physical-validation
claim. `cm1-control.cdl` is an analytic CM1-like netCDF control under CC0-1.0;
it is not CM1 solver output. Generate the disposable `cm1out_000001.nc` with
`ncgen -4`. The sounding and namelist are minimal conventional text inputs.
`nimbus-sounding.txt` is the checked hydrostatic Studio projection of that
exact sounding, and `nimbus-smoke.yaml` is the bounded Nimbus side of the
workflow.

Run the maintained qualification from the repository root:

```sh
make qualify-cm1-guided-smoke
```

The qualifier uses the production `bin/nimbus_sim`, creates all generated data
under a temporary directory, requires exactly 30 seconds with 10-second output,
and rejects missing/partial frames or invalid launch/runtime provenance. It
does not execute CM1 and never writes into this fixture directory.
