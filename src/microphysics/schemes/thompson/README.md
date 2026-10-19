# Thompson data and implementation contract

Nimbus pins its pure-C++ Thompson implementation to the WRF v4.7.1 source
identity recorded in the table headers. The activation reference fixture is
`tests/fixtures/reference/thompson/CCN_ACTIVATE.BIN`; its SHA-256 is
`f2b8d3916560f9046f89f8ac5f32c5292a1800498fd75301e422f147c82a3dbd`.
These are scientific inputs, not disposable run outputs.

`make thompson-data` installs the existing canonical inputs from the
[pinned runtime-table release](https://github.com/vtiradoegas2025/Nimbus/releases/tag/runtime-tables-v1):

- `data/thompson/ccn_activation.ntact`
- `data/thompson/collection_v1.ntctbl`
- `data/thompson/cold_process_v1.ntcold`

Startup is load-only. Missing, corrupt or incompatible required data rejects
with the preparation command; it must not silently use a replacement table.
The provisioner verifies the archive and every member against
[`thompson_tables.json`](../../../../scripts/dependencies/thompson_tables.json)
before replacing installed files. The scientific hashes remain those in the
runtime headers. The bundle includes upstream source, its license, the original
activation input and provenance. An offline archive can be supplied using
`python3 scripts/dependencies/fetch_thompson_tables.py --archive PATH`.

`make thompson-generate-data` retains the converter/generators for maintainer
reconstruction. `test-thompson-data-determinism` checks threaded generation on
the selected toolchain. Different compilers and math libraries do not guarantee
identical floating-point bytes, so this is separate from portable provisioning.
Never replace a pinned identity merely to accept a different generator result.
Keep source, license and identities with redistributed tables.

Microphysical mass/number state, process operators and sedimentation have
separate permanent tests. Supported runtime combinations are controlled by the
current compiled catalog and strict admission, including their native/backend
and checkpoint requirements. This guide does not limit Thompson to a historical
CPU-only snapshot or claim universal qualification.

See [the mathematics guide](../../../../docs/MATHEMATICS.pdf) and
[scientific references](../../../../docs/foundationalScience.md) for assumptions
and the distinction between a publication, pinned implementation and completed
coupled validation.
