# Thompson data and implementation contract

Nimbus pins its pure-C++ Thompson implementation to the WRF v4.7.1 source
identity recorded in the table headers. The activation reference fixture is
`tests/fixtures/reference/thompson/CCN_ACTIVATE.BIN`; its SHA-256 is
`f2b8d3916560f9046f89f8ac5f32c5292a1800498fd75301e422f147c82a3dbd`.
These are scientific inputs, not disposable run outputs.

`make thompson-data` performs offline preparation of:

- `data/thompson/ccn_activation.ntact`
- `data/thompson/collection_v1.ntctbl`
- `data/thompson/cold_process_v1.ntcold`

Startup is load-only. Missing, corrupt or incompatible required data rejects
with the preparation command; it must not silently use a replacement table.
The converter/generators and `test-thompson-data-determinism` check the portable
representation and threaded generation. Keep the source reference, license and
hash identities with any redistributed tables.

Microphysical mass/number state, process operators and sedimentation have
separate permanent tests. Supported runtime combinations are controlled by the
current compiled catalog and strict admission, including their native/backend
and checkpoint requirements. This guide does not limit Thompson to a historical
CPU-only snapshot or claim universal qualification.

See [the mathematics guide](../../../../docs/MATHEMATICS.pdf) and
[scientific references](../../../../docs/foundationalScience.md) for assumptions
and the distinction between a publication, pinned implementation and completed
coupled validation.
