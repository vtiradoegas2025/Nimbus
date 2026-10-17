# Domain-history reference inputs

These compact records are inputs to the maintained tracking, reference-frontier
and evidence-parser tests. They preserve historical values and provenance needed
by those contracts. The full run directories, fields and campaign artifacts are
archived outside the repository.

They are not current qualification receipts. A new physical campaign must
produce and validate its own artifacts. Maintained tools write fresh evidence
to generated output locations or an explicit caller-supplied path, rather than
overwriting these references.

Reviewed frontier policies retain their original bytes, including historical
paths, because subsequent amendments bind their exact SHA-256 identities. The
[frontier comparator](../../../../scripts/qualification/nested_domains/compare_moving_nested_supercell_reference.py)
resolves the retired evidence and configuration namespaces to maintained test
fixtures before verifying the unchanged hash. It does not search by
basename, accept parent traversal, or permit an alias to escape its fixture
directory. These location aliases do not renew a review or qualification claim.

Two compact Vulkan report inputs are also retained byte-for-byte: the 2026-08-19
lifetime-chain report and the reference-runtime report bound by the v13 policy.
Their original `tests/artifacts/` names have individual exact-path aliases;
there is no general artifact-directory or basename search. Complete field/run
outputs remain external archival material.
