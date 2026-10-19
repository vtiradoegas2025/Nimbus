# Generated Metal unserialized-control fixtures

These maintained fixtures replace the four-case test's dependency on historical
report directories. Exactly 75 YAML files used by
`tests/core/test_generated_metal_unserialized_controls.cpp` are promoted:
35 retained group witnesses, four source controls and their 36 control
mutations. The separately maintained `generated_metal_canonical_v1.yaml`
baseline remains unchanged.

All historical source files were verified present before copying. Every promoted
YAML preserves the source bytes, including comments, defaults and invalid
control mutations. No historical report was changed and no additional YAML,
probe output, log, request or generated artifact was promoted. The fixtures
contain no embedded report-directory dependencies.

[index.json](index.json) records each historical source and maintained fixture
path, byte count and SHA-256, plus the C++ source identity before/after its two
fixture-directory replacements. The four test cases, assertions, typed default
guards and scientific/configuration policy are unchanged. Root owns the
maintained Make target and its bounded execution evidence; no build or simulator
execution is implied by this fixture index.

This closes only the report-fixture dependency increment. Capability readiness,
full retained inventory/build closure, required campaigns, external scientific
review and full qualification are separate obligations.
