# Performance contract fixtures

These small JSON inputs exercise the release-policy parsers, source bindings,
exact-output requirements and statistical acceptance rules. They retain the
values needed by the mutation tests while the original build products and
campaign directories are archived outside the source tree.

Both threshold policies remain `draft_not_activated`. Historical observation
values and hashes are test inputs, not current performance measurements or
permission to promote release readiness. Tests also create synthetic workloads
and mutate their identities to prove rejection and activation requirements.

If these fixtures move, update their hash-bound references through the
maintained validators. Do not loosen a threshold or turn a rejected observation
into a passing one to make a test green.
