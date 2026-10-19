# Stochastic perturbations

The chaos module adds controlled stochastic variability to initial conditions
and selected physical sources. `extensions.chaos.scheme: boundary_layer`
perturbs the PBL's conserved source and matching surface-flux publication; it
does not replace the physical scheme selected by `boundary_layer.scheme`.

Public configuration, field views, extension interfaces and persistent runtime
ownership are separated under `include/chaos`. RNG, correlation filtering and
concrete scheme details remain private here. Seed and member identity are
scientific inputs. Restart must preserve stochastic state and cadence rather
than silently reseeding the process.

Moving-domain noise is tied to its declared Earth/frame convention. Reusing an
array index after relocation must not accidentally reuse an unrelated noise
sample. A source perturbation must preserve the process's conservation and
shared-donor contract; scaling an already formed sedimentation divergence cell
by cell would not preserve its surface flux.

The runtime stages noise epochs and physical source changes before publishing.
Failure restores the preceding accepted generation and clock. Native generation
is required only for profiles that explicitly admit that ownership; a requested
native owner cannot be replaced silently when fallback is prohibited.

Read the moving-grid and coupling chapters in the
[mathematical guide](../../docs/MATHEMATICS.pdf) and the current catalog
before selecting a motion/physics/backend combination. Tests and permanent
fixtures describe the supported reproducibility contract; historical run counts
are not an availability rule.
