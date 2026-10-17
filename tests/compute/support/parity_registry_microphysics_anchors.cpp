/**
 * @file parity_registry_microphysics_anchors.cpp
 * @brief Compile-time anchors for the microphysics CPU references named in
 *        the parity registry (GPU.md Phase 1, section 2.4).
 *
 * Split from parity_registry.cpp because the dynamics scheme headers there
 * pull in simulation.hpp, whose legacy global-scope physics constants
 * (T0, L_v, L_f, ...) collide with the function-local
 * `using namespace microphysics_constants` directives in thermodynamics.hpp
 * when both header families share one translation unit. (Namespacing those
 * simulation globals is the eventual fix; it is out of scope here.)
 *
 * compute_sedimentation is private on all three schemes, so the public
 * compute_tendencies anchors each class; the sedimentation registry rows
 * name their private kernel explicitly.
 */
#include "microphysics/schemes/kessler/kessler.hpp"
#include "microphysics/schemes/lin/lin.hpp"
#include "microphysics/schemes/thompson/thompson.hpp"

#include <type_traits>

namespace {

/// Same contract as the anchor macro in parity_registry.cpp: a renamed CPU
/// entry point breaks the build here, without ODR-using its definition.
#define NIMBUS_PARITY_ANCHOR(symbol) \
    static_assert(std::is_member_function_pointer_v<decltype(&symbol)> || \
                      std::is_pointer_v<decltype(&symbol)>, \
                  "parity registry anchor: " #symbol)

NIMBUS_PARITY_ANCHOR(KesslerScheme::compute_tendencies);
NIMBUS_PARITY_ANCHOR(ThompsonScheme::compute_tendencies);
NIMBUS_PARITY_ANCHOR(LinScheme::compute_tendencies);

#undef NIMBUS_PARITY_ANCHOR

}  // namespace
