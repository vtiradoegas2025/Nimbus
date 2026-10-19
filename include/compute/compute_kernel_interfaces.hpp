#pragma once

/**
 * @file compute_kernel_interfaces.hpp
 * @brief Compatibility umbrella for backend kernel dispatch interfaces.
 *
 * Family-specific users may include headers from compute/dispatch directly.
 * Existing callers can continue including this header without source changes.
 */

#include "compute/dispatch/common.hpp"
#include "compute/dispatch/domain_recycling.hpp"
#include "compute/dispatch/advection.hpp"
#include "compute/dispatch/dynamics.hpp"
#include "compute/dispatch/microphysics.hpp"
#include "compute/dispatch/turbulence.hpp"
#include "compute/dispatch/diffusion.hpp"
#include "compute/dispatch/boundary_layer.hpp"
#include "compute/dispatch/radiation.hpp"
#include "compute/dispatch/chaos.hpp"
#include "compute/dispatch/terrain.hpp"
