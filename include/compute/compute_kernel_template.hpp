#pragma once

/**
 * @file compute_kernel_template.hpp
 * @brief Compatibility umbrella for backend dispatch bridges.
 *
 * New code may include the kernel-family header it uses from
 * compute/bridge/. This umbrella preserves the existing public include.
 */

#include "compute/bridge/advection.hpp"
#include "compute/bridge/diffusion.hpp"
#include "compute/bridge/dynamics.hpp"
#include "compute/bridge/microphysics.hpp"
#include "compute/bridge/residency.hpp"
#include "compute/bridge/turbulence.hpp"
#include "compute/bridge/vertical_flux.hpp"
