/**
 * @file factory.cpp
 * @brief Implementation for the chaos module.
 *
 * Provides executable logic for the chaos runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/chaos subsystem.
 */

#include "factory.hpp"
#include "schemes/none/none.hpp"
#include "schemes/initial_conditions/initial_conditions.hpp"
#include "schemes/boundary_layer/boundary_layer.hpp"
#include "schemes/full_stochastic/full_stochastic.hpp"
#include "util/scheme_factory.hpp"

namespace
{
const tmv::SchemeRegistry<chaos::ChaosScheme> registry({
    {"none",               [] { return std::make_unique<chaos::NoneScheme>(); }},
    {"initial_conditions", [] { return std::make_unique<chaos::InitialConditionsScheme>(); }},
    {"boundary_layer",     [] { return std::make_unique<chaos::BoundaryLayerScheme>(); }},
    {"full_stochastic",    [] { return std::make_unique<chaos::FullStochasticScheme>(); }},
}, {
    {"pbl",                              "boundary_layer"},
    {"pbl_perturbation",                 "boundary_layer"},
    {"pbl_perturbations",                "boundary_layer"},
    {"bl_perturbation",                  "boundary_layer"},
    {"bl_perturbations",                 "boundary_layer"},
    {"boundary_layer_perturbation",      "boundary_layer"},
    {"boundary_layer_perturbations",     "boundary_layer"},
    {"ic",                               "initial_conditions"},
    {"full",                             "full_stochastic"},
});
}

namespace chaos
{

std::unique_ptr<ChaosScheme> create_chaos_scheme(const std::string& scheme_name)
{
    return registry.create("chaos", scheme_name);
}

}

/**
 * @brief Creates the chaos scheme.
 */
std::unique_ptr<chaos::ChaosScheme> create_chaos_scheme(const std::string& scheme_name)
{
    return chaos::create_chaos_scheme(scheme_name);
}

/**
 * @brief Gets the available chaos schemes.
 */
std::vector<std::string> get_available_chaos_schemes()
{
    return registry.available_ids();
}
