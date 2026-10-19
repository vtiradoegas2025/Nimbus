#pragma once

struct SimulationStateView;

#include "radar/radar_base.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace tmv
{

// Owns the selected production diagnostic and reusable product operators.
struct RadarRuntimeState
{
    static std::unique_ptr<RadarSchemeBase> clone_operator(
        const std::unique_ptr<RadarSchemeBase>& source)
    {
        if (!source)
        {
            return nullptr;
        }
        std::unique_ptr<RadarSchemeBase> copy = source->clone();
        if (!copy)
        {
            throw std::runtime_error(
                "radar operator does not support runtime snapshots: "
                + source->name());
        }
        return copy;
    }

    std::unique_ptr<RadarSchemeBase> selected_scheme;
    RadarConfig selected_config;
    RadarProductSelection product_selection;
    RadarOut selected_output;
    RadarOut polarimetric_output;

    std::unique_ptr<RadarSchemeBase> reflectivity_operator;
    std::unique_ptr<RadarSchemeBase> velocity_operator;
    std::unique_ptr<RadarSchemeBase> zdr_operator;

    int nr = -1;
    int nth = -1;
    int nz = -1;
    double radar_x = 0.0;
    double radar_y = 0.0;
    double radar_z = 0.0;

    RadarRuntimeState() = default;
    RadarRuntimeState(const RadarRuntimeState& other)
        : selected_config(other.selected_config),
          product_selection(other.product_selection),
          selected_output(other.selected_output),
          polarimetric_output(other.polarimetric_output),
          nr(other.nr), nth(other.nth), nz(other.nz),
          radar_x(other.radar_x), radar_y(other.radar_y), radar_z(other.radar_z)
    {
        selected_scheme = clone_operator(other.selected_scheme);
        reflectivity_operator = clone_operator(other.reflectivity_operator);
        velocity_operator = clone_operator(other.velocity_operator);
        zdr_operator = clone_operator(other.zdr_operator);
    }
    RadarRuntimeState& operator=(const RadarRuntimeState& other)
    {
        if (this != &other)
        {
            RadarRuntimeState copy(other);
            *this = std::move(copy);
        }
        return *this;
    }
    RadarRuntimeState(RadarRuntimeState&&) noexcept = default;
    RadarRuntimeState& operator=(RadarRuntimeState&&) noexcept = default;

    void reset() noexcept;
};

extern RadarRuntimeState& global_radar_runtime_state;
}

// Compatibility binding for the production-selected diagnostic operator.
extern std::unique_ptr<tmv::RadarSchemeBase>& radar_scheme;
extern RadarProductSelection global_radar_product_selection;
void initialize_radar(const std::string& scheme_name);
void initialize_radar(const SimulationStateView& execution_state, const std::string& scheme_name);
void initialize_radar_products(const RadarProductSelection& selection);
void initialize_radar_products(const SimulationStateView& execution_state, const RadarProductSelection& selection);
void calculate_radar_observables();
void calculate_radar_observables(const SimulationStateView& execution_state);
void calculate_radar_reflectivity();
void calculate_radar_reflectivity(const SimulationStateView& execution_state);

using RadarRuntimeState = tmv::RadarRuntimeState;
