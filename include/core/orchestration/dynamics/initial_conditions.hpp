#pragma once

#include <vector>

struct SimulationStateView;

// Wind is horizontally uniform and follows the bound configuration hodograph.
void apply_cartesian_wind_initialization(const SimulationStateView& state);
void apply_cartesian_wind_initialization();

// Adds the configured spherical Gaussian theta and cloud-water perturbations.
// Moist runs scale qv with saturation mixing ratio so the thermal preserves
// parcel RH before applying the explicit initial condensate anomaly.
void apply_cartesian_bubble_initialization(const SimulationStateView& state);
void apply_cartesian_bubble_initialization();

// Samples the one-dimensional reference atmosphere at each terrain column's
// physical scalar height and diagnoses dry density from the stored-float EOS.
void apply_cartesian_terrain_thermodynamic_initialization(
    const SimulationStateView& state,
    const std::vector<double>& reference_pressure,
    const std::vector<double>& reference_theta,
    const std::vector<double>& reference_qv);
