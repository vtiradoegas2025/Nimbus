#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace tmv::chaos
{

struct ChaosConfig
{
    std::string scheme_id = "none";

    std::uint64_t seed = 42;
    int member_id = 0;
    bool native_generation_required = false;

    // Noise evolution cadence. Fresh physics tendencies are always scaled
    // from their deterministic baseline using the current noise field.
    double dt_chaos = 60.0;
    double tau_t = 21600.0;

    double Lx = 50000.0;
    double Ly = 50000.0;
    // Historical correlation-backend vocabulary retained for checkpoint and
    // internal-config compatibility. Built-in schemes now use the common
    // translation-invariant Earth control lattice and log this mapping.
    std::string filter_id = "recursive_gaussian";

    double xi_max = 2.0;
    std::string taper_id = "none";
    double taper_z1 = 1000.0;
    double taper_z2 = 3000.0;

    std::vector<std::string> apply_to_ic;
    std::vector<std::string> apply_to_tendencies;

    std::unordered_map<std::string, double> sigma_ic;
    std::unordered_map<std::string, double> alpha_tend;

    ChaosConfig()
    {
        sigma_ic["u"] = 0.5;
        sigma_ic["v"] = 0.5;
        sigma_ic["w"] = 0.1;
        sigma_ic["theta"] = 0.5;
        sigma_ic["qv"] = 0.0005;
        sigma_ic["qc"] = 0.0;
        sigma_ic["qr"] = 0.0;

        alpha_tend["microphysics"] = 0.3;
        alpha_tend["pbl"] = 0.2;
        alpha_tend["diffusion"] = 0.1;
    }
};

} // namespace tmv::chaos

using tmv::chaos::ChaosConfig;

#ifndef NIMBUS_CHAOS_NAMESPACE_ALIAS
#define NIMBUS_CHAOS_NAMESPACE_ALIAS
namespace chaos = tmv::chaos;
#endif
