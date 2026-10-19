#include "radar/radar.hpp"
#include "radar/factory.hpp"
#include "radar/runtime.hpp"
#include <algorithm>
#include <cmath>

namespace
{

constexpr float kRadarLinearMin = 0.0f;
constexpr float kRadarLinearMax = 1.0e12f;
constexpr float kRadarZdrMin = -12.0f;
constexpr float kRadarZdrMax = 12.0f;

}


namespace tmv 
{

namespace
{

RadarConfig product_config(const char* scheme_id, const RadarStateView& state)
{
    RadarConfig config;
    config.scheme_id = scheme_id;
    config.has_qr = state.qr != nullptr;
    config.has_qs = state.qs != nullptr;
    config.has_qg = state.qg != nullptr;
    config.has_qh = state.qh != nullptr;
    config.has_qi = state.qi != nullptr;
    config.has_Nr = state.Nr != nullptr;
    config.has_Ns = state.Ns != nullptr;
    config.has_Ng = state.Ng != nullptr;
    config.has_Nh = state.Nh != nullptr;
    config.has_Ni = state.Ni != nullptr;
    return config;
}

void prepare_product_runtime(RadarRuntimeState& runtime, const RadarStateView& state)
{
    if (runtime.nr == state.NR && runtime.nth == state.NTH && runtime.nz == state.NZ)
    {
        return;
    }

    runtime.reflectivity_operator.reset();
    runtime.velocity_operator.reset();
    runtime.zdr_operator.reset();
    runtime.nr = state.NR;
    runtime.nth = state.NTH;
    runtime.nz = state.NZ;
}

RadarSchemeBase& prepare_operator(std::unique_ptr<RadarSchemeBase>& scheme,
                                  const RadarConfig& config,
                                  const RadarStateView& state)
{
    if (!scheme)
    {
        scheme = create_radar_scheme(config.scheme_id);
        scheme->initialize(config, state.NR, state.NTH, state.NZ);
    }
    return *scheme;
}

} // namespace

void RadarSystem::compute_all_observables(RadarRuntimeState& runtime,
                                          const RadarStateView& state,
                                          RadarOut& output,
                                          double radar_x,
                                          double radar_y,
                                          double radar_z)
{
    RadarProductSelection selection;
    selection.reflectivity = true;
    selection.radial_velocity = true;
    selection.zdr = true;
    selection.radar_x = radar_x;
    selection.radar_y = radar_y;
    selection.radar_z = radar_z;
    selection.apply_scatterer_correction =
        state.qr != nullptr || state.qs != nullptr || state.qg != nullptr ||
        state.qh != nullptr || state.qi != nullptr;
    compute_observables(runtime, selection, state, output);
}

void RadarSystem::compute_observables(RadarRuntimeState& runtime,
                                      const RadarProductSelection& selection,
                                      const RadarStateView& state,
                                      RadarOut& output)
{
    if (!selection.reflectivity && !selection.radial_velocity && !selection.zdr &&
        !selection.kdp && !selection.rhohv)
    {
        throw std::invalid_argument("radar requires at least one selected product");
    }
    prepare_product_runtime(runtime, state);

    RadarConfig reflectivity_config = product_config(RadarSchemes::REFLECTIVITY, state);
    reflectivity_config.operator_tier = selection.reflectivity_tier;

    RadarConfig velocity_config = product_config(RadarSchemes::VELOCITY, state);
    velocity_config.radar_x = selection.radar_x;
    velocity_config.radar_y = selection.radar_y;
    velocity_config.radar_z = selection.radar_z;
    velocity_config.sampling = selection.velocity_sampling;
    velocity_config.apply_scatterer_correction =
        selection.apply_scatterer_correction;

    RadarConfig zdr_config = product_config(RadarSchemes::ZDR, state);
    zdr_config.operator_tier = selection.zdr_tier;

    RadarOut candidate;
    if (selection.reflectivity)
    {
        prepare_operator(runtime.reflectivity_operator, reflectivity_config, state)
            .compute(reflectivity_config, state, candidate);
    }
    if (selection.radial_velocity)
    {
        const bool velocity_was_initialized = runtime.velocity_operator != nullptr;
        const bool radar_location_changed =
            runtime.radar_x != selection.radar_x ||
            runtime.radar_y != selection.radar_y ||
            runtime.radar_z != selection.radar_z;
        RadarSchemeBase& velocity = prepare_operator(
            runtime.velocity_operator, velocity_config, state);
        if (velocity_was_initialized && radar_location_changed)
        {
            velocity.initialize(velocity_config, state.NR, state.NTH, state.NZ);
        }
        velocity.compute(velocity_config, state, candidate);
    }
    if (selection.zdr || selection.kdp || selection.rhohv)
    {
        prepare_operator(runtime.zdr_operator, zdr_config, state)
            .compute(zdr_config, state, runtime.polarimetric_output);
        candidate.NR = runtime.polarimetric_output.NR;
        candidate.NTH = runtime.polarimetric_output.NTH;
        candidate.NZ = runtime.polarimetric_output.NZ;
        candidate.ZH_dBZ = runtime.polarimetric_output.ZH_dBZ;
        candidate.ZV_dBZ = runtime.polarimetric_output.ZV_dBZ;
        candidate.ZDR_dB = runtime.polarimetric_output.ZDR_dB;
        candidate.KDP_deg_km = runtime.polarimetric_output.KDP_deg_km;
        candidate.rhohv = runtime.polarimetric_output.rhohv;
    }

    output = std::move(candidate);
    runtime.product_selection = selection;
    runtime.radar_x = selection.radar_x;
    runtime.radar_y = selection.radar_y;
    runtime.radar_z = selection.radar_z;
}

void RadarSystem::compute_observables(const RadarProductSelection& selection,
                                      const RadarStateView& state,
                                      RadarOut& output)
{
    compute_observables(global_radar_runtime_state, selection, state, output);
}

void RadarSystem::compute_all_observables(const RadarStateView& state,
                                          RadarOut& output,
                                          double radar_x,
                                          double radar_y,
                                          double radar_z)
{
    compute_all_observables(
        global_radar_runtime_state, state, output, radar_x, radar_y, radar_z);
}

void RadarSystem::compute_reflectivity(RadarRuntimeState& runtime,
                                       const RadarStateView& state,
                                       RadarOut& output)
{
    prepare_product_runtime(runtime, state);
    RadarConfig config = product_config(RadarSchemes::REFLECTIVITY, state);
    config.operator_tier = RadarTiers::Reflectivity::FAST_DA;
    prepare_operator(runtime.reflectivity_operator, config, state).compute(config, state, output);
    validate_radar_output_field(output.Ze_linear, output.NR, output.NTH,
        output.NZ, kRadarLinearMin, kRadarLinearMax, "Ze_linear");
}

void RadarSystem::compute_reflectivity(const RadarStateView& state, RadarOut& output)
{
    compute_reflectivity(global_radar_runtime_state, state, output);
}

void RadarSystem::compute_velocity(RadarRuntimeState& runtime,
                                   const RadarStateView& state,
                                   RadarOut& output,
                                   double radar_x,
                                   double radar_y,
                                   double radar_z)
{
    prepare_product_runtime(runtime, state);
    RadarConfig config = product_config(RadarSchemes::VELOCITY, state);
    config.radar_x = radar_x;
    config.radar_y = radar_y;
    config.radar_z = radar_z;
    config.apply_scatterer_correction =
        config.has_qr || config.has_qs || config.has_qg || config.has_qh || config.has_qi;

    const bool velocity_was_initialized = runtime.velocity_operator != nullptr;
    const bool radar_location_changed =
        runtime.radar_x != radar_x || runtime.radar_y != radar_y || runtime.radar_z != radar_z;
    RadarSchemeBase& velocity = prepare_operator(runtime.velocity_operator, config, state);
    if (velocity_was_initialized && radar_location_changed)
    {
        velocity.initialize(config, state.NR, state.NTH, state.NZ);
    }
    velocity.compute(config, state, output);
    validate_radar_output_field(output.Vr, output.NR, output.NTH, output.NZ,
        -RadarConstants::velocity_abs_max_ms,
        RadarConstants::velocity_abs_max_ms, "Vr");
    runtime.radar_x = radar_x;
    runtime.radar_y = radar_y;
    runtime.radar_z = radar_z;
}

void RadarSystem::compute_velocity(const RadarStateView& state,
                                   RadarOut& output,
                                   double radar_x,
                                   double radar_y,
                                   double radar_z)
{
    compute_velocity(global_radar_runtime_state, state, output, radar_x, radar_y, radar_z);
}

void RadarSystem::compute_zdr(RadarRuntimeState& runtime,
                              const RadarStateView& state,
                              RadarOut& output)
{
    prepare_product_runtime(runtime, state);
    RadarConfig config = product_config(RadarSchemes::ZDR, state);
    config.operator_tier = RadarTiers::ZDR::POLARIMETRIC_FO;
    prepare_operator(runtime.zdr_operator, config, state).compute(config, state, output);
    validate_radar_output_field(output.ZDR_dB, output.NR, output.NTH,
        output.NZ, kRadarZdrMin, kRadarZdrMax, "ZDR_dB");
}

void RadarSystem::compute_zdr(const RadarStateView& state, RadarOut& output)
{
    compute_zdr(global_radar_runtime_state, state, output);
}

}  // namespace tmv
