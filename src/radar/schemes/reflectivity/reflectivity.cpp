/**
 * @file reflectivity.cpp
 * @brief Implementation for the radar module.
 *
 * Provides executable logic for the radar runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/radar subsystem.
 */

#include "reflectivity.hpp"
#include "util/log.hpp"

#include <cmath>
#include <algorithm>
#include <cstddef>

namespace
{
constexpr float kZeMaxLinear = 1.0e12f;
constexpr float kZeMaxDbz = 120.0f;

// Single-moment Marshall-Palmer reflectivity (science audit): for an
// exponential PSD, Z[mm^6/m^3] = A_x * (rho_d*q)^1.75 with
// A_x = 1e18 * 720 * N0^(-3/4) * (pi*rho_x)^(-7/4), using the LFO83 N0/rho_x
// the fall speeds use (N0 rain 8e6, snow 3e6, graupel 4e5, hail 4e4 m^-4;
// rho_x 1000/100/400/913 kg/m^3) and the Smith-1984 ice-dielectric factor
// |K_ice|^2/|K_water|^2 = 0.176/0.93 on the frozen species. This replaces the
// old q^2 * (rho_x/1000)^2 heuristic - the rain Rayleigh coefficient (3.63e9)
// mis-applied to every species with the wrong exponent, which buried frozen
// species ~15-20 dBZ low. Reference dBZ at q = 1 g/kg_dry, rho_d = 1.2:
// rain 44.5, snow 58.0, graupel 54.0, hail 55.2.
constexpr double kZeCoeffRain    = 3.63080e9;
constexpr double kZeCoeffSnow    = 8.06323e10;
constexpr double kZeCoeffGraupel = 3.22998e10;
constexpr double kZeCoeffHail    = 4.28530e10;
constexpr double kZeReflExponent = 1.75;

// Local dry-air density (kg_dry/m^3) for converting a dry-air mixing ratio to a mass
// concentration (mass = rho_d * q). Uses the prognostic dry density supplied by
// the radar state. The shared preflight requires this field to be present and
// physical. A fixed density biased dBZ
// high aloft: at rho ~ 0.4 vs the old 1.2 the linear Z ran ~(1.2/0.4)^1.75
// ~ 7x high (~ +8 dB).
double local_dry_air_density(const Field3D* rho_d, int i, int j, int k)
{
    return static_cast<double>((*rho_d)[i][j][k]);
}

void ensure_field_shape(Field3D& field, int nr, int nth, int nz, float init_value = 0.0f)
{
    if (field.size_r() != nr || field.size_th() != nth || field.size_z() != nz)
    {
        field.resize(nr, nth, nz, init_value);
        return;
    }
    field.fill(init_value);
}

void prepare_reflectivity_output(RadarOut& out, int nr, int nth, int nz)
{
    out.NR = nr;
    out.NTH = nth;
    out.NZ = nz;

    ensure_field_shape(out.Ze_linear, nr, nth, nz, 0.0f);
    ensure_field_shape(out.Ze_rain, nr, nth, nz, 0.0f);
    ensure_field_shape(out.Ze_snow, nr, nth, nz, 0.0f);
    ensure_field_shape(out.Ze_graupel, nr, nth, nz, 0.0f);
    ensure_field_shape(out.Ze_hail, nr, nth, nz, 0.0f);
    ensure_field_shape(out.Ze_ice, nr, nth, nz, 0.0f);
    ensure_field_shape(out.Z_dBZ, nr, nth, nz, 0.0f);
    ensure_field_shape(out.ZH_dBZ, nr, nth, nz, 0.0f);
    ensure_field_shape(out.ZV_dBZ, nr, nth, nz, 0.0f);
}

}

namespace tmv
{

/**
 * @brief Initializes the reflectivity scheme.
 */

const ReflectivityScheme::HydrometeorProps ReflectivityScheme::rain_props = {
    1000.0,
    842.0,
    0.8,
    0.0
};

/**
 * @brief Initializes the snow properties.
 */
const ReflectivityScheme::HydrometeorProps ReflectivityScheme::snow_props = {
    100.0,
    2.0e6,
    0.25,
    0.0
};

/**
 * @brief Initializes the graupel properties.
 */
const ReflectivityScheme::HydrometeorProps ReflectivityScheme::graupel_props = {
    400.0,
    4.0e5,
    0.5,
    0.0
};

/**
 * @brief Initializes the hail properties.
 */
const ReflectivityScheme::HydrometeorProps ReflectivityScheme::hail_props = {
    900.0,
    2.0e4,
    0.5,
    0.0
};

/**
 * @brief Initializes the ice properties.
 */
const ReflectivityScheme::HydrometeorProps ReflectivityScheme::ice_props = {
    917.0,
    2.0e6,
    0.25,
    0.0
};

/**
 * @brief Initializes the reflectivity scheme.
 */
void ReflectivityScheme::initialize(const RadarConfig& config, int NR, int NTH, int NZ) 
{
    NR_ = NR;
    NTH_ = NTH;
    NZ_ = NZ;

    tmv::log_info("Initialized reflectivity radar scheme (", NR, "x", NTH, "x", NZ, ")");
    tmv::log_info("Operator tier: ", config.operator_tier);
}

void ReflectivityScheme::invalidate_generation_state_after_domain_recycling() noexcept
{
    candidate_output_.invalidate_generation_values();
    rho_ = nullptr;
}

/**
 * @brief Computes the reflectivity.
 */
void ReflectivityScheme::compute(const RadarConfig& config, const RadarStateView& state, RadarOut& out) 
{
    validate_radar_dimensions(state, NR_, NTH_, NZ_, "reflectivity");
    if (config.operator_tier != "fast_da"
        && config.operator_tier != "psd_moment")
        throw std::invalid_argument(
            "Unknown reflectivity operator tier: " + config.operator_tier);
    validate_radar_hydrometeor_state(
        config, state, "reflectivity", /*require_density=*/true);
    prepare_reflectivity_output(candidate_output_, NR_, NTH_, NZ_);

    // Cache the local dry-air density field for this evaluation (mass = rho_d*q);
    // null -> the representative-density fallback in local_dry_air_density().
    rho_ = state.rho;

    if (config.operator_tier == "fast_da")
    {
        compute_fast_da(config, state, candidate_output_);
    }
    else if (config.operator_tier == "psd_moment")
    {
        compute_psd_moment(config, state, candidate_output_);
    }
    rho_ = nullptr;
    validate_radar_output_field(candidate_output_.Ze_linear,
        NR_, NTH_, NZ_, 0.0f, kZeMaxLinear, "Ze_linear");
    validate_radar_output_field(candidate_output_.Ze_rain,
        NR_, NTH_, NZ_, 0.0f, kZeMaxLinear, "Ze_rain");
    validate_radar_output_field(candidate_output_.Ze_snow,
        NR_, NTH_, NZ_, 0.0f, kZeMaxLinear, "Ze_snow");
    validate_radar_output_field(candidate_output_.Ze_graupel,
        NR_, NTH_, NZ_, 0.0f, kZeMaxLinear, "Ze_graupel");
    validate_radar_output_field(candidate_output_.Ze_hail,
        NR_, NTH_, NZ_, 0.0f, kZeMaxLinear, "Ze_hail");
    validate_radar_output_field(candidate_output_.Ze_ice,
        NR_, NTH_, NZ_, 0.0f, kZeMaxLinear, "Ze_ice");
    validate_radar_output_field(candidate_output_.Z_dBZ,
        NR_, NTH_, NZ_, -40.0f, kZeMaxDbz, "Z_dBZ");
    validate_radar_output_field(candidate_output_.ZH_dBZ,
        NR_, NTH_, NZ_, -40.0f, kZeMaxDbz, "ZH_dBZ");
    validate_radar_output_field(candidate_output_.ZV_dBZ,
        NR_, NTH_, NZ_, -40.0f, kZeMaxDbz, "ZV_dBZ");
    out.NR = NR_;
    out.NTH = NTH_;
    out.NZ = NZ_;
    std::swap(out.Ze_linear, candidate_output_.Ze_linear);
    std::swap(out.Ze_rain, candidate_output_.Ze_rain);
    std::swap(out.Ze_snow, candidate_output_.Ze_snow);
    std::swap(out.Ze_graupel, candidate_output_.Ze_graupel);
    std::swap(out.Ze_hail, candidate_output_.Ze_hail);
    std::swap(out.Ze_ice, candidate_output_.Ze_ice);
    std::swap(out.Z_dBZ, candidate_output_.Z_dBZ);
    std::swap(out.ZH_dBZ, candidate_output_.ZH_dBZ);
    std::swap(out.ZV_dBZ, candidate_output_.ZV_dBZ);
}

/**
 * @brief Computes the fast DA reflectivity.
 */
void ReflectivityScheme::compute_fast_da(const RadarConfig& config, const RadarStateView& state, RadarOut& out) {

    if (state.qr && config.has_qr) 
    {
        const auto* Nr_ptr = config.has_Nr ? state.Nr : nullptr;
        compute_species_reflectivity(*state.qr, Nr_ptr, rain_props, out.Ze_rain);
    }

    if (state.qs && config.has_qs) 
    {
        const auto* Ns_ptr = config.has_Ns ? state.Ns : nullptr;
        compute_species_reflectivity(*state.qs, Ns_ptr, snow_props, out.Ze_snow);
    }

    if (state.qg && config.has_qg) 
    {
        const auto* Ng_ptr = config.has_Ng ? state.Ng : nullptr;
        compute_species_reflectivity(*state.qg, Ng_ptr, graupel_props, out.Ze_graupel);
    }

    if (state.qh && config.has_qh) 
    {
        const auto* Nh_ptr = config.has_Nh ? state.Nh : nullptr;
        compute_species_reflectivity(*state.qh, Nh_ptr, hail_props, out.Ze_hail);
    }

    if (state.qi && config.has_qi) 
    {
        const auto* Ni_ptr = config.has_Ni ? state.Ni : nullptr;
        compute_species_reflectivity(*state.qi, Ni_ptr, ice_props, out.Ze_ice);
    }

    #pragma omp parallel for collapse(2) if(static_cast<std::size_t>(NR_) * NTH_ * NZ_ >= 32768)
    for (int i = 0; i < NR_; ++i) 
    {
        for (int j = 0; j < NTH_; ++j) 
        {
            for (int k = 0; k < NZ_; ++k) 
            {
                float Ze_total = out.Ze_rain[i][j][k] + out.Ze_snow[i][j][k] +
                                out.Ze_graupel[i][j][k] + out.Ze_hail[i][j][k] +
                                out.Ze_ice[i][j][k];
                out.Ze_linear[i][j][k] = Ze_total;
                out.Z_dBZ[i][j][k] = linear_to_dbz(Ze_total);

                out.ZH_dBZ[i][j][k] = out.Z_dBZ[i][j][k];
                out.ZV_dBZ[i][j][k] = out.Z_dBZ[i][j][k];
            }
        }
    }
}

/**
 * @brief Computes the PSD moment reflectivity.
 */
void ReflectivityScheme::compute_psd_moment(const RadarConfig& config, const RadarStateView& state, RadarOut& out) 
{

    if (state.qr && config.has_qr && config.has_Nr) 
    {
        const auto* Nr_ptr = state.Nr;
        compute_moment_reflectivity(*state.qr, Nr_ptr, rain_props, out.Ze_rain, "rain");
    }
 
    if (state.qs && config.has_qs && config.has_Ns) 
    {
        const auto* Ns_ptr = state.Ns;
        compute_moment_reflectivity(*state.qs, Ns_ptr, snow_props, out.Ze_snow, "snow");
    }

    if (state.qg && config.has_qg && config.has_Ng) 
    {
        const auto* Ng_ptr = state.Ng;
        compute_moment_reflectivity(*state.qg, Ng_ptr, graupel_props, out.Ze_graupel, "graupel");
    }

    if (state.qh && config.has_qh && config.has_Nh) 
    {
        const auto* Nh_ptr = state.Nh;
        compute_moment_reflectivity(*state.qh, Nh_ptr, hail_props, out.Ze_hail, "hail");
    }

    if (state.qi && config.has_qi && config.has_Ni) 
    {
        const auto* Ni_ptr = state.Ni;
        compute_moment_reflectivity(*state.qi, Ni_ptr, ice_props, out.Ze_ice, "ice");
    }

    if (state.qr && config.has_qr && !config.has_Nr) 
    {
        compute_species_reflectivity(*state.qr, nullptr, rain_props, out.Ze_rain);
    }

    if (state.qs && config.has_qs && !config.has_Ns) 
    {
        compute_species_reflectivity(*state.qs, nullptr, snow_props, out.Ze_snow);
    }

    if (state.qg && config.has_qg && !config.has_Ng) 
    {
        compute_species_reflectivity(*state.qg, nullptr, graupel_props, out.Ze_graupel);
    }

    if (state.qh && config.has_qh && !config.has_Nh) 
    {
        compute_species_reflectivity(*state.qh, nullptr, hail_props, out.Ze_hail);
    }

    if (state.qi && config.has_qi && !config.has_Ni) 
    {
        compute_species_reflectivity(*state.qi, nullptr, ice_props, out.Ze_ice);
    }

    #pragma omp parallel for collapse(2) if(static_cast<std::size_t>(NR_) * NTH_ * NZ_ >= 32768)
    for (int i = 0; i < NR_; ++i)
    {
        for (int j = 0; j < NTH_; ++j) 
        {
            for (int k = 0; k < NZ_; ++k) 
            {
                float Ze_total = out.Ze_rain[i][j][k] + out.Ze_snow[i][j][k] +
                                out.Ze_graupel[i][j][k] + out.Ze_hail[i][j][k] +
                                out.Ze_ice[i][j][k];
                out.Ze_linear[i][j][k] = Ze_total;
                out.Z_dBZ[i][j][k] = linear_to_dbz(Ze_total);

                out.ZH_dBZ[i][j][k] = out.Z_dBZ[i][j][k];
                out.ZV_dBZ[i][j][k] = out.Z_dBZ[i][j][k];
            }
        }
    }
}

/**
 * @brief Computes the species reflectivity.
 */
void ReflectivityScheme::compute_species_reflectivity(
    const Field3D& q,
    const Field3D* Nt,
    const HydrometeorProps& props,
    Field3D& Ze_out) {

    if (Nt) 
    {
        std::string species_name = "unknown";
        if (&props == &rain_props) species_name = "rain";
        else if (&props == &snow_props) species_name = "snow";
        else if (&props == &graupel_props) species_name = "graupel";
        else if (&props == &hail_props) species_name = "hail";
        else if (&props == &ice_props) species_name = "ice";

        compute_moment_reflectivity(q, Nt, props, Ze_out, species_name);
        return;
    }

    #pragma omp parallel for collapse(2) if(static_cast<std::size_t>(NR_) * NTH_ * NZ_ >= 32768)
    for (int i = 0; i < NR_; ++i) 
    {
        for (int j = 0; j < NTH_; ++j) 
        {
            for (int k = 0; k < NZ_; ++k) 
            {
                const float q_val = q[i][j][k];

                if (q_val <= 0.0f) {
                    Ze_out[i][j][k] = 0.0f;
                    continue;
                }


                // Per-species Marshall-Palmer coefficient (see kZeCoeff*),
                // selected by props identity (matches the moment path's style).
                double A_x = kZeCoeffRain;
                if (&props == &snow_props)         A_x = kZeCoeffSnow;
                else if (&props == &graupel_props) A_x = kZeCoeffGraupel;
                else if (&props == &hail_props)    A_x = kZeCoeffHail;

                const double mass = local_dry_air_density(rho_, i, j, k) * static_cast<double>(q_val);  // kg/m^3
                double Ze = A_x * std::pow(mass, kZeReflExponent);

                Ze_out[i][j][k] = static_cast<float>(Ze);
            }
        }
    }
}

/**
 * @brief Computes the moment reflectivity.
 */
void ReflectivityScheme::compute_moment_reflectivity(
    const Field3D& q,
    const Field3D* Nt,
    const HydrometeorProps& props,
    Field3D& Ze_out,
    const std::string& species_name) {

    if (!Nt) 
    {
        tmv::log_warn("compute_moment_reflectivity called without number concentrations for ",
                      species_name);
        compute_species_reflectivity(q, nullptr, props, Ze_out);
        return;
    }

    #pragma omp parallel for collapse(2) if(static_cast<std::size_t>(NR_) * NTH_ * NZ_ >= 32768)
    for (int i = 0; i < NR_; ++i) 
    {
        for (int j = 0; j < NTH_; ++j) 
        {
            for (int k = 0; k < NZ_; ++k) 
            {
                const float q_val = q[i][j][k];
                const float Nt_val = (*Nt)[i][j][k];

                if (q_val <= 0.0f || Nt_val <= 0.0f) 
                {
                    Ze_out[i][j][k] = 0.0f;
                    continue;
                }


                const double rho_d = local_dry_air_density(rho_, i, j, k);
                const double q_mass = q_val * rho_d;

                double Nt_m3 = Nt_val;

                double rho_ratio = props.density / 1000.0;

                double C = 3.6e9;

                double Ze = C * rho_ratio * rho_ratio * q_mass * q_mass / Nt_m3;

                Ze_out[i][j][k] = static_cast<float>(Ze);
            }
        }
    }
}

}  // namespace tmv
