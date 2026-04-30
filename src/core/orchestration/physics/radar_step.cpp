/**
 * @file radar_step.cpp
 * @brief Radar reflectivity computation and initialization.
 *
 * Computes radar reflectivity diagnostics via the active radar scheme,
 * with a microphysics-based fallback. Also provides radar scheme initialization.
 *
 * Extracted from src/core/equations.cpp.
 */

#include "core/simulation.hpp"
#include "microphysics/microphysics_base.hpp"
#include "radar/radar_base.hpp"
#include "radar/factory.hpp"
#include "util/log.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

extern std::unique_ptr<MicrophysicsScheme> microphysics_scheme;
extern std::unique_ptr<RadarSchemeBase> radar_scheme;

void initialize_radar(const std::string& scheme_name)
{
    try
    {
        radar_scheme = create_radar_scheme(scheme_name);

        RadarConfig config;
        config.scheme_id = scheme_name;
        config.operator_tier = "fast_da";
        if (scheme_name == "zdr")
        {
            config.operator_tier = "polarimetric_fo";
        }

        config.has_qr = true;
        config.has_qs = true;
        config.has_qg = true;
        config.has_qh = true;
        config.has_qi = true;

        radar_scheme->initialize(config, NR, NTH, NZ);
        tmv::log_info("Initialized radar scheme: ", scheme_name);
    }
    catch (const std::runtime_error& e)
    {
        tmv::log_error("Error initializing radar: ", e.what());
        tmv::log_info("Radar scheme initialization failed, radar calculations disabled");
    }
}

void calculate_radar_reflectivity()
{
    if (radar_reflectivity.size_r() != NR || radar_reflectivity.size_th() != NTH || radar_reflectivity.size_z() != NZ)
        radar_reflectivity.resize(NR, NTH, NZ, 0.0f);

    constexpr float radar_linear_min = 0.0f;
    constexpr float radar_linear_max = 1.0e12f;

    auto sanitize_linear_reflectivity_field = [&](Field3D& field, const char* source_tag) {
        if (field.empty())
        {
            return;
        }
        int sanitized = 0;
        float* const data = field.data();
        const std::size_t count = field.size();
        #pragma omp parallel for reduction(+:sanitized)
        for (long long idx = 0; idx < static_cast<long long>(count); ++idx)
        {
            const float old_value = data[idx];
            float new_value = old_value;
            if (!std::isfinite(static_cast<double>(new_value)) || new_value < radar_linear_min)
            {
                new_value = radar_linear_min;
            }
            else if (new_value > radar_linear_max)
            {
                new_value = radar_linear_max;
            }
            if (new_value != old_value)
            {
                ++sanitized;
            }
            data[idx] = new_value;
        }
        if (sanitized > 0)
        {
            tmv::log_warn("[RADAR GUARD] sanitized ", sanitized, " reflectivity samples from ", source_tag);
        }
    };

    auto apply_microphysics_fallback = [&]() -> bool
    {
        if (!microphysics_scheme)
        {
            tmv::log_warn("[RADAR GUARD] no microphysics scheme available for radar fallback.");
            return false;
        }

        Field3D radar_dbz;
        try
        {
            microphysics_scheme->compute_radar_reflectivity(
                qc, qr, qi, qs, qg, qh, radar_dbz
            );
        }
        catch (const std::exception& e)
        {
            tmv::log_warn("[RADAR GUARD] microphysics fallback reflectivity failed: ", e.what(),
                         ". Keeping previous reflectivity field for this step.");
            return false;
        }

        if (radar_dbz.size_r() != NR || radar_dbz.size_th() != NTH || radar_dbz.size_z() != NZ)
        {
            tmv::log_warn("[RADAR GUARD] microphysics fallback returned unexpected dBZ field shape; "
                         "leaving reflectivity unchanged for this step.");
            return false;
        }

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < NR; ++i)
        {
            for (int j = 0; j < NTH; ++j)
            {
                for (int k = 0; k < NZ; ++k)
                {
                    float z_dbz = static_cast<float>(radar_dbz[i][j][k]);
                    if (!std::isfinite(static_cast<double>(z_dbz)))
                    {
                        radar_reflectivity[i][j][k] = radar_linear_min;
                        continue;
                    }

                    z_dbz = std::clamp(z_dbz, -120.0f, 120.0f);
                    float z_linear = std::pow(10.0f, z_dbz / 10.0f);
                    if (!std::isfinite(static_cast<double>(z_linear)))
                    {
                        z_linear = radar_linear_max;
                    }
                    radar_reflectivity[i][j][k] =
                        std::clamp(z_linear, radar_linear_min, radar_linear_max);
                }
            }
        }

        sanitize_linear_reflectivity_field(radar_reflectivity, "microphysics_fallback");
        return true;
    };

    if (!radar_scheme)
    {
        tmv::log_warn("Radar scheme not initialized, using microphysics fallback");
        apply_microphysics_fallback();
        return;
    }

    RadarStateView state_view;
    state_view.NR = NR;
    state_view.NTH = NTH;
    state_view.NZ = NZ;

    state_view.u = &u;
    state_view.v = &v;
    state_view.w = &w;

    state_view.qr = &qr;
    state_view.qs = &qs;
    state_view.qg = &qg;
    state_view.qh = &qh;
    state_view.qi = &qi;

    RadarConfig config;
    config.scheme_id = "reflectivity";
    config.operator_tier = "fast_da";
    config.has_qr = true;
    config.has_qs = true;
    config.has_qg = true;
    config.has_qh = true;
    config.has_qi = true;

    RadarOut radar_out;
    radar_out.initialize(NR, NTH, NZ);

    try
    {
        radar_scheme->compute(config, state_view, radar_out);
    }
    catch (const std::exception& e)
    {
        tmv::log_warn("[RADAR GUARD] radar scheme compute failed: ", e.what(),
                      ". Attempting microphysics fallback.");
        apply_microphysics_fallback();
        return;
    }

    int sanitized_reflectivity = 0;

    #pragma omp parallel for collapse(2) reduction(+:sanitized_reflectivity)
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 0; k < NZ; ++k)
            {
                float z_linear = static_cast<float>(radar_out.Ze_linear[i][j][k]);
                if (!std::isfinite(static_cast<double>(z_linear)) || z_linear < radar_linear_min)
                {
                    if (z_linear != radar_linear_min)
                    {
                        ++sanitized_reflectivity;
                    }
                    z_linear = radar_linear_min;
                }
                else if (z_linear > radar_linear_max)
                {
                    ++sanitized_reflectivity;
                    z_linear = radar_linear_max;
                }
                radar_reflectivity[i][j][k] = z_linear;
            }
        }
    }

    if (sanitized_reflectivity > 0)
    {
        tmv::log_warn("[RADAR GUARD] sanitized ", sanitized_reflectivity, " reflectivity samples from radar scheme output");
    }
}
