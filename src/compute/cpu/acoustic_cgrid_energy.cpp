#include "cpu_backend.hpp"

#include "core/infra/physical_constants.hpp"
#include "diagnostics/deterministic_reduction.hpp"
#include "dynamics/acoustic_energy_report.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <stdexcept>

namespace nimbus::compute::cpu_internal
{

bool CpuComputeBackend::dispatch_acoustic_energy_path(
    const AcousticSubstepFusedCartesianCgridDispatch& d,
    const AcousticCgridContext& context)
{
    const bool attribute_energy = d.energy_report_host != nullptr;
    const int nx = context.nx;
    const int ny = context.ny;
    const int nz = context.nz;
    const double inv_dx = context.inv_dx;
    const double inv_dy = context.inv_dy;
    const double inv_dz = context.inv_dz;
    const double dt_small = context.dt_small;
    const double cell_volume = context.cell_volume;
    const auto idx = [ny, nz](int i, int j, int k) -> std::size_t
    {
        return (static_cast<std::size_t>(i) * static_cast<std::size_t>(ny)
                + static_cast<std::size_t>(j))
                   * static_cast<std::size_t>(nz)
               + static_cast<std::size_t>(k);
    };
    const auto finite_or_zero = [](double value)
    {
        return std::isfinite(value) ? value : 0.0;
    };
    const auto safe_rho = [](double value)
    {
        return std::isfinite(value) && value > 1.0e-6 ? value : 1.0;
    };
    const auto clamp_value = [](float value, float fallback,
                                float lo, float hi)
    {
        if (!std::isfinite(value)) value = fallback;
        return std::clamp(value, lo, hi);
    };

    const auto total_water = [&](std::size_t c)
    {
        return static_cast<double>(d.qv_host[c])
            + static_cast<double>(d.qc_host[c])
            + static_cast<double>(d.qr_host[c])
            + static_cast<double>(d.qi_host[c])
            + static_cast<double>(d.qs_host[c])
            + static_cast<double>(d.qg_host[c])
            + static_cast<double>(d.qh_host[c]);
    };
    const auto mixture_mass = [&](std::size_t c)
    {
        return static_cast<double>(d.rho[c])
            * (1.0 + total_water(c)) * cell_volume;
    };

    if (attribute_energy)
    {
        const double prior_values[] = {
            d.energy_report_host
                ->continuity_eos_dry_internal_change_j,
            d.energy_report_host
                ->continuity_eos_water_sensible_change_j,
            d.energy_report_host
                ->continuity_eos_phase_reference_change_j,
            d.energy_report_host
                ->continuity_carrier_kinetic_change_j,
            d.energy_report_host
                ->continuity_gravitational_change_j,
            d.energy_report_host
                ->continuity_subgrid_tke_change_j,
            d.energy_report_host
                ->continuity_carrier_requested_change_j,
            d.energy_report_host
                ->pressure_gradient_requested_linear_work_j,
            d.energy_report_host
                ->pressure_gradient_finite_step_correction_j,
            d.energy_report_host
                ->pressure_gradient_storage_rounding_j,
            d.energy_report_host
                ->pressure_gradient_committed_work_j,
            d.energy_report_host->boundary_committed_work_j,
            d.energy_report_host
                ->boundary_radiative_requested_linear_work_j,
            d.energy_report_host
                ->boundary_radiative_finite_step_correction_j,
            d.energy_report_host
                ->boundary_radiative_storage_rounding_j,
            d.energy_report_host
                ->boundary_radiative_committed_work_j,
            d.energy_report_host
                ->boundary_constraint_history_committed_work_j,
            d.energy_report_host->committed_inventory_change_j,
            cell_volume,
        };
        if (std::any_of(
                std::begin(prior_values), std::end(prior_values),
                [](double value) { return !std::isfinite(value); })
            || d.energy_report_host
                   ->pressure_gradient_finite_step_correction_j < 0.0)
        {
            throw std::invalid_argument(
                "C-grid acoustic energy attribution received an invalid "
                "existing report or cell volume");
        }

        // Validate the complete SCI-140 input before the fused operator
        // mutates state. This traversal exists only when attribution is
        // explicitly requested.
        for (int i = 1; i < nx - 1; ++i)
            for (int j = 1; j < ny - 1; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    const std::size_t c = idx(i, j, k);
                    const double values[] = {
                        static_cast<double>(d.rho[c]),
                        static_cast<double>(d.p[c]),
                        static_cast<double>(d.theta_data[c]),
                        static_cast<double>(d.qv_host[c]),
                        static_cast<double>(d.qc_host[c]),
                        static_cast<double>(d.qr_host[c]),
                        static_cast<double>(d.qi_host[c]),
                        static_cast<double>(d.qs_host[c]),
                        static_cast<double>(d.qg_host[c]),
                        static_cast<double>(d.qh_host[c]),
                        static_cast<double>(d.tke_host[c]),
                        static_cast<double>(d.u[idx(i - 1, j, k)]),
                        static_cast<double>(d.u[c]),
                        static_cast<double>(d.v[idx(i, j - 1, k)]),
                        static_cast<double>(d.v[c]),
                        k == 0 ? 0.0
                            : static_cast<double>(
                                d.w[idx(i, j, k - 1)]),
                        static_cast<double>(d.w[c]),
                    };
                    if (std::any_of(
                            std::begin(values), std::end(values),
                            [](double value)
                            {
                                return !std::isfinite(value);
                            })
                        || values[0] <= 0.0 || values[1] <= 0.0
                        || values[2] <= 0.0
                        || values[3] < 0.0 || values[4] < 0.0
                        || values[5] < 0.0 || values[6] < 0.0
                        || values[7] < 0.0 || values[8] < 0.0
                        || values[9] < 0.0 || values[10] < 0.0)
                    {
                        throw std::invalid_argument(
                            "C-grid acoustic energy attribution rejected "
                            "non-finite or nonphysical SCI-140 input");
                    }
                }
    }

    double dry_internal_change = 0.0;
    double water_sensible_change = 0.0;
    double phase_reference_change = 0.0;
    double carrier_kinetic_change = 0.0;
    double carrier_kinetic_requested_change = 0.0;
    double gravitational_change = 0.0;
    double subgrid_tke_change = 0.0;
    double requested_pressure_work = 0.0;
    double finite_step_pressure_work = 0.0;
    double committed_pressure_work = 0.0;
    std::uint64_t invalid_energy_cells = 0;
    continuity_energy_partials_.assign(
        static_cast<std::size_t>(nx), std::array<double, 8>{});
    pressure_work_partials_.assign(
        static_cast<std::size_t>(nx), std::array<double, 3>{});

    // The same one-team, three-pass dependency structure as the native
    // CPU-009 kernel: continuity must finish before density/EOS, which
    // must finish before pressure-gradient momentum.
    #pragma omp parallel
    {
        #pragma omp for collapse(2)
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    double drho = 0.0;
                    if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1)
                    {
                        const std::size_t c = idx(i, j, k);
                        const std::size_t im = idx(i - 1, j, k);
                        const std::size_t jm = idx(i, j - 1, k);

                        double fx_pos = 0.0;
                        if (i + 1 < nx)
                        {
                            const std::size_t ip = idx(i + 1, j, k);
                            fx_pos = 0.5 * (static_cast<double>(d.rho[c])
                                          + static_cast<double>(d.rho[ip]))
                                   * static_cast<double>(d.u[c]);
                        }
                        const double fx_neg =
                            0.5 * (static_cast<double>(d.rho[im])
                                 + static_cast<double>(d.rho[c]))
                            * static_cast<double>(d.u[im]);

                        double fy_pos = 0.0;
                        if (j + 1 < ny)
                        {
                            const std::size_t jp = idx(i, j + 1, k);
                            fy_pos = 0.5 * (static_cast<double>(d.rho[c])
                                          + static_cast<double>(d.rho[jp]))
                                   * static_cast<double>(d.v[c]);
                        }
                        const double fy_neg =
                            0.5 * (static_cast<double>(d.rho[jm])
                                 + static_cast<double>(d.rho[c]))
                            * static_cast<double>(d.v[jm]);

                        double fz_pos = 0.0;
                        if (k + 1 < nz)
                        {
                            const std::size_t kp = idx(i, j, k + 1);
                            fz_pos = 0.5 * (static_cast<double>(d.rho[c])
                                          + static_cast<double>(d.rho[kp]))
                                   * static_cast<double>(d.w[c]);
                        }
                        else
                        {
                            // The caller closes w at the bulk top face:
                            // zero for a rigid lid, extrapolated for the
                            // C-grid open-top contract.
                            fz_pos = static_cast<double>(d.rho[c])
                                   * static_cast<double>(d.w[c]);
                        }
                        double fz_neg = 0.0;
                        if (k > 0)
                        {
                            const std::size_t km = idx(i, j, k - 1);
                            fz_neg = 0.5 * (static_cast<double>(d.rho[km])
                                          + static_cast<double>(d.rho[c]))
                                   * static_cast<double>(d.w[km]);
                        }
                        drho = -((fx_pos - fx_neg) * inv_dx
                                + (fy_pos - fy_neg) * inv_dy
                                + (fz_pos - fz_neg) * inv_dz);
                    }
                    d.drho_scratch[idx(i, j, k)] =
                        static_cast<float>(finite_or_zero(drho));
                }

        #pragma omp for schedule(static)
        for (int i = 0; i < nx; ++i)
        {
            auto& energy_partial =
                continuity_energy_partials_[static_cast<std::size_t>(i)];
            for (int j = 0; j < ny; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    const std::size_t c = idx(i, j, k);
                    const double rho_old =
                        static_cast<double>(d.rho[c]);
                    const double p_old =
                        static_cast<double>(d.p[c]);
                    double rho_new = static_cast<double>(d.rho[c])
                        + static_cast<double>(d.drho_scratch[c]) * dt_small;
                    if (!std::isfinite(rho_new) || rho_new <= 0.0)
                    {
                        rho_new = std::max(0.1, d.rho0_base_host[k]);
                    }
                    const float rho_stored = clamp_value(
                        static_cast<float>(rho_new), 1.0f,
                        d.rho_min, d.rho_max);
                    d.rho[c] = rho_stored;

                    double p_new = static_cast<double>(d.p_ref_fallback);
                    const double th = static_cast<double>(d.theta_data[c]);
                    if (th > 0.0)
                    {
                        p_new = physical_constants::reference_pressure_pa
                            * std::pow(
                                static_cast<double>(rho_stored)
                                    * physical_constants::gas_constant_dry_air_jkgk
                                    * th
                                    / physical_constants::reference_pressure_pa,
                                d.gamma_host);
                    }
                    if (!std::isfinite(p_new) || p_new <= 0.0)
                    {
                        p_new = static_cast<double>(d.p_ref_fallback);
                    }
                    d.p[c] = clamp_value(static_cast<float>(p_new),
                                         101325.0f, d.p_min, d.p_max);

                    if (attribute_energy
                        && i >= 1 && i < nx - 1
                        && j >= 1 && j < ny - 1)
                    {
                        constexpr double cv_d =
                            physical_constants::specific_heat_cp_jkgk
                            - physical_constants::gas_constant_dry_air_jkgk;
                        constexpr double cv_v =
                            physical_constants::specific_heat_water_vapor_cp_jkgk
                            - physical_constants::gas_constant_water_vapor_jkgk;
                        constexpr double c_l =
                            physical_constants::specific_heat_liquid_water_jkgk;
                        constexpr double c_i =
                            physical_constants::specific_heat_ice_jkgk;
                        constexpr double t_ref =
                            physical_constants::freezing_temperature_k;
                        constexpr double vapor_reference =
                            physical_constants::latent_heat_vaporization_jkg
                            - physical_constants::gas_constant_water_vapor_jkgk
                                * t_ref;
                        constexpr double ice_reference =
                            -physical_constants::latent_heat_fusion_jkg;
                        constexpr double kappa =
                            physical_constants::gas_constant_dry_air_jkgk
                            / physical_constants::specific_heat_cp_jkgk;

                        const double theta =
                            static_cast<double>(d.theta_data[c]);
                        const double qv =
                            static_cast<double>(d.qv_host[c]);
                        const double qc =
                            static_cast<double>(d.qc_host[c]);
                        const double qr =
                            static_cast<double>(d.qr_host[c]);
                        const double qi =
                            static_cast<double>(d.qi_host[c]);
                        const double qs =
                            static_cast<double>(d.qs_host[c]);
                        const double qg =
                            static_cast<double>(d.qg_host[c]);
                        const double qh =
                            static_cast<double>(d.qh_host[c]);
                        const double tke =
                            static_cast<double>(d.tke_host[c]);
                        const double p_stored =
                            static_cast<double>(d.p[c]);
                        const double rho_stored =
                            static_cast<double>(d.rho[c]);
                        const double q_liquid = qc + qr;
                        const double q_ice = qi + qs + qg + qh;
                        const double q_total = qv + q_liquid + q_ice;
                        const double temperature_old = theta * std::pow(
                            p_old
                                / physical_constants::reference_pressure_pa,
                            kappa);
                        const double temperature_new = theta * std::pow(
                            p_stored
                                / physical_constants::reference_pressure_pa,
                            kappa);
                        const double values[] = {
                            rho_old, rho_stored, p_old, p_stored, theta,
                            qv, qc, qr, qi, qs, qg, qh, tke,
                            temperature_old, temperature_new, cell_volume,
                        };
                        if (std::any_of(
                                std::begin(values), std::end(values),
                                [](double value)
                                {
                                    return !std::isfinite(value);
                                })
                            || rho_old <= 0.0 || rho_stored <= 0.0
                            || p_old <= 0.0 || p_stored <= 0.0
                            || theta <= 0.0 || temperature_old <= 0.0
                            || temperature_new <= 0.0
                            || qv < 0.0 || qc < 0.0 || qr < 0.0
                            || qi < 0.0 || qs < 0.0 || qg < 0.0
                            || qh < 0.0 || tke < 0.0
                            || !(q_total > -1.0)
                            || !(cell_volume > 0.0))
                        {
                            energy_partial[7] += 1.0;
                        }
                        else
                        {
                            const double dry_mass_old =
                                rho_old * cell_volume;
                            const double dry_mass_new =
                                rho_stored * cell_volume;
                            const double mixture_mass_old =
                                dry_mass_old * (1.0 + q_total);
                            const double mixture_mass_new =
                                dry_mass_new * (1.0 + q_total);
                            const double delta_t_old =
                                temperature_old - t_ref;
                            const double delta_t_new =
                                temperature_new - t_ref;

                            energy_partial[0] += cv_d * (
                                dry_mass_new * delta_t_new
                                - dry_mass_old * delta_t_old);
                            energy_partial[1] += (
                                qv * cv_v + q_liquid * c_l + q_ice * c_i)
                                * (dry_mass_new * delta_t_new
                                   - dry_mass_old * delta_t_old);
                            energy_partial[2] +=
                                (qv * vapor_reference
                                 + q_ice * ice_reference)
                                * (dry_mass_new - dry_mass_old);

                            const double u_w = static_cast<double>(
                                d.u[idx(i - 1, j, k)]);
                            const double u_e =
                                static_cast<double>(d.u[c]);
                            const double v_s = static_cast<double>(
                                d.v[idx(i, j - 1, k)]);
                            const double v_n =
                                static_cast<double>(d.v[c]);
                            const double w_b = k == 0 ? 0.0
                                : static_cast<double>(
                                    d.w[idx(i, j, k - 1)]);
                            const double w_t =
                                static_cast<double>(d.w[c]);
                            const double speed_face_sum =
                                u_w * u_w + u_e * u_e
                                + v_s * v_s + v_n * v_n
                                + w_b * w_b + w_t * w_t;
                            energy_partial[3] += 0.25
                                * (mixture_mass_new - mixture_mass_old)
                                * speed_face_sum;
                            energy_partial[4] +=
                                dt_small
                                * static_cast<double>(
                                    d.drho_scratch[c])
                                * cell_volume * (1.0 + q_total)
                                * 0.25 * speed_face_sum;
                            energy_partial[5] +=
                                (mixture_mass_new - mixture_mass_old)
                                * physical_constants::gravity_ms2
                                * ((static_cast<double>(k) + 0.5)
                                   * d.dz_host);
                            energy_partial[6] +=
                                (dry_mass_new - dry_mass_old) * tke;
                        }
                    }
                }
        }

        #pragma omp for schedule(static)
        for (int i = 0; i < nx; ++i)
        {
            auto& work_partial =
                pressure_work_partials_[static_cast<std::size_t>(i)];
            for (int j = 0; j < ny; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    const std::size_t c = idx(i, j, k);
                    float du = 0.0f;
                    float dv = 0.0f;
                    float dw = 0.0f;
                    if (i >= 1 && i < nx - 1
                        && j >= 1 && j < ny - 1 && k < nz - 1)
                    {
                        const std::size_t ip = idx(i + 1, j, k);
                        const std::size_t jp = idx(i, j + 1, k);
                        const std::size_t kp = idx(i, j, k + 1);
                        const double rho_x = 0.5 * (
                            static_cast<double>(d.rho[c])
                            + static_cast<double>(d.rho[ip]));
                        const double rho_y = 0.5 * (
                            static_cast<double>(d.rho[c])
                            + static_cast<double>(d.rho[jp]));
                        const double rho_z = 0.5 * (
                            static_cast<double>(d.rho[c])
                            + static_cast<double>(d.rho[kp]));
                        const double dp0_dz =
                            (d.p0_base_host[k + 1] - d.p0_base_host[k])
                            * inv_dz;
                        du = static_cast<float>(finite_or_zero(
                            -(static_cast<double>(d.p[ip])
                              - static_cast<double>(d.p[c]))
                            * inv_dx / safe_rho(rho_x)));
                        dv = static_cast<float>(finite_or_zero(
                            -(static_cast<double>(d.p[jp])
                              - static_cast<double>(d.p[c]))
                            * inv_dy / safe_rho(rho_y)));
                        dw = static_cast<float>(finite_or_zero(
                            -(((static_cast<double>(d.p[kp])
                                - static_cast<double>(d.p[c])) * inv_dz)
                              - dp0_dz)
                            / safe_rho(rho_z)));
                    }

                    const double u_old = static_cast<double>(d.u[c]);
                    const double v_old = static_cast<double>(d.v[c]);
                    const double w_old = static_cast<double>(d.w[c]);
                    const double delta_u =
                        static_cast<double>(du) * dt_small;
                    const double delta_v =
                        static_cast<double>(dv) * dt_small;
                    const double delta_w =
                        static_cast<double>(dw) * dt_small;
                    d.u[c] = clamp_value(static_cast<float>(
                        u_old + delta_u),
                        0.0f, -d.wind_clamp_h, d.wind_clamp_h);
                    d.v[c] = clamp_value(static_cast<float>(
                        v_old + delta_v),
                        0.0f, -d.wind_clamp_h, d.wind_clamp_h);
                    d.w[c] = clamp_value(static_cast<float>(
                        w_old + delta_w),
                        0.0f, -d.wind_clamp_v, d.wind_clamp_v);

                    if (attribute_energy
                        && i >= 1 && i < nx - 1
                        && j >= 1 && j < ny - 1 && k < nz - 1)
                    {
                        // A physical scalar cell owns half of each
                        // adjacent face control volume. At a lateral
                        // boundary only the interior half belongs to the
                        // inventory; the subsequent BC pass owns any
                        // mirrored/periodic half-face change.
                        double u_mass = 0.5 * mixture_mass(c);
                        if (i + 1 < nx - 1)
                            u_mass += 0.5 * mixture_mass(
                                idx(i + 1, j, k));
                        double v_mass = 0.5 * mixture_mass(c);
                        if (j + 1 < ny - 1)
                            v_mass += 0.5 * mixture_mass(
                                idx(i, j + 1, k));
                        const double w_mass = 0.5 * (
                            mixture_mass(c)
                            + mixture_mass(idx(i, j, k + 1)));

                        work_partial[0] +=
                            u_mass * u_old * delta_u
                            + v_mass * v_old * delta_v
                            + w_mass * w_old * delta_w;
                        work_partial[1] += 0.5 * (
                            u_mass * delta_u * delta_u
                            + v_mass * delta_v * delta_v
                            + w_mass * delta_w * delta_w);
                        work_partial[2] += 0.5 * (
                            u_mass
                                * (static_cast<double>(d.u[c])
                                       * static_cast<double>(d.u[c])
                                   - u_old * u_old)
                            + v_mass
                                * (static_cast<double>(d.v[c])
                                       * static_cast<double>(d.v[c])
                                   - v_old * v_old)
                            + w_mass
                                * (static_cast<double>(d.w[c])
                                       * static_cast<double>(d.w[c])
                                   - w_old * w_old));
                    }
                }
        }
    }
    if (attribute_energy)
    {
        std::array<tmv::diagnostics::CompensatedSum, 8> energy_merged{};
        for (const auto& partial : continuity_energy_partials_)
            for (std::size_t n = 0; n < partial.size(); ++n)
                energy_merged[n].add(partial[n]);
        dry_internal_change = energy_merged[0].value();
        water_sensible_change = energy_merged[1].value();
        phase_reference_change = energy_merged[2].value();
        carrier_kinetic_change = energy_merged[3].value();
        carrier_kinetic_requested_change = energy_merged[4].value();
        gravitational_change = energy_merged[5].value();
        subgrid_tke_change = energy_merged[6].value();
        invalid_energy_cells = static_cast<std::uint64_t>(
            energy_merged[7].value());

        std::array<tmv::diagnostics::CompensatedSum, 3> work_merged{};
        for (const auto& partial : pressure_work_partials_)
            for (std::size_t n = 0; n < partial.size(); ++n)
                work_merged[n].add(partial[n]);
        requested_pressure_work = work_merged[0].value();
        finite_step_pressure_work = work_merged[1].value();
        committed_pressure_work = work_merged[2].value();

        if (invalid_energy_cells != 0)
        {
            throw std::runtime_error(
                "C-grid acoustic energy attribution rejected non-finite "
                "or nonphysical SCI-140 state");
        }
        const double storage_rounding =
            committed_pressure_work - requested_pressure_work
            - finite_step_pressure_work;
        const double committed_inventory =
            dry_internal_change + water_sensible_change
            + phase_reference_change + carrier_kinetic_change
            + gravitational_change + subgrid_tke_change
            + committed_pressure_work;
        const double values[] = {
            dry_internal_change, water_sensible_change,
            phase_reference_change, carrier_kinetic_change,
            carrier_kinetic_requested_change,
            gravitational_change, subgrid_tke_change,
            requested_pressure_work, finite_step_pressure_work,
            storage_rounding, committed_pressure_work,
            committed_inventory,
        };
        if (std::any_of(
                std::begin(values), std::end(values),
                [](double value) { return !std::isfinite(value); })
            || finite_step_pressure_work < 0.0)
        {
            throw std::runtime_error(
                "C-grid acoustic energy attribution produced an invalid ledger");
        }

        CartesianCGridAcousticEnergyReportV1 next =
            *d.energy_report_host;
        next.continuity_eos_dry_internal_change_j +=
            dry_internal_change;
        next.continuity_eos_water_sensible_change_j +=
            water_sensible_change;
        next.continuity_eos_phase_reference_change_j +=
            phase_reference_change;
        next.continuity_carrier_kinetic_change_j +=
            carrier_kinetic_change;
        next.continuity_carrier_requested_change_j +=
            carrier_kinetic_requested_change;
        next.continuity_gravitational_change_j +=
            gravitational_change;
        next.continuity_subgrid_tke_change_j +=
            subgrid_tke_change;
        next.pressure_gradient_requested_linear_work_j +=
            requested_pressure_work;
        next.pressure_gradient_finite_step_correction_j +=
            finite_step_pressure_work;
        next.pressure_gradient_storage_rounding_j += storage_rounding;
        next.pressure_gradient_committed_work_j +=
            committed_pressure_work;
        next.committed_inventory_change_j += committed_inventory;
        ++next.substep_count;
        next.complete = true;
        const double cumulative_values[] = {
            next.continuity_eos_dry_internal_change_j,
            next.continuity_eos_water_sensible_change_j,
            next.continuity_eos_phase_reference_change_j,
            next.continuity_carrier_kinetic_change_j,
            next.continuity_gravitational_change_j,
            next.continuity_subgrid_tke_change_j,
            next.continuity_carrier_requested_change_j,
            next.pressure_gradient_requested_linear_work_j,
            next.pressure_gradient_finite_step_correction_j,
            next.pressure_gradient_storage_rounding_j,
            next.pressure_gradient_committed_work_j,
            next.boundary_committed_work_j,
            next.boundary_radiative_requested_linear_work_j,
            next.boundary_radiative_finite_step_correction_j,
            next.boundary_radiative_storage_rounding_j,
            next.boundary_radiative_committed_work_j,
            next.boundary_constraint_history_committed_work_j,
            next.committed_inventory_change_j,
        };
        if (std::any_of(
                std::begin(cumulative_values),
                std::end(cumulative_values),
                [](double value) { return !std::isfinite(value); }))
        {
            throw std::overflow_error(
                "C-grid acoustic energy attribution overflowed its cumulative report");
        }
        *d.energy_report_host = next;
    }
    ++acoustic_fused_dispatches_;
    return true;
}

} // namespace nimbus::compute::cpu_internal
