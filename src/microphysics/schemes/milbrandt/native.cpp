#include "coupled_storage.hpp"
#include "milbrandt.hpp"
#include "compute/backend_runtime.hpp"
#include "microphysics/base/thermodynamics.hpp"
#include "microphysics/milbrandt_cold_parameters.hpp"
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tmv
{
bool MilbrandtScheme::native_execution_selected() const noexcept
{
    const auto* runtime = compute_runtime();
    return runtime && runtime->backend && runtime->active_kind != ComputeBackendKind::Cpu && !runtime->fallback_active;
}

void MilbrandtScheme::compute_native_tendencies(const std::array<const Field3D*, 10>& state,
                                                double                                dt,
                                                const std::array<Field3D*, 8>&        output)
{
    auto* runtime = compute_runtime();
    auto* kernels = runtime && runtime->backend ? runtime->backend->microphysics_kernels() : nullptr;
    if (!kernels)
        throw std::runtime_error("native Milbrandt process ownership is unavailable");
    for (auto op : {MicrophysicsKernelOp::MilbrandtCloudAdjustment,
                    MicrophysicsKernelOp::MilbrandtWarmRain,
                    MicrophysicsKernelOp::MilbrandtCold,
                    MicrophysicsKernelOp::MilbrandtMelting,
                    MicrophysicsKernelOp::MilbrandtSedimentation})
        if (!kernels->supports(op))
            throw std::runtime_error("native Milbrandt process ownership is unavailable");
    const std::array<double, 25> parameters{a_g_,     a_h_, a_i_, a_r_, a_s_, alpha_g_, alpha_h_, alpha_i_, alpha_r_,
                                            alpha_s_, b_g_, b_h_, b_i_, b_r_, b_s_,     c_g_,     c_h_,     c_i_,
                                            c_r_,     c_s_, d_g_, d_h_, d_i_, d_r_,     d_s_};
    if (parameters != milbrandt_detail::default_cold_parameters)
        throw std::invalid_argument("native Milbrandt requires default population parameters");
    const int  nx = state[0]->size_r(), ny = state[0]->size_th(), nz = state[0]->size_z();
    const auto matches = [&](const Field3D& f) { return f.size_r() == nx && f.size_th() == ny && f.size_z() == nz; };
    if (nx <= 0 || ny <= 0 || nz <= 0 || !std::isfinite(dt) || dt <= 0)
        throw std::invalid_argument("invalid native Milbrandt extent or timestep");
    const std::array<const Field3D*, 6> moments{&Nc_, &Nr_, &Ni_, &Ns_, &Ng_, &Nh_};
    for (auto f : state)
        if (!f || !matches(*f))
            throw std::invalid_argument("native Milbrandt input shape mismatch");
    for (auto f : moments)
        if (!matches(*f))
            throw std::invalid_argument("native Milbrandt requires prepared number state");
    const Field3D* wind = vertical_velocity_field();
    if (wind && !matches(*wind))
        throw std::invalid_argument("native Milbrandt velocity shape mismatch");
    for (std::size_t f = 0; f < output.size(); ++f)
    {
        if (!output[f])
            throw std::invalid_argument("null native Milbrandt destination");
        if (output[f] == wind)
            throw std::invalid_argument("native Milbrandt output aliases velocity input");
        for (auto input : state)
            if (output[f] == input)
                throw std::invalid_argument("native Milbrandt output aliases input");
        for (auto input : moments)
            if (output[f] == input)
                throw std::invalid_argument("native Milbrandt output aliases private state");
        for (std::size_t q = 0; q < f; ++q)
            if (output[f] == output[q])
                throw std::invalid_argument("native Milbrandt outputs overlap");
    }
    const auto domain    = physical_cell_domain(nx, ny, nz);
    const auto configure = [&](auto& d)
    {
        d.nx      = nx;
        d.ny      = ny;
        d.nz      = nz;
        d.i_begin = domain.i_begin;
        d.i_end   = domain.i_end;
        d.j_begin = domain.j_begin;
        d.j_end   = domain.j_end;
        d.k_begin = domain.k_begin;
        d.k_end   = domain.k_end;
        d.dt      = dt;
    };
    Field3D temperature(nx, ny, nz, 150), zero(nx, ny, nz, 0), dz_high(nx, ny, nz, 1), dz_low(nx, ny, nz, 0);
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                for (auto f : state)
                    if (!std::isfinite((*f)(i, j, k)))
                        throw std::invalid_argument("nonfinite native Milbrandt input");
                const double dz = vertical_spacing_m(i, j, k);
                if (!std::isfinite(dz) || dz <= 0)
                    throw std::invalid_argument("invalid native Milbrandt layer thickness");
                temperature(i, j, k) =
                    float(thermodynamics::theta_to_temperature((*state[0])(i, j, k), (*state[1])(i, j, k)));
                dz_high(i, j, k) = float(dz);
                dz_low(i, j, k)  = float(dz - dz_high(i, j, k));
            }
    std::array<Field3D, 7> cloud;
    for (auto& f : cloud)
        f.resize(nx, ny, nz, 0);
    MilbrandtCloudAdjustmentDispatch cloud_dispatch;
    configure(cloud_dispatch);
    cloud_dispatch.state                   = {temperature.data(),
                                              state[1]->data(),
                                              state[2]->data(),
                                              state[3]->data(),
                                              state[4]->data(),
                                              Nc_.data(),
                                              wind ? wind->data() : zero.data()};
    cloud_dispatch.face_staggered_velocity = vertical_velocity_is_face_staggered();
    cloud_dispatch.lower_boundary_vertical_velocity =
        lower_boundary_vertical_velocity().empty() ? nullptr : lower_boundary_vertical_velocity().data();
    for (std::size_t f = 0; f < cloud.size(); ++f)
        cloud_dispatch.output[f] = cloud[f].data();
    if (!kernels->dispatch_milbrandt_cloud_adjustment(cloud_dispatch))
        throw std::runtime_error("native Milbrandt cloud adjustment failed");
    std::array<Field3D, 14> rates;
    for (auto& f : rates)
        f.resize(nx, ny, nz, 0);
    rates[0] = std::move(cloud[5]);
    rates[6] = std::move(cloud[4]);
    rates[7] = std::move(cloud[3]);
    rates[8] = std::move(cloud[6]);
    constexpr std::array<int, 6> warm_fields{0, 1, 6, 7, 8, 9};
    std::array<Field3D, 6>       warm;
    for (auto& f : warm)
        f.resize(nx, ny, nz, 0);
    MilbrandtWarmRainDispatch warm_dispatch;
    configure(warm_dispatch);
    warm_dispatch.state = {cloud[0].data(),
                           state[1]->data(),
                           state[2]->data(),
                           cloud[1].data(),
                           cloud[2].data(),
                           state[5]->data(),
                           Nc_.data(),
                           Nr_.data()};
    for (int f = 0; f < 6; ++f)
    {
        warm_dispatch.rates[f]  = rates[warm_fields[f]].data();
        warm_dispatch.output[f] = warm[f].data();
    }
    if (!kernels->dispatch_milbrandt_warm_rain(warm_dispatch))
        throw std::runtime_error("native Milbrandt warm rain failed");
    for (int f = 0; f < 6; ++f)
        rates[warm_fields[f]] = std::move(warm[f]);
    std::array<Field3D, 3> process{std::move(cloud[1]), std::move(cloud[2]), *state[5]};
    temperature = std::move(cloud[0]);
    zero        = Field3D{};
    constexpr std::array<int, 3> water_rates{6, 0, 1};
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                for (int f = 0; f < 3; ++f)
                    process[f](i, j, k) =
                        float(std::max(0.0, double((*state[3 + f])(i, j, k)) + dt * rates[water_rates[f]](i, j, k)));
                temperature(i, j, k) =
                    float(thermodynamics::theta_to_temperature((*state[0])(i, j, k), (*state[1])(i, j, k)) +
                          dt * rates[7](i, j, k));
            }
    std::array<Field3D, 14> cold;
    for (auto& f : cold)
        f.resize(nx, ny, nz, 0);
    MilbrandtColdDispatch cold_dispatch;
    configure(cold_dispatch);
    cold_dispatch.hail_processes = hail_processes_;
    cold_dispatch.state          = {temperature.data(),
                                    state[1]->data(),
                                    state[2]->data(),
                                    process[0].data(),
                                    process[1].data(),
                                    process[2].data(),
                                    state[6]->data(),
                                    state[7]->data(),
                                    state[8]->data(),
                                    state[9]->data(),
                                    Nc_.data(),
                                    Nr_.data(),
                                    Ni_.data(),
                                    Ns_.data(),
                                    Ng_.data(),
                                    Nh_.data()};
    for (int f = 0; f < 14; ++f)
    {
        cold_dispatch.rates[f]  = rates[f].data();
        cold_dispatch.output[f] = cold[f].data();
    }
    if (!kernels->dispatch_milbrandt_cold(cold_dispatch))
        throw std::runtime_error("native Milbrandt cold processes failed");
    rates = std::move(cold);
    std::array<Field3D, 11> melting;
    for (auto& f : melting)
        f.resize(nx, ny, nz, 0);
    MilbrandtMeltingDispatch melting_dispatch;
    configure(melting_dispatch);
    melting_dispatch.state = {temperature.data(),
                              state[1]->data(),
                              state[2]->data(),
                              process[0].data(),
                              state[6]->data(),
                              state[7]->data(),
                              state[8]->data(),
                              state[9]->data(),
                              Ni_.data(),
                              Ns_.data(),
                              Ng_.data(),
                              Nh_.data()};
    for (int f = 0; f < 11; ++f)
        melting_dispatch.output[f] = melting[f].data();
    if (!kernels->dispatch_milbrandt_melting(melting_dispatch))
        throw std::runtime_error("native Milbrandt melting failed");
    constexpr std::array<int, 11> melt_fields{2, 3, 4, 5, 1, 7, 9, 10, 11, 12, 13};
    for (int f = 0; f < 11; ++f)
        for (std::size_t n = 0; n < state[0]->size(); ++n)
            rates[melt_fields[f]].data()[n] += melting[f].data()[n];
    melting = {};
    Field3D surface(nx, ny, 1, 0);
    for (int species = 0; species < 5; ++species)
    {
        std::array<Field3D, 3> sediment;
        for (auto& f : sediment)
            f.resize(nx, ny, nz, 0);
        MilbrandtSedimentationDispatch sediment_dispatch;
        configure(sediment_dispatch);
        sediment_dispatch.species = species;
        sediment_dispatch.state   = {state[5 + species]->data(),
                                     moments[1 + species]->data(),
                                     state[2]->data(),
                                     dz_high.data(),
                                     dz_low.data(),
                                     rates[1 + species].data(),
                                     rates[9 + species].data()};
        for (int f = 0; f < 3; ++f)
            sediment_dispatch.output[f] = sediment[f].data();
        if (!kernels->dispatch_milbrandt_sedimentation(sediment_dispatch))
            throw std::runtime_error("native Milbrandt sedimentation failed");
        rates[1 + species] = std::move(sediment[0]);
        rates[9 + species] = std::move(sediment[1]);
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                surface(i, j, 0) += sediment[2](i, j, 0);
    }
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                std::array<float, 4> cloud_rates{
                    rates[0](i, j, k), rates[6](i, j, k), rates[7](i, j, k), rates[8](i, j, k)};
                if (!milbrandt_detail::finalize_cloud_storage((*state[4])(i, j, k),
                                                              process[1](i, j, k),
                                                              (*state[3])(i, j, k),
                                                              (*state[2])(i, j, k),
                                                              Nc_(i, j, k),
                                                              dt,
                                                              cloud_rates))
                    throw std::runtime_error("native Milbrandt cloud tendency exceeds the process storage envelope");
                rates[0](i, j, k) = cloud_rates[0];
                rates[6](i, j, k) = cloud_rates[1];
                rates[7](i, j, k) = cloud_rates[2];
                rates[8](i, j, k) = cloud_rates[3];
                for (int f = 0; f < 6; ++f)
                {
                    const double old       = (*moments[f])(i, j, k);
                    rates[8 + f](i, j, k)  = float(std::max(-old / dt, double(rates[8 + f](i, j, k))));
                    const double candidate = old + dt * rates[8 + f](i, j, k);
                    if (!std::isfinite(old) || old < 0 || !std::isfinite(candidate) ||
                        candidate > std::numeric_limits<float>::max() || candidate < -1e-5 * std::max(1.0, old))
                        throw std::runtime_error("native Milbrandt number update is invalid");
                }
                rates[7](i, j, k) =
                    float(thermodynamics::temperature_tendency_to_theta(rates[7](i, j, k), (*state[1])(i, j, k)));
                for (auto& f : rates)
                    if (!std::isfinite(f(i, j, k)))
                        throw std::runtime_error("nonfinite native Milbrandt coupled tendency");
            }
    constexpr std::array<int, 8> destinations{7, 6, 0, 1, 2, 3, 4, 5};
    for (int f = 0; f < 8; ++f)
        *output[f] = std::move(rates[destinations[f]]);
    const std::array<Field3D*, 6> number_rates{&dNc_dt_, &dNr_dt_, &dNi_dt_, &dNs_dt_, &dNg_dt_, &dNh_dt_};
    for (int f = 0; f < 6; ++f)
        *number_rates[f] = std::move(rates[8 + f]);
    surface_sedimentation_sink_rate_ = std::move(surface);
}
} // namespace tmv
