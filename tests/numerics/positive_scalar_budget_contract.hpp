#pragma once

#include "compute/cartesian_scalar_capture.hpp"
#include "core/runtime/simulation_state.hpp"
#include <cstdint>
#include <cstring>

#include "compute/compute_backend.hpp"
#include "core/infra/physical_constants.hpp"
#include "core/runtime/simulation.hpp"
#include <iomanip>
#include <iostream>

namespace nimbus::testing
{
inline void exercise_positive_scalar_contract(ComputeBackend& backend)
{
    auto* kernels = backend.advection_kernels();
    REQUIRE(kernels != nullptr);
    constexpr int          nx = 5, ny = 3, nz = 1;
    constexpr double       dx = 500, dy = 750, dz = 250, dt = 2;
    SimulationStorageState storage{{nx, ny, nz, dx, dz, dt, dy}};
    storage.grid.geometry.initialize(nx, ny, nz, dx, dz, dy, CoordinateSystem::Cartesian, StaggerType::CGrid);
    storage.configuration.coordinate_system = CoordinateSystem::Cartesian;
    storage.configuration.stagger_type      = StaggerType::CGrid;
    auto& f                                 = storage.fields;
    f.rho.resize(nx, ny, nz, 1);
    f.u.resize(nx, ny, nz, 0);
    f.v.resize(nx, ny, nz, 0);
    f.w.resize(nx, ny, nz, 0);
    f.theta.resize(nx, ny, nz, 300);
    f.p.resize(nx, ny, nz, 100000);
    for (auto* water : {&f.qv, &f.qc, &f.qr, &f.qi, &f.qs, &f.qg, &f.qh, &f.tke})
        water->resize(nx, ny, nz, 0);
    f.theta(1, 1, 0)     = 280;
    f.u(1, 1, 0)         = 1;
    const auto state     = make_simulation_state_view(storage);
    Field3D    rho_begin = f.rho, mfx(nx, ny, nz, 0), zero(nx, ny, nz, 0);
    mfx(1, 1, 0)         = 1;
    f.rho(1, 1, 0)       = float(1 - dt / dx);
    f.rho(2, 1, 0)       = float(1 + dt / dx);
    auto&   input        = f.qv;
    Field3D output(nx, ny, nz, 0);
    input(1, 1, 0) = std::numeric_limits<float>::min();
    const float*                                inputs[]{input.data()};
    float*                                      outputs[]{output.data()};
    float                                       kappas[]{0};
    int                                         pd[]{1};
    tmv::ScalarBoundaryTransportReport          boundary;
    tmv::ScalarBoundaryTransportReport*         boundaries[]{&boundary};
    tmv::ScalarMoistEnergyTransportReportV1     energy;
    tmv::ScalarMoistEnergyTransportReportV1*    energies[]{&energy};
    tmv::ScalarMoistEnergyKind                  kinds[]{tmv::ScalarMoistEnergyKind::WaterVapor};
    ConservativeCartesianCgridTransportDispatch d{};
    d.execution_state        = &state;
    d.u                      = f.u.data();
    d.v                      = f.v.data();
    d.w                      = f.w.data();
    d.rho_data               = f.rho.data();
    d.q_in                   = inputs;
    d.q_out                  = outputs;
    d.dt_kappas              = kappas;
    d.positive_definite      = pd;
    d.n_scalars              = 1;
    d.nx                     = nx;
    d.ny                     = ny;
    d.nz                     = nz;
    d.inv_dx                 = float(1 / dx);
    d.inv_dy                 = float(1 / dy);
    d.inv_dz                 = float(1 / dz);
    d.inv_dx2                = d.inv_dx * d.inv_dx;
    d.inv_dy2                = d.inv_dy * d.inv_dy;
    d.inv_dz2                = d.inv_dz * d.inv_dz;
    d.dt_step                = float(dt);
    d.rho_floor              = .05f;
    d.use_supplied_mass_flux = true;
    d.density_begin          = rho_begin.data();
    d.mass_flux_x            = mfx.data();
    d.mass_flux_y            = zero.data();
    d.mass_flux_z            = zero.data();
    d.weno                   = false;
    d.host_boundary_reports  = boundaries;
    d.host_energy_kinds      = kinds;
    d.host_energy_reports    = energies;
    REQUIRE(kernels->dispatch_advect_scalars_conservative_cartesian_cgrid(d));
    REQUIRE(boundary.complete);
    REQUIRE(energy.complete);
    REQUIRE(boundary.outward_boundary_inventory == 0);
    constexpr double cv_v =
        physical_constants::specific_heat_water_vapor_cp_jkgk - physical_constants::gas_constant_water_vapor_jkgk;
    // The independently rounded rate is round(FLT_MIN * float(1/500) / 2^-149) units.
    const double rounded_rate = 16777.0 * double(std::numeric_limits<float>::denorm_min());
    const double expected     = dt * dx * dy * dz * rounded_rate * cv_v * 20.0;
    const double rounding     = 512 * std::numeric_limits<double>::epsilon() * dt * dx * dy * dz * rounded_rate * 3.0e6;
    INFO("expected tiny transfer energy=" << expected << " actual=" << energy.limited_face_state_change_j);
    REQUIRE(energy.limited_face_state_change_j > 0);
    REQUIRE(std::abs(energy.limited_face_state_change_j - expected) <= rounding);
    REQUIRE(std::isfinite(energy.storage_rounding_j));
    for (std::size_t i = 0; i < output.size(); ++i)
        REQUIRE(output.data()[i] >= 0);

    Field3D                            second      = input, first_out(nx, ny, nz, 123), second_out(nx, ny, nz, 456);
    const Field3D                      first_saved = first_out, second_saved = second_out;
    const float*                       batch_inputs[]{input.data(), second.data()};
    float*                             batch_outputs[]{first_out.data(), second_out.data()};
    float                              batch_kappas[]{0, 0};
    int                                batch_pd[]{1, 1};
    tmv::ScalarBoundaryTransportReport b1, b2;
    b1.outward_boundary_inventory = 123;
    b2.outward_boundary_inventory = 456;
    tmv::ScalarBoundaryTransportReport* batch_reports[]{&b1, &b2};
    d.n_scalars                 = 2;
    d.q_in                      = batch_inputs;
    d.q_out                     = batch_outputs;
    d.dt_kappas                 = batch_kappas;
    d.positive_definite         = batch_pd;
    d.host_boundary_reports     = batch_reports;
    d.host_energy_kinds         = nullptr;
    d.host_energy_reports       = nullptr;
    std::uint32_t negative_word = 0x80000001u;
    float         negative;
    std::memcpy(&negative, &negative_word, sizeof(negative));
    for (std::size_t index : {std::size_t(4), std::size_t(0)})
    {
        const float saved    = second.data()[index];
        second.data()[index] = negative;
        REQUIRE_FALSE(kernels->dispatch_advect_scalars_conservative_cartesian_cgrid(d));
        REQUIRE(std::memcmp(first_out.data(), first_saved.data(), first_out.size() * sizeof(float)) == 0);
        REQUIRE(std::memcmp(second_out.data(), second_saved.data(), second_out.size() * sizeof(float)) == 0);
        REQUIRE_FALSE(b1.complete);
        REQUIRE_FALSE(b2.complete);
        REQUIRE(b1.outward_boundary_inventory == 123);
        REQUIRE(b2.outward_boundary_inventory == 456);
        second.data()[index] = saved;
    }
    REQUIRE(kernels->dispatch_advect_scalars_conservative_cartesian_cgrid(d));
    REQUIRE(b1.complete);
    REQUIRE(b2.complete);
    REQUIRE(std::memcmp(first_out.data(), second_out.data(), first_out.size() * sizeof(float)) == 0);

    d.host_boundary_reports = nullptr;
    nimbus::compute::CartesianScalarCapture capture;
    REQUIRE(capture.prepare(d));
    Field3D negative_candidate                                  = input;
    negative_candidate.data()[4]                                = negative;
    const Field3D                                  first_before = first_out, second_before = second_out;
    tmv::FlatCGridDeviceScalarAttributionWorkspace workspace;
    REQUIRE_THROWS(nimbus::compute::publish_cartesian_scalar_batch(
        d,
        capture,
        nullptr,
        input.size(),
        [&](int scalar) { return scalar == 0 ? input.data() : negative_candidate.data(); },
        workspace,
        [](int) -> const float* { return nullptr; },
        [](void* destination, const void* source, std::size_t bytes) { std::memcpy(destination, source, bytes); }));
    REQUIRE(std::memcmp(first_out.data(), first_before.data(), first_out.size() * sizeof(float)) == 0);
    REQUIRE(std::memcmp(second_out.data(), second_before.data(), second_out.size() * sizeof(float)) == 0);

    Field3D      signed_input(nx, ny, nz, -.25f), signed_output(nx, ny, nz, 0);
    const float* signed_inputs[]{signed_input.data()};
    float*       signed_outputs[]{signed_output.data()};
    int          signed_pd[]{0};
    d.n_scalars         = 1;
    d.q_in              = signed_inputs;
    d.q_out             = signed_outputs;
    d.positive_definite = signed_pd;
    d.density_begin     = f.rho.data();
    d.mass_flux_x       = zero.data();
    REQUIRE(kernels->dispatch_advect_scalars_conservative_cartesian_cgrid(d));
    REQUIRE(std::memcmp(signed_input.data(), signed_output.data(), signed_input.size() * sizeof(float)) == 0);
}

inline void exercise_positive_scalar_scales(ComputeBackend& backend)
{
    constexpr int    nx = 18, ny = 18, nz = 18;
    constexpr double dt_step = 2.0;
    NR                       = nx;
    NTH                      = ny;
    NZ                       = nz;
    dr                       = 500.0;
    dtheta                   = 750.0;
    dz                       = 250.0;
    global_stagger_type      = StaggerType::CGrid;
    global_grid_geometry.initialize(nx, ny, nz, dr, dz, dtheta, CoordinateSystem::Cartesian, StaggerType::CGrid);
    SimulationStorageState storage{{nx, ny, nz, dr, dz, dt_step, dtheta}};
    storage.grid.geometry                   = global_grid_geometry;
    storage.configuration.coordinate_system = CoordinateSystem::Cartesian;
    storage.configuration.stagger_type      = StaggerType::CGrid;
    storage.configuration.lateral_bc_west   = tmv::init::config::LateralBC::RigidSlip;
    storage.configuration.lateral_bc_east   = tmv::init::config::LateralBC::RigidSlip;
    storage.configuration.lateral_bc_south  = tmv::init::config::LateralBC::RigidSlip;
    storage.configuration.lateral_bc_north  = tmv::init::config::LateralBC::RigidSlip;
    storage.configuration.bottom_bc         = tmv::init::config::BottomBC::RigidSlip;
    storage.configuration.top_bc            = tmv::init::config::TopBC::Rigid;
    auto& u                                 = storage.fields.u;
    auto& v                                 = storage.fields.v;
    auto& w                                 = storage.fields.w;
    auto& rho                               = storage.fields.rho;
    u.resize(nx, ny, nz, 0.0f);
    v.resize(nx, ny, nz, 0.0f);
    w.resize(nx, ny, nz, 0.0f);
    rho.resize(nx, ny, nz, 0.0f);
    auto  transport_view = make_simulation_state_view(storage);
    auto* kernels        = backend.advection_kernels();
    REQUIRE(kernels != nullptr);
    const auto interior_rho_q_sum = [](const Field3D& density, const Field3D& scalar)
    {
        double sum = 0.0;
        for (int i = 1; i < nx - 1; ++i)
            for (int j = 1; j < ny - 1; ++j)
                for (int k = 0; k < nz; ++k)
                    sum += double(density(i, j, k)) * double(scalar(i, j, k));
        return sum;
    };
    const auto run_metal_batch = [&](const float* const*                   inputs,
                                     float* const*                         outputs,
                                     const float*                          kappas,
                                     int                                   count,
                                     double                                dt,
                                     const int*                            positive,
                                     const CartesianCGridTransportContext* context)
    {
        ConservativeCartesianCgridTransportDispatch d{};
        d.execution_state        = &transport_view;
        d.u                      = u.data();
        d.v                      = v.data();
        d.w                      = w.data();
        d.rho_data               = rho.data();
        d.q_in                   = inputs;
        d.q_out                  = outputs;
        d.dt_kappas              = kappas;
        d.n_scalars              = count;
        d.positive_definite      = positive;
        d.nx                     = nx;
        d.ny                     = ny;
        d.nz                     = nz;
        d.inv_dx                 = float(1.0 / dr);
        d.inv_dy                 = float(1.0 / dtheta);
        d.inv_dz                 = float(1.0 / dz);
        d.inv_dx2                = d.inv_dx * d.inv_dx;
        d.inv_dy2                = d.inv_dy * d.inv_dy;
        d.inv_dz2                = d.inv_dz * d.inv_dz;
        d.dt_step                = float(dt);
        d.rho_floor              = 0.05f;
        d.density_begin          = context->density_begin.data();
        d.mass_flux_x            = context->mass_flux_x.data();
        d.mass_flux_y            = context->mass_flux_y.data();
        d.mass_flux_z            = context->mass_flux_z.data();
        d.use_supplied_mass_flux = true;
        return kernels->dispatch_advect_scalars_conservative_cartesian_cgrid(d);
    };
    constexpr double                   step_dt = 2.0;
    const std::array<std::uint32_t, 9> words{
        0u, 1u, 0x000014fbu, 0x007fffffu, 0x00800000u, 0x01000000u, 0x10000000u, 0x2f800000u, 0x3a83126fu};
    for (bool flowing : {false, true})
        for (std::uint32_t word : words)
        {
            float concentration = 0;
            std::memcpy(&concentration, &word, sizeof(word));
            rho.resize(NR, NTH, NZ, 0.92935240268707275f);
            u.resize(NR, NTH, NZ, 0.0f);
            v.resize(NR, NTH, NZ, 0.0f);
            w.resize(NR, NTH, NZ, 0.0f);
            if (flowing)
                for (int i = 1; i < NR - 2; ++i)
                    for (int j = 1; j < NTH - 1; ++j)
                        for (int k = 0; k < NZ; ++k)
                            u(i, j, k) = float(1.4 * std::sin(6.283185307179586 * double(i) / (NR - 1)));
            const Field3D                  rho_begin = rho;
            CartesianCGridTransportContext context;
            context.begin(transport_view, NR, NTH, NZ, step_dt);
            context.accumulate(transport_view,
                               rho_begin,
                               u,
                               v,
                               w,
                               step_dt,
                               nullptr,
                               nullptr,
                               tmv::AcousticMassFluxPrecision::NativeFloat);
            context.complete(transport_view);
            for (int i = 1; i < NR - 1; ++i)
                for (int j = 1; j < NTH - 1; ++j)
                    for (int k = 0; k < NZ; ++k)
                        rho(i, j, k) = float(
                            double(rho_begin(i, j, k)) -
                            step_dt *
                                ((double(context.mass_flux_x(i, j, k)) - double(context.mass_flux_x(i - 1, j, k))) /
                                     dr +
                                 (double(context.mass_flux_y(i, j, k)) - double(context.mass_flux_y(i, j - 1, k))) /
                                     dtheta +
                                 (double(context.mass_flux_z(i, j, k)) -
                                  (k ? double(context.mass_flux_z(i, j, k - 1)) : 0.0)) /
                                     dz));
            Field3D      input(NR, NTH, NZ, concentration), output = input;
            const double before = interior_rho_q_sum(rho_begin, input);
            const float* ins[]{input.data()};
            float*       outs[]{output.data()};
            const float  kappas[]{0};
            const int    pd[]{1};
            REQUIRE(run_metal_batch(ins, outs, kappas, 1, step_dt, pd, &context));
            double maximum_difference = 0;
            float  minimum = concentration, maximum = concentration;
            for (int i = 1; i < NR - 1; ++i)
                for (int j = 1; j < NTH - 1; ++j)
                    for (int k = 0; k < NZ; ++k)
                    {
                        minimum = std::min(minimum, output(i, j, k));
                        maximum = std::max(maximum, output(i, j, k));
                        maximum_difference =
                            std::max(maximum_difference, std::abs(double(output(i, j, k)) - double(concentration)));
                    }
            const double  after       = interior_rho_q_sum(rho, output);
            std::uint32_t center_word = 0;
            const float   center      = output(6, 4, 3);
            std::memcpy(&center_word, &center, sizeof(center_word));
            INFO("flowing=" << flowing << " input bits=" << word << " center bits=" << center_word << " maximum change="
                            << maximum_difference << " mass before=" << before << " after=" << after);
            REQUIRE(minimum >= 0.0f);
            REQUIRE(std::abs(after - before) / std::max(std::abs(before), 1e-30) < 3.0e-6);
            if (!flowing)
                REQUIRE(std::memcmp(input.data(), output.data(), input.size() * sizeof(float)) == 0);
        }
}
} // namespace nimbus::testing
