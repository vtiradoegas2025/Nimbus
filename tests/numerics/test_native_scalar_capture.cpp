#include "catch2/catch.hpp"
#include "compute/backend_runtime.hpp"
#include "compute/bridge/advection.hpp"
#include "core/runtime/simulation_state.hpp"
#include "boundary_conditions/factory.hpp"
#include "dynamics/native_cgrid_boundary.hpp"
#include "numerics/advection/advection_cartesian.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <omp.h>

namespace
{
using Register = tmv::CartesianInterfaceFluxRegister;

void same_register(const Register& actual, const Register& expected, double tolerance = 0)
{
    const auto& a = actual.scalar();
    const auto& e = expected.scalar();
    CHECK(a.contribution_count == e.contribution_count);
    CHECK(a.accumulated_dt_s == e.accumulated_dt_s);
    const std::array<const std::vector<double>*, 4> aa{&a.west, &a.east, &a.south, &a.north};
    const std::array<const std::vector<double>*, 4> ee{&e.west, &e.east, &e.south, &e.north};
    for (int side = 0; side < 4; ++side)
    {
        REQUIRE(aa[side]->size() == ee[side]->size());
        for (std::size_t bin = 0; bin < aa[side]->size(); ++bin)
        {
            if (tolerance == 0)
                CHECK((*aa[side])[bin] == (*ee[side])[bin]);
            else
                CHECK((*aa[side])[bin] == Approx((*ee[side])[bin]).epsilon(0).margin(tolerance));
        }
    }
}

void exercise(const char* backend)
{
    constexpr int          nx = 14, ny = 14, nz = 8;
    constexpr double       dx = 8, dy = 16, dz = 4, dt = 4;
    SimulationStorageState storage{{nx, ny, nz, dx, dz, dt, dy}};
    storage.configuration.coordinate_system = CoordinateSystem::Cartesian;
    storage.configuration.stagger_type      = StaggerType::CGrid;
    storage.grid.geometry.initialize(nx, ny, nz, dx, dz, dy, CoordinateSystem::Cartesian, StaggerType::CGrid);
    auto& f = storage.fields;
    for (auto* field : {&f.u, &f.v, &f.w, &f.rho, &f.theta, &f.qv, &f.qc, &f.qr, &f.qi, &f.qs, &f.qg, &f.qh, &f.p})
        field->resize(nx, ny, nz, 0);
    f.rho.fill(1);
    f.p.fill(100000);
    auto                       view = make_simulation_state_view(storage);
    ComputeBackendRuntimeState runtime;
    runtime.config.backend        = backend;
    runtime.config.allow_fallback = false;
    std::string error;
    INFO(error);
    REQUIRE(initialize_compute_backend_runtime(runtime, CoordinateSystem::Cartesian, error));
    auto* kernels = runtime.backend->advection_kernels();
    REQUIRE(kernels);

    for (bool weno : {false, true})
        for (double transport_dt : {dt, .7})
            for (bool exterior : {false, true})
                for (int ratio : {1, 2, 3})
                    for (float sign : {-1.0f, 1.0f})
                    {
                        CAPTURE(backend, exterior, ratio, sign, transport_dt, weno);
                        f.u.fill(sign * 8);
                        f.v.fill(-sign * 4);
                        f.theta.fill(300);
                        f.qv.fill(0);
                        const tmv::CartesianInterfaceFootprint footprint =
                            exterior ? tmv::CartesianInterfaceFootprint{1, 13, 1, 13, 0, nz}
                                     : tmv::CartesianInterfaceFootprint{3, 9, 3, 9, 1, nz - 1};
                        for (int i = 0; i < nx; ++i)
                            for (int j = 0; j < ny; ++j)
                                for (int k = 0; k < nz; ++k)
                                {
                                    f.theta(i, j, k) += .125f * i + .25f * j;
                                    if (i == footprint.i_begin || i == footprint.i_end - 1 || j == footprint.j_begin ||
                                        j == footprint.j_end - 1)
                                        f.qv(i, j, k) = .01f;
                                }
                        tmv::CartesianCGridTransportContext context;
                        context.begin(view, nx, ny, nz, transport_dt);
                        context.accumulate(view, f.rho, f.u, f.v, f.w, transport_dt);
                        context.complete(view);
                        std::array<Register, 3> actual, expected;
                        for (auto* registers : {&actual, &expected})
                            for (auto& reg : *registers)
                                reg.configure(nx, ny, nz, footprint, ratio, ratio);
                        const std::array<Field3D, 3> original{f.theta, f.qv, f.qc};
                        std::array<Field3D, 3>       reference;
                        for (int s = 0; s < 3; ++s)
                        {
                            tmv::advect_scalar_conservative_3d_cartesian(
                                view,
                                original[s],
                                reference[s],
                                transport_dt,
                                s != 0,
                                &context,
                                nullptr,
                                tmv::ScalarMoistEnergyKind::None,
                                nullptr,
                                weno ? tmv::CartesianFaceReconstruction::WENO5
                                     : tmv::CartesianFaceReconstruction::MonotonicCentered,
                                &expected[s]);
                        }
                        const float*                                inputs[]{f.theta.data(), f.qv.data(), f.qc.data()};
                        float*                                      outputs[]{f.theta.data(), f.qv.data(), f.qc.data()};
                        const float                                 kappas[]{0, 0, 0};
                        const int                                   pd[]{0, 1, 1};
                        Register*                                   registers[]{&actual[0], &actual[1], &actual[2]};
                        ConservativeCartesianCgridTransportDispatch d{};
                        d.u                        = f.u.data();
                        d.v                        = f.v.data();
                        d.w                        = f.w.data();
                        d.rho_data                 = f.rho.data();
                        d.q_in                     = inputs;
                        d.q_out                    = outputs;
                        d.dt_kappas                = kappas;
                        d.positive_definite        = pd;
                        d.n_scalars                = 3;
                        d.nx                       = nx;
                        d.ny                       = ny;
                        d.nz                       = nz;
                        d.inv_dx                   = 1.0f / dx;
                        d.inv_dy                   = 1.0f / dy;
                        d.inv_dz                   = 1.0f / dz;
                        d.inv_dx2                  = d.inv_dx * d.inv_dx;
                        d.inv_dy2                  = d.inv_dy * d.inv_dy;
                        d.inv_dz2                  = d.inv_dz * d.inv_dz;
                        d.dt_step                  = float(transport_dt);
                        d.dt_step_host             = transport_dt;
                        d.rho_floor                = 1e-6f;
                        d.use_supplied_mass_flux   = true;
                        d.density_begin            = context.density_begin.data();
                        d.mass_flux_x              = context.mass_flux_x.data();
                        d.mass_flux_y              = context.mass_flux_y.data();
                        d.mass_flux_z              = context.mass_flux_z.data();
                        d.execution_state          = &view;
                        d.host_interface_registers = registers;
                        d.weno                     = weno;
                        REQUIRE(kernels->dispatch_advect_scalars_conservative_cartesian_cgrid(d));
                        for (int s = 0; s < 3; ++s)
                        {
                            same_register(actual[s], expected[s], s == 0 ? .02 : .0001);
                            double change = 0, scale = 0;
                            for (int i = footprint.i_begin; i < footprint.i_end; ++i)
                                for (int j = footprint.j_begin; j < footprint.j_end; ++j)
                                    for (int k = footprint.k_begin; k < footprint.k_end; ++k)
                                    {
                                        const std::size_t n      = (i * ny + j) * nz + k;
                                        const double      before = original[s].data()[n], after = outputs[s][n];
                                        CHECK(after == Approx(reference[s].data()[n]).margin(s == 0 ? 2e-4 : 1e-7));
                                        if (s != 0)
                                            CHECK(after >= 0);
                                        change += (after - before) * dx * dy * dz;
                                        scale += (std::abs(after) + std::abs(before)) * dx * dy * dz;
                                    }
                            const auto& flux = actual[s].scalar();
                            const auto  sum  = [](const auto& values)
                            { return std::accumulate(values.begin(), values.end(), 0.0); };
                            const double outward = sum(flux.east) - sum(flux.west) + sum(flux.north) - sum(flux.south);
                            CHECK(std::abs(change + outward) <= 4 * std::numeric_limits<float>::epsilon() * scale);
                        }
                        Register unlimited;
                        unlimited.configure(nx, ny, nz, footprint, ratio, ratio);
                        Field3D unlimited_field;
                        tmv::advect_scalar_conservative_3d_cartesian(
                            view,
                            original[1],
                            unlimited_field,
                            transport_dt,
                            false,
                            &context,
                            nullptr,
                            tmv::ScalarMoistEnergyKind::None,
                            nullptr,
                            weno ? tmv::CartesianFaceReconstruction::WENO5
                                 : tmv::CartesianFaceReconstruction::MonotonicCentered,
                            &unlimited);
                        const auto& limited_face = sign > 0 ? actual[1].scalar().east : actual[1].scalar().west;
                        const auto& raw_face     = sign > 0 ? unlimited.scalar().east : unlimited.scalar().west;
                        if (transport_dt == dt)
                            CHECK(std::abs(std::accumulate(limited_face.begin(), limited_face.end(), 0.0)) <
                                  .5 * std::abs(std::accumulate(raw_face.begin(), raw_face.end(), 0.0)));
                        const std::array<Field3D, 3> first{f.theta, f.qv, f.qc};
                        const auto                   first_registers = actual;
                        for (int workers : {1, 4})
                        {
                            omp_set_num_threads(workers);
                            std::memcpy(f.theta.data(), original[0].data(), f.theta.size() * sizeof(float));
                            std::memcpy(f.qv.data(), original[1].data(), f.qv.size() * sizeof(float));
                            for (auto& reg : actual)
                                reg.reset_interval();
                            REQUIRE(kernels->dispatch_advect_scalars_conservative_cartesian_cgrid(d));
                            CHECK(std::memcmp(first[0].data(), f.theta.data(), f.theta.size() * sizeof(float)) == 0);
                            CHECK(std::memcmp(first[1].data(), f.qv.data(), f.qv.size() * sizeof(float)) == 0);
                            same_register(actual[0], first_registers[0]);
                            same_register(actual[1], first_registers[1]);
                            same_register(actual[2], first_registers[2]);
                        }
                        std::memcpy(f.theta.data(), original[0].data(), f.theta.size() * sizeof(float));
                        std::memcpy(f.qv.data(), original[1].data(), f.qv.size() * sizeof(float));
                        d.use_supplied_mass_flux = false;
                        REQUIRE(kernels->dispatch_advect_scalars_conservative_cartesian_cgrid(d));
                        for (int scalar = 0; scalar < 3; ++scalar)
                        {
                            const auto& values = actual[scalar].scalar();
                            CHECK(values.contribution_count == 2);
                            CHECK(values.accumulated_dt_s == 2 * transport_dt);
                            const auto&                                     before = first_registers[scalar].scalar();
                            const std::array<const std::vector<double>*, 4> aa{
                                &values.west, &values.east, &values.south, &values.north};
                            const std::array<const std::vector<double>*, 4> bb{
                                &before.west, &before.east, &before.south, &before.north};
                            for (int side = 0; side < 4; ++side)
                                for (std::size_t bin = 0; bin < aa[side]->size(); ++bin)
                                    CHECK((*aa[side])[bin] == 2 * (*bb[side])[bin]);
                        }
                        actual                   = first_registers;
                        d.use_supplied_mass_flux = true;
                        registers[1]             = registers[0];
                        CHECK_FALSE(kernels->dispatch_advect_scalars_conservative_cartesian_cgrid(d));
                        registers[1] = &actual[1];
                        outputs[1]   = outputs[0] + 1;
                        CHECK_FALSE(kernels->dispatch_advect_scalars_conservative_cartesian_cgrid(d));
                        outputs[1] = f.qv.data();
                        tmv::ScalarBoundaryTransportReport boundary[2];
                        boundary[0].complete = true;
                        tmv::ScalarBoundaryTransportReport*      reports[]{&boundary[0], &boundary[1], nullptr};
                        tmv::ScalarMoistEnergyTransportReportV1  energy;
                        tmv::ScalarMoistEnergyTransportReportV1* energy_reports[]{nullptr, &energy, nullptr};
                        const tmv::ScalarMoistEnergyKind         kinds[]{tmv::ScalarMoistEnergyKind::None,
                                                                         tmv::ScalarMoistEnergyKind::WaterVapor,
                                                                         tmv::ScalarMoistEnergyKind::None};
                        d.host_boundary_reports = reports;
                        d.host_energy_reports   = energy_reports;
                        d.host_energy_kinds     = kinds;
                        f.p(4, 4, 3)            = std::numeric_limits<float>::quiet_NaN();
                        CHECK_THROWS(kernels->dispatch_advect_scalars_conservative_cartesian_cgrid(d));
                        CHECK(boundary[0].complete);
                        CHECK_FALSE(boundary[1].complete);
                        CHECK_FALSE(energy.complete);
                        CHECK(std::memcmp(first[0].data(), f.theta.data(), f.theta.size() * sizeof(float)) == 0);
                        CHECK(std::memcmp(first[1].data(), f.qv.data(), f.qv.size() * sizeof(float)) == 0);
                        same_register(actual[0], first_registers[0]);
                        same_register(actual[1], first_registers[1]);
                        same_register(actual[2], first_registers[2]);
                        f.p(4, 4, 3) = 100000;
                    }
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
                f.qc(i, j, k) = float(20 + i + j) * std::numeric_limits<float>::denorm_min();
    const auto                          initial = f.qc;
    tmv::CartesianCGridTransportContext context;
    context.begin(view, nx, ny, nz, .125);
    context.accumulate(view, f.rho, f.u, f.v, f.w, .125);
    context.complete(view);
    Register interface;
    interface.configure(nx, ny, nz, {1, nx - 1, 1, ny - 1, 0, nz});
    Register*                                   interfaces[]{&interface};
    const float*                                input[]{f.qc.data()};
    float*                                      output[]{f.qc.data()};
    const float                                 kappa[]{0};
    const int                                   positive[]{1};
    tmv::ScalarBoundaryTransportReport          boundary;
    tmv::ScalarBoundaryTransportReport*         boundaries[]{&boundary};
    tmv::ScalarMoistEnergyTransportReportV1     energy;
    tmv::ScalarMoistEnergyTransportReportV1*    energies[]{&energy};
    const tmv::ScalarMoistEnergyKind            kinds[]{tmv::ScalarMoistEnergyKind::LiquidWater};
    ConservativeCartesianCgridTransportDispatch d;
    d.u                        = f.u.data();
    d.v                        = f.v.data();
    d.w                        = f.w.data();
    d.rho_data                 = f.rho.data();
    d.q_in                     = input;
    d.q_out                    = output;
    d.dt_kappas                = kappa;
    d.positive_definite        = positive;
    d.n_scalars                = 1;
    d.nx                       = nx;
    d.ny                       = ny;
    d.nz                       = nz;
    d.inv_dx                   = 1.0f / dx;
    d.inv_dy                   = 1.0f / dy;
    d.inv_dz                   = 1.0f / dz;
    d.inv_dx2                  = d.inv_dx * d.inv_dx;
    d.inv_dy2                  = d.inv_dy * d.inv_dy;
    d.inv_dz2                  = d.inv_dz * d.inv_dz;
    d.dt_step                  = .125f;
    d.dt_step_host             = .125;
    d.rho_floor                = 1e-6f;
    d.use_supplied_mass_flux   = true;
    d.density_begin            = context.density_begin.data();
    d.mass_flux_x              = context.mass_flux_x.data();
    d.mass_flux_y              = context.mass_flux_y.data();
    d.mass_flux_z              = context.mass_flux_z.data();
    d.execution_state          = &view;
    d.host_interface_registers = interfaces;
    d.host_boundary_reports    = boundaries;
    d.host_energy_kinds        = kinds;
    d.host_energy_reports      = energies;
    REQUIRE(kernels->dispatch_advect_scalars_conservative_cartesian_cgrid(d));
    REQUIRE(boundary.complete);
    REQUIRE(energy.complete);
    double before = 0, after = 0, transport = 0, scale = 0;
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz; ++k)
            {
                before += double(context.density_begin(i, j, k)) * initial(i, j, k) * dx * dy * dz;
                after += double(f.rho(i, j, k)) * f.qc(i, j, k) * dx * dy * dz;
            }
    const auto& flux = interface.scalar();
    for (std::size_t n = 0; n < flux.west.size(); ++n)
        transport += flux.east[n] - flux.west[n];
    for (std::size_t n = 0; n < flux.south.size(); ++n)
        transport += flux.north[n] - flux.south[n];
    scale = std::abs(before) + std::abs(after) + std::abs(transport);
    REQUIRE(scale > 0);
    CHECK(std::abs(boundary.outward_boundary_inventory - transport) <=
          256 * std::numeric_limits<double>::epsilon() * scale);
    CHECK(std::abs(after - before + transport - boundary.closure_residual) <=
          256 * std::numeric_limits<double>::epsilon() * scale);
}
} // namespace

TEST_CASE("Metal scalar interfaces capture the applied donor-limited flux atomically", "[metal][native_interface]")
{
    exercise("metal");
}

TEST_CASE("Vulkan scalar interfaces capture the applied donor-limited flux atomically", "[vulkan][native_interface]")
{
    exercise("vulkan");
}

namespace
{
void exercise_east_radiative_coupling(const char* backend)
{
    using namespace tmv;
    using L = init::config::LateralBC;
    constexpr int nx = 10, ny = 9, nz = 7;
    constexpr double dx = 120, dy = 150, dz = 80, dt = .125;
    ComputeBackendRuntimeState runtime;
    runtime.config.backend = backend;
    runtime.config.allow_fallback = false;
    std::string error;
    REQUIRE(initialize_compute_backend_runtime(runtime, CoordinateSystem::Cartesian, error));
    INFO(error);
    REQUIRE_FALSE(runtime.fallback_active);
    auto* boundary = runtime.backend->dynamics_kernels();
    auto* advection = runtime.backend->advection_kernels();
    REQUIRE(boundary);
    REQUIRE(advection);
    std::array<float, 4> east_probe{};
    int selector = 0;
    for (const L east : {L::Radiative, L::ZeroGradient, L::Rigid, L::RigidSlip})
    {
        CAPTURE(backend, selector);
        SimulationStorageState cpu{{nx, ny, nz, dx, dz, dt, dy}};
        cpu.grid.geometry.initialize(nx, ny, nz, dx, dz, dy, CoordinateSystem::Cartesian, StaggerType::CGrid);
        cpu.configuration.coordinate_system = CoordinateSystem::Cartesian;
        cpu.configuration.stagger_type = StaggerType::CGrid;
        cpu.configuration.lateral_bc_west = L::RigidSlip;
        cpu.configuration.lateral_bc_east = east;
        cpu.configuration.lateral_bc_south = L::RigidSlip;
        cpu.configuration.lateral_bc_north = L::RigidSlip;
        cpu.configuration.bottom_bc = init::config::BottomBC::RigidNoSlip;
        cpu.configuration.top_bc = init::config::TopBC::Open;
        fields::visit(cpu.fields, fields::prognostic,
                      [&](const char*, Field3D& field) { field.resize(nx, ny, nz, 0); });
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    const float corner = std::exp(-.4f * ((i - 8) * (i - 8) + (j - 1) * (j - 1)));
                    cpu.fields.u(i, j, k) = 2.0f + .08f * i + .03f * k;
                    cpu.fields.v(i, j, k) = -.4f + .04f * j;
                    cpu.fields.w(i, j, k) = .03f * k;
                    cpu.fields.rho(i, j, k) = 1.1f + .001f * i;
                    cpu.fields.p(i, j, k) = 100000.0f + 3.0f * i + 2.0f * j;
                    cpu.fields.theta(i, j, k) = 300.0f + 1.5f * corner + .02f * k;
                    cpu.fields.qv(i, j, k) = .004f + .001f * corner;
                }
        auto gpu = cpu;
        auto cpu_view = make_simulation_state_view(cpu);
        auto scheme = create_cartesian_cgrid_bc_scheme(cpu_view);
        CartesianCGridAcousticEnergyReportV1 cpu_energy, gpu_energy;
        std::vector<float> first_east_wave;
        for (int step = 0; step < 3; ++step)
        {
            cpu.fields.u(7, 1, 2) += .125f;
            gpu.fields.u(7, 1, 2) += .125f;
            scheme->apply_acoustic_attributed(dt / 6, cpu_energy);
            apply_native_cgrid_acoustic_boundary(gpu, *boundary, dt / 6, &gpu_energy);
            for (const auto pair : {std::pair{&cpu.fields.u, &gpu.fields.u},
                                    std::pair{&cpu.fields.v, &gpu.fields.v},
                                    std::pair{&cpu.fields.w, &gpu.fields.w},
                                    std::pair{&cpu.fields.p, &gpu.fields.p},
                                    std::pair{&cpu.fields.theta, &gpu.fields.theta},
                                    std::pair{&cpu.fields.qv, &gpu.fields.qv}})
                REQUIRE(std::memcmp(pair.first->data(), pair.second->data(),
                                    pair.first->size() * sizeof(float)) == 0);
            CHECK(cpu.runtime.boundary_conditions.radiative_u_east ==
                  gpu.runtime.boundary_conditions.radiative_u_east);
            if (step == 0)
                first_east_wave = cpu.runtime.boundary_conditions.radiative_u_east;
        }
        REQUIRE(cpu.runtime.boundary_conditions.radiative_u_east.size() == std::size_t(ny * nz));
        if (east == L::Radiative)
            CHECK(cpu.runtime.boundary_conditions.radiative_u_east != first_east_wave);
        east_probe[selector++] = cpu.fields.u(nx - 2, 1, 2);

        CartesianCGridTransportContext context;
        context.begin(cpu_view, nx, ny, nz, dt);
        context.accumulate(cpu_view, cpu.fields.rho, cpu.fields.u, cpu.fields.v, cpu.fields.w, dt);
        context.complete(cpu_view);
        const auto rho_begin = cpu.fields.rho;
        for (int i = 1; i < nx - 1; ++i)
            for (int j = 1; j < ny - 1; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    const double fx = context.mass_flux_x(i, j, k) - context.mass_flux_x(i - 1, j, k);
                    const double fy = context.mass_flux_y(i, j, k) - context.mass_flux_y(i, j - 1, k);
                    const double fz = context.mass_flux_z(i, j, k) -
                        (k > 0 ? context.mass_flux_z(i, j, k - 1) : 0.0);
                    cpu.fields.rho(i, j, k) = float(rho_begin(i, j, k) -
                        dt * (fx / dx + fy / dy + fz / dz));
                }
        gpu.fields.rho = cpu.fields.rho;
        const Field3D theta_in = cpu.fields.theta, qv_in = cpu.fields.qv;
        Field3D theta_expected, qv_expected;
        ScalarBoundaryTransportReport theta_boundary, qv_boundary;
        Register theta_flux, qv_flux, native_theta_flux, native_qv_flux;
        for (auto* flux : {&theta_flux, &qv_flux, &native_theta_flux, &native_qv_flux})
            flux->configure(nx, ny, nz, {1, nx - 1, 1, ny - 1, 0, nz});
        advect_scalar_conservative_3d_cartesian(cpu_view, theta_in, theta_expected, dt, false,
            &context, &theta_boundary, ScalarMoistEnergyKind::None, nullptr,
            CartesianFaceReconstruction::MonotonicCentered, &theta_flux);
        advect_scalar_conservative_3d_cartesian(cpu_view, qv_in, qv_expected, dt, true,
            &context, &qv_boundary, ScalarMoistEnergyKind::None, nullptr,
            CartesianFaceReconstruction::MonotonicCentered, &qv_flux);
        auto gpu_view = make_simulation_state_view(gpu);
        const float* inputs[]{gpu.fields.theta.data(), gpu.fields.qv.data()};
        float* outputs[]{gpu.fields.theta.data(), gpu.fields.qv.data()};
        const float kappas[]{0, 0};
        const int positive[]{0, 1};
        Register* registers[]{&native_theta_flux, &native_qv_flux};
        ScalarBoundaryTransportReport native_theta_boundary, native_qv_boundary;
        ScalarBoundaryTransportReport* reports[]{&native_theta_boundary, &native_qv_boundary};
        ConservativeCartesianCgridTransportDispatch dispatch{};
        dispatch.execution_state = &gpu_view;
        dispatch.u = gpu.fields.u.data();
        dispatch.v = gpu.fields.v.data();
        dispatch.w = gpu.fields.w.data();
        dispatch.rho_data = gpu.fields.rho.data();
        dispatch.q_in = inputs;
        dispatch.q_out = outputs;
        dispatch.dt_kappas = kappas;
        dispatch.positive_definite = positive;
        dispatch.n_scalars = 2;
        dispatch.nx = nx;
        dispatch.ny = ny;
        dispatch.nz = nz;
        dispatch.inv_dx = float(1 / dx);
        dispatch.inv_dy = float(1 / dy);
        dispatch.inv_dz = float(1 / dz);
        dispatch.inv_dx2 = dispatch.inv_dx * dispatch.inv_dx;
        dispatch.inv_dy2 = dispatch.inv_dy * dispatch.inv_dy;
        dispatch.inv_dz2 = dispatch.inv_dz * dispatch.inv_dz;
        dispatch.dt_step = float(dt);
        dispatch.dt_step_host = dt;
        dispatch.rho_floor = 1e-6f;
        dispatch.use_supplied_mass_flux = true;
        dispatch.density_begin = context.density_begin.data();
        dispatch.mass_flux_x = context.mass_flux_x.data();
        dispatch.mass_flux_y = context.mass_flux_y.data();
        dispatch.mass_flux_z = context.mass_flux_z.data();
        dispatch.host_interface_registers = registers;
        dispatch.host_boundary_reports = reports;
        REQUIRE(advection->dispatch_advect_scalars_conservative_cartesian_cgrid(dispatch));
        REQUIRE(theta_boundary.complete);
        REQUIRE(qv_boundary.complete);
        REQUIRE(native_theta_boundary.complete);
        REQUIRE(native_qv_boundary.complete);
        same_register(native_theta_flux, theta_flux, .05);
        same_register(native_qv_flux, qv_flux, .0001);
        const std::array<const ScalarBoundaryTransportReport*, 2> cpu_reports{
            &theta_boundary, &qv_boundary};
        const std::array<const ScalarBoundaryTransportReport*, 2> native_reports{
            &native_theta_boundary, &native_qv_boundary};
        const std::array<const Field3D*, 2> initial_scalars{&theta_in, &qv_in};
        for (std::size_t scalar = 0; scalar < initial_scalars.size(); ++scalar)
        {
            double physical_inventory = 0;
            for (int i = 1; i < nx - 1; ++i)
                for (int j = 1; j < ny - 1; ++j)
                    for (int k = 0; k < nz; ++k)
                    {
                        const auto index = std::size_t((i * ny + j) * nz + k);
                        physical_inventory += dx * dy * dz * std::max({
                            std::abs(double(context.density_begin(i, j, k)) *
                                     (*initial_scalars[scalar])(i, j, k)),
                            std::abs(double(cpu.fields.rho(i, j, k)) *
                                     (scalar == 0 ? theta_expected(i, j, k) : qv_expected(i, j, k))),
                            std::abs(double(gpu.fields.rho(i, j, k)) * outputs[scalar][index])});
                    }
            const auto reports_pair = std::pair{cpu_reports[scalar], native_reports[scalar]};
            // Match the conserved-inventory gamma_256 report contract in
            // test_advect_conservative_cartesian_cgrid_gpu_parity.cpp.
            const double rounded_operations = 256.0 * std::numeric_limits<float>::epsilon();
            REQUIRE(rounded_operations < 1);
            const double limit = rounded_operations / (1.0 - rounded_operations) *
                std::max(physical_inventory, 1.0);
            CAPTURE(scalar, physical_inventory, limit);
            for (const auto* report : {reports_pair.first, reports_pair.second})
            {
                const double scale = std::max({1.0, std::abs(report->committed_state_change),
                    std::abs(report->flux_implied_state_change), std::abs(report->closure_residual)});
                CHECK(std::abs(report->committed_state_change -
                    report->flux_implied_state_change - report->closure_residual) <=
                    32 * std::numeric_limits<double>::epsilon() * scale);
            }
            const double committed_difference = std::abs(reports_pair.second->committed_state_change -
                reports_pair.first->committed_state_change);
            std::cout << "[east-radiative-report] selector=" << selector - 1
                      << " scalar=" << scalar << " inventory=" << physical_inventory
                      << " committed_abs_error=" << committed_difference
                      << " gamma256_limit=" << limit << '\n';
            CHECK(std::abs(reports_pair.second->outward_boundary_inventory -
                           reports_pair.first->outward_boundary_inventory) <= limit);
            CHECK(std::abs(reports_pair.second->committed_state_change -
                           reports_pair.first->committed_state_change) <= limit);
            CHECK(std::abs(reports_pair.second->closure_residual) <= limit);
        }
        for (std::size_t n = 0; n < theta_expected.size(); ++n)
        {
            CHECK(std::abs(double(outputs[0][n] - theta_expected.data()[n])) <= 2e-4);
            CHECK(std::abs(double(outputs[1][n] - qv_expected.data()[n])) <= 2e-7);
        }
        const auto& east_theta = native_theta_flux.scalar().east;
        const auto east_outflow = std::accumulate(east_theta.begin(), east_theta.end(), 0.0);
        const auto south_outflow = std::accumulate(native_theta_flux.scalar().south.begin(),
                                                  native_theta_flux.scalar().south.end(), 0.0);
        if (east == L::Radiative || east == L::ZeroGradient)
            CHECK(std::abs(east_outflow) > 1);
        else
            CHECK(std::abs(east_outflow) <= .05);
        CHECK(std::abs(south_outflow) <= .05);
        const auto saved_theta = gpu.fields.theta;
        const auto saved_qv = gpu.fields.qv;
        const auto saved_wave = gpu.runtime.boundary_conditions.radiative_u_east;
        gpu.fields.p(4, 4, 3) = std::numeric_limits<float>::quiet_NaN();
        ScalarMoistEnergyTransportReportV1 late_energy;
        ScalarMoistEnergyTransportReportV1* energy_reports[]{nullptr, &late_energy};
        const ScalarMoistEnergyKind kinds[]{ScalarMoistEnergyKind::None, ScalarMoistEnergyKind::WaterVapor};
        dispatch.host_energy_reports = energy_reports;
        dispatch.host_energy_kinds = kinds;
        CHECK_THROWS(advection->dispatch_advect_scalars_conservative_cartesian_cgrid(dispatch));
        CHECK(std::memcmp(saved_theta.data(), gpu.fields.theta.data(), saved_theta.size() * sizeof(float)) == 0);
        CHECK(std::memcmp(saved_qv.data(), gpu.fields.qv.data(), saved_qv.size() * sizeof(float)) == 0);
        CHECK(saved_wave == gpu.runtime.boundary_conditions.radiative_u_east);
        CHECK_FALSE(late_energy.complete);
    }
    CHECK(east_probe[0] != east_probe[1]);
    CHECK(east_probe[0] != east_probe[2]);
}
}

TEST_CASE("Metal east-radiative corner couples native acoustic history to supplied-flux TVD atomically",
          "[metal][native_interface][east_radiative]")
{
    exercise_east_radiative_coupling("metal");
}

namespace
{
void exercise_population_support(const char* backend)
{
    constexpr int          nx = 14, ny = 6, nz = 4;
    constexpr double       dt = .125;
    SimulationStorageState storage{{nx, ny, nz, 1, 1, dt, 1}};
    storage.configuration.coordinate_system = CoordinateSystem::Cartesian;
    storage.configuration.stagger_type      = StaggerType::CGrid;
    storage.grid.geometry.initialize(nx, ny, nz, 1, 1, 1, CoordinateSystem::Cartesian, StaggerType::CGrid);
    auto& f = storage.fields;
    for (auto* field : {&f.u, &f.v, &f.w, &f.rho, &f.theta, &f.qv, &f.qc, &f.qr, &f.qi, &f.qs, &f.qg, &f.qh, &f.p})
        field->resize(nx, ny, nz, 0);
    f.rho.fill(1);
    f.u.fill(-1);
    auto                                view = make_simulation_state_view(storage);
    tmv::CartesianCGridTransportContext context;
    context.begin(view, nx, ny, nz, dt);
    context.accumulate(view, f.rho, f.u, f.v, f.w, dt);
    context.complete(view);
    ComputeBackendRuntimeState runtime;
    runtime.config.backend        = backend;
    runtime.config.allow_fallback = false;
    std::string error;
    REQUIRE(initialize_compute_backend_runtime(runtime, CoordinateSystem::Cartesian, error));
    auto* kernels = runtime.backend->advection_kernels();
    REQUIRE(kernels);
    for (bool weno : {false, true})
        for (int scenario : {0, 1, 2})
            for (float scale : {1e-30f, 1e-10f, 1.0f, 1e10f, 1e30f})
            {
                const bool   front = scenario != 0;
                const double step  = scenario == 2 ? 1.0 : dt;
                context.begin(view, nx, ny, nz, step);
                context.accumulate(view, f.rho, f.u, f.v, f.w, step);
                context.complete(view);
                CAPTURE(backend, weno, scenario, scale);
                Field3D source(nx, ny, nz, 0), reference, candidate(nx, ny, nz, -1);
                for (int i = 0; i < nx; ++i)
                    for (int j = 0; j < ny; ++j)
                        for (int k = 0; k < nz; ++k)
                            source(i, j, k) =
                                scale * (front ? (i < 5    ? 0.0f
                                                  : i == 5 ? 1.0f
                                                           : 4.0f)
                                               : float(1.5 + .4 * std::sin(.7 * i) + .2 * std::cos(.4 * j) + .1 * k));
                if (scenario == 2)
                    for (int i = 0; i < nx; ++i)
                        for (int j = 0; j < ny; ++j)
                            for (int k = 0; k < nz; ++k)
                                source(i, j, k) = i == 5 ? scale : 0;
                tmv::ScalarBoundaryTransportReport expected_report, report;
                const auto reconstruction = weno ? tmv::CartesianFaceReconstruction::WENO5PositiveSupport
                                                 : tmv::CartesianFaceReconstruction::MonotonicCenteredPositiveSupport;
                tmv::advect_scalar_conservative_3d_cartesian(view,
                                                             source,
                                                             reference,
                                                             step,
                                                             true,
                                                             &context,
                                                             &expected_report,
                                                             tmv::ScalarMoistEnergyKind::None,
                                                             nullptr,
                                                             reconstruction);
                const float*                                inputs[]{source.data()};
                float*                                      outputs[]{candidate.data()};
                const float                                 kappas[]{0};
                const int                                   positive[]{1};
                tmv::ScalarBoundaryTransportReport*         reports[]{&report};
                ConservativeCartesianCgridTransportDispatch d{};
                d.u                 = f.u.data();
                d.v                 = f.v.data();
                d.w                 = f.w.data();
                d.rho_data          = f.rho.data();
                d.q_in              = inputs;
                d.q_out             = outputs;
                d.dt_kappas         = kappas;
                d.positive_definite = positive;
                d.n_scalars         = 1;
                d.nx                = nx;
                d.ny                = ny;
                d.nz                = nz;
                d.inv_dx = d.inv_dy = d.inv_dz = d.inv_dx2 = d.inv_dy2 = d.inv_dz2 = 1;
                d.dt_step                                                          = float(step);
                d.dt_step_host                                                     = step;
                d.rho_floor                                                        = 1e-6f;
                d.use_supplied_mass_flux                                           = true;
                d.density_begin                                                    = context.density_begin.data();
                d.mass_flux_x                                                      = context.mass_flux_x.data();
                d.mass_flux_y                                                      = context.mass_flux_y.data();
                d.mass_flux_z                                                      = context.mass_flux_z.data();
                d.execution_state                                                  = &view;
                d.host_boundary_reports                                            = reports;
                d.weno                                                             = weno;
                d.preserve_positive_support                                        = true;
                REQUIRE(kernels->dispatch_advect_scalars_conservative_cartesian_cgrid(d));
                REQUIRE(report.complete);
                for (int i = 1; i < nx - 1; ++i)
                    for (int j = 1; j < ny - 1; ++j)
                        for (int k = 0; k < nz; ++k)
                        {
                            CHECK(std::isfinite(candidate(i, j, k)));
                            CHECK(candidate(i, j, k) >= 0);
                            CHECK(std::abs(double(candidate(i, j, k)) - reference(i, j, k)) <= 3e-6 * scale);
                        }
                CHECK(std::abs(report.flux_implied_state_change - expected_report.flux_implied_state_change) <=
                      1e-4 * scale);
                if (front)
                {
                    CHECK(candidate(4, 2, 1) > 0);
                    if (!weno && scenario == 1)
                        CHECK(std::abs(double(candidate(4, 2, 1)) - .125 * scale) <= 1e-7 * scale);
                    if (!weno && scenario == 2)
                    {
                        CHECK(candidate(5, 2, 1) > 0);
                        CHECK(candidate(5, 2, 1) < 1e-4 * scale);
                        CHECK(std::abs(double(candidate(4, 2, 1)) + candidate(5, 2, 1) - scale) < 2e-7 * scale);
                    }
                }
                const auto retained = candidate;
                d.positive_definite = nullptr;
                CHECK_FALSE(kernels->dispatch_advect_scalars_conservative_cartesian_cgrid(d));
                CHECK(std::memcmp(candidate.data(), retained.data(), candidate.size() * sizeof(float)) == 0);
            }
}
} // namespace

TEST_CASE("Metal particle transport retains wet-front support and scaled WENO weights", "[metal][particle-support]")
{
    exercise_population_support("metal");
}

TEST_CASE("Vulkan particle transport retains wet-front support and scaled WENO weights", "[vulkan][particle-support]")
{
    exercise_population_support("vulkan");
}

TEST_CASE("Particle-support batches reject a late signed carrier before publication", "[particle-support][cpu][atomic]")
{
    constexpr int          nx = 6, ny = 5, nz = 3;
    SimulationStorageState storage{{nx, ny, nz, 1, 1, .25, 1}};
    storage.configuration.coordinate_system = CoordinateSystem::Cartesian;
    storage.configuration.stagger_type      = StaggerType::CGrid;
    storage.grid.geometry.initialize(nx, ny, nz, 1, 1, 1, CoordinateSystem::Cartesian, StaggerType::CGrid);
    storage.fields.rho                       = Field3D(nx, ny, nz, 1);
    storage.fields.u                         = Field3D(nx, ny, nz, 1);
    storage.fields.v                         = Field3D(nx, ny, nz, 0);
    storage.fields.w                         = Field3D(nx, ny, nz, 0);
    auto                                view = make_simulation_state_view(storage);
    tmv::CartesianCGridTransportContext context;
    context.begin(view, nx, ny, nz, .25);
    context.accumulate(view, storage.fields.rho, storage.fields.u, storage.fields.v, storage.fields.w, .25);
    context.complete(view);
    Field3D first(nx, ny, nz, 0), second(nx, ny, nz, 3);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
                first(i, j, k) = float(1 + i + 2 * j + 3 * k);
    const auto                    retained = first;
    tmv::CartesianScalarTransport fields[]{{&first, 0, true}, {&second, 0, false}};
    CHECK_THROWS_AS(
        tmv::advect_scalars_conservative_3d_cartesian(
            view, fields, 2, .25, &context, tmv::CartesianFaceReconstruction::MonotonicCenteredPositiveSupport),
        std::invalid_argument);
    CHECK(std::memcmp(first.data(), retained.data(), first.size() * sizeof(float)) == 0);
}

namespace
{
void exercise_mapped_update(const char* backend, bool positive)
{
    ComputeBackendRuntimeState runtime;
    runtime.config.backend        = backend;
    runtime.config.allow_fallback = false;
    std::string error;
    REQUIRE(initialize_compute_backend_runtime(runtime, CoordinateSystem::Cartesian, error));
    auto* kernels = runtime.backend->terrain_kernels();
    REQUIRE(kernels);
    REQUIRE(kernels->supports_mapped_scalar_transport());
    constexpr int nx = 8, ny = 7, nz = 4;
    Field3D       scalar(nx, ny, nz), density_begin(nx, ny, nz), density_end(nx, ny, nz);
    Field3D       jacobian(nx, ny, nz, 1), output(nx, ny, nz, -777);
    Field3D       flux_x(nx, ny, nz, 0), flux_y(nx, ny, nz, 0), flux_z(nx, ny, nz, 0);
    struct Rates
    {
        Field3D gain;
        Field3D loss;
        bool    complete = false;
    };
    Rates rates{Field3D(nx, ny, nz, 0), Field3D(nx, ny, nz, 0)};
    for (bool weno : {false, true})
        for (int scenario = 0; scenario < (positive ? 9 : 7); ++scenario)
        {
            CAPTURE(backend, positive, weno, scenario);
            flux_x.fill(0);
            flux_y.fill(0);
            flux_z.fill(0);
            rates.complete = false;
            for (int i = 0; i < nx; ++i)
                for (int j = 0; j < ny; ++j)
                    for (int k = 0; k < nz; ++k)
                    {
                        scalar(i, j, k) = positive ? .0005985849420540035f
                                                   : (i % 2 ? 1.0f : -1.0f) * (305.7652893066406f + .125f * (j + k));
                        density_begin(i, j, k) = .9460299015045166f;
                        density_end(i, j, k)   = density_begin(i, j, k);
                    }
            if (scenario == 1)
            {
                scalar.fill(1);
                density_begin.fill(1e-30f);
                density_end.fill(1);
            }
            if (scenario == 2)
            {
                scalar.fill(std::numeric_limits<float>::max());
                density_begin.fill(1);
                density_end.fill(1.0090323686599731f);
            }
            if (scenario == 3 || scenario == 4)
            {
                scalar.fill(scenario == 3 ? 1e30f : 1e-30f);
                density_begin.fill(scalar(0, 0, 0));
                density_end = density_begin;
            }
            if (scenario == 5)
                density_end.fill(.9463298916816711f);
            if (scenario == 6)
                density_end.fill(1e-7f);
            if (scenario == 7)
            {
                flux_x.fill(.75f);
                flux_y.fill(-.3f);
                flux_z.fill(.125f);
                density_end.fill(.9463298916816711f);
            }
            if (scenario == 8)
            {
                scalar.fill(0);
                for (int j = 0; j < ny; ++j)
                    for (int k = 0; k < nz; ++k)
                        scalar(4, j, k) = .0005985868629068136f;
                flux_x.fill(128);
            }
            MappedTerrainScalarTransportReport   report;
            MappedTerrainScalarTransportDispatch d;
            d.scalar_begin              = scalar.data();
            d.density_begin             = density_begin.data();
            d.density_end               = density_end.data();
            d.mass_flux_x               = flux_x.data();
            d.mass_flux_y               = flux_y.data();
            d.mass_flux_zeta            = flux_z.data();
            d.jacobian                  = jacobian.data();
            d.scalar_end                = output.data();
            d.nx                        = nx;
            d.ny                        = ny;
            d.nz                        = nz;
            d.dx                        = 8;
            d.dy                        = 16;
            d.dzeta                     = 4;
            d.dt                        = .125f;
            d.density_floor             = scenario == 6 ? 1e-6f : 1e-35f;
            d.weno                      = weno;
            d.positive_definite         = positive;
            d.preserve_positive_support = positive;
            d.applied_observer_context  = &rates;
            d.applied_observer          = [](void* context,
                                             const float*,
                                             const float*,
                                             const float*,
                                             const float*,
                                             const float* gain,
                                             const float* loss)
            {
                auto& captured = *static_cast<Rates*>(context);
                std::copy_n(gain, captured.gain.size(), captured.gain.data());
                std::copy_n(loss, captured.loss.size(), captured.loss.data());
                captured.complete = true;
            };
            d.report = &report;
            REQUIRE(kernels->dispatch_mapped_scalar_transport(d));
            REQUIRE(report.complete);
            REQUIRE(rates.complete);
            for (int i = 0; i < nx; ++i)
                for (int j = 0; j < ny; ++j)
                    for (int k = 0; k < nz; ++k)
                    {
                        if (positive)
                            CHECK(output(i, j, k) >= 0);
                        const bool boundary = i == 0 || i == nx - 1 || j == 0 || j == ny - 1;
                        if (boundary || scenario == 0 || scenario == 3 || scenario == 4)
                            CHECK(output(i, j, k) == scalar(i, j, k));
                        else
                        {
                            const double retained = positive ? (double(density_begin(i, j, k)) * scalar(i, j, k) -
                                                                double(d.dt) * rates.loss(i, j, k)) +
                                                                   double(d.dt) * rates.gain(i, j, k)
                                                             : double(density_begin(i, j, k)) * scalar(i, j, k);
                            const float  expected = float(retained / std::max(density_end(i, j, k), d.density_floor));
                            const double quantum  = std::abs(double(std::nextafter(expected, 0.0f)) - expected);
                            CHECK(std::abs(double(output(i, j, k)) - expected) <= quantum);
                        }
                    }
        }
}
} // namespace

TEST_CASE("Metal mapped signed transport preserves identity and density normalization", "[metal][mapped-signed]")
{
    exercise_mapped_update("metal", false);
}

TEST_CASE("Vulkan mapped signed transport preserves identity and density normalization", "[vulkan][mapped-signed]")
{
    exercise_mapped_update("vulkan", false);
}

TEST_CASE("Metal mapped positive transport retains the reported mass budget", "[metal][mapped-positive]")
{
    exercise_mapped_update("metal", true);
}

TEST_CASE("Vulkan mapped positive transport retains the reported mass budget", "[vulkan][mapped-positive]")
{
    exercise_mapped_update("vulkan", true);
}
