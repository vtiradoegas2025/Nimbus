#pragma once

#include "boundary_layer/boundary_layer_base.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

struct MynnBatchColumnFixture
{
    std::vector<double> u;
    std::vector<double> v;
    std::vector<double> theta;
    std::vector<double> qv;
    std::vector<double> p;
    std::vector<double> rho;
    std::vector<double> z_interface;
    std::vector<double> tke;
};

struct MynnBatchFixture
{
    BoundaryLayerConfig                       config;
    SurfaceConfig                             surface;
    std::vector<MynnBatchColumnFixture>       storage;
    std::vector<BoundaryLayerColumnWorkspace> workspaces;
    std::vector<BoundaryLayerColumnStateView> views;
};

inline MynnBatchFixture make_mynn_batch_fixture(std::size_t column_count = 3, std::size_t levels = 10)
{
    MynnBatchFixture fixture;
    fixture.config.dt_pbl               = 30.0;
    fixture.config.pbl_max_height       = 1800.0;
    fixture.config.apply_surface_fluxes = true;
    fixture.surface.z0m                 = 0.1;
    fixture.surface.z0h                 = 0.01;
    fixture.storage.resize(column_count);
    fixture.workspaces.resize(column_count);
    fixture.views.resize(column_count);

    for (std::size_t column = 0; column < column_count; ++column)
    {
        auto& data = fixture.storage[column];
        data.u.resize(levels);
        data.v.resize(levels);
        data.theta.resize(levels);
        data.qv.resize(levels);
        data.p.resize(levels);
        data.rho.resize(levels);
        data.tke.resize(levels);
        data.z_interface.resize(levels + 1);

        for (std::size_t interface_index = 0; interface_index <= levels; ++interface_index)
        {
            data.z_interface[interface_index] = 100.0 * static_cast<double>(interface_index);
        }
        for (std::size_t level = 0; level < levels; ++level)
        {
            const double z_center = 0.5 * (data.z_interface[level] + data.z_interface[level + 1]);
            data.u[level] =
                7.0 + static_cast<double>(column) + 0.4 * static_cast<double>(level) * static_cast<double>(column + 1);
            data.v[level]      = 0.15 * static_cast<double>(column) - 0.05 * static_cast<double>(level);
            const double lapse = column == 1 ? -0.001 : 0.008;
            data.theta[level]  = 299.0 + static_cast<double>(column) + lapse * z_center;
            data.qv[level]     = 0.010 - 2.0e-6 * z_center + 2.0e-4 * static_cast<double>(column);
            data.p[level]      = 100000.0 * std::exp(-z_center / 8000.0);
            data.rho[level]    = 1.15 * std::exp(-z_center / 9000.0);
            data.tke[level]    = 0.08 + 0.015 * static_cast<double>((level + column) % 4);
        }

        auto& view     = fixture.views[column];
        view.u         = &data.u;
        view.v         = &data.v;
        view.theta     = &data.theta;
        view.qv        = &data.qv;
        view.p         = &data.p;
        view.rho       = &data.rho;
        view.z_int     = &data.z_interface;
        view.tke       = &data.tke;
        view.u_sfc     = 5.0 + static_cast<double>(column);
        view.v_sfc     = 0.3 * static_cast<double>(column);
        view.theta_sfc = 304.0;
        view.qv_sfc    = 0.012;
        view.z_sfc     = 10.0;
        view.workspace = &fixture.workspaces[column];
    }

    return fixture;
}
