#include "catch2/catch.hpp"
#include "core/runtime/simulation_state.hpp"
#include "numerics/advection/advection_cartesian.hpp"
#include <cmath>
#include <cstring>
#include <omp.h>

TEST_CASE("Native retained flux uses fused rounding with compiler contraction disabled",
          "[retained-flux][arithmetic]")
{
    const int workers = GENERATE(1, 4);
    struct RestoreThreads
    {
        int previous = omp_get_max_threads();
        ~RestoreThreads() { omp_set_num_threads(previous); }
    } restore_threads;
    omp_set_num_threads(workers);
    SimulationStorageState state{{6, 6, 6, 100, 100, 1, 100}};
    state.configuration.stagger_type = StaggerType::CGrid;
    state.grid.geometry.initialize(6, 6, 6, 100, 100, 100,
                                   CoordinateSystem::Cartesian, StaggerType::CGrid);
    const auto view = make_simulation_state_view(state);
    const float density_value = 1.0f + std::ldexp(1.0f, -23);
    const float velocity_value = 1.0f - std::ldexp(1.0f, -23);
    const double dt_large = 1.0 / (1.0 + std::ldexp(1.0, -46));
    const double weight = 1.0 / dt_large;
    const double flux = static_cast<double>(density_value) * velocity_value;
    volatile double separately_rounded = weight * flux;
    const float unfused = static_cast<float>(-1.0 + separately_rounded);
    const float expected = static_cast<float>(std::fma(weight, flux, -1.0));
    REQUIRE(unfused == 0.0f);
    REQUIRE(expected != unfused);
    tmv::CartesianCGridTransportContext context;
    context.begin(view, 6, 6, 6, dt_large);
    context.mass_flux_x.fill(-1.0f);
    context.mass_flux_y.fill(-1.0f);
    context.mass_flux_z.fill(-1.0f);
    Field3D density(6, 6, 6, density_value), velocity(6, 6, 6, velocity_value);
    context.accumulate(view, density, velocity, velocity, velocity, 1.0, nullptr, nullptr,
                       tmv::AcousticMassFluxPrecision::NativeCompensated);
    for (const auto* field : {&context.mass_flux_x, &context.mass_flux_y, &context.mass_flux_z})
    {
        const float actual = (*field)(2, 2, 2);
        CHECK(std::memcmp(&actual, &expected, sizeof(float)) == 0);
    }
    CHECK(context.substep_count() == 1);
    CHECK(context.accumulated_dt() == 1.0);
    CHECK(context.density_begin(2, 2, 2) == density_value);
}
