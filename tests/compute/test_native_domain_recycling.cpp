#include <catch2/catch.hpp>

#include "compute/backend_runtime.hpp"
#include "core/infra/domain_recycling.hpp"

#include <array>
#include <limits>
#include <string>
#include <vector>

TEST_CASE("Native recycling mixes terrain inflow fields and vertical profiles atomically",
          "[domain_recycling][terrain][metal][vulkan][device]")
{
    const std::string backend_name = GENERATE("metal", "vulkan");
    const int         shift        = GENERATE(-2, 2);
    CAPTURE(backend_name, shift);
    ComputeBackendRuntimeState runtime;
    runtime.config.backend        = backend_name;
    runtime.config.allow_fallback = false;
    std::string error;
    REQUIRE(initialize_compute_backend_runtime(runtime, CoordinateSystem::Cartesian, error));
    REQUIRE_FALSE(runtime.fallback_active);
    auto&                 backend = *runtime.backend;
    constexpr int         nx = 8, ny = 9, nz = 4;
    constexpr std::size_t count = 6;
    using Location              = tmv::domain::HorizontalFieldLocation;
    const auto plan =
        tmv::domain::plan_whole_cell_shift_for_physical_domain(nx, ny, nz, shift, -1, 1, nx - 1, 1, ny - 1);
    std::array<Field3D, count>                                   actual, expected, inflow;
    std::array<std::vector<float>, count>                        profiles;
    std::array<HomogeneousDomainFieldShiftDispatch, count>       dispatches{};
    std::array<tmv::domain::HomogeneousFieldShiftBinding, count> references{};
    std::array<std::string, count>                               names;
    std::array<const char*, count>                               ids{};
    const std::array<Location, 3> locations{Location::ScalarCenter, Location::XFace, Location::YFace};
    for (std::size_t f = 0; f < count; ++f)
    {
        actual[f].resize(nx, ny, nz, 0.0f);
        inflow[f].resize(nx, ny, nz, 0.0f);
        profiles[f].assign(nz, -float(f + 1));
        for (std::size_t index = 0; index < actual[f].size(); ++index)
        {
            actual[f].data()[index] = float(f * 10000 + index);
            inflow[f].data()[index] = -float(f * 10000 + index + 1);
        }
        expected[f]           = actual[f];
        const auto owner      = tmv::domain::plan_whole_cell_shift_for_horizontal_location(plan, locations[f % 3]);
        auto&      d          = dispatches[f];
        d.source              = actual[f].data();
        d.destination         = actual[f].data();
        d.nr                  = nx;
        d.nth                 = ny;
        d.nz                  = nz;
        d.shift_i_cells       = shift;
        d.shift_j_cells       = -1;
        d.destination_i_begin = owner.destination_i_begin;
        d.destination_i_end   = owner.destination_i_end;
        d.destination_j_begin = owner.destination_j_begin;
        d.destination_j_end   = owner.destination_j_end;
        d.source_i_begin      = owner.source_i_begin;
        d.source_i_end        = owner.source_i_end;
        d.source_j_begin      = owner.source_j_begin;
        d.source_j_end        = owner.source_j_end;
        d.leading_profile     = f % 2 ? profiles[f].data() : nullptr;
        d.leading_field       = f % 2 ? nullptr : inflow[f].data();
        names[f]              = "mapped_recycle_" + std::to_string(f);
        ids[f]                = names[f].c_str();
        references[f]         = {
            names[f], &expected[f], f % 2 ? &profiles[f] : nullptr, locations[f % 3], f % 2 ? nullptr : &inflow[f]};
        REQUIRE(backend.dispatch_homogeneous_domain_field_shift(d));
        tmv::domain::execute_homogeneous_field_shift(plan, {references[f]});
        REQUIRE(std::equal(actual[f].data(), actual[f].data() + actual[f].size(), expected[f].data()));
    }
    backend.begin_resident_step();
    const HomogeneousDomainFieldShiftBatchDispatch batch{dispatches.data(), ids.data(), int(count)};
    for (int iteration = 0; iteration < 3; ++iteration)
    {
        CAPTURE(iteration);
        for (std::size_t f = 0; f < count; ++f)
        {
            for (auto& value : profiles[f])
            {
                value -= 10.0f;
            }
            for (std::size_t index = 0; index < inflow[f].size(); ++index)
            {
                inflow[f].data()[index] -= 100.0f;
            }
        }
        REQUIRE(backend.dispatch_homogeneous_domain_field_shift_batch(batch));
        tmv::domain::execute_homogeneous_field_shift(plan, {references.begin(), references.end()});
        for (std::size_t f = 0; f < count; ++f)
        {
            REQUIRE(std::equal(actual[f].data(), actual[f].data() + actual[f].size(), expected[f].data()));
        }
    }
    const auto original = dispatches.back();
    for (int invalid = 0; invalid < 3; ++invalid)
    {
        dispatches.back() = original;
        if (invalid == 0)
        {
            dispatches.back().leading_field = inflow.back().data();
        }
        else if (invalid == 1)
        {
            dispatches.back().leading_profile = nullptr;
        }
        else
        {
            dispatches.back().shift_i_cells = std::numeric_limits<int>::max();
        }
        REQUIRE_FALSE(backend.dispatch_homogeneous_domain_field_shift(dispatches.back()));
        REQUIRE_FALSE(backend.dispatch_homogeneous_domain_field_shift_batch(batch));
        for (std::size_t f = 0; f < count; ++f)
        {
            REQUIRE(std::equal(actual[f].data(), actual[f].data() + actual[f].size(), expected[f].data()));
        }
    }
}
