#include <catch2/catch.hpp>

#include "compute/backend_runtime.hpp"
#include "core/output/output_config.hpp"
#include "core/runtime/process_state.hpp"
#include "core/runtime/runtime_config.hpp"
#include "core/runtime/simulation_state.hpp"

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>

#ifndef NIMBUS_ENABLE_METAL
#error "The injected mixed-open admission target requires the Metal factory call site"
#endif

namespace
{
enum class AdmissionFault
{
    MissingFamily,
    MissingOperation,
    MissingRetainedFlux,
    StorageRange,
    DeviceHeap,
    AvailableBudget,
    Fitting
};

AdmissionFault selected_fault = AdmissionFault::Fitting;
int factory_calls = 0;
int initialized_calls = 0;
int dispatch_calls = 0;
int allocation_boundary_calls = 0;

struct AllocationBoundaryReached : std::runtime_error
{
    AllocationBoundaryReached() : std::runtime_error("injected field-allocation boundary")
    {
    }
};

class AdmissionOnlyMetal final : public ComputeBackend, public AdvectionComputeKernels
{
public:
    std::string name() const override
    {
        return "injected-metal-admission-only";
    }

    bool initialize(std::string&) override
    {
        ++initialized_calls;
        return true;
    }

    void shutdown() override
    {
    }

    ComputeMemoryProperties memory_properties() const override
    {
        ComputeMemoryProperties result;
        result.accelerator = true;
        result.unified_host_device_memory = true;
        result.device_local_heap_bytes = selected_fault == AdmissionFault::DeviceHeap
            ? 1024 * 1024 : 4 * 1024 * 1024;
        result.max_storage_buffer_bytes = selected_fault == AdmissionFault::StorageRange
            ? 18 * 18 * 18 * sizeof(float) - 1 : 1024 * 1024;
        result.device_budget_is_live = selected_fault == AdmissionFault::AvailableBudget;
        result.device_available_budget_bytes = 1024 * 1024;
        return result;
    }

    AdvectionComputeKernels* advection_kernels() override
    {
        return selected_fault == AdmissionFault::MissingFamily ? nullptr : this;
    }

    const AdvectionComputeKernels* advection_kernels() const override
    {
        return selected_fault == AdmissionFault::MissingFamily ? nullptr : this;
    }

    bool supports(AdvectionKernelOp operation) const override
    {
        return selected_fault != AdmissionFault::MissingOperation &&
            operation == AdvectionKernelOp::ConservativeCartesianCgridTransport;
    }

    bool supports_supplied_cartesian_cgrid_mass_flux() const override
    {
        return selected_fault != AdmissionFault::MissingRetainedFlux;
    }

    bool dispatch_advect_scalars_conservative_cartesian_cgrid(
        const ConservativeCartesianCgridTransportDispatch&) override
    {
        ++dispatch_calls;
        throw std::runtime_error("admission-only backend must never execute physics");
    }
};

class RestoreStorage
{
public:
    RestoreStorage() : prior_(std::move(global_simulation_storage))
    {
        global_simulation_storage = SimulationStorageState{};
    }

    ~RestoreStorage()
    {
        shutdown_compute_backend_runtime();
        global_simulation_storage = std::move(prior_);
    }

private:
    SimulationStorageState prior_;
};

void require_unallocated_fields()
{
    const auto& fields = global_simulation_storage.fields;
    const std::array<const Field3D*, 15> primary{
        &fields.rho, &fields.p, &fields.u, &fields.v, &fields.w,
        &fields.theta, &fields.qv, &fields.qc, &fields.qr, &fields.qi,
        &fields.qs, &fields.qg, &fields.qh, &fields.tracer, &fields.tke};
    for (const auto* field : primary)
    {
        CHECK(field->empty());
        CHECK(field->capacity() == 0);
    }
}

void exercise_admission(AdmissionFault fault, bool reject_memory, const char* expected)
{
    RestoreStorage restore;
    selected_fault = fault;
    factory_calls = initialized_calls = dispatch_calls = allocation_boundary_calls = 0;
    const auto root = std::filesystem::temp_directory_path() /
        (std::string("nimbus_mixed_admission_only_") +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    REQUIRE(std::filesystem::create_directory(root));
    std::ifstream maintained("tests/fixtures/reference/catalog/flat_mixed_boundary_metal_exploratory_v1.yaml");
    REQUIRE(maintained.good());
    std::string source(std::istreambuf_iterator<char>{maintained}, {});
    if (reject_memory)
    {
        const std::string original = "    memory_admission: warn\n";
        const auto offset = source.find(original);
        REQUIRE(offset != std::string::npos);
        REQUIRE(source.find(original, offset + original.size()) == std::string::npos);
        source.replace(offset, original.size(), "    memory_admission: reject\n");
    }
    const auto fixture = root / "exact-mixed.yaml";
    std::ofstream(fixture) << source;
    int duration = 1, cadence = 1;
    std::string output_root = (root / "output").string();
    OutputConfig output;
    const auto load = [&]()
    {
        load_config(fixture.string(), duration, cadence, output_root, &output,
                    nullptr, nullptr, -1, -1, nullptr, false, nullptr, nullptr, 1);
    };
    if (expected != nullptr)
    {
        CHECK_THROWS_WITH(load(), expected);
        CHECK(allocation_boundary_calls == 0);
    }
    else
    {
        CHECK_THROWS_AS(load(), AllocationBoundaryReached);
        CHECK(allocation_boundary_calls == 1);
    }
    CHECK(factory_calls == 1);
    CHECK(initialized_calls == 1);
    CHECK(dispatch_calls == 0);
    CHECK_FALSE(std::filesystem::exists(root / "output"));
    CHECK(global_simulation_storage.configuration.admission.native_dry_tvd);
    const auto& config = global_simulation_storage.configuration;
    CHECK(config.lateral_bc_west == tmv::init::config::LateralBC::RigidSlip);
    CHECK(config.lateral_bc_east == tmv::init::config::LateralBC::RigidSlip);
    CHECK(config.lateral_bc_south == tmv::init::config::LateralBC::RigidSlip);
    CHECK(config.lateral_bc_north == tmv::init::config::LateralBC::RigidSlip);
    CHECK(config.bottom_bc == tmv::init::config::BottomBC::RigidNoSlip);
    CHECK(config.top_bc == tmv::init::config::TopBC::Open);
    CHECK(global_simulation_storage.grid.nr == 18);
    CHECK(global_simulation_storage.grid.nth == 18);
    CHECK(global_simulation_storage.grid.nz == 18);
    CHECK(global_simulation_storage.grid.dx == 500);
    CHECK(global_simulation_storage.grid.dy == 500);
    CHECK(global_simulation_storage.grid.dz == 250);
    CHECK(global_simulation_storage.grid.dt == .1);
    CHECK(global_simulation_storage.time_s == 0);
    require_unallocated_fields();
    std::cout << "[mixed-open-admission-only] fault=" << static_cast<int>(fault)
              << " reject_memory=" << reject_memory
              << " allocation_boundary_calls=" << allocation_boundary_calls
              << " field_elements=" << global_simulation_storage.fields.rho.size()
              << " dispatch_calls=" << dispatch_calls << '\n';
}
}

std::unique_ptr<ComputeBackend> create_metal_compute_backend()
{
    ++factory_calls;
    return std::make_unique<AdmissionOnlyMetal>();
}

void resize_fields()
{
    ++allocation_boundary_calls;
    throw AllocationBoundaryReached{};
}

TEST_CASE("Exact mixed-open preallocation rejects unavailable conservative owners",
          "[mixed_open][preallocation][injected_factory]")
{
    for (const auto fault : {AdmissionFault::MissingFamily, AdmissionFault::MissingOperation,
                            AdmissionFault::MissingRetainedFlux})
    {
        exercise_admission(fault, false,
            "native conservative transport requires retained mass-flux ownership");
    }
}

TEST_CASE("Exact mixed-open preallocation rejects undersized storage and device heap",
          "[mixed_open][preallocation][injected_factory]")
{
    for (const bool reject_memory : {false, true})
    {
        exercise_admission(AdmissionFault::StorageRange, reject_memory,
            "requested full field exceeds the active accelerator's maximum storage-buffer range before field allocation");
    }
    exercise_admission(AdmissionFault::DeviceHeap, true,
        "accelerator working-set estimate exceeds the 80% device-local heap planning budget and hardware.memory_admission=reject");
}

TEST_CASE("Exact mixed-open fitting control stops before field allocation or physics",
          "[mixed_open][preallocation][injected_factory]")
{
    exercise_admission(AdmissionFault::Fitting, true, nullptr);
}

TEST_CASE("Live device budget rejects before allocating fields and preserves warning policy",
          "[mixed_open][preallocation][injected_factory]")
{
    exercise_admission(AdmissionFault::AvailableBudget, true,
        "accelerator working-set estimate exceeds 80% of the currently available device-local memory budget; "
        "free GPU memory or reduce the requested working set (hardware.memory_admission=reject)");
    exercise_admission(AdmissionFault::AvailableBudget, false, nullptr);
}
