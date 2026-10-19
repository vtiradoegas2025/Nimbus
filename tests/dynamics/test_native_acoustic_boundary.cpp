#include "catch2/catch.hpp"
#include "compute/backend_runtime.hpp"
#include "core/runtime/simulation_state.hpp"
#include "boundary_conditions/factory.hpp"
#include "dynamics/native_cgrid_boundary.hpp"
#include "dynamics/acoustic_energy_report.hpp"
#include <array>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <random>
#include <omp.h>
namespace boundary_arithmetic
{
using uint = std::uint32_t;
using std::max;
uint floatBitsToUint(float value)
{
    uint bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}
float uintBitsToFloat(uint bits)
{
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}
void umulExtended(uint a, uint b, uint& high, uint& low)
{
    const std::uint64_t product = std::uint64_t(a) * b;
    high = uint(product >> 32);
    low = uint(product);
}
#include "../../vulkan/shaders/compute/dynamics/boundary_float32.glsl"
}
namespace
{
using namespace tmv;
SimulationStorageState initial(int nx, int ny, int nz, float scale = 1)
{
    SimulationStorageState s{{nx, ny, nz, 100, 100, .1, 100}};
    s.grid.geometry.initialize(nx, ny, nz, 100, 100, 100, CoordinateSystem::Cartesian, StaggerType::CGrid);
    s.configuration.stagger_type             = StaggerType::CGrid;
    s.configuration.domain_frame.velocity_ms = {2, -3};
    fields::visit(s.fields, fields::prognostic, [&](const char*, Field3D& field) { field.resize(nx, ny, nz, 0); });
    for (std::size_t n = 0; n < s.fields.u.size(); ++n)
    {
        s.fields.u.data()[n]     = scale * float(int(n % 19) - 9);
        s.fields.v.data()[n]     = scale * float(int(n % 17) - 8);
        s.fields.w.data()[n]     = scale * float(int(n % 13) - 6);
        s.fields.rho.data()[n]   = 1 + .001f * n;
        s.fields.p.data()[n]     = 80000 + float(n);
        s.fields.theta.data()[n] = 300 + .001f * n;
        s.fields.qv.data()[n]    = .005f + .000001f * n;
    }
    s.runtime.boundary_conditions.radiative_u_west  = {19};
    s.runtime.boundary_conditions.radiative_u_east  = {20};
    s.runtime.boundary_conditions.radiative_v_south = {21};
    s.runtime.boundary_conditions.radiative_v_north = {22};
    return s;
}
void equal(const PrognosticFieldStorage& a, const PrognosticFieldStorage& b)
{
    const std::array<const Field3D*, 8> lhs{&a.u, &a.v, &a.w, &a.rho, &a.p, &a.theta, &a.qv, &a.tke};
    const std::array<const Field3D*, 8> rhs{&b.u, &b.v, &b.w, &b.rho, &b.p, &b.theta, &b.qv, &b.tke};
    for (std::size_t f = 0; f < lhs.size(); ++f)
    {
        CAPTURE(f);
        REQUIRE(lhs[f]->size() == rhs[f]->size());
        CHECK(std::memcmp(lhs[f]->data(), rhs[f]->data(), lhs[f]->size() * sizeof(float)) == 0);
    }
}
void equal(const BoundaryConditionWaveState& a, const BoundaryConditionWaveState& b)
{
    CHECK(a.radiative_u_west == b.radiative_u_west);
    CHECK(a.radiative_u_east == b.radiative_u_east);
    CHECK(a.radiative_v_south == b.radiative_v_south);
    CHECK(a.radiative_v_north == b.radiative_v_north);
}
class Reject final : public DynamicsComputeKernels
{
public:
    bool dispatch_cgrid_boundary(const CgridBoundaryDispatch& d) override
    {
        d.output[0][0] = 999;
        return false;
    }
};
void exercise(const char* backend)
{
    ComputeBackendRuntimeState runtime;
    runtime.config.backend        = backend;
    runtime.config.allow_fallback = false;
    std::string error;
    REQUIRE(initialize_compute_backend_runtime(runtime, CoordinateSystem::Cartesian, error));
    INFO(error);
    REQUIRE_FALSE(runtime.fallback_active);
    auto* kernels = runtime.backend->dynamics_kernels();
    REQUIRE(kernels);
    const int threads = omp_get_max_threads();
    omp_set_num_threads(1);
    using L = init::config::LateralBC;
    using B = init::config::BottomBC;
    using T = init::config::TopBC;
    for (int encoded = 0; encoded < 625; ++encoded)
        for (int vertical = 0; vertical < 6; ++vertical)
        {
            CAPTURE(backend, encoded, vertical);
            auto cpu = initial(3, 3, 2), gpu = initial(3, 3, 2);
            int  code                         = encoded;
            cpu.configuration.lateral_bc_west = L(code % 5);
            code /= 5;
            cpu.configuration.lateral_bc_east = L(code % 5);
            code /= 5;
            cpu.configuration.lateral_bc_south = L(code % 5);
            code /= 5;
            cpu.configuration.lateral_bc_north          = L(code % 5);
            cpu.configuration.bottom_bc                 = B(vertical % 2);
            cpu.configuration.top_bc                    = T(vertical / 2);
            gpu.configuration                           = cpu.configuration;
            auto                                 view   = make_simulation_state_view(cpu);
            auto                                 scheme = create_cartesian_cgrid_bc_scheme(view);
            CartesianCGridAcousticEnergyReportV1 cpu_energy, gpu_energy;
            for (int step = 0; step < 3; ++step)
            {
                CAPTURE(step);
                if (step > 0)
                {
                    cpu.fields.u(1, 1, 0) += .25f;
                    gpu.fields.u(1, 1, 0) += .25f;
                }
                scheme->apply_acoustic_attributed(.125, cpu_energy);
                const auto submissions_before = runtime.backend->successful_dispatch_count();
                const auto faces_before = runtime.backend->successful_kernel_dispatch_count(
                    nimbus::compute::KernelId::AcousticBcFaceCartesianCgrid);
                apply_native_cgrid_acoustic_boundary(gpu, *kernels, .125, &gpu_energy);
                CHECK(runtime.backend->successful_dispatch_count() == submissions_before + 1);
                CHECK(runtime.backend->successful_kernel_dispatch_count(
                    nimbus::compute::KernelId::AcousticBcFaceCartesianCgrid) == faces_before + 5);
                const std::array<double, 6> expected{cpu_energy.boundary_committed_work_j,
                                                     cpu_energy.boundary_radiative_requested_linear_work_j,
                                                     cpu_energy.boundary_radiative_finite_step_correction_j,
                                                     cpu_energy.boundary_radiative_committed_work_j,
                                                     cpu_energy.boundary_constraint_history_committed_work_j,
                                                     cpu_energy.committed_inventory_change_j};
                const std::array<double, 6> actual{gpu_energy.boundary_committed_work_j,
                                                   gpu_energy.boundary_radiative_requested_linear_work_j,
                                                   gpu_energy.boundary_radiative_finite_step_correction_j,
                                                   gpu_energy.boundary_radiative_committed_work_j,
                                                   gpu_energy.boundary_constraint_history_committed_work_j,
                                                   gpu_energy.committed_inventory_change_j};
                for (std::size_t f = 0; f < expected.size(); ++f)
                {
                    CAPTURE(f, expected[f], actual[f]);
                    CHECK(std::abs(expected[f] - actual[f]) <= 3e-12 * std::max(1.0, std::abs(expected[f])) + 1e-6);
                }
                CHECK(gpu_energy.boundary_application_count == cpu_energy.boundary_application_count);
                equal(cpu.fields, gpu.fields);
                equal(cpu.runtime.boundary_conditions, gpu.runtime.boundary_conditions);
            }
        }
    auto s                          = initial(6, 5, 8);
    using L                         = init::config::LateralBC;
    s.configuration.lateral_bc_west = s.configuration.lateral_bc_east = s.configuration.lateral_bc_south =
        s.configuration.lateral_bc_north                              = L::Radiative;
    const auto                       fields                           = s.fields;
    const BoundaryConditionWaveState wave                             = s.runtime.boundary_conditions;
    Reject                           reject;
    REQUIRE_THROWS(apply_native_cgrid_acoustic_boundary(s, reject, .1));
    equal(fields, s.fields);
    equal(wave, s.runtime.boundary_conditions);
    CartesianCGridAcousticEnergyReportV1 incomplete;
    incomplete.boundary_application_count = 1;
    REQUIRE_THROWS(apply_native_cgrid_acoustic_boundary(s, *kernels, .1, &incomplete));
    CHECK_FALSE(incomplete.boundary_complete);
    CHECK(incomplete.boundary_application_count == 1);
    equal(fields, s.fields);
    equal(wave, s.runtime.boundary_conditions);
    REQUIRE_THROWS(apply_native_cgrid_acoustic_boundary(s, *kernels, -1));
    equal(fields, s.fields);
    equal(wave, s.runtime.boundary_conditions);
    omp_set_num_threads(threads);
    CHECK(runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::AcousticBcFaceCartesianCgrid) >
          0);
}

void exercise_gradual_underflow(const char* backend)
{
    ComputeBackendRuntimeState runtime;
    runtime.config.backend = backend;
    runtime.config.allow_fallback = false;
    std::string error;
    REQUIRE(initialize_compute_backend_runtime(runtime, CoordinateSystem::Cartesian, error));
    INFO(error);
    auto* kernels = runtime.backend->dynamics_kernels();
    REQUIRE(kernels);
    const int threads = omp_get_max_threads();
    omp_set_num_threads(1);
    for (int exponent = -149; exponent <= 4; ++exponent)
        for (double dt : {0.03125, 0.005, 0.125})
        {
            CAPTURE(backend, exponent, dt);
            auto cpu = initial(6, 5, 8, std::ldexp(1.0f, exponent));
            cpu.configuration.domain_frame.velocity_ms = {0, 0};
            using L = init::config::LateralBC;
            cpu.configuration.lateral_bc_west = cpu.configuration.lateral_bc_east =
                cpu.configuration.lateral_bc_south = cpu.configuration.lateral_bc_north = L::Radiative;
            for (auto* field : {&cpu.fields.u, &cpu.fields.v, &cpu.fields.w})
                for (std::size_t n = 0; n < field->size(); ++n)
                {
                    if (n % 7 == 0) field->data()[n] = std::nextafter(field->data()[n], INFINITY);
                    if (n % 11 == 0) field->data()[n] = -0.0f;
                }
            auto gpu = cpu;
            auto view = make_simulation_state_view(cpu);
            auto scheme = create_cartesian_cgrid_bc_scheme(view);
            for (int step = 0; step < 3; ++step)
            {
                scheme->apply_acoustic(dt);
                apply_native_cgrid_acoustic_boundary(gpu, *kernels, dt);
                equal(cpu.fields, gpu.fields);
                equal(cpu.runtime.boundary_conditions, gpu.runtime.boundary_conditions);
            }
        }
    omp_set_num_threads(threads);
}
} // namespace
TEST_CASE("Boundary binary32 arithmetic preserves IEEE rounding", "[native_acoustic_boundary][arithmetic]")
{
    using namespace boundary_arithmetic;
    std::mt19937 random(3219);
    for (int n = 0; n < 3000000; ++n)
    {
        const uint a = random(), b = random();
        if ((a & 0x7f800000u) == 0x7f800000u || (b & 0x7f800000u) == 0x7f800000u) continue;
        const float x = uintBitsToFloat(a), y = uintBitsToFloat(b);
        const float sum = x + y, product = x * y;
        if (floatBitsToUint(sum) != floatBitsToUint(boundary_add(x, y)) ||
            floatBitsToUint(product) != floatBitsToUint(boundary_mul(x, y)))
        {
            CAPTURE(a, b);
            FAIL("radiative boundary arithmetic differs from binary32 reference");
        }
    }
    CHECK(floatBitsToUint(boundary_add(-0.0f, -0.0f)) == 0x80000000u);
    CHECK(floatBitsToUint(boundary_sub(-0.0f, 0.0f)) == 0x80000000u);
    CHECK(floatBitsToUint(boundary_add(-0.0f, 0.0f)) == 0u);
    CHECK(floatBitsToUint(boundary_mul(-0.0f, 2.0f)) == 0x80000000u);
}
TEST_CASE("Native acoustic boundary fields and history match CPU on Metal", "[metal][native_acoustic_boundary]")
{
    exercise("metal");
}
TEST_CASE("Native acoustic boundary fields and history match CPU on Vulkan", "[vulkan][native_acoustic_boundary]")
{
    exercise("vulkan");
}
TEST_CASE("Native radiative boundaries preserve binary32 gradual underflow on Vulkan", "[vulkan][native_acoustic_boundary][underflow]")
{
    exercise_gradual_underflow("vulkan");
}
TEST_CASE("Native radiative boundaries preserve binary32 gradual underflow on Metal", "[metal][native_acoustic_boundary][underflow]")
{
    exercise_gradual_underflow("metal");
}
