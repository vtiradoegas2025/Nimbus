// Runtime parity harness and compiled-registry tests.
#include "catch2/catch.hpp"
#include "compute/parity_check.hpp"
#include "compute/support/parity_registry.hpp"
#include "compute/compute_backend.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace par = nimbus::parity;

namespace
{

// Enables a clean global harness and restores it on exit.
struct ParityEnabledGuard
{
    bool saved;

    ParityEnabledGuard() : saved(global_compute_backend_config.validate_parity)
    {
        unsetenv("NIMBUS_VALIDATE_PARITY");
        unsetenv("NIMBUS_PARITY_EVERY");
        par::reset();
        global_compute_backend_config.validate_parity = true;
    }

    ~ParityEnabledGuard()
    {
        global_compute_backend_config.validate_parity = saved;
        unsetenv("NIMBUS_VALIDATE_PARITY");
        unsetenv("NIMBUS_PARITY_EVERY");
        par::reset();
    }
};

std::size_t idx(int i, int j, int k, int nth, int nz)
{
    return (static_cast<std::size_t>(i) * static_cast<std::size_t>(nth)
            + static_cast<std::size_t>(j))
           * static_cast<std::size_t>(nz) + static_cast<std::size_t>(k);
}

}  // namespace

TEST_CASE("parity tolerances match GPU audit section 2.3", "[compute][parity]")
{
    const auto p = par::tolerance_for(par::Tier::Pointwise);
    REQUIRE(p.rel == Approx(1e-5));
    REQUIRE(p.abs == Approx(1e-6));

    const auto s = par::tolerance_for(par::Tier::Stencil);
    REQUIRE(s.rel == Approx(1e-4));
    REQUIRE(s.abs == Approx(5e-5));

    const auto m = par::tolerance_for(par::Tier::Microphysics);
    REQUIRE(m.rel == Approx(1e-4));
    REQUIRE(m.abs == Approx(1e-7));

    const auto h = par::tolerance_for(par::Tier::MicrophysicsHeating);
    REQUIRE(h.rel == Approx(1e-4));
    REQUIRE(h.abs == Approx(6e-4));
}

TEST_CASE("parity sample() refuses when the harness is disabled", "[compute][parity]")
{
    const bool saved = global_compute_backend_config.validate_parity;
    unsetenv("NIMBUS_VALIDATE_PARITY");
    global_compute_backend_config.validate_parity = false;
    par::reset();

    REQUIRE_FALSE(par::enabled());
    REQUIRE_FALSE(par::sample("any_kernel"));

    global_compute_backend_config.validate_parity = saved;
    par::reset();
}

TEST_CASE("parity enabled() honors the environment override", "[compute][parity]")
{
    const bool saved = global_compute_backend_config.validate_parity;
    global_compute_backend_config.validate_parity = false;
    setenv("NIMBUS_VALIDATE_PARITY", "1", 1);
    par::reset();

    REQUIRE(par::enabled());
    REQUIRE(par::sample("any_kernel"));

    unsetenv("NIMBUS_VALIDATE_PARITY");
    global_compute_backend_config.validate_parity = saved;
    par::reset();
}

TEST_CASE("Caller-owned parity runtimes isolate cadence and results",
          "[compute][parity][runtime][ownership]")
{
    unsetenv("NIMBUS_VALIDATE_PARITY");
    unsetenv("NIMBUS_PARITY_EVERY");
    par::ParityRuntimeState first;
    par::ParityRuntimeState second;

    REQUIRE(par::sample(first, true, "owned_kernel"));
    REQUIRE(par::sample(second, true, "owned_kernel"));
    REQUIRE(first.call_counters.at("owned_kernel") == 1);
    REQUIRE(second.call_counters.at("owned_kernel") == 1);

    constexpr int n = 3;
    std::vector<float> cpu(static_cast<std::size_t>(n) * n * n, 1.0f);
    std::vector<float> gpu(cpu);
    gpu[idx(1, 1, 1, n, n)] = 2.0f;
    REQUIRE_FALSE(par::check_field(
        first, "owned_kernel", "field", cpu.data(), gpu.data(),
        n, n, n, par::Tier::Pointwise,
        par::ComparisonDomain::StrictInterior));
    REQUIRE(par::check_field(
        second, "owned_kernel", "field", cpu.data(), cpu.data(),
        n, n, n, par::Tier::Pointwise,
        par::ComparisonDomain::StrictInterior));
    REQUIRE(par::any_breach(first));
    REQUIRE_FALSE(par::any_breach(second));
    REQUIRE(par::total_samples(first) == 1);
    REQUIRE(par::total_samples(second) == 1);

    par::ParityRuntimeState snapshot(first);
    par::reset(first);
    REQUIRE(par::total_samples(first) == 0);
    REQUIRE(par::total_samples(snapshot) == 1);
    REQUIRE(par::any_breach(snapshot));
}

TEST_CASE("check_field passes identical fields and in-tolerance noise", "[compute][parity]")
{
    ParityEnabledGuard guard;
    const int n = 8;
    std::vector<float> cpu(static_cast<std::size_t>(n) * n * n, 1.0f);
    std::vector<float> gpu(cpu);

    REQUIRE(par::check_field("kernel", "field", cpu.data(), gpu.data(),
                             n, n, n, par::Tier::Pointwise));

    // Noise inside the pointwise band (tol = 1e-6 + 1e-5*1.0 = 1.1e-5).
    for (auto& g : gpu) g = 1.0f + 5e-6f;
    REQUIRE(par::check_field("kernel", "field", cpu.data(), gpu.data(),
                             n, n, n, par::Tier::Pointwise));

    REQUIRE_FALSE(par::any_breach());
    REQUIRE(par::total_samples() == 2);
}

TEST_CASE("check_field flags an interior breach", "[compute][parity]")
{
    ParityEnabledGuard guard;
    const int n = 8;
    std::vector<float> cpu(static_cast<std::size_t>(n) * n * n, 1.0f);
    std::vector<float> gpu(cpu);
    gpu[idx(3, 4, 5, n, n)] = 1.01f;  // diff 1e-2 >> pointwise band

    REQUIRE_FALSE(par::check_field("kernel", "field", cpu.data(), gpu.data(),
                                   n, n, n, par::Tier::Pointwise));
    REQUIRE(par::any_breach());
}

TEST_CASE("check_field rejects non-finite values from either path",
          "[compute][parity]")
{
    ParityEnabledGuard guard;
    const int n = 8;
    const std::size_t bad = idx(3, 4, 5, n, n);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();

    auto rejects = [&](const char* field, float cpu_value, float gpu_value)
    {
        par::reset();
        global_compute_backend_config.validate_parity = true;
        std::vector<float> cpu(static_cast<std::size_t>(n) * n * n, 1.0f);
        std::vector<float> gpu(cpu);
        cpu[bad] = cpu_value;
        gpu[bad] = gpu_value;

        REQUIRE_FALSE(par::check_field("nonfinite", field,
                                       cpu.data(), gpu.data(), n, n, n,
                                       par::Tier::Pointwise));
        REQUIRE(par::any_breach());
        REQUIRE(par::total_samples() == 1);
    };

    rejects("cpu_nan", nan, 1.0f);
    rejects("gpu_nan", 1.0f, nan);
    rejects("both_nan", nan, nan);
    rejects("cpu_inf", inf, 1.0f);
    rejects("gpu_inf", 1.0f, -inf);
    // Matching infinities are still invalid state, not numerical agreement.
    rejects("both_inf", inf, inf);
}

TEST_CASE("check_field rejects an empty strict-interior domain",
          "[compute][parity]")
{
    ParityEnabledGuard guard;

    auto rejects = [&](const char* field, int nr, int nth, int nz)
    {
        par::reset();
        global_compute_backend_config.validate_parity = true;
        const std::size_t size = static_cast<std::size_t>(nr)
                               * static_cast<std::size_t>(nth)
                               * static_cast<std::size_t>(nz);
        std::vector<float> cpu(size, 1.0f);
        std::vector<float> gpu(cpu);

        REQUIRE_FALSE(par::check_field("empty_interior", field,
                                       cpu.data(), gpu.data(), nr, nth, nz,
                                       par::Tier::Pointwise));
        REQUIRE(par::any_breach());
        REQUIRE(par::total_samples() == 1);
    };

    // A strict interior exists only when every dimension is at least three.
    rejects("nr", 2, 8, 8);
    rejects("nth", 8, 2, 8);
    rejects("nz", 8, 8, 2);
}

TEST_CASE("pure field comparison fails closed without a GPU",
          "[compute][parity][comparison]")
{
    constexpr int nr = 4;
    constexpr int nth = 5;
    constexpr int nz = 6;
    const auto full = par::ComparisonDomain::FullField;
    const par::Tolerance exact{0.0, 0.0};
    std::vector<float> cpu(static_cast<std::size_t>(nr) * nth * nz, 1.0f);
    std::vector<float> candidate(cpu);

    auto result = par::compare_field(cpu.data(), candidate.data(),
                                     nr, nth, nz, exact, full);
    REQUIRE(result.passed);
    REQUIRE(result.invalid_reason == nullptr);
    REQUIRE(result.cells_compared == cpu.size());

    // Matching NaN and matching infinity are invalid state, not parity.
    candidate[idx(0, 0, 0, nth, nz)] =
        std::numeric_limits<float>::quiet_NaN();
    cpu[idx(0, 0, 0, nth, nz)] = candidate[idx(0, 0, 0, nth, nz)];
    result = par::compare_field(cpu.data(), candidate.data(),
                                nr, nth, nz, exact, full);
    REQUIRE_FALSE(result.passed);
    REQUIRE(result.nonfinite_cells == 1);

    cpu.assign(cpu.size(), 1.0f);
    candidate = cpu;
    candidate[idx(0, 0, 0, nth, nz)] =
        std::numeric_limits<float>::infinity();
    cpu[idx(0, 0, 0, nth, nz)] = candidate[idx(0, 0, 0, nth, nz)];
    result = par::compare_field(cpu.data(), candidate.data(),
                                nr, nth, nz, exact, full);
    REQUIRE_FALSE(result.passed);
    REQUIRE(result.nonfinite_cells == 1);

    // Empty domains, null fields, and invalid tolerances all fail closed.
    result = par::compare_field(cpu.data(), candidate.data(),
                                2, nth, nz, exact,
                                par::ComparisonDomain::StrictInterior);
    REQUIRE_FALSE(result.passed);
    REQUIRE(result.cells_compared == 0);
    REQUIRE(std::string(result.invalid_reason) == "empty_strict_interior");

    result = par::compare_field(nullptr, candidate.data(),
                                nr, nth, nz, exact, full);
    REQUIRE_FALSE(result.passed);
    REQUIRE(std::string(result.invalid_reason) == "null_input");

    result = par::compare_field(cpu.data(), candidate.data(), nr, nth, nz,
                                {0.0, -1.0}, full);
    REQUIRE_FALSE(result.passed);
    REQUIRE(std::string(result.invalid_reason) == "invalid_tolerance");
}

TEST_CASE("microphysics tiers carry the knife-edge cell allowance", "[compute][parity]")
{
    ParityEnabledGuard guard;
    // 30x30x30 interior = 21952... use 32^3: interior 30^3 = 27000 cells,
    // allowance = 27000/10000 = 2 cells for microphysics tiers.
    const int n = 32;
    std::vector<float> cpu(static_cast<std::size_t>(n) * n * n, 1.0f);
    std::vector<float> gpu(cpu);
    gpu[idx(5, 5, 5, n, n)] = 2.0f;   // one knife-edge cell

    // Microphysics: one bad cell is within the allowance -> no breach.
    REQUIRE(par::check_field("mp_kernel", "f", cpu.data(), gpu.data(),
                             n, n, n, par::Tier::Microphysics));
    REQUIRE_FALSE(par::any_breach());

    // Three bad cells exceed the 2-cell allowance -> breach.
    gpu[idx(6, 6, 6, n, n)] = 2.0f;
    gpu[idx(7, 7, 7, n, n)] = 2.0f;
    REQUIRE_FALSE(par::check_field("mp_kernel", "f", cpu.data(), gpu.data(),
                                   n, n, n, par::Tier::Microphysics));
    REQUIRE(par::any_breach());

    par::reset();
    global_compute_backend_config.validate_parity = true;

    // The finite knife-edge allowance must never hide non-finite state.
    std::vector<float> gpu_nonfinite(cpu);
    gpu_nonfinite[idx(5, 5, 5, n, n)] =
        std::numeric_limits<float>::quiet_NaN();
    REQUIRE_FALSE(par::check_field("mp_kernel", "nonfinite", cpu.data(),
                                   gpu_nonfinite.data(), n, n, n,
                                   par::Tier::Microphysics));
    REQUIRE(par::any_breach());

    par::reset();
    global_compute_backend_config.validate_parity = true;

    // Stencil tier: zero allowance - a single bad cell breaches.
    std::vector<float> gpu2(cpu);
    gpu2[idx(5, 5, 5, n, n)] = 2.0f;
    REQUIRE_FALSE(par::check_field("stencil_kernel", "f", cpu.data(),
                                   gpu2.data(), n, n, n, par::Tier::Stencil));
    REQUIRE(par::any_breach());
}

TEST_CASE("check_field horizon scales both tolerance bands", "[compute][parity]")
{
    ParityEnabledGuard guard;
    const int n = 8;
    std::vector<float> cpu(static_cast<std::size_t>(n) * n * n, 0.0f);
    std::vector<float> gpu(cpu);
    // 3e-4 vs stencil abs 5e-5: breaches at horizon 1, passes at horizon 8
    // (band 4e-4). cpu = 0 so the relative term contributes nothing.
    gpu[idx(3, 4, 5, n, n)] = 3e-4f;

    REQUIRE_FALSE(par::check_field("batched", "f", cpu.data(), gpu.data(),
                                   n, n, n, par::Tier::Stencil, 1));
    REQUIRE(par::check_field("batched", "f", cpu.data(), gpu.data(),
                             n, n, n, par::Tier::Stencil, 8));

    // The relative band scales the same way (the KW78 radiative faces
    // gave boundary cells recursive per-substep state, so systematic
    // rounding compounds in the relative measure too): 2e-4 on an O(1)
    // value breaches stencil rel=1e-4 at horizon 1 but passes the 8e-4
    // band at horizon 8.
    std::vector<float> cpu2(static_cast<std::size_t>(n) * n * n, 1.0f);
    std::vector<float> gpu2(cpu2);
    gpu2[idx(3, 4, 5, n, n)] = 1.0002f;
    REQUIRE_FALSE(par::check_field("batched", "g", cpu2.data(), gpu2.data(),
                                   n, n, n, par::Tier::Stencil, 1));
    REQUIRE(par::check_field("batched", "g", cpu2.data(), gpu2.data(),
                             n, n, n, par::Tier::Stencil, 8));

    // Bug-signal separation is horizon-invariant: a 0.5% relative error
    // still breaches at horizon 8 (band 8e-4).
    std::vector<float> gpu3(cpu2);
    gpu3[idx(3, 4, 5, n, n)] = 1.005f;
    REQUIRE_FALSE(par::check_field("batched", "h", cpu2.data(), gpu3.data(),
                                   n, n, n, par::Tier::Stencil, 8));
}

TEST_CASE("check_field ignores boundary cells", "[compute][parity]")
{
    ParityEnabledGuard guard;
    const int n = 8;
    std::vector<float> cpu(static_cast<std::size_t>(n) * n * n, 1.0f);
    std::vector<float> gpu(cpu);
    // Corrupt only boundary cells: i=0 face, j=n-1 face, k=0 face. GPU
    // kernels write interior cells; boundaries belong to BC passes, so the
    // comparison excludes them.
    gpu[idx(0, 4, 4, n, n)] = 99.0f;
    gpu[idx(4, n - 1, 4, n, n)] = 99.0f;
    gpu[idx(4, 4, 0, n, n)] = 99.0f;

    REQUIRE(par::check_field("kernel", "field", cpu.data(), gpu.data(),
                             n, n, n, par::Tier::Pointwise));
    REQUIRE_FALSE(par::any_breach());
}

TEST_CASE("full-field parity checks boundary and singleton domains",
          "[compute][parity][domain]")
{
    ParityEnabledGuard guard;
    const int nr = 5, nth = 6, nz = 7;
    std::vector<float> cpu(static_cast<std::size_t>(nr) * nth * nz, 1.0f);
    std::vector<float> gpu(cpu);

    REQUIRE(par::check_field("full", "identical", cpu.data(), gpu.data(),
                             nr, nth, nz, par::Tier::Pointwise,
                             par::ComparisonDomain::FullField));

    // A strict-interior comparison would miss this west-ghost corruption.
    gpu[idx(0, 2, 3, nth, nz)] = 2.0f;
    REQUIRE_FALSE(par::check_field("full", "west_boundary",
                                   cpu.data(), gpu.data(), nr, nth, nz,
                                   par::Tier::Pointwise,
                                   par::ComparisonDomain::FullField));

    par::reset();
    global_compute_backend_config.validate_parity = true;
    float one_cpu = 1.0f;
    float one_gpu = 1.0f;
    REQUIRE(par::check_field("full", "singleton", &one_cpu, &one_gpu,
                             1, 1, 1, par::Tier::Pointwise,
                             par::ComparisonDomain::FullField));
    one_gpu = 2.0f;
    REQUIRE_FALSE(par::check_field("full", "singleton_breach",
                                   &one_cpu, &one_gpu, 1, 1, 1,
                                   par::Tier::Pointwise,
                                   par::ComparisonDomain::FullField));
}

TEST_CASE("boundary-shell parity is disjoint from the strict interior",
          "[compute][parity][domain]")
{
    ParityEnabledGuard guard;
    const int nr = 5, nth = 6, nz = 7;
    std::vector<float> cpu(static_cast<std::size_t>(nr) * nth * nz, 1.0f);
    std::vector<float> gpu(cpu);

    REQUIRE(std::string(par::domain_name(
                par::ComparisonDomain::BoundaryShell)) == "boundary-shell");
    REQUIRE(par::check_field("shell", "identical", cpu.data(), gpu.data(),
                             nr, nth, nz, par::Tier::Pointwise,
                             par::ComparisonDomain::BoundaryShell));

    gpu[idx(0, 2, 3, nth, nz)] = 2.0f;
    REQUIRE_FALSE(par::check_field("shell", "boundary", cpu.data(), gpu.data(),
                                   nr, nth, nz, par::Tier::Pointwise,
                                   par::ComparisonDomain::BoundaryShell));

    par::reset();
    global_compute_backend_config.validate_parity = true;
    gpu = cpu;
    gpu[idx(2, 3, 3, nth, nz)] = 2.0f;
    REQUIRE(par::check_field("shell", "interior_ignored",
                             cpu.data(), gpu.data(), nr, nth, nz,
                             par::Tier::Pointwise,
                             par::ComparisonDomain::BoundaryShell));

    // A one-cell field is all boundary and must remain non-vacuous.
    float one_cpu = 1.0f;
    float one_gpu = 2.0f;
    REQUIRE_FALSE(par::check_field("shell", "singleton",
                                   &one_cpu, &one_gpu, 1, 1, 1,
                                   par::Tier::Pointwise,
                                   par::ComparisonDomain::BoundaryShell));
}

TEST_CASE("face parity domains select every physical face non-vacuously",
          "[compute][parity][domain]")
{
    ParityEnabledGuard guard;
    const int nr = 5, nth = 6, nz = 7;
    const std::size_t cells = static_cast<std::size_t>(nr) * nth * nz;
    std::vector<float> cpu(cells, 1.0f);

    struct FaceCase
    {
        par::ComparisonDomain domain;
        const char* name;
        int i;
        int j;
        int k;
    };
    const FaceCase faces[] = {
        {par::ComparisonDomain::WestFace,   "west-face",   0,      2,       3},
        {par::ComparisonDomain::EastFace,   "east-face",   nr - 1, 2,       3},
        {par::ComparisonDomain::SouthFace,  "south-face",  2,      0,       3},
        {par::ComparisonDomain::NorthFace,  "north-face",  2,      nth - 1, 3},
        {par::ComparisonDomain::BottomFace, "bottom-face", 2,      3,       0},
        {par::ComparisonDomain::TopFace,    "top-face",    2,      3,       nz - 1},
    };

    for (const FaceCase& face : faces)
    {
        INFO("domain=" << face.name);
        REQUIRE(std::string(par::domain_name(face.domain)) == face.name);

        par::reset();
        global_compute_backend_config.validate_parity = true;
        std::vector<float> gpu(cpu);
        REQUIRE(par::check_field("face", face.name, cpu.data(), gpu.data(),
                                 nr, nth, nz, par::Tier::Pointwise,
                                 face.domain));
        gpu[idx(face.i, face.j, face.k, nth, nz)] = 2.0f;
        REQUIRE_FALSE(par::check_field("face", face.name,
                                       cpu.data(), gpu.data(), nr, nth, nz,
                                       par::Tier::Pointwise, face.domain));

        // A face comparison must not silently widen to the full field.
        par::reset();
        global_compute_backend_config.validate_parity = true;
        gpu = cpu;
        gpu[idx(2, 3, 3, nth, nz)] = 2.0f;
        REQUIRE(par::check_field("face", face.name, cpu.data(), gpu.data(),
                                 nr, nth, nz, par::Tier::Pointwise,
                                 face.domain));
    }
}

TEST_CASE("comparison domains reject invalid shapes and null fields",
          "[compute][parity][domain]")
{
    ParityEnabledGuard guard;
    float value = 1.0f;
    const par::ComparisonDomain domains[] = {
        par::ComparisonDomain::StrictInterior,
        par::ComparisonDomain::FullField,
        par::ComparisonDomain::BoundaryShell,
        par::ComparisonDomain::WestFace,
        par::ComparisonDomain::EastFace,
        par::ComparisonDomain::SouthFace,
        par::ComparisonDomain::NorthFace,
        par::ComparisonDomain::BottomFace,
        par::ComparisonDomain::TopFace,
    };

    for (const auto domain : domains)
    {
        INFO("domain=" << par::domain_name(domain));
        par::reset();
        global_compute_backend_config.validate_parity = true;
        REQUIRE_FALSE(par::check_field("invalid", "zero_dimension",
                                       &value, &value, 0, 4, 4,
                                       par::Tier::Pointwise, domain));
        REQUIRE(par::any_breach());
        REQUIRE(par::total_samples() == 1);
    }

    par::reset();
    global_compute_backend_config.validate_parity = true;
    REQUIRE_FALSE(par::check_field("invalid", "negative_dimension",
                                   &value, &value, -1, 4, 4,
                                   par::Tier::Pointwise,
                                   par::ComparisonDomain::FullField));

    par::reset();
    global_compute_backend_config.validate_parity = true;
    REQUIRE_FALSE(par::check_field("invalid", "null_cpu",
                                   nullptr, &value, 1, 1, 1,
                                   par::Tier::Pointwise,
                                   par::ComparisonDomain::FullField));
    REQUIRE_FALSE(par::check_field("invalid", "null_gpu",
                                   &value, nullptr, 1, 1, 1,
                                   par::Tier::Pointwise,
                                   par::ComparisonDomain::FullField));
}

TEST_CASE("sample cadence respects NIMBUS_PARITY_EVERY", "[compute][parity]")
{
    ParityEnabledGuard guard;
    setenv("NIMBUS_PARITY_EVERY", "3", 1);
    par::reset();
    global_compute_backend_config.validate_parity = true;

    // Calls 0..8 for one kernel: sampled at 0, 3, 6.
    REQUIRE(par::sample("cadenced"));
    REQUIRE_FALSE(par::sample("cadenced"));
    REQUIRE_FALSE(par::sample("cadenced"));
    REQUIRE(par::sample("cadenced"));
    REQUIRE_FALSE(par::sample("cadenced"));
    REQUIRE_FALSE(par::sample("cadenced"));
    REQUIRE(par::sample("cadenced"));

    // Counters are per kernel: a different kernel starts at its own 0.
    REQUIRE(par::sample("other_kernel"));
}

TEST_CASE("invalid parity cadence text safely uses the default",
          "[compute][parity][validation]")
{
    ParityEnabledGuard guard;
    const char* invalid_values[] = {
        "3trailing", "-2", "999999999999999999999999999999999999",
    };
    for (const char* value : invalid_values)
    {
        setenv("NIMBUS_PARITY_EVERY", value, 1);
        par::reset();
        REQUIRE(par::sample("checked-cadence"));
        REQUIRE(par::sample("checked-cadence"));
    }
}

TEST_CASE("parity registry lists every compute shader", "[compute][parity]")
{
    const auto& reg = par::registry();
    REQUIRE(reg.size() == nimbus::compute::kernel_count());
    const char* validation_reason = nullptr;
    const bool valid_registry = par::validate_registry(reg, &validation_reason);
    INFO("parity registry validation reason=" <<
         (validation_reason != nullptr ? validation_reason : "none"));
    REQUIRE(valid_registry);
    REQUIRE(validation_reason == nullptr);

    std::size_t orphans = 0;
    std::size_t active = 0;
    std::size_t deferred = 0;
    std::size_t full_field = 0;
    std::size_t runtime_covered = 0;
    std::size_t hardware_covered = 0;
    std::size_t metal_hardware_covered = 0;
    std::size_t accumulated_horizon = 0;
    for (const auto& e : reg)
    {
        REQUIRE(e.shader != nullptr);
        REQUIRE(e.shader[0] != '\0');
        REQUIRE(e.cpu_reference != nullptr);
        REQUIRE(e.cpu_reference[0] != '\0');
        REQUIRE(std::string(par::domain_name(e.comparison_domain)) != "invalid");
        if (e.coverage.runtime_sample != nullptr) ++runtime_covered;
        if (e.coverage.hardware_test != nullptr)
        {
            ++hardware_covered;
            REQUIRE(std::filesystem::exists(e.coverage.hardware_test));
        }
        if (e.coverage.metal_hardware_test != nullptr)
        {
            ++metal_hardware_covered;
            REQUIRE(std::filesystem::exists(e.coverage.metal_hardware_test));
        }
        if (e.coverage.horizon == par::HorizonRule::AccumulatedSubsteps)
            ++accumulated_horizon;
        if (e.comparison_domain == par::ComparisonDomain::FullField)
        {
            ++full_field;
            const std::string shader = e.shader;
            REQUIRE((
                shader == "dynamics/acoustic_bc_face_cartesian_cgrid.comp" ||
                shader == "dynamics/conservative_reduction.comp" ||
                shader == "acoustic/acoustic_continuity_massflux_cartesian_agrid.comp" ||
                shader == "acoustic/acoustic_retained_flux_fp64.comp" || shader == "diffusion/diffusion.comp" ||
                shader == "diffusion/implicit_diffusion_stencil.comp" ||
                shader == "diffusion/implicit_diffusion_vector.comp" || shader == "turbulence/tke_closure.comp" ||
                shader == "turbulence/tke_tendency.comp" ||
                shader == "microphysics/thompson_moment_sedimentation.comp" ||
                shader == "microphysics/thompson_cloud_adjustment.comp" ||
                shader == "microphysics/thompson_warm_rain.comp" || shader == "microphysics/milbrandt_warm_rain.comp" ||
                shader == "microphysics/milbrandt_cloud_adjustment.comp" ||
                shader == "microphysics/milbrandt_cold.comp" || shader == "time_stepping/rk_combination.comp" ||
                shader == "time_stepping/rk_tendency.comp" || shader == "boundary_layer/slab_column.comp" ||
                shader == "radiation/simple_grey.comp" || shader == "terrain/held_sources.comp" ||
                shader == "terrain/column_remap.comp" || shader == "terrain/reference_atmosphere.comp" ||
                shader == "terrain/moving_energy.comp" || shader == "terrain/moving_momentum.comp" ||
                shader == "terrain/earth_geometry.comp" || shader == "terrain/conservative_remap.comp" ||
                shader == "turbulence/collocated_sgs.comp" || shader == "microphysics/milbrandt_sedimentation.comp" ||
                shader == "microphysics/milbrandt_melting.comp" || shader == "microphysics/thompson_cloud_phase.comp" ||
                shader == "microphysics/thompson_freezing.comp" || shader == "microphysics/thompson_deposition.comp" ||
                shader == "microphysics/thompson_completion.comp" ||
                shader == "domain/domain_recycling_homogeneous_field.comp" ||
                shader == "chaos/earth_control_evolution.comp" ||
                shader == "acoustic/acoustic_pressure_eos_cartesian_agrid.comp" ||
                shader == "acoustic/acoustic_momentum_cartesian_agrid.comp" ||
                shader == "acoustic/acoustic_boundary_cartesian_agrid.comp" ||
                shader == "terrain/mapped_mass_continuity.comp" || shader == "terrain/mapped_fast_momentum.comp" ||
                shader == "terrain/mapped_slow_tendencies.comp" || shader == "terrain/mapped_scalar_update.comp" ||
                shader == "terrain/mapped_scalar_limiter.comp" || shader == "terrain/mapped_state_update.comp" ||
                shader == "terrain/mapped_flux_accumulation.comp" || shader == "terrain/mapped_eos.comp" ||
                shader == "terrain/mapped_boundary.comp" || shader == "boundary_layer/mynn_level25_stability.comp" ||
                shader == "radiation/rrtmg_longwave_transfer.comp" ||
                shader == "radiation/rrtmg_shortwave_transfer.comp" ||
                shader == "radiation/rrtmg_longwave_transfer_batch.comp" ||
                shader == "radiation/rrtmg_shortwave_transfer_batch.comp"));
        }
        switch (e.status)
        {
            case par::ShaderStatus::Active:   ++active;   break;
            case par::ShaderStatus::Deferred: ++deferred; break;
            case par::ShaderStatus::Orphan:   ++orphans;  break;
        }
    }
    // Two Thompson shaders and the removed transport point filter are
    // compiled provenance artifacts but never wired into execution.
    REQUIRE(orphans == 3);
    REQUIRE(active > 0);
    REQUIRE(deferred == 0);
    // Boundary ownership, native diffusion, recycling, and mapped state/flux
    // transactions are compared over the complete field.
    REQUIRE(full_field == 52);
    REQUIRE(runtime_covered > 0);
    REQUIRE(hardware_covered > 0);
    REQUIRE(metal_hardware_covered + 1 == active);
    REQUIRE(accumulated_horizon == 9);

    const auto find_shader = [&](const char* shader) -> const par::ParityEntry& {
        const auto it = std::find_if(reg.begin(), reg.end(), [&](const auto& e) {
            return std::string(e.shader) == shader;
        });
        REQUIRE(it != reg.end());
        return *it;
    };
    REQUIRE(std::string(find_shader("advection/tvd_vertical_flux.comp")
                            .coverage.runtime_sample) == "tvd_vertical_flux");
    REQUIRE(std::string(find_shader("acoustic/acoustic_pressure_cartesian.comp")
                            .coverage.hardware_test) ==
            "tests/vulkan/test_gpu_parity.cpp");
    REQUIRE(std::string(find_shader("acoustic/acoustic_pressure_cartesian.comp")
                            .coverage.metal_hardware_test) ==
            "tests/metal/test_metal_compute_backend.cpp");
}

TEST_CASE("kernel manifest lookup rejects invalid identifiers",
          "[compute][manifest]")
{
    namespace compute = nimbus::compute;
    REQUIRE(compute::find_kernel(compute::KernelId::Count) == nullptr);
    REQUIRE_THROWS_AS(
        compute::kernel_manifest_entry(compute::KernelId::Count),
        std::out_of_range);
}

TEST_CASE("parity identifiers fail closed without constructing null strings",
          "[compute][parity][validation]")
{
    nimbus::parity::ParityRuntimeState runtime;
    float field[27]{};

    REQUIRE_FALSE(nimbus::parity::sample(runtime, true, nullptr));
    REQUIRE(runtime.call_counters.empty());

    REQUIRE_FALSE(nimbus::parity::check_field(
        runtime, nullptr, "theta", field, field, 3, 3, 3,
        nimbus::parity::Tier::Pointwise,
        nimbus::parity::ComparisonDomain::FullField));
    REQUIRE(nimbus::parity::any_breach(runtime));
    REQUIRE(nimbus::parity::total_samples(runtime) == 1);
    REQUIRE(runtime.stats.size() == 1);
    REQUIRE(std::get<0>(runtime.stats.begin()->first) ==
            "<invalid-kernel>");
}

TEST_CASE("active parity entries cannot lose all coverage",
          "[compute][parity][coverage]")
{
    auto broken = par::registry();
    const auto active = std::find_if(broken.begin(), broken.end(), [](const auto& e) {
        return e.status == par::ShaderStatus::Active;
    });
    REQUIRE(active != broken.end());
    active->coverage.runtime_sample = nullptr;
    active->coverage.hardware_test = nullptr;
    active->coverage.metal_hardware_test = nullptr;

    const char* reason = nullptr;
    REQUIRE_FALSE(par::validate_registry(broken, &reason));
    REQUIRE(reason != nullptr);
    REQUIRE(std::string(reason) ==
            "active parity entry needs runtime or hardware coverage");
}

TEST_CASE("FP64 retained-flux coverage keeps the Vulkan-only boundary explicit", "[compute][parity][coverage]")
{
    auto       entries = par::registry();
    const auto fp64    = std::find_if(
        entries.begin(),
        entries.end(),
        [](const auto& entry) { return std::string(entry.shader) == "acoustic/acoustic_retained_flux_fp64.comp"; });
    REQUIRE(fp64 != entries.end());
    REQUIRE(fp64->coverage.required_hardware == par::RequiredHardware::Vulkan);
    REQUIRE(fp64->coverage.metal_hardware_test == nullptr);
    REQUIRE(std::string(fp64->coverage.hardware_test) == "tests/vulkan/test_acoustic_input_reuse.cpp");
    REQUIRE(par::validate_registry(entries));

    SECTION("the FP64 helper cannot claim Metal coverage")
    {
        fp64->coverage.metal_hardware_test = "tests/metal/test_metal_compute_backend.cpp";
        REQUIRE_FALSE(par::validate_registry(entries));
    }
    SECTION("other kernels cannot drop their Metal requirement")
    {
        const auto shared =
            std::find_if(entries.begin(),
                         entries.end(),
                         [](const auto& entry)
                         {
                             return entry.status == par::ShaderStatus::Active &&
                                    entry.coverage.required_hardware == par::RequiredHardware::VulkanAndMetal;
                         });
        REQUIRE(shared != entries.end());
        shared->coverage.required_hardware   = par::RequiredHardware::Vulkan;
        shared->coverage.metal_hardware_test = nullptr;
        REQUIRE_FALSE(par::validate_registry(entries));
    }
}
