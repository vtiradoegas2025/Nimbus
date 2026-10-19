#include "catch2/catch.hpp"
#include "compute/backend_runtime.hpp"
#include "compute/rk_tendency_contract.hpp"
#include "numerics/time_stepping/native_rk.hpp"
#include "numerics/advection/weno5_face.hpp"
#include "numerics/advection/advection_cartesian.hpp"
#include <array>
#include <cmath>
#include <limits>
#include <iomanip>
#include <vector>

namespace
{
void exercise(const char* name)
{
    ComputeBackendRuntimeState runtime;
    runtime.config.backend        = name;
    runtime.config.allow_fallback = false;
    std::string error;
    const bool  initialized = initialize_compute_backend_runtime(runtime, CoordinateSystem::Cartesian, error);
    INFO(error);
    REQUIRE(initialized);
    REQUIRE_FALSE(runtime.fallback_active);
    REQUIRE(runtime.backend->supports_rk_tendency());
    constexpr int                     nx = 9, ny = 8, nz = 7, count = nx * ny * nz;
    const auto                        at = [](int i, int j, int k) { return (i * ny + j) * nz + k; };
    std::array<std::vector<float>, 7> input;
    for (auto& f : input)
        f.resize(count);
    RkTendencyDispatch d;
    d.nx    = nx;
    d.ny    = ny;
    d.nz    = nz;
    d.dx    = 100.25;
    d.dy    = 90.125;
    d.dz    = 30.375;
    d.dt    = .25;
    d.gamma = 1.4;
    for (int f = 0; f < 7; ++f)
        d.input[f] = input[f].data();
    tmv::NativeRkTendencyWorkspace workspace;
    std::uint64_t                  dispatches = 0;
    const auto                     run        = [&](int bank = 0) -> const tmv::NativeRkTendencyWorkspace::Fields&
    {
        ++dispatches;
        return workspace.run(*runtime.backend, d, bank);
    };
    const auto close = [&](float actual, double expected)
    {
        CAPTURE(name, actual, expected);
        INFO("operation=" << int(d.operation) << " actual=" << std::scientific << std::setprecision(17) << actual
                          << " expected=" << expected);
        CHECK(std::abs(double(actual) - expected) <= 2e-7 * std::max(std::abs(expected), 1e-35));
    };
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                int n           = at(i, j, k);
                input[0][n]     = float(std::sin(i * .2 + j * .3 + k * .4));
                input[1][n]     = float(std::cos(i * .3 - j * .4 + k * .2));
                input[2][n]     = k == nz - 1 ? 0 : float(std::sin(i * .5 - j * .3 + k * .1));
                input[3][n]     = float(.25 + .03 * k + .01 * i);
                input[4][n]     = float(100000 - 200.25 * k + 3.5 * i - 2.25 * j);
                const double p0 = 100000 - 200.123456789 * k;
                input[5][n]     = float(p0);
                input[6][n]     = float(p0 - float(p0));
            }
    const auto& acoustic = run();
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz; ++k)
            {
                int          n = at(i, j, k), x = nz * ny, y = nz;
                const auto   rho_face = [&](int a, int b) { return .5 * (double(input[3][a]) + input[3][b]); };
                const double div      = (double(input[0][n]) - input[0][n - x]) / d.dx +
                                        (double(input[1][n]) - input[1][n - y]) / d.dy +
                                        (double(input[2][n]) - (k > 0 ? input[2][n - 1] : 0)) / d.dz;
                close(acoustic[4].data()[n], -d.gamma * input[4][n] * div);
                const double mass_div =
                    (rho_face(n, n + x) * input[0][n] - rho_face(n - x, n) * input[0][n - x]) / d.dx +
                    (rho_face(n, n + y) * input[1][n] - rho_face(n - y, n) * input[1][n - y]) / d.dy +
                    ((k < nz - 1 ? rho_face(n, n + 1) : input[3][n]) * input[2][n] -
                     (k > 0 ? rho_face(n - 1, n) * input[2][n - 1] : 0)) /
                        d.dz;
                close(acoustic[3].data()[n], -mass_div);
                if (k == nz - 1)
                    continue;
                close(acoustic[0].data()[n], -(double(input[4][n + x]) - input[4][n]) / d.dx / rho_face(n, n + x));
                close(acoustic[1].data()[n], -(double(input[4][n + y]) - input[4][n]) / d.dy / rho_face(n, n + y));
                const double p0lo = double(input[5][n]) + input[6][n], p0hi = double(input[5][n + 1]) + input[6][n + 1];
                close(acoustic[2].data()[n],
                      -((double(input[4][n + 1]) - input[4][n]) / d.dz - (p0hi - p0lo) / d.dz) / rho_face(n, n + 1));
            }
    d.operation                            = RkTendencyOperation::MassFlux;
    const auto&                       mass = run();
    std::array<std::vector<float>, 3> retained;
    for (int f = 0; f < 3; ++f)
        retained[f].assign(mass[f].data(), mass[f].data() + count);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                int n = at(i, j, k);
                if (i < nx - 1)
                    close(retained[0][n], .5 * (double(input[3][n]) + input[3][n + ny * nz]) * input[0][n]);
                if (j < ny - 1)
                    close(retained[1][n], .5 * (double(input[3][n]) + input[3][n + nz]) * input[1][n]);
                if (k < nz - 1)
                    close(retained[2][n], .5 * (double(input[3][n]) + input[3][n + 1]) * input[2][n]);
            }
    std::array<std::vector<float>, 3> reference_flux;
    for (auto& f : reference_flux)
        f.resize(count);
    std::vector<float> limiter(count);
    for (bool weno : {false, true})
        for (bool periodic : {false, true})
            for (double amplitude : {300., .001, 1e-20})
            {
                d.weno       = weno;
                d.periodic_x = periodic;
                d.periodic_y = periodic;
                for (int n = 0; n < count; ++n)
                    input[0][n] = float(amplitude * (.5 + .5 * std::sin(n * .7)));
                d.input          = {input[0].data(),
                                    retained[0].data(),
                                    retained[1].data(),
                                    retained[2].data(),
                                    input[3].data(),
                                    input[0].data(),
                                    input[0].data()};
                d.operation      = RkTendencyOperation::ScalarFlux;
                const auto& flux = run();
                for (int axis = 0; axis < 3; ++axis)
                    for (int i = 0; i < nx; ++i)
                        for (int j = 0; j < ny; ++j)
                            for (int k = 0; k < nz; ++k)
                            {
                                int    n = at(i, j, k), position = axis == 0 ? i : (axis == 1 ? j : k),
                                       extent  = axis == 0 ? nx : (axis == 1 ? ny : nz),
                                       stride  = axis == 0 ? ny * nz : (axis == 1 ? nz : 1);
                                double carrier = retained[axis][n], face = 0;
                                if (position < extent - 1)
                                {
                                    int center    = carrier >= 0 ? n : n + stride,
                                        pos       = carrier >= 0 ? position : position + 1,
                                        direction = carrier >= 0 ? stride : -stride;
                                    face          = input[0][center];
                                    if (periodic && axis < 2 && (position == 0 || position == extent - 2))
                                        face = input[0][n + ((carrier >= 0 ? extent - 2 : 1) - position) * stride];
                                    else if (weno && pos >= 2 && pos + 2 < extent)
                                        face = weno5_face_value(input[0][center - 2 * direction],
                                                                input[0][center - direction],
                                                                input[0][center],
                                                                input[0][center + direction],
                                                                input[0][center + 2 * direction]);
                                    else if (pos >= 1 && pos + 1 < extent)
                                    {
                                        double left  = face - input[0][center - stride],
                                               right = input[0][center + stride] - face;
                                        double slope = left * right <= 0
                                                           ? 0
                                                           : std::copysign(std::min({.5 * std::abs(left + right),
                                                                                     2 * std::abs(left),
                                                                                     2 * std::abs(right)}),
                                                                           left);
                                        face += (carrier >= 0 ? .5 : -.5) * slope;
                                    }
                                }
                                reference_flux[axis][n] = float(carrier * face);
                                INFO("flux axis=" << axis << " n=" << n << " carrier=" << std::scientific << carrier
                                                  << " face=" << face << " amplitude=" << amplitude);
                                close(flux[axis].data()[n], reference_flux[axis][n]);
                            }
                for (bool positive : {false, true})
                {
                    d.positive         = positive;
                    d.dt               = 1000;
                    d.input            = {input[0].data(),
                                          flux[0].data(),
                                          flux[1].data(),
                                          flux[2].data(),
                                          input[3].data(),
                                          input[0].data(),
                                          input[0].data()};
                    d.operation        = RkTendencyOperation::DonorLimiter;
                    const auto& ratios = run(1);
                    for (int i = 0; i < nx; ++i)
                        for (int j = 0; j < ny; ++j)
                            for (int k = 0; k < nz; ++k)
                            {
                                int    n    = at(i, j, k);
                                double rate = 0;
                                for (int axis = 0; axis < 3; ++axis)
                                {
                                    int    pos     = axis == 0 ? i : (axis == 1 ? j : k),
                                           extent  = axis == 0 ? nx : (axis == 1 ? ny : nz),
                                           stride  = axis == 0 ? ny * nz : (axis == 1 ? nz : 1);
                                    double spacing = axis == 0 ? d.dx : (axis == 1 ? d.dy : d.dz);
                                    if (pos < extent - 1)
                                        rate += std::max(double(flux[axis].data()[n]), 0.) / spacing;
                                    if (pos > 0)
                                        rate -= std::min(double(flux[axis].data()[n - stride]), 0.) / spacing;
                                }
                                double r = rate > 0 ? std::min(1., double(input[3][n]) * input[0][n] / d.dt / rate) : 1;
                                limiter[n] = tmv::cartesian_advection_detail::conservative_pd_limiter_factor(r);
                                close(ratios[0].data()[n], limiter[n]);
                            }
                    d.operation          = RkTendencyOperation::LimitedDivergence;
                    d.input[5]           = ratios[0].data();
                    const auto& tendency = run(2);
                    for (int i = 1; i < nx - 1; ++i)
                        for (int j = 1; j < ny - 1; ++j)
                            for (int k = 0; k < nz; ++k)
                            {
                                int    n          = at(i, j, k);
                                double divergence = 0;
                                for (int axis = 0; axis < 3; ++axis)
                                {
                                    int  pos     = axis == 0 ? i : (axis == 1 ? j : k),
                                         extent  = axis == 0 ? nx : (axis == 1 ? ny : nz),
                                         stride  = axis == 0 ? ny * nz : (axis == 1 ? nz : 1);
                                    auto limited = [&](int face, int face_pos)
                                    {
                                        if (face_pos < 0 || face_pos >= extent - 1)
                                            return 0.;
                                        double value = flux[axis].data()[face];
                                        int    donor = value > 0 ? face : face + stride;
                                        if (periodic && axis < 2 && (face_pos == 0 || face_pos == extent - 2))
                                            donor = face + ((value > 0 ? extent - 2 : 1) - face_pos) * stride;
                                        return value * (positive ? ratios[0].data()[donor] : 1.);
                                    };
                                    divergence += (limited(n, pos) - limited(n - stride, pos - 1)) /
                                                  (axis == 0 ? d.dx : (axis == 1 ? d.dy : d.dz));
                                }
                                close(tendency[0].data()[n], -divergence);
                                if (positive)
                                    CHECK(double(input[3][n]) * input[0][n] + d.dt * tendency[0].data()[n] >=
                                          -1e-6 * amplitude);
                            }
                }
            }
    d.periodic_x = d.periodic_y = true;
    for (auto operation : {RkTendencyOperation::ScalarHalo,
                           RkTendencyOperation::HorizontalWindHalo,
                           RkTendencyOperation::VerticalWindHalo,
                           RkTendencyOperation::VolumeConcentrationHalo})
    {
        d.operation        = operation;
        d.input            = {input[0].data(),
                              input[3].data(),
                              input[0].data(),
                              input[0].data(),
                              input[0].data(),
                              input[0].data(),
                              input[0].data()};
        const auto& closed = run();
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    int    n = at(i, j, k), si = i == 0 ? nx - 2 : (i == nx - 1 ? 1 : i),
                           sj       = j == 0 ? ny - 2 : (j == ny - 1 ? 1 : j);
                    int    sk       = operation == RkTendencyOperation::HorizontalWindHalo && k == nz - 1 ? nz - 2 : k;
                    int    source   = at(si, sj, sk);
                    double expected = input[0][source];
                    if (operation == RkTendencyOperation::VerticalWindHalo && k == nz - 1)
                        expected = 0;
                    if (operation == RkTendencyOperation::VolumeConcentrationHalo && source != n)
                        expected = expected / input[3][source] * input[3][n];
                    close(closed[0].data()[n], expected);
                }
    }
    {
        const auto         saved = d;
        std::vector<float> source(count), density(count, 2);
        for (int n = 0; n < count; ++n)
            source[n] = std::array<float, 6>{-1e20f, -1e-30f, 0, 1e-30f, 2, 1e20f}[n % 6];
        const auto original = source;
        d.input[0]          = source.data();
        d.input[1]          = density.data();
        d.periodic_x = d.periodic_y = false;
        d.open_top                  = true;
        for (auto operation : {RkTendencyOperation::NonnegativeInput, RkTendencyOperation::NonnegativeMixingRatio})
        {
            d.operation        = operation;
            const auto& result = run();
            for (int n = 0; n < count; ++n)
            {
                const double expected =
                    std::max(0.0f, source[n]) / (operation == RkTendencyOperation::NonnegativeMixingRatio ? 2.0 : 1.0);
                close(result[0].data()[n], expected);
                for (int f = 1; f < 5; ++f)
                    CHECK(result[f].data()[n] == 0);
            }
            CHECK(source == original);
        }
        std::array<std::vector<float>, 5> untouched;
        for (std::size_t f = 0; f < untouched.size(); ++f)
        {
            untouched[f].assign(count, 7);
            d.output[f] = untouched[f].data();
        }
        source.back() = std::numeric_limits<float>::quiet_NaN();
        CHECK(runtime.backend->dispatch_rk_tendency(d) == RkCombinationStatus::InvalidInput);
        for (const auto& values : untouched)
            CHECK(std::all_of(values.begin(), values.end(), [](float value) { return value == 7; }));
        source.back()  = original.back();
        density.back() = 0;
        CHECK(runtime.backend->dispatch_rk_tendency(d) == RkCombinationStatus::InvalidInput);
        d = saved;
    }
    for (auto operation : {RkTendencyOperation::Add, RkTendencyOperation::MixingRatio})
    {
        d.operation        = operation;
        const auto& result = run();
        for (int n = 0; n < count; ++n)
            close(result[0].data()[n],
                  operation == RkTendencyOperation::Add ? double(input[0][n]) + input[3][n]
                                                        : double(input[0][n]) / input[3][n]);
    }
    {
        const auto                        saved = d;
        std::array<std::vector<float>, 7> slow;
        for (auto& field : slow)
            field.assign(count, 0);
        std::fill(slow[0].begin(), slow[0].end(), 1e10f);
        std::fill(slow[1].begin(), slow[1].end(), 1e-30f);
        std::fill(slow[4].begin(), slow[4].end(), 1.0f);
        for (std::size_t f = 0; f < slow.size(); ++f)
            d.input[f] = slow[f].data();
        d.operation          = RkTendencyOperation::DonorLimiter;
        d.dt                 = 1;
        d.dx                 = 1;
        const auto& inactive = run();
        for (int n = 0; n < count; ++n)
            CHECK(inactive[0].data()[n] == 1.0f);
        d = saved;
    }
    const auto original_scalar = input[0];
    for (int n = 0; n < count; ++n)
        input[0][n] = std::ldexp(float(1 + n % 13), -145);
    d.operation             = RkTendencyOperation::ScalarHalo;
    d.input                 = {input[0].data()};
    const auto& tiny_halo   = run();
    int         lost_values = 0;
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                int si = i == 0 ? nx - 2 : (i == nx - 1 ? 1 : i);
                int sj = j == 0 ? ny - 2 : (j == ny - 1 ? 1 : j);
                if (tiny_halo[0].data()[at(i, j, k)] != input[0][at(si, sj, k)])
                    ++lost_values;
            }
    CHECK(lost_values == 0);
    std::copy(original_scalar.begin(), original_scalar.end(), input[0].begin());
    d.collocated = true;
    d.operation  = RkTendencyOperation::MassFlux;
    for (int f = 0; f < 7; ++f)
        d.input[f] = input[f].data();
    const auto& collocated_flux = run();
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                int n = at(i, j, k);
                for (int axis = 0; axis < 3; ++axis)
                {
                    int position = axis == 0 ? i : (axis == 1 ? j : k), extent = axis == 0 ? nx : (axis == 1 ? ny : nz),
                        stride      = axis == 0 ? ny * nz : (axis == 1 ? nz : 1);
                    double expected = 0;
                    if (position < extent - 1 && (axis < 2 || (k > 0 && k < nz - 2)))
                    {
                        int lo = n, hi = n + stride;
                        if (axis < 2 && (position == 0 || position == extent - 2))
                        {
                            lo = n + (extent - 2 - position) * stride;
                            hi = n + (1 - position) * stride;
                        }
                        expected =
                            .25 * (double(input[3][lo]) + input[3][hi]) * (double(input[axis][lo]) + input[axis][hi]);
                    }
                    close(collocated_flux[axis].data()[n], expected);
                }
            }
    d.operation = RkTendencyOperation::Divergence;
    d.input     = {input[0].data(), collocated_flux[0].data(), collocated_flux[1].data(), collocated_flux[2].data()};
    const auto& divergence = run(1);
    double      total_rate = 0, absolute_rate = 0;
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                int n = at(i, j, k);
                if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0 || k == nz - 1)
                    CHECK(divergence[0].data()[n] == 0);
                else
                {
                    double expected =
                        -(double(collocated_flux[0].data()[n]) - collocated_flux[0].data()[n - ny * nz]) / d.dx -
                        (double(collocated_flux[1].data()[n]) - collocated_flux[1].data()[n - nz]) / d.dy -
                        (double(collocated_flux[2].data()[n]) - collocated_flux[2].data()[n - 1]) / d.dz;
                    close(divergence[0].data()[n], expected);
                    total_rate += divergence[0].data()[n];
                    absolute_rate += std::abs(divergence[0].data()[n]);
                }
            }
    CHECK(std::abs(total_rate) <= 2 * std::numeric_limits<float>::epsilon() * absolute_rate);
    for (auto operation : {RkTendencyOperation::BasePerturbationHalo,
                           RkTendencyOperation::ScalarHalo,
                           RkTendencyOperation::VerticalWindHalo,
                           RkTendencyOperation::VolumeConcentrationHalo})
    {
        d.operation        = operation;
        d.input            = {input[0].data(), input[3].data(), input[5].data()};
        const auto& closed = run();
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    int    n = at(i, j, k), si = i == 0 ? nx - 2 : (i == nx - 1 ? 1 : i),
                           sj = j == 0 ? ny - 2 : (j == ny - 1 ? 1 : j), sk = std::clamp(k, 1, nz - 2);
                    int    source   = at(si, sj, sk);
                    double expected = input[0][source];
                    if (operation == RkTendencyOperation::BasePerturbationHalo && sk != k)
                    {
                        float perturbation = input[0][source] - input[5][source];
                        expected           = float(input[5][n] + perturbation);
                    }
                    if (operation == RkTendencyOperation::VerticalWindHalo && sk != k)
                        expected = 0;
                    if (operation == RkTendencyOperation::VolumeConcentrationHalo && source != n)
                        expected = expected / input[3][source] * input[3][n];
                    close(closed[0].data()[n], expected);
                }
    }
    d.collocated = false;
    d.operation  = RkTendencyOperation::Acoustic;
    for (int n = 0; n < count; ++n)
    {
        input[0][n] = input[1][n] = input[2][n] = input[6][n] = 0;
        input[4][n] = input[5][n] = float(100000 - 125 * (n % nz));
    }
    for (int f = 0; f < 7; ++f)
        d.input[f] = input[f].data();
    const auto& rest = run();
    for (const auto& field : rest)
        for (std::size_t n = 0; n < field.size(); ++n)
            CHECK(field.data()[n] == 0);
    d.operation = RkTendencyOperation::Add;
    d.input[1]  = input[3].data();
    std::array<std::vector<float>, 5> untouched;
    for (int f = 0; f < 5; ++f)
    {
        untouched[f].assign(count, 17);
        d.output[f] = untouched[f].data();
    }
    const auto unchanged = [&]
    {
        for (const auto& f : untouched)
            for (float value : f)
                CHECK(value == 17);
    };
    const auto invalid = [&]
    {
        CHECK(runtime.backend->dispatch_rk_tendency(d) == RkCombinationStatus::InvalidInput);
        unchanged();
    };
    d.input[0] = nullptr;
    invalid();
    d.input[0] = input[0].data();
    d.nx       = std::numeric_limits<int>::max();
    invalid();
    d.nx = nx;
    d.nz = -1;
    invalid();
    d.nz = nz;
    d.dx = 0;
    invalid();
    d.dx = 100.25;
    d.dt = std::numeric_limits<double>::infinity();
    invalid();
    d.dt       = .25;
    d.input[0] = d.output[0] + 1;
    invalid();
    d.input[0]  = input[0].data();
    d.output[1] = d.output[0] + 1;
    invalid();
    d.output[1]     = untouched[1].data();
    input[0].back() = std::numeric_limits<float>::quiet_NaN();
    invalid();
    input[0].back() = 1;
    d.operation     = RkTendencyOperation::MixingRatio;
    input[3].back() = 0;
    invalid();
    input[3].back() = 1;
    d.operation     = RkTendencyOperation::Add;
    input[0].back() = input[3].back() = std::numeric_limits<float>::max();
    CHECK(runtime.backend->dispatch_rk_tendency(d) == RkCombinationStatus::NonFinite);
    ++dispatches;
    unchanged();
    CHECK(runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::RkTendency) == dispatches);
}
} // namespace
TEST_CASE("Metal native RK acoustic and transport tendencies preserve direct RHS contracts", "[metal][rk]")
{
    exercise("metal");
}
TEST_CASE("Vulkan native RK acoustic and transport tendencies preserve direct RHS contracts", "[vulkan][rk]")
{
    exercise("vulkan");
}
