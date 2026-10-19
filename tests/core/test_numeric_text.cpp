#include "catch2/catch.hpp"
#include "util/numeric_text.hpp"

#include <cerrno>
#include <cfenv>
#include <cstring>
#include <limits>
#include <random>
#include <thread>
#include <vector>

TEST_CASE("Canonical numbers preserve signed zero, precision and notation boundaries", "[core][numeric-text]")
{
    using tmv::strutil::format_double_roundtrip;
    for (const auto& entry : std::vector<std::pair<double, std::string>>{
             {0.0, "0"}, {-0.0, "-0"}, {0.0001, "0.0001"}, {0.00001, "1e-05"},
             {1e14, "100000000000000"}, {1e15, "1e+15"}, {1.2345678901234567, "1.2345678901234567"},
             {std::numeric_limits<double>::denorm_min(), "5e-324"},
             {std::numeric_limits<double>::max(), "1.7976931348623157e+308"},
             {std::numeric_limits<double>::infinity(), "inf"}})
    {
        CAPTURE(entry.first);
        REQUIRE(format_double_roundtrip(entry.first) == entry.second);
    }
}

#if defined(__APPLE__)
TEST_CASE("Compatibility formatter preserves the caller's rounding, exceptions and errno", "[core][numeric-text]")
{
    std::fenv_t original;
    std::fegetenv(&original);
    const int original_errno = errno;
    struct Restore
    {
        std::fenv_t environment;
        int error;
        ~Restore() { std::fesetenv(&environment); errno = error; }
    } restore{original, original_errno};
    for (int mode : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO})
    {
        std::fesetround(mode);
        std::feclearexcept(FE_ALL_EXCEPT);
        std::feraiseexcept(FE_DIVBYZERO);
        errno = EDOM;
        const auto text = tmv::strutil::format_double_roundtrip_legacy(1e-200);
        const int error = errno, rounding = std::fegetround(), exceptions = std::fetestexcept(FE_ALL_EXCEPT);
        REQUIRE(text == "1e-200");
        REQUIRE(error == EDOM);
        REQUIRE(rounding == mode);
        REQUIRE(exceptions == FE_DIVBYZERO);
    }
}

TEST_CASE("Compatibility formatter is independent across concurrent callers", "[core][numeric-text]")
{
    std::vector<std::thread> threads;
    std::vector<int> failures(8, 0);
    for (std::size_t i = 0; i < failures.size(); ++i)
        threads.emplace_back([i, &failures] {
            std::fesetround(i % 2 ? FE_UPWARD : FE_DOWNWARD);
            for (int j = 0; j < 128; ++j)
                if (tmv::strutil::format_double_roundtrip_legacy(1.2345678901234567) != "1.2345678901234567")
                    ++failures[i];
        });
    for (auto& thread : threads) thread.join();
    for (int failures_on_thread : failures) REQUIRE(failures_on_thread == 0);
}

#if __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ >= 130300
TEST_CASE("Compatibility canonical strings match libc++ across binary powers and seeded bits", "[core][numeric-text][oracle]")
{
    const auto compare = [](double value) {
        REQUIRE(tmv::strutil::format_double_roundtrip_legacy(value) ==
                tmv::strutil::format_double_roundtrip(value));
    };
    std::mt19937_64 random(18231);
    for (int i = 0; i < 100000; ++i)
    {
        const auto bits = random();
        double value;
        std::memcpy(&value, &bits, sizeof(value));
        compare(value);
    }
    for (int exponent = -1074; exponent <= 1023; ++exponent)
    {
        const double value = std::ldexp(1.0, exponent);
        for (double v : {value, -value, std::nextafter(value, 0.0), std::nextafter(value, INFINITY)}) compare(v);
    }
}
#endif
#endif
