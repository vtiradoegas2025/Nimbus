#pragma once

#include <charconv>
#include <cmath>
#include <stdexcept>
#include <string>
#if defined(__APPLE__)
#include <cerrno>
#include <cfenv>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <xlocale.h>
#endif

namespace tmv::strutil
{
#if defined(__APPLE__)
inline std::string format_double_roundtrip_legacy(double value)
{
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const auto fraction = bits & 0x000fffffffffffffull;
    if ((bits & 0x7ff0000000000000ull) == 0x7ff0000000000000ull)
    {
        const std::string sign = bits >> 63 ? "-" : "";
        if (!fraction) return sign + "inf";
        if (!(fraction & 0x0008000000000000ull)) return sign + "nan(snan)";
        return bits == 0xfff8000000000000ull ? "-nan(ind)" : sign + "nan";
    }
    if (value == 0.0) return std::signbit(value) ? "-0" : "0";
    struct Environment
    {
        int previous_errno = errno;
        std::fenv_t saved;
        Environment()
        {
            if (std::feholdexcept(&saved) != 0) throw std::runtime_error("cannot preserve floating-point environment");
            std::fesetround(FE_TONEAREST);
        }
        ~Environment() { std::fesetenv(&saved); errno = previous_errno; }
    } environment;
    struct Locale
    {
        locale_t value = newlocale(LC_NUMERIC_MASK, "C", nullptr);
        ~Locale() { if (value) freelocale(value); }
    };
    static const Locale locale;
    if (!locale.value) throw std::runtime_error("cannot create numeric locale");
    char buffer[64];
    bool found = false;
    for (int digits = 1; digits <= 17; ++digits)
    {
        // At binary powers the rounding interval is asymmetric. The nearest
        // decimal may miss it while the adjacent decimal still round-trips.
        for (const int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD})
        {
            std::fesetround(rounding);
            const int length = snprintf_l(buffer, sizeof(buffer), locale.value, "%.*e", digits - 1, value);
            std::fesetround(FE_TONEAREST);
            if (length < 0 || static_cast<std::size_t>(length) >= sizeof(buffer))
                throw std::runtime_error("cannot format configuration number");
            char* end = nullptr;
            const double parsed = strtod_l(buffer, &end, locale.value);
            if (end && *end == '\0' && parsed == value) { found = true; break; }
        }
        if (found) break;
    }
    if (!found) throw std::runtime_error("cannot round-trip configuration number");
    std::string scientific(buffer);
    const auto e = scientific.find('e');
    const int exponent = std::stoi(scientific.substr(e + 1));
    std::string digits = scientific.substr(0, e);
    std::string sign;
    if (digits.front() == '-') { sign = "-"; digits.erase(0, 1); }
    const auto dot = digits.find('.');
    if (dot != std::string::npos) digits.erase(dot, 1);
    const int point = exponent + 1;
    std::string fixed;
    if (point <= 0) fixed = "0." + std::string(-point, '0') + digits;
    else if (point >= static_cast<int>(digits.size())) fixed = digits + std::string(point - digits.size(), '0');
    else fixed = digits.substr(0, point) + "." + digits.substr(point);
    fixed = sign + fixed;
    const double magnitude = std::abs(value);
    return magnitude >= 1e-4 && magnitude < 1e15 ? fixed : scientific;
}
#endif

inline std::string format_double_roundtrip(double value)
{
#if defined(__APPLE__) && __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ < 130300
    return format_double_roundtrip_legacy(value);
#else
    char buffer[64];
    const double magnitude = std::abs(value);
    const auto format = magnitude == 0.0 || (magnitude >= 1e-4 && magnitude < 1e15)
        ? std::chars_format::fixed : std::chars_format::general;
    const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value, format);
    if (result.ec != std::errc{})
        throw std::runtime_error("cannot format configuration number");
    return {buffer, result.ptr};
#endif
}
}
