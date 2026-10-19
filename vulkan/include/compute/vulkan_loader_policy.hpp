#pragma once

#include <cerrno>
#include <cstdlib>
#include <mutex>
#include <string>
#include <system_error>
#include <utility>

namespace tmv_vulkan::loader_policy
{
inline constexpr const char* environment_name = "VK_LOADER_DISABLE_DYNAMIC_LIBRARY_UNLOADING";

inline std::mutex& environment_mutex()
{
    static std::mutex mutex;
    return mutex;
}

template <typename Read, typename Write, typename Open>
bool open_with_lifetime_policy(Read&& read, Write&& write, Open&& open, std::string& error)
{
    const std::lock_guard<std::mutex> lock(environment_mutex());
    if (read(environment_name) == nullptr)
    {
        const int result = write(environment_name, "1");
        if (result != 0)
        {
            error = std::string("failed to configure ") + environment_name + ": " +
                    std::error_code(result, std::generic_category()).message();
            return false;
        }
    }
    std::forward<Open>(open)();
    return true;
}

template <typename Open>
bool open_with_lifetime_policy(Open&& open, std::string& error)
{
    // Repeated ICD unloading can exhaust static TLS even after every Vulkan
    // object is destroyed. Configure library lifetime before loader constructors run.
    return open_with_lifetime_policy([](const char* name) { return std::getenv(name); },
                                     [](const char* name, const char* value)
                                     {
#if defined(_WIN32)
                                         return _putenv_s(name, value);
#else
                                         return setenv(name, value, 0) == 0 ? 0 : (errno != 0 ? errno : EINVAL);
#endif
                                     },
                                     std::forward<Open>(open),
                                     error);
}
} // namespace tmv_vulkan::loader_policy
