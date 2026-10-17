#pragma once

#include <filesystem>
#include <cstdint>
#include <string>
#include <vector>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

namespace tmv::packaged_runtime
{
inline std::filesystem::path frameworks_for_executable(const std::filesystem::path& executable)
{
    auto directory = executable.parent_path();
    if (directory.filename() == "MacOS" && directory.parent_path().filename() == "Contents")
        return directory.parent_path() / "Frameworks";
    if (directory.filename() == "bin" && directory.parent_path().filename() == "Resources" &&
        directory.parent_path().parent_path().filename() == "Contents")
        return directory.parent_path().parent_path() / "Frameworks";
    return {};
}

inline std::string library(const char* name)
{
#if defined(__APPLE__)
    std::uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    if (size == 0) return {};
    std::vector<char> storage(size);
    if (_NSGetExecutablePath(storage.data(), &size) != 0) return {};
    std::error_code error;
    const auto executable = std::filesystem::weakly_canonical(std::filesystem::path(storage.data()), error);
    if (error) return {};
    const auto directory = frameworks_for_executable(executable);
    if (directory.empty()) return {};
    return (directory / name).string();
#else
    (void)name;
#endif
    return {};
}
}
