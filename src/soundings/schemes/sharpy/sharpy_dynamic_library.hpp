#pragma once

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <filesystem>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>
#include <algorithm>
#include <cctype>
#include "util/packaged_runtime.hpp"

namespace tmv::soundings::sharpy
{

inline void append_packaged_windows_library_candidates(
    std::vector<std::string>& candidates, const std::string& filename_prefix)
{
#if defined(_WIN32)
    std::vector<wchar_t> executable_path(32768, L'\0');
    const DWORD length = GetModuleFileNameW(
        nullptr, executable_path.data(), static_cast<DWORD>(executable_path.size()));
    if (length == 0 || length >= executable_path.size())
        return;
    const std::filesystem::path directory =
        std::filesystem::path(executable_path.data()).parent_path();
    std::error_code ec;
    for (std::filesystem::directory_iterator it(directory, ec), end;
         !ec && it != end; it.increment(ec))
    {
        if (!it->is_regular_file(ec) || ec)
        {
            ec.clear();
            continue;
        }
        std::string name = it->path().filename().string();
        std::string lower = name;
        std::transform(lower.begin(), lower.end(), lower.begin(),
                       [](unsigned char value)
                       { return static_cast<char>(std::tolower(value)); });
        if (lower.rfind(filename_prefix, 0) == 0 &&
            lower.size() >= 4 && lower.substr(lower.size() - 4) == ".dll")
            candidates.push_back(std::move(name));
    }
#else
    (void)candidates;
    (void)filename_prefix;
#endif
}

class SharedLibraryHandle
{
public:
    SharedLibraryHandle() = default;
    explicit SharedLibraryHandle(void* handle) : handle_(handle)
    {
    }

    SharedLibraryHandle(const SharedLibraryHandle&)            = delete;
    SharedLibraryHandle& operator=(const SharedLibraryHandle&) = delete;

    SharedLibraryHandle(SharedLibraryHandle&& other) noexcept : handle_(other.handle_)
    {
        other.handle_ = nullptr;
    }

    SharedLibraryHandle& operator=(SharedLibraryHandle&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            handle_       = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    ~SharedLibraryHandle()
    {
        reset();
    }

    void* get() const
    {
        return handle_;
    }
    explicit operator bool() const
    {
        return handle_ != nullptr;
    }

private:
    void reset()
    {
        if (handle_ != nullptr)
        {
#if defined(_WIN32)
            FreeLibrary(static_cast<HMODULE>(handle_));
#else
            dlclose(handle_);
#endif
            handle_ = nullptr;
        }
    }

    void* handle_ = nullptr;
};

inline SharedLibraryHandle
open_shared_library_first(const std::vector<std::string>& candidates, std::string& opened_path, std::string& error_out)
{
    std::ostringstream errors;
    auto selected = candidates;
#if defined(__APPLE__)
    for (const auto& candidate : candidates)
    {
        if (candidate.find('/') != std::string::npos || std::filesystem::path(candidate).extension() != ".dylib") continue;
        const auto bundled = tmv::packaged_runtime::library(candidate.c_str());
        if (!bundled.empty()) { selected = {bundled}; break; }
    }
#endif
    for (const std::string& candidate : selected)
    {
#if defined(_WIN32)
        void* handle = static_cast<void*>(LoadLibraryW(
            std::filesystem::path(candidate).c_str()));
#else
        dlerror();
        void* handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
        if (handle != nullptr)
        {
            opened_path = candidate;
            return SharedLibraryHandle(handle);
        }

#if defined(_WIN32)
        errors << candidate << ": "
               << std::system_category().message(static_cast<int>(GetLastError()))
               << "; ";
#else
        const char* dl_err = dlerror();
        errors << candidate << ": " << (dl_err ? dl_err : "unknown dlopen error") << "; ";
#endif
    }

    error_out = errors.str();
    return SharedLibraryHandle{};
}

template <typename FnPtr>
bool load_symbol(void* handle, const char* symbol, FnPtr& out)
{
#if defined(_WIN32)
    const FARPROC raw = GetProcAddress(static_cast<HMODULE>(handle), symbol);
    if (raw == nullptr)
    {
        out = nullptr;
        return false;
    }
    out = reinterpret_cast<FnPtr>(raw);
    return true;
#else
    dlerror();
    void*       raw   = dlsym(handle, symbol);
    const char* error = dlerror();
    if (raw == nullptr || error != nullptr)
    {
        out = nullptr;
        return false;
    }
    out = reinterpret_cast<FnPtr>(raw);
    return true;
#endif
}

} // namespace tmv::soundings::sharpy
