#pragma once

#include <filesystem>
#include <string>
#include <system_error>
#include <utility>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#endif

/**
 * @file durable_file.hpp
 * @brief Make published output survive power loss, not just process death.
 *
 * `rename` is atomic with respect to other processes, so a reader never
 * observes a half-written manifest. It is not *durable*: after a power cut the
 * rename can land while the payload's data blocks have not, leaving a manifest
 * that advertises fields full of zeros. Studio's completeness check passes,
 * because file size is metadata and metadata survives -- the contents are what
 * is lost.
 *
 * Closing an ofstream only pushes bytes into the page cache. Reaching stable
 * storage requires an OS durability primitive: fsync/F_FULLFSYNC on Unix and
 * FlushFileBuffers on Windows. POSIX directory handles are flushed as well;
 * Windows couples publication to MoveFileExW(MOVEFILE_WRITE_THROUGH), since it
 * does not expose a supported directory-fsync equivalent.
 */
namespace nimbus::output
{

namespace detail
{

#if defined(_WIN32)
inline std::string describe_windows_error(const char* what,
                                          const std::filesystem::path& path,
                                          DWORD error_code)
{
    return std::string(what) + " '" + path.string() + "': " +
           std::system_category().message(static_cast<int>(error_code));
}

inline bool flush_windows_path(const std::filesystem::path& path,
                               DWORD                       desired_access,
                               DWORD                       flags,
                               bool                        flush_handle,
                               const char*                 what,
                               std::string&                error)
{
    const HANDLE handle = CreateFileW(path.c_str(), desired_access,
                                      FILE_SHARE_READ | FILE_SHARE_WRITE |
                                          FILE_SHARE_DELETE,
                                      nullptr, OPEN_EXISTING, flags, nullptr);
    if (handle == INVALID_HANDLE_VALUE)
    {
        error = describe_windows_error(what, path, GetLastError());
        return false;
    }
    if (!flush_handle)
    {
        BY_HANDLE_FILE_INFORMATION information{};
        if (GetFileInformationByHandle(handle, &information) == 0)
        {
            error = describe_windows_error("failed to inspect directory", path, GetLastError());
            CloseHandle(handle);
            return false;
        }
        if ((information.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0)
        {
            error = describe_windows_error("expected a directory", path, ERROR_DIRECTORY);
            CloseHandle(handle);
            return false;
        }
    }
    // FlushFileBuffers requires a handle opened with GENERIC_WRITE. Windows
    // does not expose a supported directory-fsync equivalent, so directory
    // calls validate/open the directory while publication itself uses
    // MoveFileExW(MOVEFILE_WRITE_THROUGH) below.
    const bool ok = !flush_handle || FlushFileBuffers(handle) != 0;
    if (!ok)
    {
        error = describe_windows_error("failed to flush", path, GetLastError());
    }
    CloseHandle(handle);
    return ok;
}
#else
inline std::string describe_errno(const char* what, const std::filesystem::path& path)
{
    return std::string(what) + " '" + path.string() + "': " + std::strerror(errno);
}

inline bool sync_file_descriptor(int fd)
{
#if defined(__APPLE__)
    // fsync on macOS only pushes as far as the drive's own write cache;
    // F_FULLFSYNC asks the drive to flush that cache too. Some filesystems do
    // not implement it, and there fsync is the strongest guarantee available.
    if (::fcntl(fd, F_FULLFSYNC) == 0)
    {
        return true;
    }
    if (errno != ENOTSUP && errno != EINVAL && errno != EOPNOTSUPP)
    {
        return false;
    }
#endif
    return ::fsync(fd) == 0;
}
#endif

} // namespace detail

/** Flush an already written and closed file to stable storage. */
inline bool sync_file_to_disk(const std::filesystem::path& path, std::string& error)
{
#if defined(_WIN32)
    return detail::flush_windows_path(path, GENERIC_READ | GENERIC_WRITE,
                                      FILE_ATTRIBUTE_NORMAL, true,
                                      "failed to open for durable flush", error);
#else
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0)
    {
        error = detail::describe_errno("failed to open for fsync", path);
        return false;
    }
    const bool ok = detail::sync_file_descriptor(fd);
    if (!ok)
    {
        error = detail::describe_errno("failed to fsync", path);
    }
    ::close(fd);
    return ok;
#endif
}

/**
 * Complete the platform's directory-publication barrier. POSIX flushes the
 * directory handle. Windows validates that the directory remains reachable;
 * its actual rename durability is requested by atomic_replace_file with
 * MOVEFILE_WRITE_THROUGH.
 */
inline bool sync_directory_to_disk(const std::filesystem::path& path, std::string& error)
{
#if defined(_WIN32)
    return detail::flush_windows_path(path, FILE_READ_ATTRIBUTES,
                                      FILE_FLAG_BACKUP_SEMANTICS, false,
                                      "failed to open directory for durable flush", error);
#else
    const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY);
    if (fd < 0)
    {
        error = detail::describe_errno("failed to open directory for fsync", path);
        return false;
    }
    // Some filesystems reject fsync on a directory descriptor; there the entry
    // is made durable by other means, so EINVAL is not a failure.
    const bool ok = ::fsync(fd) == 0 || errno == EINVAL;
    if (!ok)
    {
        error = detail::describe_errno("failed to fsync directory", path);
    }
    ::close(fd);
    return ok;
#endif
}

/**
 * Atomically publish `temporary` at `destination`, replacing an older file.
 *
 * `std::filesystem::rename` has replacement semantics on POSIX but deliberately
 * does not replace an existing destination on Windows.  Nimbus republishes
 * manifests, checkpoints, trajectories and history products, so relying on
 * that difference either breaks the second write or tempts callers to unlink
 * first (which creates a reader-visible gap).  Use the native single-operation
 * replacement primitive on each platform instead.
 *
 * Both paths require source and destination to be on the same filesystem.
 * Windows additionally requests write-through so the replacement is not left
 * pending only in the system cache.  Callers that need the full crash-durable
 * contract should flush the temporary payload before this call and its parent
 * directory after it.
 */
inline bool atomic_replace_file(const std::filesystem::path& temporary,
                                const std::filesystem::path& destination,
                                std::string&                 error)
{
    error.clear();
#if defined(_WIN32)
    if (MoveFileExW(temporary.c_str(), destination.c_str(),
                    MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) == 0)
    {
        error = detail::describe_windows_error(
            "failed to atomically replace", destination, GetLastError());
        return false;
    }
    return true;
#else
    std::error_code ec;
    std::filesystem::rename(temporary, destination, ec);
    if (ec)
    {
        error = "failed to atomically replace '" + destination.string() +
                "': " + ec.message();
        return false;
    }
    return true;
#endif
}

/**
 * Flush every regular file directly inside `directory`.
 *
 * Deliberately keyed on directory contents rather than a list of paths the
 * caller tracked: a step directory holds npy, zfp3d, csv, sidecar and
 * validation products written by several code paths, and a list would silently
 * go stale the next time a writer is added. Non-recursive -- step directories
 * are flat.
 */
inline bool sync_directory_contents_to_disk(const std::filesystem::path& directory,
                                            std::string&                 error)
{
    std::error_code ec;
    std::filesystem::directory_iterator it(directory, ec);
    if (ec)
    {
        error = "failed to list '" + directory.string() + "' for fsync: " + ec.message();
        return false;
    }
    for (const auto& entry : it)
    {
        if (!entry.is_regular_file(ec) || ec)
        {
            ec.clear();
            continue;
        }
        if (!sync_file_to_disk(entry.path(), error))
        {
            return false;
        }
    }
    return true;
}

/**
 * Make every payload durable before invoking the completion-manifest
 * publisher. A failed payload sync must not call the publisher at all.
 */
template <typename PublishManifest>
inline bool publish_manifest_after_durable_payloads(
    const std::filesystem::path& payload_directory,
    PublishManifest&&            publish_manifest,
    std::string&                 error)
{
    if (!sync_directory_contents_to_disk(payload_directory, error))
    {
        return false;
    }
    return std::forward<PublishManifest>(publish_manifest)();
}

} // namespace nimbus::output
