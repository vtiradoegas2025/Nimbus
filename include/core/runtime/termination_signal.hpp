#pragma once

#include <csignal>
#include <string>

namespace tmv::runtime
{

/** Signal number requested by the process handler, or zero when none. */
int requested_termination_signal() noexcept;

/** Stable diagnostic name for the supported termination signals. */
const char* termination_signal_name(int signal_number) noexcept;

/**
 * Installs the native graceful-stop bridge. POSIX uses SIGINT/SIGTERM handlers;
 * Windows combines console-control handling with the per-process named event
 * used by Nimbus Studio. Handler context performs only a sig_atomic_t store.
 * Previous process handlers/resources are restored on scope exit.
 */
class ScopedTerminationSignalHandlers
{
  public:
    ScopedTerminationSignalHandlers() noexcept;
    ~ScopedTerminationSignalHandlers();

    ScopedTerminationSignalHandlers(const ScopedTerminationSignalHandlers&)            = delete;
    ScopedTerminationSignalHandlers& operator=(const ScopedTerminationSignalHandlers&) = delete;

    bool               installed() const noexcept { return installed_; }
    const std::string& error_message() const noexcept { return error_; }

  private:
#if !defined(_WIN32)
    using Handler = void (*)(int);
    Handler previous_sigint_  = SIG_DFL;
    Handler previous_sigterm_ = SIG_DFL;
#endif
    bool   installed_         = false;
    std::string error_;
};

} // namespace tmv::runtime
