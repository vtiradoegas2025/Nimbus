#include "core/runtime/termination_signal.hpp"

#include <cerrno>
#include <cstring>
#include <system_error>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace tmv::runtime
{
namespace
{

volatile std::sig_atomic_t termination_signal = 0;

#if defined(_WIN32)
HANDLE termination_event = nullptr;

std::wstring termination_event_name()
{
    return L"Local\\Nimbus_Terminate_" + std::to_wstring(GetCurrentProcessId());
}
#endif

extern "C" void record_termination_signal(int signal_number) noexcept
{
    if (termination_signal == 0)
        termination_signal = static_cast<std::sig_atomic_t>(signal_number);
}

#if defined(_WIN32)
BOOL WINAPI record_console_control(DWORD control_type)
{
    switch (control_type)
    {
    case CTRL_C_EVENT:
        record_termination_signal(SIGINT);
        return TRUE;
    case CTRL_BREAK_EVENT:
    case CTRL_CLOSE_EVENT:
    case CTRL_LOGOFF_EVENT:
    case CTRL_SHUTDOWN_EVENT:
        record_termination_signal(SIGTERM);
        return TRUE;
    default:
        return FALSE;
    }
}
#endif

} // namespace

int requested_termination_signal() noexcept
{
#if defined(_WIN32)
    if (termination_signal == 0 && termination_event != nullptr &&
        WaitForSingleObject(termination_event, 0) == WAIT_OBJECT_0)
    {
        record_termination_signal(SIGTERM);
    }
#endif
    return static_cast<int>(termination_signal);
}

const char* termination_signal_name(int signal_number) noexcept
{
    switch (signal_number)
    {
    case SIGINT:
#if defined(_WIN32)
        return "CTRL_C";
#else
        return "SIGINT";
#endif
    case SIGTERM:
#if defined(_WIN32)
        return "CTRL_BREAK";
#else
        return "SIGTERM";
#endif
    default:
        return "UNKNOWN";
    }
}

ScopedTerminationSignalHandlers::ScopedTerminationSignalHandlers() noexcept
{
    termination_signal = 0;
#if defined(_WIN32)
    termination_event = CreateEventW(nullptr, TRUE, FALSE,
                                     termination_event_name().c_str());
    if (termination_event == nullptr)
    {
        error_ = "failed to create Windows termination event: " +
                 std::system_category().message(static_cast<int>(GetLastError()));
        return;
    }
    if (SetConsoleCtrlHandler(record_console_control, TRUE) == 0)
    {
        error_ = "failed to install Windows console control handler: " +
                 std::system_category().message(static_cast<int>(GetLastError()));
        CloseHandle(termination_event);
        termination_event = nullptr;
        return;
    }
    installed_ = true;
#else
    errno               = 0;
    previous_sigint_    = std::signal(SIGINT, record_termination_signal);
    if (previous_sigint_ == SIG_ERR)
    {
        error_ = "failed to install SIGINT handler: " + std::string(std::strerror(errno));
        return;
    }
    errno            = 0;
    previous_sigterm_ = std::signal(SIGTERM, record_termination_signal);
    if (previous_sigterm_ == SIG_ERR)
    {
        const int failure_errno = errno;
        std::signal(SIGINT, previous_sigint_);
        error_ = "failed to install SIGTERM handler: " + std::string(std::strerror(failure_errno));
        return;
    }
    installed_ = true;
#endif
}

ScopedTerminationSignalHandlers::~ScopedTerminationSignalHandlers()
{
    if (!installed_)
        return;
#if defined(_WIN32)
    SetConsoleCtrlHandler(record_console_control, FALSE);
    if (termination_event != nullptr)
    {
        CloseHandle(termination_event);
        termination_event = nullptr;
    }
#else
    std::signal(SIGINT, previous_sigint_);
    std::signal(SIGTERM, previous_sigterm_);
#endif
}

} // namespace tmv::runtime
