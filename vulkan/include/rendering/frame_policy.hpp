/**
 * @file frame_policy.hpp
 * @brief Pure native-viewer redraw and event-wait policy.
 */

#pragma once

#include <algorithm>

namespace vkcpp::frame_policy
{

[[nodiscard]] constexpr bool should_redraw(bool first_frame,
                                           bool backend_dirty,
                                           bool framebuffer_resized,
                                           bool capture_pending,
                                           bool forced_frame_count)
{
    return first_frame || backend_dirty || framebuffer_resized || capture_pending ||
           forced_frame_count;
}

[[nodiscard]] inline double wait_timeout_seconds(bool first_frame,
                                                 bool active_deadline,
                                                 float viewer_fps)
{
    if (first_frame) { return 0.0; }
    if (!active_deadline) { return 0.05; }
    return 1.0 / static_cast<double>(std::max(viewer_fps, 1.0f));
}

[[nodiscard]] constexpr bool qualification_resize_due(int configured_frame,
                                                      int rendered_frames,
                                                      bool already_requested)
{
    return !already_requested && configured_frame >= 0 &&
           rendered_frames == configured_frame;
}

[[nodiscard]] constexpr bool duration_complete(double configured_seconds,
                                               double elapsed_seconds)
{
    return configured_seconds > 0.0 && elapsed_seconds >= configured_seconds;
}

[[nodiscard]] constexpr bool interval_capture_allowed(int configured_limit,
                                                      int completed_or_pending)
{
    return configured_limit <= 0 || completed_or_pending < configured_limit;
}

}  // namespace vkcpp::frame_policy
