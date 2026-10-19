#pragma once

namespace tmv
{
inline float kw78_updated_face(
    float previous, float phase_speed, float normal_gradient, float dt_rad, float inv_spacing, bool negative_outgoing)
{
    if (negative_outgoing ? phase_speed >= 0.0f : phase_speed <= 0.0f)
        return previous;
    // Keep the four FP32 operations identical to the precise device boundary update.
    volatile float dt_phase  = dt_rad * phase_speed;
    volatile float tendency  = dt_phase * normal_gradient;
    volatile float increment = tendency * inv_spacing;
    return previous - increment;
}
} // namespace tmv
