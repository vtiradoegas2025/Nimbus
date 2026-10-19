#include "init/config/config_v1.hpp"

#include <algorithm>
#include <cctype>

namespace tmv::init::config
{

namespace
{

bool internal_moving_nest_qualification = false;

std::string to_lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return s;
}

} // namespace

void set_internal_moving_nest_qualification(bool enabled)
{
    internal_moving_nest_qualification = enabled;
}

bool internal_moving_nest_qualification_enabled()
{
    return internal_moving_nest_qualification;
}

// ─────────────────────────────────────────────────────────────
// LateralBC
// ─────────────────────────────────────────────────────────────

bool parse_lateral_bc(const std::string& s, LateralBC& out)
{
    const std::string v = to_lower(s);
    if (v == "periodic")
    {
        out = LateralBC::Periodic;
        return true;
    }
    if (v == "radiative")
    {
        out = LateralBC::Radiative;
        return true;
    }
    if (v == "zero_gradient")
    {
        out = LateralBC::ZeroGradient;
        return true;
    }
    if (v == "open")
    {
        out = LateralBC::LegacyOpen;
        return true;
    }
    if (v == "rigid")
    {
        out = LateralBC::Rigid;
        return true;
    }
    if (v == "rigid_slip")
    {
        out = LateralBC::RigidSlip;
        return true;
    }
    return false;
}

const char* lateral_bc_name(LateralBC v)
{
    switch (v)
    {
    case LateralBC::Periodic:
        return "periodic";
    case LateralBC::Radiative:
        return "radiative";
    case LateralBC::ZeroGradient:
        return "zero_gradient";
    case LateralBC::LegacyOpen:
        return "open";
    case LateralBC::Rigid:
        return "rigid";
    case LateralBC::RigidSlip:
        return "rigid_slip";
    }
    return "zero_gradient";
}

LateralBC resolve_legacy_lateral_bc(LateralBC v, StaggerType staggering)
{
    if (v != LateralBC::LegacyOpen)
    {
        return v;
    }
    return staggering == StaggerType::CGrid ? LateralBC::Radiative : LateralBC::ZeroGradient;
}

// ─────────────────────────────────────────────────────────────
// BottomBC
// ─────────────────────────────────────────────────────────────

bool parse_bottom_bc(const std::string& s, BottomBC& out)
{
    const std::string v = to_lower(s);
    if (v == "rigid_noslip")
    {
        out = BottomBC::RigidNoSlip;
        return true;
    }
    if (v == "rigid_slip")
    {
        out = BottomBC::RigidSlip;
        return true;
    }
    return false;
}

const char* bottom_bc_name(BottomBC v)
{
    switch (v)
    {
    case BottomBC::RigidNoSlip:
        return "rigid_noslip";
    case BottomBC::RigidSlip:
        return "rigid_slip";
    }
    return "rigid_noslip";
}

// ─────────────────────────────────────────────────────────────
// TopBC
// ─────────────────────────────────────────────────────────────

bool parse_top_bc(const std::string& s, TopBC& out)
{
    const std::string v = to_lower(s);
    if (v == "rigid")
    {
        out = TopBC::Rigid;
        return true;
    }
    if (v == "rigid_with_sponge")
    {
        out = TopBC::RigidWithSponge;
        return true;
    }
    if (v == "open")
    {
        out = TopBC::Open;
        return true;
    }
    return false;
}

const char* top_bc_name(TopBC v)
{
    switch (v)
    {
    case TopBC::Rigid:
        return "rigid";
    case TopBC::RigidWithSponge:
        return "rigid_with_sponge";
    case TopBC::Open:
        return "open";
    }
    return "rigid_with_sponge";
}

// ─────────────────────────────────────────────────────────────
// DampingMode
// ─────────────────────────────────────────────────────────────

bool parse_damping_mode(const std::string& s, DampingMode& out)
{
    const std::string v = to_lower(s);
    if (v == "explicit")
    {
        out = DampingMode::Explicit;
        return true;
    }
    if (v == "implicit")
    {
        out = DampingMode::Implicit;
        return true;
    }
    return false;
}

const char* damping_mode_name(DampingMode v)
{
    switch (v)
    {
    case DampingMode::Explicit:
        return "explicit";
    case DampingMode::Implicit:
        return "implicit";
    }
    return "explicit";
}

// ─────────────────────────────────────────────────────────────
// CoriolisMode
// ─────────────────────────────────────────────────────────────

bool parse_coriolis_mode(const std::string& s, CoriolisMode& out)
{
    const std::string v = to_lower(s);
    if (v == "f_plane")
    {
        out = CoriolisMode::FPlane;
        return true;
    }
    if (v == "beta_plane")
    {
        out = CoriolisMode::BetaPlane;
        return true;
    }
    return false;
}

const char* coriolis_mode_name(CoriolisMode v)
{
    switch (v)
    {
    case CoriolisMode::FPlane:
        return "f_plane";
    case CoriolisMode::BetaPlane:
        return "beta_plane";
    }
    return "f_plane";
}

// ─────────────────────────────────────────────────────────────
// Domain motion and lifetime planning
// ─────────────────────────────────────────────────────────────

bool parse_time_integrator(const std::string& s, TimeIntegrator& out)
{
    const std::string v = to_lower(s);
    if (v == "split_explicit")
    {
        out = TimeIntegrator::SplitExplicit;
        return true;
    }
    if (v == "rk3" || v == "ssprk3")
    {
        out = TimeIntegrator::SSPRK3;
        return true;
    }
    if (v == "rk4")
    {
        out = TimeIntegrator::RK4;
        return true;
    }
    return false;
}

const char* time_integrator_name(TimeIntegrator v)
{
    switch (v)
    {
    case TimeIntegrator::SplitExplicit:
        return "split_explicit";
    case TimeIntegrator::SSPRK3:
        return "rk3";
    case TimeIntegrator::RK4:
        return "rk4";
    }
    return "split_explicit";
}

bool parse_domain_motion_mode(const std::string& s, DomainMotionMode& out)
{
    const std::string v = to_lower(s);
    if (v == "stationary")
    {
        out = DomainMotionMode::Stationary;
        return true;
    }
    if (v == "constant_velocity")
    {
        out = DomainMotionMode::ConstantVelocity;
        return true;
    }
    if (v == "tracked_window")
    {
        out = DomainMotionMode::TrackedWindow;
        return true;
    }
    if (v == "moving_nest")
    {
        out = DomainMotionMode::MovingNest;
        return true;
    }
    return false;
}

const char* domain_motion_mode_name(DomainMotionMode v)
{
    switch (v)
    {
    case DomainMotionMode::Stationary:
        return "stationary";
    case DomainMotionMode::ConstantVelocity:
        return "constant_velocity";
    case DomainMotionMode::TrackedWindow:
        return "tracked_window";
    case DomainMotionMode::MovingNest:
        return "moving_nest";
    }
    return "stationary";
}

bool parse_nesting_feedback(const std::string& s, NestingFeedback& out)
{
    const std::string value = to_lower(s);
    if (value == "one_way")
    {
        out = NestingFeedback::OneWay;
        return true;
    }
    if (value == "two_way")
    {
        out = NestingFeedback::TwoWay;
        return true;
    }
    return false;
}

const char* nesting_feedback_name(NestingFeedback v)
{
    switch (v)
    {
    case NestingFeedback::OneWay:
        return "one_way";
    case NestingFeedback::TwoWay:
        return "two_way";
    }
    return "one_way";
}

bool parse_lifetime_planning_action(const std::string& s, LifetimePlanningAction& out)
{
    const std::string v = to_lower(s);
    if (v == "warn")
    {
        out = LifetimePlanningAction::Warn;
        return true;
    }
    if (v == "error")
    {
        out = LifetimePlanningAction::Error;
        return true;
    }
    return false;
}

const char* lifetime_planning_action_name(LifetimePlanningAction v)
{
    switch (v)
    {
    case LifetimePlanningAction::Warn:
        return "warn";
    case LifetimePlanningAction::Error:
        return "error";
    }
    return "error";
}

bool parse_storm_motion_source(const std::string& s, StormMotionSource& out)
{
    const std::string v = to_lower(s);
    if (v == "bunkers_right")
    {
        out = StormMotionSource::BunkersRight;
        return true;
    }
    if (v == "bunkers_left")
    {
        out = StormMotionSource::BunkersLeft;
        return true;
    }
    if (v == "explicit")
    {
        out = StormMotionSource::Explicit;
        return true;
    }
    return false;
}

const char* storm_motion_source_name(StormMotionSource v)
{
    switch (v)
    {
    case StormMotionSource::BunkersRight:
        return "bunkers_right";
    case StormMotionSource::BunkersLeft:
        return "bunkers_left";
    case StormMotionSource::Explicit:
        return "explicit";
    }
    return "bunkers_right";
}

bool parse_storm_branch_preference(const std::string& s, StormBranchPreference& out)
{
    const std::string v = to_lower(s);
    if (v == "continuity")
    {
        out = StormBranchPreference::Continuity;
        return true;
    }
    if (v == "bunkers_right")
    {
        out = StormBranchPreference::BunkersRight;
        return true;
    }
    if (v == "bunkers_left")
    {
        out = StormBranchPreference::BunkersLeft;
        return true;
    }
    return false;
}

const char* storm_branch_preference_name(StormBranchPreference v)
{
    switch (v)
    {
    case StormBranchPreference::Continuity:
        return "continuity";
    case StormBranchPreference::BunkersRight:
        return "bunkers_right";
    case StormBranchPreference::BunkersLeft:
        return "bunkers_left";
    }
    return "continuity";
}

// ─────────────────────────────────────────────────────────────
} // namespace tmv::init::config
