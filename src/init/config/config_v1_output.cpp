#include "init/config/config_v1.hpp"

#include <algorithm>
#include <cctype>

namespace tmv::init::config
{

namespace
{

std::string to_lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return s;
}

} // namespace

// ─────────────────────────────────────────────────────────────
// OutputPreset
// ─────────────────────────────────────────────────────────────

bool parse_output_preset(const std::string& s, OutputPreset& out)
{
    const std::string v = to_lower(s);
    if (v == "minimal")
    {
        out = OutputPreset::Minimal;
        return true;
    }
    if (v == "radar_ready")
    {
        out = OutputPreset::RadarReady;
        return true;
    }
    if (v == "radar_zdr_exploratory")
    {
        out = OutputPreset::RadarZdrExploratory;
        return true;
    }
    if (v == "radar_polarimetric_ready")
    {
        out = OutputPreset::RadarPolarimetricReady;
        return true;
    }
    if (v == "presentation")
    {
        out = OutputPreset::Presentation;
        return true;
    }
    if (v == "studio_full")
    {
        out = OutputPreset::StudioFull;
        return true;
    }
    if (v == "dynamics_full")
    {
        out = OutputPreset::DynamicsFull;
        return true;
    }
    if (v == "science_validation")
    {
        out = OutputPreset::ScienceValidation;
        return true;
    }
    return false;
}

const char* output_preset_name(OutputPreset v)
{
    switch (v)
    {
    case OutputPreset::Minimal:
        return "minimal";
    case OutputPreset::RadarReady:
        return "radar_ready";
    case OutputPreset::RadarZdrExploratory:
        return "radar_zdr_exploratory";
    case OutputPreset::RadarPolarimetricReady:
        return "radar_polarimetric_ready";
    case OutputPreset::Presentation:
        return "presentation";
    case OutputPreset::StudioFull:
        return "studio_full";
    case OutputPreset::DynamicsFull:
        return "dynamics_full";
    case OutputPreset::ScienceValidation:
        return "science_validation";
    }
    return "minimal";
}

bool parse_qualification_diagnostics_preset(
    const std::string& s, QualificationDiagnosticsPreset& out)
{
    const std::string v = to_lower(s);
    if (v == "none")
    {
        out = QualificationDiagnosticsPreset::None;
        return true;
    }
    if (v == "moving_reference_v1")
    {
        out = QualificationDiagnosticsPreset::MovingReferenceV1;
        return true;
    }
    return false;
}

const char* qualification_diagnostics_preset_name(
    QualificationDiagnosticsPreset v)
{
    switch (v)
    {
    case QualificationDiagnosticsPreset::None:
        return "none";
    case QualificationDiagnosticsPreset::MovingReferenceV1:
        return "moving_reference_v1";
    }
    return "none";
}

// ─────────────────────────────────────────────────────────────
// OutputFormat
// ─────────────────────────────────────────────────────────────

bool parse_output_format(const std::string& s, OutputFormat& out)
{
    const std::string v = to_lower(s);
    if (v == "npy_3d")
    {
        out = OutputFormat::Npy3D;
        return true;
    }
    if (v == "netcdf")
    {
        out = OutputFormat::Netcdf;
        return true;
    }
    if (v == "zfp_low")
    {
        out = OutputFormat::ZfpLow;
        return true;
    }
    if (v == "zfp_medium")
    {
        out = OutputFormat::ZfpMedium;
        return true;
    }
    if (v == "zfp_high")
    {
        out = OutputFormat::ZfpHigh;
        return true;
    }
    return false;
}

const char* output_format_name(OutputFormat v)
{
    switch (v)
    {
    case OutputFormat::Npy3D:
        return "npy_3d";
    case OutputFormat::Netcdf:
        return "netcdf";
    case OutputFormat::ZfpLow:
        return "zfp_low";
    case OutputFormat::ZfpMedium:
        return "zfp_medium";
    case OutputFormat::ZfpHigh:
        return "zfp_high";
    }
    return "npy_3d";
}

// ─────────────────────────────────────────────────────────────
// ComputeBackend
// ─────────────────────────────────────────────────────────────

bool parse_compute_backend(const std::string& s, ComputeBackend& out)
{
    const std::string v = to_lower(s);
    if (v == "cpu")
    {
        out = ComputeBackend::CPU;
        return true;
    }
    // Preserve the historical generic GPU spelling as Vulkan. Metal remains
    // explicit so a portable configuration never changes backend implicitly.
    if (v == "gpu" || v == "vulkan" || v == "vk")
    {
        out = ComputeBackend::GPU;
        return true;
    }
    if (v == "metal" || v == "mtl")
    {
        out = ComputeBackend::Metal;
        return true;
    }
    return false;
}

const char* compute_backend_name(ComputeBackend v)
{
    switch (v)
    {
    case ComputeBackend::CPU:
        return "cpu";
    case ComputeBackend::GPU:
        return "gpu";
    case ComputeBackend::Metal:
        return "metal";
    }
    return "cpu";
}

bool parse_memory_admission_policy(const std::string& s, MemoryAdmissionPolicy& out)
{
    const std::string v = to_lower(s);
    if (v == "warn")
    {
        out = MemoryAdmissionPolicy::Warn;
        return true;
    }
    if (v == "reject")
    {
        out = MemoryAdmissionPolicy::Reject;
        return true;
    }
    return false;
}

const char* memory_admission_policy_name(MemoryAdmissionPolicy v)
{
    switch (v)
    {
    case MemoryAdmissionPolicy::Warn:
        return "warn";
    case MemoryAdmissionPolicy::Reject:
        return "reject";
    }
    return "warn";
}

// ─────────────────────────────────────────────────────────────
// ChaosScheme
// ─────────────────────────────────────────────────────────────

bool parse_chaos_scheme(const std::string& s, ChaosScheme& out)
{
    const std::string v = to_lower(s);
    if (v == "none")
    {
        out = ChaosScheme::None;
        return true;
    }
    if (v == "initial_conditions")
    {
        out = ChaosScheme::InitialConditions;
        return true;
    }
    if (v == "boundary_layer")
    {
        out = ChaosScheme::BoundaryLayer;
        return true;
    }
    if (v == "full_stochastic")
    {
        out = ChaosScheme::FullStochastic;
        return true;
    }
    return false;
}

const char* chaos_scheme_name(ChaosScheme v)
{
    switch (v)
    {
    case ChaosScheme::None:
        return "none";
    case ChaosScheme::InitialConditions:
        return "initial_conditions";
    case ChaosScheme::BoundaryLayer:
        return "boundary_layer";
    case ChaosScheme::FullStochastic:
        return "full_stochastic";
    }
    return "none";
}

// ─────────────────────────────────────────────────────────────
// ValidationGuardMode
// ─────────────────────────────────────────────────────────────

bool parse_validation_guard_mode(const std::string& s, ValidationGuardMode& out)
{
    const std::string v = to_lower(s);
    if (v == "off")
    {
        out = ValidationGuardMode::Off;
        return true;
    }
    if (v == "warn")
    {
        out = ValidationGuardMode::Warn;
        return true;
    }
    if (v == "error")
    {
        out = ValidationGuardMode::Error;
        return true;
    }
    return false;
}

const char* validation_guard_mode_name(ValidationGuardMode v)
{
    switch (v)
    {
    case ValidationGuardMode::Off:
        return "off";
    case ValidationGuardMode::Warn:
        return "warn";
    case ValidationGuardMode::Error:
        return "error";
    }
    return "warn";
}

// ─────────────────────────────────────────────────────────────
// ExecutionMode
// ─────────────────────────────────────────────────────────────

bool parse_execution_mode(const std::string& s, ExecutionMode& out)
{
    const std::string v = to_lower(s);
    if (v == "exploratory")
    {
        out = ExecutionMode::Exploratory;
        return true;
    }
    if (v == "qualified")
    {
        out = ExecutionMode::Qualified;
        return true;
    }
    return false;
}

const char* execution_mode_name(ExecutionMode v)
{
    switch (v)
    {
    case ExecutionMode::Exploratory:
        return "exploratory";
    case ExecutionMode::Qualified:
        return "qualified";
    }
    return "exploratory";
}
} // namespace tmv::init::config
