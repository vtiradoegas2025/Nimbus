#pragma once

#include <string>
#include <unordered_map>

#include "diagnostics/field_contract.hpp"

namespace tmv
{

enum class GuardMode
{
    Off,       // no validation
    Warn,      // validate + record/log violations, but do not modify or abort (observe)
    Sanitize,  // validate + clamp out-of-bounds and replace non-finite
    Strict,    // validate + abort the run on a violation
};

enum class GuardFailOn
{
    NonFinite,
    Bounds,
    Both,
};

enum class StrictGuardScope
{
    RequiredOnly,
    ExportedNow,
};

struct ValidationPolicy
{
    GuardMode mode = GuardMode::Sanitize;
    GuardFailOn fail_on = GuardFailOn::Both;
    StrictGuardScope strict_scope = StrictGuardScope::RequiredOnly;
    std::unordered_map<std::string, FieldBounds> field_overrides;
};

}  // namespace tmv
