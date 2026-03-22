#pragma once

#include <string>
#include <string_view>
#include <vector>

/**
 * @file field_contract.hpp
 * @brief Field metadata contract used by validation and reporting systems.
 *
 * Captures field identity, dimensionality, implementation status,
 * bounds expectations, and requirement tier for runtime QA checks.
 * Contract queries are used by diagnostics and export guards.
 */

namespace tmv
{

enum class FieldDimensionality
{
    Volume3D,
    Surface2D,
    Column2D,
    CrossSection,
    Trajectory,
    RadarSynthetic,
};

enum class FieldImplementationStatus
{
    ExportedNow,
    ComputedNotExported,
    NotImplemented,
};

enum class FieldRequirementTier
{
    RequiredNow,
    ReportOnly,
};

struct FieldBounds
{
    bool has_min = false;
    bool has_max = false;
    double min_value = 0.0;
    double max_value = 0.0;
};

struct SeverityPolicy
{
    bool check_nonfinite = true;
    bool check_bounds = true;
};

struct FieldContract
{
    std::string id;
    std::string units;
    std::string description;
    FieldDimensionality dimensionality = FieldDimensionality::Volume3D;
    FieldImplementationStatus status = FieldImplementationStatus::NotImplemented;
    FieldBounds default_bounds{};
    SeverityPolicy severity{};
    std::vector<std::string> aliases;
    FieldRequirementTier requirement = FieldRequirementTier::ReportOnly;
};

/**
 * @brief Returns the complete CM1 field contract table.
 * @return Immutable list of field contracts.
 */
const std::vector<FieldContract>& cm1_field_contracts();

/**
 * @brief Finds a contract by canonical id or alias.
 * @param id_or_alias Contract id or alias.
 * @return Pointer to matched contract, or null if not found.
 */
const FieldContract* find_field_contract(std::string_view id_or_alias);

/**
 * @brief Filters contracts by implementation status.
 * @param status Target status value.
 * @return Matched contract pointers.
 */
std::vector<const FieldContract*> contracts_with_status(FieldImplementationStatus status);

/**
 * @brief Filters contracts by requirement tier.
 * @param requirement Target requirement tier.
 * @return Matched contract pointers.
 */
std::vector<const FieldContract*> contracts_with_requirement(FieldRequirementTier requirement);

/**
 * @brief Returns required-now contracts still not implemented.
 * @return Inventory of missing required field ids.
 */
std::vector<std::string> known_not_implemented_required_now_inventory();

/**
 * @brief Returns report-only contracts still not implemented.
 * @return Inventory of backlog field ids.
 */
std::vector<std::string> known_not_implemented_backlog_inventory();

/**
 * @brief Converts dimensionality enum to text.
 */
const char* to_string(FieldDimensionality value);

/**
 * @brief Converts implementation-status enum to text.
 */
const char* to_string(FieldImplementationStatus value);

/**
 * @brief Converts requirement-tier enum to text.
 */
const char* to_string(FieldRequirementTier value);

} // namespace tmv
