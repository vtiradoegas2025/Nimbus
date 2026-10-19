/**
 * @file test_field_contract.cpp
 * @brief Unit tests for field contract definitions and coverage.
 */
#include "catch2/catch.hpp"
#include "core/field/field_bounds.hpp"
#include "diagnostics/derived_field_math.hpp"
#include "diagnostics/field_contract.hpp"

#include <unordered_set>

TEST_CASE("cm1_field_contracts returns non-empty list", "[diagnostics][contract]")
{
    const auto& contracts = tmv::cm1_field_contracts();
    REQUIRE_FALSE(contracts.empty());
    REQUIRE(contracts.size() > 10);
}

TEST_CASE("Each contract has a non-empty id", "[diagnostics][contract]")
{
    const auto& contracts = tmv::cm1_field_contracts();
    for (const auto& c : contracts)
    {
        REQUIRE_FALSE(c.id.empty());
    }
}

TEST_CASE("Each contract has a valid status", "[diagnostics][contract]")
{
    const auto& contracts = tmv::cm1_field_contracts();
    for (const auto& c : contracts)
    {
        bool valid = (c.status == tmv::FieldImplementationStatus::ExportedNow ||
                      c.status == tmv::FieldImplementationStatus::ComputedNotExported ||
                      c.status == tmv::FieldImplementationStatus::NotImplemented);
        REQUIRE(valid);
    }
}

TEST_CASE("Required-now fields with ExportedNow status meet floor", "[diagnostics][contract]")
{
    const auto& contracts         = tmv::cm1_field_contracts();
    int         required_exported = 0;
    for (const auto& c : contracts)
    {
        if (c.requirement == tmv::FieldRequirementTier::RequiredNow &&
            c.status == tmv::FieldImplementationStatus::ExportedNow)
        {
            ++required_exported;
        }
    }
    // The contract coverage guard requires >= 20
    REQUIRE(required_exported >= 20);
}

TEST_CASE("Core prognostic fields are in the contract", "[diagnostics][contract]")
{
    REQUIRE(tmv::find_field_contract("u") != nullptr);
    REQUIRE(tmv::find_field_contract("w") != nullptr);
    REQUIRE(tmv::find_field_contract("theta") != nullptr);
    REQUIRE(tmv::find_field_contract("p") != nullptr);
    REQUIRE(tmv::find_field_contract("rho") != nullptr);
    REQUIRE(tmv::find_field_contract("qv") != nullptr);
    REQUIRE(tmv::find_field_contract("qr") != nullptr);
    const auto* tke = tmv::find_field_contract("tke");
    REQUIRE(tke != nullptr);
    REQUIRE(tke->default_bounds.has_min);
    REQUIRE(tke->default_bounds.min_value == Approx(0.0));
}

TEST_CASE("Field contract lookup remains ASCII case insensitive", "[diagnostics][contract][lookup]")
{
    const auto* canonical = tmv::find_field_contract("angular_momentum_tendency");
    REQUIRE(canonical != nullptr);
    REQUIRE(tmv::find_field_contract("ANGULAR_MOMENTUM_TENDENCY") == canonical);
    REQUIRE(tmv::find_field_contract("Z") == tmv::find_field_contract("reflectivity_dbz"));
}

TEST_CASE("SCI-024 field metadata declares the dry carrier basis", "[diagnostics][contract][SCI-024]")
{
    const auto* rho_contract = tmv::find_field_contract("rho");
    REQUIRE(rho_contract != nullptr);
    REQUIRE(rho_contract->description.find("dry-air density") != std::string::npos);

    for (const char* id : {"qv", "qc", "qr", "qi", "qs", "qg", "qh"})
    {
        const auto* contract = tmv::find_field_contract(id);
        REQUIRE(contract != nullptr);
        REQUIRE(contract->description.find("dry-air mixing ratio") != std::string::npos);
    }
}

TEST_CASE("Potential-temperature metadata matches the runtime clamp", "[diagnostics][contract][theta]")
{
    const auto* contract = tmv::find_field_contract("theta");
    REQUIRE(contract != nullptr);
    REQUIRE(contract->default_bounds.has_min);
    REQUIRE(contract->default_bounds.has_max);
    REQUIRE(contract->default_bounds.min_value == Approx(theta_min_k));
    REQUIRE(contract->default_bounds.max_value == Approx(theta_max_k));
}

TEST_CASE("Cold-pool contract publishes the qualified density-potential-temperature deficit",
          "[diagnostics][contract][cold_pool]")
{
    const auto* contract = tmv::find_field_contract("cold_pool_boundary");
    REQUIRE(contract != nullptr);
    REQUIRE(contract->units == "K");
    REQUIRE(contract->dimensionality == tmv::FieldDimensionality::Surface2D);
    REQUIRE(contract->status == tmv::FieldImplementationStatus::ExportedNow);
    REQUIRE(contract->grid_location ==
            tmv::FieldGridLocation::SurfaceCellCenter);
    REQUIRE(contract->state_semantic == tmv::FieldStateSemantic::Diagnostic);
    REQUIRE(contract->description.find(
                tmv::derived::cold_pool_boundary_algorithm_id) !=
            std::string::npos);
    REQUIRE(contract->default_bounds.has_min);
    REQUIRE(contract->default_bounds.min_value == Approx(0.0));
    REQUIRE(contract->default_bounds.has_max);
    REQUIRE(contract->default_bounds.max_value == Approx(100.0));
}

TEST_CASE("Cartesian history identities carry explicit CM1 interoperability metadata", "[diagnostics][contract][cm1]")
{
    const auto* u                      = tmv::find_field_contract("u");
    const auto* v                      = tmv::find_field_contract("v");
    const auto* w                      = tmv::find_field_contract("w");
    const auto* p                      = tmv::find_field_contract("p");
    const auto* p_prime                = tmv::find_field_contract("p_prime");
    const auto* theta                  = tmv::find_field_contract("theta");
    const auto* theta_prime            = tmv::find_field_contract("theta_prime");
    const auto* reflectivity           = tmv::find_field_contract("reflectivity_dbz");
    const auto* qv                     = tmv::find_field_contract("qv");
    const auto* qc                     = tmv::find_field_contract("qc");
    const auto* vorticity_r            = tmv::find_field_contract("vorticity_r");
    const auto* vorticity_theta        = tmv::find_field_contract("vorticity_theta");
    const auto* vorticity_z            = tmv::find_field_contract("vorticity_z");
    const auto* buoyancy               = tmv::find_field_contract("buoyancy");
    const auto* potential_vorticity    = tmv::find_field_contract("potential_vorticity");
    const auto* composite_reflectivity = tmv::find_field_contract("composite_reflectivity");
    const auto* accumulated_rainfall   = tmv::find_field_contract("accumulated_rainfall");
    const auto* precip_rate            = tmv::find_field_contract("precip_rate");
    const auto* cape                   = tmv::find_field_contract("cape");
    const auto* updraft_helicity       = tmv::find_field_contract("updraft_helicity");
    const auto* precipitable_water     = tmv::find_field_contract("precipitable_water");
    const auto* liquid_water_path      = tmv::find_field_contract("liquid_water_path");
    const auto* cloud_water_path       = tmv::find_field_contract("cloud_water_path");
    REQUIRE(u != nullptr);
    REQUIRE(v != nullptr);
    REQUIRE(w != nullptr);
    REQUIRE(p != nullptr);
    REQUIRE(p_prime != nullptr);
    REQUIRE(theta != nullptr);
    REQUIRE(theta_prime != nullptr);
    REQUIRE(reflectivity != nullptr);
    REQUIRE(qv != nullptr);
    REQUIRE(qc != nullptr);
    REQUIRE(vorticity_r != nullptr);
    REQUIRE(vorticity_theta != nullptr);
    REQUIRE(vorticity_z != nullptr);
    REQUIRE(buoyancy != nullptr);
    REQUIRE(potential_vorticity != nullptr);
    REQUIRE(composite_reflectivity != nullptr);
    REQUIRE(accumulated_rainfall != nullptr);
    REQUIRE(precip_rate != nullptr);
    REQUIRE(cape != nullptr);
    REQUIRE(updraft_helicity != nullptr);
    REQUIRE(precipitable_water != nullptr);
    REQUIRE(liquid_water_path != nullptr);
    REQUIRE(cloud_water_path != nullptr);
    REQUIRE(u->description.find("Cartesian") != std::string::npos);
    REQUIRE(v->description.find("Cartesian") != std::string::npos);
    REQUIRE(u->cm1_aliases == std::vector<std::string>{"uinterp"});
    REQUIRE(v->cm1_aliases == std::vector<std::string>{"vinterp"});
    REQUIRE(w->cm1_aliases == std::vector<std::string>{"winterp"});
    REQUIRE(p->cm1_aliases == std::vector<std::string>{"prs"});
    REQUIRE(p_prime->cm1_aliases == std::vector<std::string>{"prspert"});
    REQUIRE(theta->cm1_aliases == std::vector<std::string>{"th"});
    REQUIRE(theta_prime->cm1_aliases == std::vector<std::string>{"thpert"});
    REQUIRE(reflectivity->cm1_aliases == std::vector<std::string>{"dbz"});
    REQUIRE(qv->cm1_aliases == std::vector<std::string>{"qv"});
    REQUIRE(qc->cm1_aliases.empty());
    REQUIRE(vorticity_r->cm1_aliases.empty());
    REQUIRE(vorticity_theta->cm1_aliases.empty());
    REQUIRE(vorticity_z->cm1_aliases == std::vector<std::string>{"zvort"});
    REQUIRE(buoyancy->cm1_aliases == std::vector<std::string>{"buoyancy"});
    REQUIRE(potential_vorticity->cm1_aliases == std::vector<std::string>{"pv"});
    REQUIRE(composite_reflectivity->cm1_aliases == std::vector<std::string>{"cref"});
    REQUIRE(accumulated_rainfall->cm1_aliases == std::vector<std::string>{"rain"});
    REQUIRE(precip_rate->cm1_aliases == std::vector<std::string>{"prate"});
    REQUIRE(cape->cm1_aliases == std::vector<std::string>{"cape"});
    REQUIRE(updraft_helicity->cm1_aliases == std::vector<std::string>{"uh"});
    REQUIRE(precipitable_water->cm1_aliases == std::vector<std::string>{"pwat"});
    REQUIRE(liquid_water_path->cm1_aliases == std::vector<std::string>{"lwp"});
    REQUIRE(cloud_water_path->cm1_aliases == std::vector<std::string>{"cwp"});
    REQUIRE(precipitable_water->status == tmv::FieldImplementationStatus::ExportedNow);
    REQUIRE(liquid_water_path->status == tmv::FieldImplementationStatus::ExportedNow);
    REQUIRE(cloud_water_path->status == tmv::FieldImplementationStatus::ExportedNow);
    REQUIRE(u->grid_location == tmv::FieldGridLocation::CellCenter);
    REQUIRE(u->state_semantic == tmv::FieldStateSemantic::TotalState);
    REQUIRE(p_prime->state_semantic == tmv::FieldStateSemantic::Perturbation);
    REQUIRE(theta_prime->state_semantic == tmv::FieldStateSemantic::Perturbation);
    REQUIRE(reflectivity->state_semantic == tmv::FieldStateSemantic::Diagnostic);
    REQUIRE(tmv::find_field_contract("ua") == nullptr);
    REQUIRE(tmv::find_field_contract("uinterp") == nullptr);
}

TEST_CASE("No duplicate field ids in contract", "[diagnostics][contract]")
{
    const auto&                     contracts = tmv::cm1_field_contracts();
    std::unordered_set<std::string> seen;
    for (const auto& c : contracts)
    {
        REQUIRE(seen.count(c.id) == 0);
        seen.insert(c.id);
    }
}

TEST_CASE("contracts_with_status filters correctly", "[diagnostics][contract]")
{
    auto exported = tmv::contracts_with_status(tmv::FieldImplementationStatus::ExportedNow);
    REQUIRE_FALSE(exported.empty());

    for (const auto* c : exported)
    {
        REQUIRE(c->status == tmv::FieldImplementationStatus::ExportedNow);
    }
}

TEST_CASE("find_field_contract returns null for unknown id", "[diagnostics][contract]")
{
    REQUIRE(tmv::find_field_contract("nonexistent_field_xyz") == nullptr);
}
