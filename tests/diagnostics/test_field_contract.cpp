/**
 * @file test_field_contract.cpp
 * @brief Unit tests for field contract definitions and coverage.
 */
#include "catch2/catch.hpp"
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
    const auto& contracts = tmv::cm1_field_contracts();
    int required_exported = 0;
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
}

TEST_CASE("No duplicate field ids in contract", "[diagnostics][contract]")
{
    const auto& contracts = tmv::cm1_field_contracts();
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
