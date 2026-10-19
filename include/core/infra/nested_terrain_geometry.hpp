#pragma once

#include <cstdint>

#include "core/infra/nested_grid.hpp"
#include "terrain/earth_fixed_geometry.hpp"

namespace tmv::domain
{

struct NestedMappedConservativeStateV1
{
    static constexpr std::uint32_t schema_version = 1;

    TerrainField3D scalar;
    TerrainField3D x_flux;
    TerrainField3D y_flux;
    TerrainField3D zeta_flux;
};

struct NestedTerrainTangentBoundaryV1
{
    static constexpr std::uint32_t schema_version = 1;

    TerrainField3D relaxation_weight;
    TerrainField3D ground_u_ms;
    TerrainField3D ground_v_ms;
    TerrainField3D ground_w_ms;
};

struct NestedTerrainTransferLedgerV1
{
    static constexpr std::uint32_t schema_version = 1;

    double        parent_time_s                         = 0.0;
    double        child_time_s                          = 0.0;
    double        parent_scalar_inventory               = 0.0;
    double        child_scalar_inventory                = 0.0;
    double        scalar_inventory_residual             = 0.0;
    double        normalized_scalar_inventory_residual  = 0.0;
    double        maximum_x_flux_residual               = 0.0;
    double        maximum_normalized_x_flux_residual    = 0.0;
    double        maximum_y_flux_residual               = 0.0;
    double        maximum_normalized_y_flux_residual    = 0.0;
    double        maximum_zeta_flux_residual            = 0.0;
    double        maximum_normalized_zeta_flux_residual = 0.0;
    double        maximum_ground_tangency_residual_ms   = 0.0;
    std::uint64_t relaxation_owned_values               = 0;
};

struct NestedMappedTerrainGeometryV1
{
    static constexpr std::uint32_t schema_version = 1;

    TerrainConfig             config;
    OneWayNestRelationshipV1  relationship;
    HorizontalPoint           parent_origin_earth_m;
    HorizontalPoint           child_origin_earth_m;
    HorizontalPoint           feature_center_earth_m;
    EarthFixedTerrainGeometry parent;
    EarthFixedTerrainGeometry child_storage;
    std::uint64_t             generation = 0;
    // Version 1 retains historical local-origin rounding; version 2 uses the parent lattice.
    std::uint32_t sampling_version = 1;
};

struct NestedActiveTerrainStateV1
{
    static constexpr std::uint32_t schema_version = 2;

    bool                            ready = false;
    NestedMappedTerrainGeometryV1   geometry;
    NestedMappedConservativeStateV1 child_state;
    NestedTerrainTangentBoundaryV1  boundary;
    NestedTerrainTransferLedgerV1   ledger;
    double                          parent_time_s     = 0.0;
    double                          child_time_s      = 0.0;
    std::uint64_t                   generation        = 0;
    std::uint64_t                   publication_count = 0;
};

using NestedActiveTerrainCandidateObserver = bool (*)(const NestedActiveTerrainStateV1&, void*);

struct PreparedNestedActiveTerrainCommitV1
{
    NestedActiveTerrainStateV1 candidate;
    NestedActiveTerrainStateV1 expected_live;
    bool                       committed = false;
};

NestedMappedTerrainGeometryV1 build_stationary_nested_terrain_geometry(const TerrainConfig&            config,
                                                                       const OneWayNestRelationshipV1& relationship,
                                                                       HorizontalPoint parent_origin_earth_m,
                                                                       std::uint64_t   generation,
                                                                       std::uint32_t   sampling_version = 1);

NestedMappedConservativeStateV1
prolong_nested_mapped_conservative_state(const NestedMappedTerrainGeometryV1&   geometry,
                                         const NestedMappedConservativeStateV1& parent,
                                         NestedTerrainTransferLedgerV1*         ledger = nullptr);

NestedMappedConservativeStateV1 restrict_nested_mapped_conservative_state(const NestedMappedTerrainGeometryV1& geometry,
                                                                          const NestedMappedConservativeStateV1& child);

TerrainField3D restrict_nested_mapped_scalar(const NestedMappedTerrainGeometryV1& geometry,
                                             const TerrainField3D&                child_scalar);

NestedTerrainTangentBoundaryV1 build_nested_terrain_tangent_boundary(const NestedMappedTerrainGeometryV1& geometry,
                                                                     double                               ground_u_ms,
                                                                     double                               ground_v_ms,
                                                                     NestedTerrainTransferLedgerV1* ledger = nullptr);

PreparedNestedActiveTerrainCommitV1
prepare_nested_active_terrain_commit(const NestedActiveTerrainStateV1&      live,
                                     const TerrainConfig&                   config,
                                     const OneWayNestRelationshipV1&        relationship,
                                     HorizontalPoint                        parent_origin_earth_m,
                                     const NestedMappedConservativeStateV1& parent_state,
                                     double                                 parent_time_s,
                                     double                                 ground_u_ms,
                                     double                                 ground_v_ms,
                                     NestedActiveTerrainCandidateObserver   observer         = nullptr,
                                     void*                                  observer_context = nullptr);

bool can_commit_nested_active_terrain(const PreparedNestedActiveTerrainCommitV1& prepared,
                                      const NestedActiveTerrainStateV1&          live) noexcept;

PreparedNestedActiveTerrainCommitV1
prepare_nested_terrain_parent_rebase(const NestedActiveTerrainStateV1& live,
                                     const OneWayNestRelationshipV1& relationship,
                                     HorizontalPoint parent_origin_earth_m);

bool commit_nested_active_terrain(PreparedNestedActiveTerrainCommitV1& prepared,
                                  NestedActiveTerrainStateV1&          live) noexcept;

} // namespace tmv::domain
