#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace tmv
{

/**
 * Exact boundary accounting for one rho-weighted scalar transport commit.
 *
 * The inventory unit is caller-defined: kg for a dimensionless dry-air mixing
 * ratio, kg*K for potential temperature, and so on. Positive boundary
 * inventory is outward. The flux term is integrated from the final shared face
 * flux after positive-definite donor limiting; the committed term is measured
 * from the stored FP32 field and retained beginning/ending dry density.
 */
struct ScalarBoundaryTransportReport
{
    double outward_boundary_inventory = 0.0;
    double flux_implied_state_change = 0.0;
    double committed_state_change = 0.0;
    double closure_residual = 0.0;
    bool complete = false;
};

/**
 * Exact correction introduced when a positive-definite passive tracer is
 * canonicalized after transport.
 *
 * The conservative donor limiter can exhaust an FP32 cell to within one or a
 * few subnormal quanta and then round the stored result just below zero.
 * Finite negative binary32 subnormals and negative zero are canonicalized to
 * +0 at this producer boundary. Positive subnormals remain physical marker
 * state; this policy does not claim CPU/device denormal parity. The physical
 * correction is measured with dry density and cell volume. It has tracer
 * marker inventory units (rho_d*C*dV), not chemical-species mass, and exactly
 * zero thermodynamic-energy coupling. Negative normal, non-finite, or
 * above-contract values are material operator failures and never publish a
 * complete report.
 */
struct PositiveDefiniteScalarRoundoffReport
{
    static constexpr std::uint32_t report_schema_version = 1;
    static constexpr std::uint32_t classification_policy_version = 1;

    double physical_inventory_correction_kg = 0.0;
    double thermodynamic_energy_correction_j = 0.0;
    float most_negative_repaired_value = 0.0f;
    std::size_t negative_subnormal_count = 0;
    std::size_t negative_zero_count = 0;
    std::size_t closure_canonicalization_count = 0;
    bool complete = false;
};

/**
 * Scalar identity in the frozen moist-mixture energy basis.
 *
 * A scalar commit is linear in its transported variable when evaluated in
 * production order: theta first, followed by the seven water species.  This
 * tag selects the matching operator-time coefficient; it is deliberately not
 * inferred from a Field3D address.
 */
enum class ScalarMoistEnergyKind : std::uint8_t
{
    None = 0,
    PotentialTemperature,
    WaterVapor,
    LiquidWater,
    IceWater,
};

/**
 * Exact moist-mixture basis-v2 attribution for one limited scalar commit.
 *
 * Positive outward_boundary_energy_j leaves the physical domain.  The exact
 * limited-face contribution satisfies
 *
 *   limited_face_state_change_j =
 *       -outward_boundary_energy_j + interior_conversion_j.
 *
 * Density-history remap is required because the retained transport equation
 * advances rho_begin*phi and recovers with rho_end.  Storage rounding is the
 * residual introduced by the committed FP32 scalar value.  The final closure
 * identity is therefore
 *
 *   committed = density_remap + limited_face_change + storage_rounding.
 */
struct ScalarMoistEnergyTransportReportV1
{
    static constexpr std::uint32_t report_schema_version = 1;
    static constexpr std::uint32_t energy_basis_version = 2;

    double dry_internal_change_j = 0.0;
    double water_sensible_change_j = 0.0;
    double phase_reference_change_j = 0.0;
    double carrier_kinetic_change_j = 0.0;
    double gravitational_change_j = 0.0;
    double subgrid_tke_change_j = 0.0;

    double density_history_remap_j = 0.0;
    double outward_boundary_energy_j = 0.0;
    double interior_conversion_j = 0.0;
    double limited_face_state_change_j = 0.0;
    double requested_state_change_j = 0.0;
    double storage_rounding_j = 0.0;
    double committed_state_change_j = 0.0;
    double closure_residual_j = 0.0;
    bool complete = false;
};

inline constexpr std::size_t thermodynamic_boundary_report_count = 8;
inline constexpr std::size_t theta_boundary_report_index = 0;
inline constexpr std::size_t qv_boundary_report_index = 1;
inline constexpr std::size_t qc_boundary_report_index = 2;
inline constexpr std::size_t qr_boundary_report_index = 3;
inline constexpr std::size_t qi_boundary_report_index = 4;
inline constexpr std::size_t qs_boundary_report_index = 5;
inline constexpr std::size_t qg_boundary_report_index = 6;
inline constexpr std::size_t qh_boundary_report_index = 7;
using ThermodynamicBoundaryTransportReports =
    std::array<ScalarBoundaryTransportReport,
               thermodynamic_boundary_report_count>;
using ThermodynamicMoistEnergyTransportReports =
    std::array<ScalarMoistEnergyTransportReportV1,
               thermodynamic_boundary_report_count>;

} // namespace tmv
