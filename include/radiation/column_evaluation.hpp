#pragma once

#include "radiation/radiation_base.hpp"

#include <cstdint>
#include <stdexcept>

namespace tmv
{

enum class RadiationColumnEvaluationFailure : std::uint8_t
{
    SchemeRejected         = 1,
    MalformedTendency      = 2,
    PowerClosure           = 3,
    NonFiniteThetaTendency = 4,
    NonFiniteCellPower     = 5,
};

class RadiationColumnEvaluationError final : public std::runtime_error
{
public:
    RadiationColumnEvaluationError(RadiationColumnEvaluationFailure failure, const char* message)
        : std::runtime_error(message), failure_(failure)
    {
    }

    RadiationColumnEvaluationFailure failure() const noexcept
    {
        return failure_;
    }

private:
    RadiationColumnEvaluationFailure failure_;
};

struct RadiationPreparedColumnView
{
    const std::vector<double>* pressure_pa               = nullptr;
    const std::vector<double>* temperature_k             = nullptr;
    const std::vector<double>* density_kgm3              = nullptr;
    const std::vector<double>* vapor_mixing_ratio        = nullptr;
    const std::vector<double>* cloud_liquid_mixing_ratio = nullptr;
    const std::vector<double>* cloud_ice_mixing_ratio    = nullptr;
    const std::vector<double>* z_center_m                = nullptr;
    const std::vector<double>* z_interface_m             = nullptr;
    const std::vector<double>* layer_thickness_m         = nullptr;

    double cosine_solar_zenith     = 0.0;
    double surface_temperature_k   = 288.0;
    double surface_albedo          = 0.2;
    double surface_emissivity      = 0.95;
    double horizontal_cell_area_m2 = 0.0;
};

// Validate and bind one prepared column to reusable storage. The returned
// view remains valid while both `column` and `workspace` remain alive.
RadiationColumnStateView prepare_radiation_column_state(const RadiationConfig&             config,
                                                        const RadiationPreparedColumnView& column,
                                                        RadiationColumnWorkspace&          workspace);

// Close a scheme result against the interface-flux power contract and convert
// it to the FP32 theta/cell-power buffers used by the runtime transaction.
double finalize_prepared_radiation_column(const RadiationPreparedColumnView& column,
                                          RadiationColumnWorkspace&          workspace,
                                          const RadiationColumnTendencies&   tendencies);

/**
 * Evaluate one already reconstructed physical radiation column.
 *
 * This is the single conversion/closure path used by normal cadence updates
 * and recycling leading-column preparation. Results are published into the caller's
 * reusable workspace only after the scheme tendency closes its reported
 * interface-flux power and every FP32 storage value is finite.
 */
double evaluate_prepared_radiation_column(RadiationSchemeBase&               scheme,
                                          const RadiationConfig&             config,
                                          const RadiationPreparedColumnView& column,
                                          RadiationColumnWorkspace&          workspace,
                                          RadiationColumnTendencies&         tendencies,
                                          RadiationColumnFluxes*             fluxes = nullptr);

const char* radiation_prepared_column_algorithm_id() noexcept;

} // namespace tmv
