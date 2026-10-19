// Time-evolving stochastic physics perturbation scheme.

#pragma once
#include "chaos/scheme.hpp"
#include "chaos/base/earth_correlated_noise.hpp"

namespace tmv::chaos
{

// Applies correlated SPPT-like multiplicative noise to physics tendencies.
class FullStochasticScheme : public ChaosScheme {
public:
    FullStochasticScheme() = default;
    FullStochasticScheme(const FullStochasticScheme&) = default;
    std::unique_ptr<ChaosScheme> clone() const override
    {
        return std::make_unique<FullStochasticScheme>(*this);
    }

    std::string name() const override { return "full_stochastic"; }
    void configure_compute_backend(ComputeBackend* backend, bool required) noexcept override;

    PortableSchemeState portable_state() const override;
    bool restore_portable_state(const PortableSchemeState& state,
                                std::string& error) override;

    void initialize(const ChaosConfig& cfg, const GridMetrics& grid) override;

    void apply_initial_conditions(
        const ChaosConfig& cfg,
        const GridMetrics& grid,
        SimulationState& state,
        ChaosDiagnostics* diag = nullptr
    ) override;

    void apply_tendencies(
        const ChaosConfig& cfg,
        const GridMetrics& grid,
        const ChaosStateView& state_view,
        ChaosTendencies& tendencies,
        ChaosDiagnostics* diag = nullptr
    ) override;

    void step_noise(
        const ChaosConfig& cfg,
        const GridMetrics& grid,
        double dt
    ) override;

    bool set_continuous_horizontal_earth_origin(
        double origin_x_m,
        double origin_y_m,
        const ChaosConfig& cfg,
        const GridMetrics& grid,
        std::string& error) override;

    bool translate_horizontal_stochastic_state(
        int shift_i_cells,
        int shift_j_cells,
        const ChaosConfig& cfg,
        const GridMetrics& grid,
        std::string& error) override;
    bool prepare_horizontal_stochastic_state_translation(
        int shift_i_cells,
        int shift_j_cells,
        const ChaosConfig& cfg,
        const GridMetrics& grid,
        std::unique_ptr<ChaosScheme>& workspace,
        HorizontalStochasticStatePhase& phase,
        std::string& error) const override;
    void exchange_prepared_horizontal_stochastic_state_translation(
        std::unique_ptr<ChaosScheme>& workspace) noexcept override;
    bool horizontal_stochastic_storage_identity(
        HorizontalStochasticStorageIdentity& identity,
        std::string& error) const override;
    bool horizontal_stochastic_state_phase(
        HorizontalStochasticStatePhase& phase,
        std::string& error) const override;
    bool initialize_nested_horizontal_stochastic_state(
        const ChaosScheme& parent,
        std::int64_t earth_origin_i_cells,
        std::int64_t earth_origin_j_cells,
        int nr,
        int nth,
        int nz,
        std::uint64_t target_epoch,
        double cadence_dt_s,
        const ChaosConfig& cfg,
        const GridMetrics& grid,
        std::string& error) override;

private:
    EarthCorrelatedNoise microphysics_noise_;
    EarthCorrelatedNoise pbl_noise_;
    EarthCorrelatedNoise diffusion_noise_;

    Field3D xi_microphysics_;
    Field3D xi_pbl_;
    Field3D xi_diffusion_;

    Field3D xi_micro_prev_;
    Field3D xi_pbl_prev_;
    Field3D xi_diff_prev_;

    uint64_t time_step_counter_ = 0;
    double cadence_dt_s_ = 0.0;
    ComputeBackend* native_backend_ = nullptr;
    bool native_generation_required_ = false;
};

}
