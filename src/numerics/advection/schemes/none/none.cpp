#include "none.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tmv
{

namespace
{
void validate_state(const AdvectionConfig& cfg,
                    const AdvectionStateView& state)
{
    if (cfg.scheme_id != "none")
        throw std::invalid_argument(
            "no-advection operator requires scheme_id=none");
    if (state.q == nullptr || state.w == nullptr || state.grid == nullptr)
        throw std::invalid_argument("no-advection state is incomplete");
    if (state.q->empty() || state.q->size_r() != state.w->size_r() ||
        state.q->size_th() != state.w->size_th() ||
        state.q->size_z() != state.w->size_z())
        throw std::invalid_argument("no-advection field shapes disagree");
    for (std::size_t index = 0; index < state.q->size(); ++index)
        if (!std::isfinite(static_cast<double>(state.q->data()[index])) ||
            !std::isfinite(static_cast<double>(state.w->data()[index])))
            throw std::invalid_argument(
                "no-advection operator received non-finite state");
}
}

void NoAdvectionScheme::initialize()
{
}

void NoAdvectionScheme::initialize(const AdvectionConfig& cfg)
{
    if (cfg.scheme_id != "none")
        throw std::invalid_argument(
            "no-advection operator requires scheme_id=none");
}

void NoAdvectionScheme::compute_flux_divergence(
    const AdvectionConfig& cfg,
    const AdvectionStateView& state,
    AdvectionTendencies& tendencies,
    AdvectionDiagnostics* diag_opt)
{
    validate_state(cfg, state);
    tendencies.dqdt_adv.resize(
        state.q->size_r(), state.q->size_th(), state.q->size_z(), 0.0f);
    std::fill_n(
        tendencies.dqdt_adv.data(), tendencies.dqdt_adv.size(), 0.0f);
    tendencies.fused_completed = false;
    if (diag_opt != nullptr)
        *diag_opt = {};
}

double NoAdvectionScheme::suggest_dt(
    const AdvectionConfig& cfg,
    const AdvectionStateView& state)
{
    validate_state(cfg, state);
    return std::numeric_limits<double>::infinity();
}

} // namespace tmv
