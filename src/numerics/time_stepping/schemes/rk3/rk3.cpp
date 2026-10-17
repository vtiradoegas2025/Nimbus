#include "rk3.hpp"
#include "util/log.hpp"
#include "numerics/time_stepping/rk_validation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tmv
{
namespace
{

bool same_shape(const Field3D& lhs, const Field3D& rhs) noexcept
{
    return lhs.size_r() == rhs.size_r()
        && lhs.size_th() == rhs.size_th()
        && lhs.size_z() == rhs.size_z();
}

} // namespace

RK3Scheme::RK3Scheme() : rhs_function_(nullptr) {}

void RK3Scheme::initialize() 
{
    initialize(TimeSteppingConfig{}, nullptr);
}

void RK3Scheme::initialize(const TimeSteppingConfig& cfg, RHSFunction rhs_func) 
{
    rhs_function_ = rhs_func;

    tmv::log_info("Initialized RK3 (SSPRK3) time stepping scheme");
    if (cfg.split_acoustic) {
        tmv::log_info("  Split-explicit acoustic mode enabled");
    }
}

void RK3Scheme::step(
    const TimeSteppingConfig&,
    TimeSteppingState& state,
    TimeSteppingDiagnostics* diag_opt
) 
{
    if (!rhs_function_) 
    {
        throw std::runtime_error("RHS function not set for RK3 scheme");
    }
    const double next_time = state.time + state.dt;
    if (!std::isfinite(state.time) || !std::isfinite(state.dt) || state.dt <= 0.0 || !std::isfinite(next_time) ||
        next_time <= state.time || !(state.time + .5 * state.dt > state.time) ||
        !(state.time + .5 * state.dt < next_time))
    {
        throw std::invalid_argument("RK3 requires finite time and a positive finite, representable timestep");
    }

    rk::validate_state(state.fields, "RK3");

    if (diag_opt)
    {
        if (diag_opt->n_steps == std::numeric_limits<int>::max())
        {
            throw std::overflow_error("RK3 diagnostic step count overflow");
        }
        if (diag_opt->dt_history.size() == diag_opt->dt_history.max_size())
        {
            throw std::length_error("RK3 timestep history is full");
        }
        diag_opt->prepare_history_append();
    }

    if (stage1_.empty() || stage1_.size() != state.fields.size()) {
        stage1_.resize(state.fields.size());
        stage2_.resize(state.fields.size());
        rhs_stage1_.resize(state.fields.size());
        rhs_stage2_.resize(state.fields.size());
    }

    compute_stage1(state, rhs_stage1_);

    compute_stage2(state, rhs_stage2_);

    compute_final(state, rhs_stage2_);

    state.time = next_time;

    if (diag_opt)
    {
        diag_opt->dt_history.push_back(state.dt);
        ++diag_opt->n_steps;
    }
}

double RK3Scheme::suggest_dt(
    const TimeSteppingConfig&,
    const TimeSteppingState&
) 
{
    return 1.0;
}

void RK3Scheme::step_coupled_unsplit_rk(
    const TimeSteppingConfig&,
    double dt,
    double time,
    const UnsplitRkCallbacks& callbacks,
    TimeSteppingDiagnostics* diagnostics)
{
    const double next_time = time + dt;
    if (!std::isfinite(time) || !std::isfinite(dt) || dt <= 0.0 || !std::isfinite(next_time) || next_time <= time ||
        !(time + .5 * dt > time) || !(time + .5 * dt < next_time))
        throw std::invalid_argument(
            "coupled SSPRK3 requires finite time and a positive finite, representable timestep");
    if (!callbacks.capture_base || !callbacks.compute_tendencies ||
        !callbacks.publish_combination || !callbacks.restore_base)
        throw std::invalid_argument(
            "coupled SSPRK3 requires a complete callback transaction");
    if (diagnostics)
    {
        if (diagnostics->n_steps == std::numeric_limits<int>::max())
            throw std::overflow_error("SSPRK3 diagnostic step count overflow");
        if (diagnostics->dt_history.size() == diagnostics->dt_history.max_size())
            throw std::length_error("SSPRK3 timestep history is full");
        diagnostics->prepare_history_append();
    }

    callbacks.capture_base();
    try
    {
        callbacks.compute_tendencies(0, time);
        callbacks.publish_combination(1, {1.0, 0.0, 0.0, 0.0}, dt);
        callbacks.compute_tendencies(1, time + dt);
        callbacks.publish_combination(2, {0.25, 0.25, 0.0, 0.0}, dt);
        callbacks.compute_tendencies(2, time + 0.5 * dt);
        callbacks.publish_combination(
            3, {1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0, 0.0}, dt);
    }
    catch (...)
    {
        callbacks.restore_base();
        throw;
    }

    if (diagnostics)
    {
        diagnostics->dt_history.push_back(dt);
        ++diagnostics->n_steps;
    }
}

void RK3Scheme::invalidate_generation_state_after_domain_recycling() noexcept
{
    for (auto& state : stage1_) state.data.fill(0.0f);
    for (auto& state : stage2_) state.data.fill(0.0f);
    for (auto& tendency : rhs_stage1_) tendency.tendencies.fill(0.0f);
    for (auto& tendency : rhs_stage2_) tendency.tendencies.fill(0.0f);
}

void RK3Scheme::compute_stage1(const TimeSteppingState& state, std::vector<NumericalTendencies>& tendencies) 
{
    rhs_function_(state.fields, state.time, tendencies);
    rk::validate_rhs(state.fields, tendencies, "RK3");
}

void RK3Scheme::compute_stage2(const TimeSteppingState& state, std::vector<NumericalTendencies>& tendencies) 
{
    add_states(state.fields, 1.0, rhs_stage1_, 1.0, state.dt, stage1_);

    rk::validate_state(stage1_, "RK3");
    rhs_function_(stage1_, state.time + state.dt, tendencies);
    rk::validate_rhs(stage1_, tendencies, "RK3");
}

void RK3Scheme::compute_final(TimeSteppingState& state, std::vector<NumericalTendencies>& tendencies) 
{
    if (tendencies.size() != state.fields.size() || stage1_.size() != state.fields.size())
    {
        throw std::runtime_error("RK3 stage storage size mismatch");
    }

    stage2_.resize(state.fields.size());
    for (size_t i = 0; i < state.fields.size(); ++i)
    {
        stage2_[i].name = state.fields[i].name;
        const Field3D& qn = state.fields[i].data;
        const Field3D& q1 = stage1_[i].data;
        const Field3D& k2 = tendencies[i].tendencies;
        if (!same_shape(qn, q1) || !same_shape(qn, k2))
        {
            throw std::runtime_error("RK3 q2 stage shape mismatch");
        }

        stage2_[i].data.resize(qn.size_r(), qn.size_th(), qn.size_z(), 0.0f);
        const float* __restrict__ qn_ptr = qn.data();
        const float* __restrict__ q1_ptr = q1.data();
        const float* __restrict__ k2_ptr = k2.data();
        float* __restrict__ q2_ptr = stage2_[i].data.data();
        const std::size_t count = qn.size();
        const double dt_local = state.dt;
        #pragma omp parallel for simd schedule(static)
        for (std::size_t idx = 0; idx < count; ++idx)
        {
            const double q2 =
                0.75 * static_cast<double>(qn_ptr[idx]) +
                0.25 * (static_cast<double>(q1_ptr[idx]) +
                        dt_local * static_cast<double>(k2_ptr[idx]));
            q2_ptr[idx] = static_cast<float>(q2);
        }
    }

    std::vector<NumericalTendencies> k3(state.fields.size());
    rk::validate_state(stage2_, "RK3");
    rhs_function_(stage2_, state.time + 0.5 * state.dt, k3);
    rk::validate_rhs(stage2_, k3, "RK3");
    if (k3.size() != state.fields.size())
    {
        throw std::runtime_error("RK3 k3 size mismatch");
    }

    std::vector<NumericalState> candidate(state.fields.size());
    for (size_t i = 0; i < state.fields.size(); ++i)
    {
        const Field3D& qn = state.fields[i].data;
        const Field3D& q2 = stage2_[i].data;
        const Field3D& k3_field = k3[i].tendencies;
        if (!same_shape(qn, q2) || !same_shape(qn, k3_field))
        {
            throw std::runtime_error("RK3 final stage shape mismatch");
        }

        candidate[i].name = state.fields[i].name;
        candidate[i].data.resize(
            qn.size_r(), qn.size_th(), qn.size_z(), 0.0f);
        const float* __restrict__ qn_ptr = qn.data();
        const float* __restrict__ q2_ptr = q2.data();
        const float* __restrict__ k3_ptr = k3_field.data();
        float* __restrict__ out_ptr = candidate[i].data.data();
        const std::size_t count = qn.size();
        const double dt_local = state.dt;
        #pragma omp parallel for simd schedule(static)
        for (std::size_t idx = 0; idx < count; ++idx)
        {
            const double qn_val = static_cast<double>(qn_ptr[idx]);
            const double updated =
                (1.0 / 3.0) * qn_val +
                (2.0 / 3.0) * (static_cast<double>(q2_ptr[idx]) +
                               dt_local * static_cast<double>(k3_ptr[idx]));
            out_ptr[idx] = static_cast<float>(updated);
        }
    }
    rk::validate_state(candidate, "RK3");
    for (std::size_t i = 0; i < state.fields.size(); ++i)
    {
        state.fields[i].data = std::move(candidate[i].data);
    }
}

void RK3Scheme::add_states(const std::vector<NumericalState>& state1, double coef1,
                          const std::vector<NumericalTendencies>& tend, double coef2, double dt,
                          std::vector<NumericalState>& result) {
    if (tend.size() != state1.size())
    {
        throw std::runtime_error("RK3 tendency count mismatch");
    }
    result.resize(state1.size());

    for (size_t i = 0; i < state1.size(); ++i) 
    {
        result[i].name = state1[i].name;
        const Field3D& base = state1[i].data;
        const Field3D& tendency = tend[i].tendencies;
        if (!same_shape(base, tendency))
        {
            throw std::runtime_error("RK3 add_states shape mismatch");
        }

        result[i].data.resize(base.size_r(), base.size_th(), base.size_z(), 0.0f);
        const float* __restrict__ base_ptr = base.data();
        const float* __restrict__ tend_ptr = tendency.data();
        float* __restrict__ out_ptr = result[i].data.data();
        const std::size_t count = base.size();
        const double scale = coef2 * dt;
        #pragma omp parallel for simd schedule(static)
        for (std::size_t idx = 0; idx < count; ++idx)
        {
            out_ptr[idx] = static_cast<float>(
                coef1 * static_cast<double>(base_ptr[idx]) +
                scale * static_cast<double>(tend_ptr[idx]));
        }
    }
}

} // namespace tmv
