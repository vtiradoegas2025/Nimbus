#include "rk4.hpp"
#include "util/log.hpp"
#include "numerics/time_stepping/rk_validation.hpp"

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

RK4Scheme::RK4Scheme() : rhs_function_(nullptr) {}

void RK4Scheme::initialize() 
{
    initialize(TimeSteppingConfig{}, nullptr);
}

void RK4Scheme::initialize(const TimeSteppingConfig&, RHSFunction rhs_func)
{
    rhs_function_ = rhs_func;

    tmv::log_info("Initialized RK4 time stepping scheme");
}

void RK4Scheme::step(
    const TimeSteppingConfig&,
    TimeSteppingState& state,
    TimeSteppingDiagnostics* diag_opt
) 
{
    if (!rhs_function_) 
    {
        throw std::runtime_error("RHS function not set for RK4 scheme");
    }
    const double next_time = state.time + state.dt;
    if (!std::isfinite(state.time) || !std::isfinite(state.dt) || state.dt <= 0.0 || !std::isfinite(next_time) ||
        next_time <= state.time || !(state.time + .5 * state.dt > state.time) ||
        !(state.time + .5 * state.dt < next_time))
    {
        throw std::invalid_argument("RK4 requires finite time and a positive finite, representable timestep");
    }

    rk::validate_state(state.fields, "RK4");

    if (diag_opt)
    {
        if (diag_opt->n_steps == std::numeric_limits<int>::max())
        {
            throw std::overflow_error("RK4 diagnostic step count overflow");
        }
        if (diag_opt->dt_history.size() == diag_opt->dt_history.max_size())
        {
            throw std::length_error("RK4 timestep history is full");
        }
        diag_opt->prepare_history_append();
    }

    if (stage1_.empty() || stage1_.size() != state.fields.size()) {
        stage1_.resize(state.fields.size());
        stage2_.resize(state.fields.size());
        stage3_.resize(state.fields.size());
        k1_.resize(state.fields.size());
        k2_.resize(state.fields.size());
        k3_.resize(state.fields.size());
        k4_.resize(state.fields.size());
    }

    compute_k1(state);
    compute_k2(state);
    compute_k3(state);
    compute_k4(state);
    update_final(state);

    state.time = next_time;

    if (diag_opt) 
    {
        diag_opt->dt_history.push_back(state.dt);
        ++diag_opt->n_steps;
    }
}

double RK4Scheme::suggest_dt(
    const TimeSteppingConfig&,
    const TimeSteppingState&
) 
{
    return 0.5;
}

void RK4Scheme::step_coupled_unsplit_rk(
    const TimeSteppingConfig&,
    double dt,
    double time,
    const UnsplitRkCallbacks& callbacks,
    TimeSteppingDiagnostics* diagnostics)
{
    const double next_time = time + dt;
    if (!std::isfinite(time) || !std::isfinite(dt) || dt <= 0.0 || !std::isfinite(next_time) || next_time <= time ||
        !(time + .5 * dt > time) || !(time + .5 * dt < next_time))
        throw std::invalid_argument("coupled RK4 requires finite time and a positive finite, representable timestep");
    if (!callbacks.capture_base || !callbacks.compute_tendencies ||
        !callbacks.publish_combination || !callbacks.restore_base)
        throw std::invalid_argument(
            "coupled RK4 requires a complete callback transaction");
    if (diagnostics)
    {
        if (diagnostics->n_steps == std::numeric_limits<int>::max())
            throw std::overflow_error("RK4 diagnostic step count overflow");
        if (diagnostics->dt_history.size() == diagnostics->dt_history.max_size())
            throw std::length_error("RK4 timestep history is full");
        diagnostics->prepare_history_append();
    }

    callbacks.capture_base();
    try
    {
        callbacks.compute_tendencies(0, time);
        callbacks.publish_combination(1, {0.5, 0.0, 0.0, 0.0}, dt);
        callbacks.compute_tendencies(1, time + 0.5 * dt);
        callbacks.publish_combination(2, {0.0, 0.5, 0.0, 0.0}, dt);
        callbacks.compute_tendencies(2, time + 0.5 * dt);
        callbacks.publish_combination(3, {0.0, 0.0, 1.0, 0.0}, dt);
        callbacks.compute_tendencies(3, time + dt);
        callbacks.publish_combination(
            4, {1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0}, dt);
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

void RK4Scheme::invalidate_generation_state_after_domain_recycling() noexcept
{
    for (auto& state : stage1_) state.data.fill(0.0f);
    for (auto& state : stage2_) state.data.fill(0.0f);
    for (auto& state : stage3_) state.data.fill(0.0f);
    for (auto& tendency : k1_) tendency.tendencies.fill(0.0f);
    for (auto& tendency : k2_) tendency.tendencies.fill(0.0f);
    for (auto& tendency : k3_) tendency.tendencies.fill(0.0f);
    for (auto& tendency : k4_) tendency.tendencies.fill(0.0f);
}

void RK4Scheme::compute_k1(const TimeSteppingState& state) 
{
    rhs_function_(state.fields, state.time, k1_);
    rk::validate_rhs(state.fields, k1_, "RK4");
}

void RK4Scheme::compute_k2(const TimeSteppingState& state) 
{
    add_states(state.fields, 1.0, k1_, 0.5, state.dt, stage1_);

    rk::validate_state(stage1_, "RK4");
    rhs_function_(stage1_, state.time + 0.5 * state.dt, k2_);
    rk::validate_rhs(stage1_, k2_, "RK4");
}

void RK4Scheme::compute_k3(const TimeSteppingState& state) 
{
    add_states(state.fields, 1.0, k2_, 0.5, state.dt, stage2_);

    rk::validate_state(stage2_, "RK4");
    rhs_function_(stage2_, state.time + 0.5 * state.dt, k3_);
    rk::validate_rhs(stage2_, k3_, "RK4");
}

void RK4Scheme::compute_k4(const TimeSteppingState& state) {
    add_states(state.fields, 1.0, k3_, 1.0, state.dt, stage3_);

    rk::validate_state(stage3_, "RK4");
    rhs_function_(stage3_, state.time + state.dt, k4_);
    rk::validate_rhs(stage3_, k4_, "RK4");
}

void RK4Scheme::update_final(TimeSteppingState& state) 
{
    if (k1_.size() != state.fields.size() ||
        k2_.size() != state.fields.size() ||
        k3_.size() != state.fields.size() ||
        k4_.size() != state.fields.size())
    {
        throw std::runtime_error("RK4 stage storage size mismatch");
    }

    std::vector<NumericalState> candidate(state.fields.size());
    for (size_t i = 0; i < state.fields.size(); ++i) 
    {
        const Field3D& values = state.fields[i].data;
        const Field3D& k1 = k1_[i].tendencies;
        const Field3D& k2 = k2_[i].tendencies;
        const Field3D& k3 = k3_[i].tendencies;
        const Field3D& k4 = k4_[i].tendencies;
        if (!same_shape(values, k1) || !same_shape(values, k2) ||
            !same_shape(values, k3) || !same_shape(values, k4))
        {
            throw std::runtime_error("RK4 update_final shape mismatch");
        }

        candidate[i].name = state.fields[i].name;
        candidate[i].data.resize(
            values.size_r(), values.size_th(), values.size_z(), 0.0f);
        const float* __restrict__ values_ptr = values.data();
        float* __restrict__ out = candidate[i].data.data();
        const float* __restrict__ k1_ptr = k1.data();
        const float* __restrict__ k2_ptr = k2.data();
        const float* __restrict__ k3_ptr = k3.data();
        const float* __restrict__ k4_ptr = k4.data();
        const std::size_t count = values.size();
        const double dt_over_6 = state.dt / 6.0;
        #pragma omp parallel for simd schedule(static)
        for (std::size_t idx = 0; idx < count; ++idx)
        {
            const double updated =
                static_cast<double>(values_ptr[idx]) +
                dt_over_6 *
                    (static_cast<double>(k1_ptr[idx]) +
                     2.0 * static_cast<double>(k2_ptr[idx]) +
                     2.0 * static_cast<double>(k3_ptr[idx]) +
                     static_cast<double>(k4_ptr[idx]));
            out[idx] = static_cast<float>(updated);
        }
    }
    rk::validate_state(candidate, "RK4");
    for (std::size_t i = 0; i < state.fields.size(); ++i)
    {
        state.fields[i].data = std::move(candidate[i].data);
    }
}

void RK4Scheme::add_states(const std::vector<NumericalState>& state1, double coef1,
                          const std::vector<NumericalTendencies>& tend, double coef2, double dt,
                          std::vector<NumericalState>& result) 
{
    if (tend.size() != state1.size())
    {
        throw std::runtime_error("RK4 tendency count mismatch");
    }
    result.resize(state1.size());

    for (size_t i = 0; i < state1.size(); ++i) 
    {
        result[i].name = state1[i].name;
        const Field3D& base = state1[i].data;
        const Field3D& tendency = tend[i].tendencies;
        if (!same_shape(base, tendency))
        {
            throw std::runtime_error("RK4 add_states shape mismatch");
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
