/**
 * @file rk4.cpp
 * @brief Implementation for the numerics module.
 *
 * Provides executable logic for the numerics runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/numerics subsystem.
 */

#include "rk4.hpp"
#include <iostream>
#include <stdexcept>

RK4Scheme::RK4Scheme() : rhs_function_(nullptr) {}

/**
 * @brief Initializes the RK4 time stepping scheme.
 */
void RK4Scheme::initialize() 
{
    initialize(TimeSteppingConfig{}, nullptr);
}

/**
 * @brief Initializes the RK4 time stepping scheme.
 */
void RK4Scheme::initialize(const TimeSteppingConfig& cfg, RHSFunction rhs_func) 
{
    config_ = cfg;
    rhs_function_ = rhs_func;

    std::cout << "Initialized RK4 time stepping scheme" << std::endl;
}

/**
 * @brief Steps the RK4 time stepping scheme.
 */
void RK4Scheme::step(
    const TimeSteppingConfig& cfg,
    TimeSteppingState& state,
    TimeSteppingDiagnostics* diag_opt
) 
{
    if (!rhs_function_) 
    {
        throw std::runtime_error("RHS function not set for RK4 scheme");
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

    state.time += state.dt;

    if (diag_opt) 
    {
        diag_opt->n_steps++;
        diag_opt->dt_history.push_back(state.dt);
    }
}

/**
 * @brief Suggests the time step.
 */  
double RK4Scheme::suggest_dt(
    const TimeSteppingConfig& cfg,
    const TimeSteppingState& state
) 
{
    return 0.5;
}

/**
 * @brief Computes the first stage.
 */
void RK4Scheme::compute_k1(const TimeSteppingState& state) 
{
    rhs_function_(state.fields, state.time, k1_);
}

/**
 * @brief Computes the second stage.
 */
void RK4Scheme::compute_k2(const TimeSteppingState& state) 
{
    add_states(state.fields, 1.0, k1_, 0.5, state.dt, stage1_);

    rhs_function_(stage1_, state.time + 0.5 * state.dt, k2_);
}

/**
 * @brief Computes the third stage.
 */
void RK4Scheme::compute_k3(const TimeSteppingState& state) 
{
    add_states(state.fields, 1.0, k2_, 0.5, state.dt, stage2_);

    rhs_function_(stage2_, state.time + 0.5 * state.dt, k3_);
}

void RK4Scheme::compute_k4(const TimeSteppingState& state) {
    add_states(state.fields, 1.0, k3_, 1.0, state.dt, stage3_);

    rhs_function_(stage3_, state.time + state.dt, k4_);
}

/**
 * @brief Updates the final stage.
 */
void RK4Scheme::update_final(TimeSteppingState& state) 
{

    for (size_t i = 0; i < state.fields.size(); ++i) 
    {
        Field3D& values = state.fields[i].data;
        const Field3D& k1 = k1_[i].tendencies;
        const Field3D& k2 = k2_[i].tendencies;
        const Field3D& k3 = k3_[i].tendencies;
        const Field3D& k4 = k4_[i].tendencies;
        if (values.size() != k1.size() || values.size() != k2.size() ||
            values.size() != k3.size() || values.size() != k4.size())
        {
            throw std::runtime_error("RK4 update_final shape mismatch");
        }

        float* __restrict__ out = values.data();
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
                static_cast<double>(out[idx]) +
                dt_over_6 *
                    (static_cast<double>(k1_ptr[idx]) +
                     2.0 * static_cast<double>(k2_ptr[idx]) +
                     2.0 * static_cast<double>(k3_ptr[idx]) +
                     static_cast<double>(k4_ptr[idx]));
            out[idx] = static_cast<float>(updated);
        }
    }
}

/**
 * @brief Copies the state.
 */
void RK4Scheme::copy_state(const std::vector<NumericalState>& src, std::vector<NumericalState>& dst) 
{
    dst.resize(src.size());

    for (size_t i = 0; i < src.size(); ++i) 
    {
        dst[i].data = src[i].data;
        dst[i].name = src[i].name;
    }
}

/**
 * @brief Adds the states.
 */
void RK4Scheme::add_states(const std::vector<NumericalState>& state1, double coef1,
                          const std::vector<NumericalTendencies>& tend, double coef2, double dt,
                          std::vector<NumericalState>& result) 
{
    result.resize(state1.size());

    for (size_t i = 0; i < state1.size(); ++i) 
    {
        result[i].name = state1[i].name;
        const Field3D& base = state1[i].data;
        const Field3D& tendency = tend[i].tendencies;
        if (base.size() != tendency.size())
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
