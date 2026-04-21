/**
 * @file rk3.cpp
 * @brief Implementation for the numerics module.
 *
 * Provides executable logic for the numerics runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/numerics subsystem.
 */

#include "rk3.hpp"
#include <iostream>
#include <algorithm>
#include <stdexcept>


/**
 * @brief Implements the RK3 time stepping scheme.
 */

RK3Scheme::RK3Scheme() : rhs_function_(nullptr) {}

/**
 * @brief Initializes the RK3 time stepping scheme.
 */
void RK3Scheme::initialize() 
{
    initialize(TimeSteppingConfig{}, nullptr);
}

/**
 * @brief Initializes the RK3 time stepping scheme.
 */
void RK3Scheme::initialize(const TimeSteppingConfig& cfg, RHSFunction rhs_func) 
{
    config_ = cfg;
    rhs_function_ = rhs_func;

    std::cout << "Initialized RK3 (SSPRK3) time stepping scheme" << std::endl;
    if (cfg.split_acoustic) {
        std::cout << "  Split-explicit acoustic mode enabled" << std::endl;
    }
}

/**
 * @brief Steps the RK3 time stepping scheme.
 */

void RK3Scheme::step(
    const TimeSteppingConfig& cfg,
    TimeSteppingState& state,
    TimeSteppingDiagnostics* diag_opt
) 
{
    (void)cfg;

    if (!rhs_function_) 
    {
        throw std::runtime_error("RHS function not set for RK3 scheme");
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
double RK3Scheme::suggest_dt(
    const TimeSteppingConfig& cfg,
    const TimeSteppingState& state
) 
{
    return 1.0;
}

/**
 * @brief Computes the first stage.
 */
void RK3Scheme::compute_stage1(const TimeSteppingState& state, std::vector<NumericalTendencies>& tendencies) 
{
    rhs_function_(state.fields, state.time, tendencies);
}

/**
 * @brief Computes the second stage.
 */
void RK3Scheme::compute_stage2(const TimeSteppingState& state, std::vector<NumericalTendencies>& tendencies) 
{
    add_states(state.fields, 1.0, rhs_stage1_, 1.0, state.dt, stage1_);

    rhs_function_(stage1_, state.time + state.dt, tendencies);
}

/**
 * @brief Computes the final stage.
 */
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
        if (qn.size() != q1.size() || qn.size() != k2.size())
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
    rhs_function_(stage2_, state.time + state.dt, k3);
    if (k3.size() != state.fields.size())
    {
        throw std::runtime_error("RK3 k3 size mismatch");
    }

    for (size_t i = 0; i < state.fields.size(); ++i)
    {
        Field3D& qn = state.fields[i].data;
        const Field3D& q2 = stage2_[i].data;
        const Field3D& k3_field = k3[i].tendencies;
        if (qn.size() != q2.size() || qn.size() != k3_field.size())
        {
            throw std::runtime_error("RK3 final stage shape mismatch");
        }

        float* __restrict__ qn_ptr = qn.data();
        const float* __restrict__ q2_ptr = q2.data();
        const float* __restrict__ k3_ptr = k3_field.data();
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
            qn_ptr[idx] = static_cast<float>(updated);
        }
    }
}

/**
 * @brief Copies the state.
 */
void RK3Scheme::copy_state(const std::vector<NumericalState>& src, std::vector<NumericalState>& dst) 
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
void RK3Scheme::add_states(const std::vector<NumericalState>& state1, double coef1,
                          const std::vector<NumericalTendencies>& tend, double coef2, double dt,
                          std::vector<NumericalState>& result) {
    result.resize(state1.size());

    for (size_t i = 0; i < state1.size(); ++i) 
    {
        result[i].name = state1[i].name;
        const Field3D& base = state1[i].data;
        const Field3D& tendency = tend[i].tendencies;
        if (base.size() != tendency.size())
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

/**
 * @brief Updates the state.
 */
void RK3Scheme::update_state(std::vector<NumericalState>& state,
                           const std::vector<NumericalTendencies>& tendencies, double dt)
{
    for (size_t i = 0; i < state.size(); ++i)
    {
        Field3D& values = state[i].data;
        const Field3D& tendency = tendencies[i].tendencies;
        if (values.size() != tendency.size())
        {
            throw std::runtime_error("RK3 update_state shape mismatch");
        }

        float* __restrict__ state_ptr = values.data();
        const float* __restrict__ tend_ptr = tendency.data();
        const std::size_t count = values.size();
        #pragma omp parallel for simd schedule(static)
        for (std::size_t idx = 0; idx < count; ++idx)
        {
            state_ptr[idx] = static_cast<float>(
                static_cast<double>(state_ptr[idx]) + dt * static_cast<double>(tend_ptr[idx]));
        }
    }
}
