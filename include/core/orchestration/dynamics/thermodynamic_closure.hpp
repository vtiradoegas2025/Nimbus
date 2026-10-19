/**
 * @file thermodynamic_closure.hpp
 * @brief Initial-state pressure-controlled thermodynamic closure.
 */

#pragma once

struct SimulationStateView;

/**
 * @brief Restores the model's dry rho-theta equation-of-state closure.
 *
 * Pressure is the controlled variable during initialization: this operation
 * leaves p, theta, and every moisture field unchanged and re-diagnoses rho
 * pointwise from
 *
 *   T   = theta * (p / p0)^(R_d / c_p)
 *   rho = p / (R_d * T).
 *
 * Moisture is deliberately excluded from this density diagnosis. The active
 * dynamics uses the dry rho-theta EOS and represents vapor buoyancy explicitly;
 * including qv here would count that contribution twice. Temperature is a
 * derived field in Nimbus, so no separate temperature storage is updated.
 *
 * Call this after any initial-condition process that changes theta, including
 * stochastic IC perturbations, and before the first dynamics step.
 * Throws before mutating rho when pressure/theta are invalid or the diagnosed
 * density lies outside the model's supported [0.01, 5] kg/m^3 range; it never
 * silently returns a state that violates the stated equation of state.
 */
void apply_initial_thermodynamic_closure(const SimulationStateView& state);
