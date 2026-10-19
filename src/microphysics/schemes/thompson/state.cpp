#include "thompson.hpp"

#include <cmath>

namespace tmv
{

PortableSchemeState ThompsonScheme::portable_state() const
{
    PortableSchemeState state;
    state.version = 3;
    state.integers = {thompson::PrognosticState::portable_version};
    state.reals = {qc0_, c_auto_, c_evap_, ccn_conc_, in_conc_};
    state.fields = prognostic_state_.portable_fields();
    return state;
}

PortableSchemeState ThompsonScheme::portable_parameter_state() const
{
    PortableSchemeState state;
    state.version = 3;
    state.integers = {thompson::PrognosticState::portable_version};
    state.reals = {qc0_, c_auto_, c_evap_, ccn_conc_, in_conc_};
    return state;
}

MicrophysicsParameterIdentity ThompsonScheme::parameter_identity() const noexcept
{
    MicrophysicsParameterIdentity identity;
    identity.version = 3;
    identity.integer_count = 1;
    identity.integers[0] = thompson::PrognosticState::portable_version;
    identity.real_count = 5;
    identity.reals = {qc0_, c_auto_, c_evap_, ccn_conc_, in_conc_};
    return identity;
}

bool ThompsonScheme::restore_portable_state(const PortableSchemeState& state,
                                            std::string& error)
{
    const bool legacy_v1 = state.version == 1 && state.reals.size() == 4
        && state.fields.empty() && state.integers.empty();
    const bool legacy_v2 = state.version == 2 && state.reals.size() == 5
        && state.fields.empty() && state.integers.empty();
    const bool current = state.version == 3 && state.reals.size() == 5
        && state.fields.size() == 10 && state.integers.size() == 1
        && state.integers[0] == thompson::PrognosticState::portable_version;
    if (!legacy_v1 && !legacy_v2 && !current)
    {
        error = "invalid Thompson portable-state layout";
        return false;
    }
    for (double value : state.reals)
    {
        if (!std::isfinite(value) || value < 0.0)
        {
            error = "invalid Thompson portable-state parameter";
            return false;
        }
    }

    thompson::PrognosticState restored_state;
    if (current)
    {
        if (!restored_state.restore_fields(state.fields, error)
            || !restored_state.bind_aerosol_profile(
                {state.reals[3],
                 thompson::PrognosticState::free_troposphere_water_friendly_aerosol_m3,
                 state.reals[4],
                 thompson::PrognosticState::free_troposphere_ice_friendly_aerosol_m3},
                error))
        {
            return false;
        }
    }

    qc0_ = state.reals[0];
    c_auto_ = state.reals[1];
    c_evap_ = legacy_v1 ? 1.0e-3 : state.reals[2];
    ccn_conc_ = state.reals[legacy_v1 ? 2 : 3];
    in_conc_ = state.reals[legacy_v1 ? 3 : 4];
    prognostic_state_ = std::move(restored_state);
    error.clear();
    return true;
}

void ThompsonScheme::prepare_prognostic_state(
    const Field3D& rho_d,
    const Field3D& qc, const Field3D& qr,
    const Field3D& qi, const Field3D& qs,
    const Field3D& qg, const Field3D& qh)
{
    ensure_activation_table();
    std::vector<double> layer_thickness(rho_d.size());
    const int nth = rho_d.size_th();
    const int nz = rho_d.size_z();
    for (int i = 0; i < rho_d.size_r(); ++i)
        for (int j = 0; j < nth; ++j)
        {
            for (int k = 0; k < nz; ++k)
            {
                const std::size_t index =
                    (static_cast<std::size_t>(i) * static_cast<std::size_t>(nth)
                     + static_cast<std::size_t>(j)) * static_cast<std::size_t>(nz)
                    + static_cast<std::size_t>(k);
                layer_thickness[index] = vertical_spacing_m(i, j, k);
            }
        }
    const std::vector<double> profile_sample_height =
        thompson::flat_profile_sampling_heights(
            rho_d.size_r(), nth, nz, layer_thickness);
    prognostic_state_.prepare(
        rho_d, qc, qr, qi, profile_sample_height,
        {ccn_conc_,
         thompson::PrognosticState::free_troposphere_water_friendly_aerosol_m3,
         in_conc_,
         thompson::PrognosticState::free_troposphere_ice_friendly_aerosol_m3},
        physical_cell_domain(
            rho_d.size_r(), rho_d.size_th(), rho_d.size_z()));
    (void)qs;
    (void)qg;
    (void)qh;
}

MicrophysicsPrognosticScalarList ThompsonScheme::prognostic_scalars_for_transport()
{
    return prognostic_state_.transport_bindings();
}

bool ThompsonScheme::validate_prognostic_update(double dt, std::string& error) const
{
    const Field3D& reference = prognostic_state_.cloud_number();
    return prognostic_state_.validate_update(
        dt, physical_cell_domain(
            reference.size_r(), reference.size_th(), reference.size_z()),
        error);
}

bool ThompsonScheme::apply_stochastic_column_multipliers(double              dt,
                                                         const Field3D&      density,
                                                         std::vector<float>& column_multipliers,
                                                         std::string&        error)
{
    return prognostic_state_.apply_column_multipliers(
        dt,
        density,
        physical_cell_domain(density.size_r(), density.size_th(), density.size_z()),
        column_multipliers,
        error);
}

void ThompsonScheme::commit_prognostic_update(double dt) noexcept
{
    const Field3D& reference = prognostic_state_.cloud_number();
    prognostic_state_.commit_update(dt,
                                    physical_cell_domain(reference.size_r(), reference.size_th(), reference.size_z()));
}

void ThompsonScheme::invalidate_transient_state_after_domain_recycling() noexcept
{
    prognostic_state_.invalidate_transient();
    surface_sedimentation_sink_rate_ = Field3D{};
}

} // namespace tmv
