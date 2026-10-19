// Milbrandt multi-moment state, process rates, and sedimentation ledger.

#include "milbrandt.hpp"
#include "warm_rain.hpp"
#include "activation.hpp"
#include "melting.hpp"
#include "cold.hpp"
#include "coupled_storage.hpp"
#include "microphysics/base/thermodynamics.hpp"
#include "core/field/field_pool.hpp"
#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <sstream>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <limits>

namespace tmv
{

/**
 * @brief Initializes the Milbrandt scheme with default parameters.
 */
MilbrandtScheme::MilbrandtScheme(
    double alpha_r, double alpha_i, double alpha_s, double alpha_g, double alpha_h,
    double c_r, double d_r, double c_i, double d_i, double c_s, double d_s,
    double c_g, double d_g, double c_h, double d_h,
    double a_r, double b_r, double a_i, double b_i, double a_s, double b_s,
    double a_g, double b_g, double a_h, double b_h,
    bool hail_processes
) : alpha_r_(alpha_r), alpha_i_(alpha_i), alpha_s_(alpha_s), alpha_g_(alpha_g), alpha_h_(alpha_h),
    c_r_(c_r), d_r_(d_r), c_i_(c_i), d_i_(d_i), c_s_(c_s), d_s_(d_s),
    c_g_(c_g), d_g_(d_g), c_h_(c_h), d_h_(d_h),
    a_r_(a_r), b_r_(b_r), a_i_(a_i), b_i_(b_i), a_s_(a_s), b_s_(b_s),
    a_g_(a_g), b_g_(b_g), a_h_(a_h), b_h_(b_h),
    hail_processes_(hail_processes) {}

MilbrandtScheme::MilbrandtScheme(const MilbrandtScheme& other)
    : MicrophysicsScheme(other),
      alpha_r_(other.alpha_r_), alpha_i_(other.alpha_i_),
      alpha_s_(other.alpha_s_), alpha_g_(other.alpha_g_),
      alpha_h_(other.alpha_h_),
      c_r_(other.c_r_), d_r_(other.d_r_),
      c_i_(other.c_i_), d_i_(other.d_i_),
      c_s_(other.c_s_), d_s_(other.d_s_),
      c_g_(other.c_g_), d_g_(other.d_g_),
      c_h_(other.c_h_), d_h_(other.d_h_),
      a_r_(other.a_r_), b_r_(other.b_r_),
      a_i_(other.a_i_), b_i_(other.b_i_),
      a_s_(other.a_s_), b_s_(other.b_s_),
      a_g_(other.a_g_), b_g_(other.b_g_),
      a_h_(other.a_h_), b_h_(other.b_h_),
      hail_processes_(other.hail_processes_),
      Nc_(other.Nc_), Nr_(other.Nr_), Ni_(other.Ni_),
      Ns_(other.Ns_), Ng_(other.Ng_), Nh_(other.Nh_),
      dNc_dt_(other.dNc_dt_), dNr_dt_(other.dNr_dt_),
      dNi_dt_(other.dNi_dt_), dNs_dt_(other.dNs_dt_),
      dNg_dt_(other.dNg_dt_), dNh_dt_(other.dNh_dt_),
      surface_sedimentation_sink_rate_(
          other.surface_sedimentation_sink_rate_),
      internal_triple_(other.internal_triple_)
{
    if (internal_triple_)
    {
        internal_triple_->candidate_valid = false;
        internal_triple_->prepared = false;
        if (internal_triple_->mode == InternalTripleMode::FrozenCompositionOnly ||
            internal_triple_->mode == InternalTripleMode::MeltingOnly ||
            internal_triple_->mode == InternalTripleMode::IceMeltingOnly ||
            internal_triple_->mode == InternalTripleMode::SnowMeltingOnly ||
            internal_triple_->mode == InternalTripleMode::WarmPrimaryOnly ||
            internal_triple_->mode == InternalTripleMode::CloudAdjustmentOnly ||
            internal_triple_->mode == InternalTripleMode::WarmCloudPrimaryOnly ||
            internal_triple_->mode == InternalTripleMode::DynamicWarmCloudPrimaryOnly ||
            internal_triple_->mode == InternalTripleMode::SubcycledWarmCloudPrimaryOnly ||
            internal_triple_->mode == InternalTripleMode::FrozenCollectionPrimaryOnly ||
            internal_triple_->mode == InternalTripleMode::ColdCloudCollectionPrimaryOnly ||
            internal_triple_->mode == InternalTripleMode::ColdCloudDiffusionPrimaryOnly ||
            is_cold_riming_mode(internal_triple_->mode))
        {
            internal_triple_->primary_endpoints.owner = nullptr;
        }
    }
}

MilbrandtScheme& MilbrandtScheme::operator=(const MilbrandtScheme& other)
{
    if (this != &other)
    {
        MilbrandtScheme copy(other);
        *this = std::move(copy);
    }
    return *this;
}

void MilbrandtScheme::configure(const MicrophysicsConfig& config)
{
    if (internal_triple_)
    {
        throw std::logic_error(
            "internal Milbrandt triple sedimentation cannot be configured");
    }
    if (config.milbrandt.triple_moment)
    {
        throw std::invalid_argument(
            "microphysics.milbrandt.triple_moment is not part of the "
            "production Milbrandt-Yau double-moment scheme");
    }
    hail_processes_ = config.milbrandt.hail_processes;
}

void MilbrandtScheme::prepare_prognostic_state(
    const Field3D& rho_d,
    const Field3D& qc, const Field3D& qr,
    const Field3D& qi, const Field3D& qs,
    const Field3D& qg, const Field3D& qh)
{
    if (internal_triple_)
    {
        throw std::logic_error(
            "internal Milbrandt triple sedimentation has no dynamics prognostic preparation");
    }
    const int nr = rho_d.size_r();
    const int nth = rho_d.size_th();
    const int nz = rho_d.size_z();
    const auto matches = [=](const Field3D& field)
    {
        return field.size_r() == nr && field.size_th() == nth
            && field.size_z() == nz;
    };
    if (!matches(Nc_) || !matches(Nr_) || !matches(Ni_)
        || !matches(Ns_) || !matches(Ng_) || !matches(Nh_))
    {
        initialize_prognostic_fields(rho_d, qc, qr, qi, qs, qg, qh);
    }
}

MicrophysicsPrognosticScalarList
MilbrandtScheme::prognostic_scalars_for_transport()
{
    if (internal_triple_)
    {
        throw std::logic_error(
            "internal Milbrandt triple sedimentation cannot enter prognostic transport or RK");
    }
    using Storage = MicrophysicsScalarStorage;
    return {
        {"milbrandt.Nc_kg-1", &Nc_, true, Storage::DryAirMixingRatio, MicrophysicsMassSpecies::Cloud},
        {"milbrandt.Nr_kg-1", &Nr_, true, Storage::DryAirMixingRatio, MicrophysicsMassSpecies::Rain},
        {"milbrandt.Ni_kg-1", &Ni_, true, Storage::DryAirMixingRatio, MicrophysicsMassSpecies::Ice},
        {"milbrandt.Ns_kg-1", &Ns_, true, Storage::DryAirMixingRatio, MicrophysicsMassSpecies::Snow},
        {"milbrandt.Ng_kg-1", &Ng_, true, Storage::DryAirMixingRatio, MicrophysicsMassSpecies::Graupel},
        {"milbrandt.Nh_kg-1", &Nh_, true, Storage::DryAirMixingRatio, MicrophysicsMassSpecies::Hail}
    };
}

bool MilbrandtScheme::validate_prognostic_update(
    double dt, std::string& error) const
{
    if (internal_triple_)
    {
        const auto& internal = *internal_triple_;
        if (internal.poisoned || !internal.candidate_valid ||
            !std::isfinite(dt) || dt != internal.candidate_dt)
        {
            error = "internal Milbrandt triple sedimentation candidate is unavailable";
            return false;
        }
        error.clear();
        return true;
    }
    if (!std::isfinite(dt) || dt <= 0.0)
    {
        error = "number-moment timestep must be positive and finite";
        return false;
    }
    const std::array<const Field3D*, 6> moments{
        &Nc_, &Nr_, &Ni_, &Ns_, &Ng_, &Nh_};
    const std::array<const Field3D*, 6> tendencies{
        &dNc_dt_, &dNr_dt_, &dNi_dt_, &dNs_dt_, &dNg_dt_, &dNh_dt_};
    constexpr std::array<const char*, 6> names{
        "Nc", "Nr", "Ni", "Ns", "Ng", "Nh"};
    if (moments.front()->empty())
    {
        error = "number moments were not initialized";
        return false;
    }
    const PhysicalCellDomain domain = physical_cell_domain(
        moments.front()->size_r(), moments.front()->size_th(),
        moments.front()->size_z());
    for (std::size_t species = 0; species < moments.size(); ++species)
    {
        const auto matches = [&](const Field3D& field)
        {
            return field.size_r() == moments.front()->size_r() && field.size_th() == moments.front()->size_th() &&
                   field.size_z() == moments.front()->size_z();
        };
        if (!matches(*moments[species]) || !matches(*tendencies[species]))
        {
            error = "number-moment tendency shape mismatch";
            return false;
        }
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
                    const double old_value = (*moments[species])[i][j][k];
                    const double rate = (*tendencies[species])[i][j][k];
                    const double candidate = old_value + dt * rate;
                    if (!std::isfinite(old_value) || old_value < 0.0 || !std::isfinite(rate) ||
                        !std::isfinite(candidate) || candidate > std::numeric_limits<float>::max() ||
                        candidate < -1.0e-5 * std::max(1.0, old_value))
                    {
                        error = std::string("invalid number-moment update for ")
                            + names[species];
                        return false;
                    }
                }
    }
    error.clear();
    return true;
}

bool MilbrandtScheme::apply_stochastic_column_multipliers(double              dt,
                                                          const Field3D&      density,
                                                          std::vector<float>& column_multipliers,
                                                          std::string&        error)
{
    if (internal_triple_)
    {
        error = "internal Milbrandt triple sedimentation cannot use stochastic column control";
        return false;
    }
    if (!std::isfinite(dt) || dt <= 0.0 || Nc_.empty())
    {
        error = "number-moment stochastic timestep/state is invalid";
        return false;
    }
    const int nr = Nc_.size_r();
    const int nth = Nc_.size_th();
    const int nz = Nc_.size_z();
    if (column_multipliers.size()
        != static_cast<std::size_t>(nr) * static_cast<std::size_t>(nth))
    {
        error = "number-moment stochastic column shape mismatch";
        return false;
    }
    const PhysicalCellDomain domain = physical_cell_domain(nr, nth, nz);
    const std::array<const Field3D*, 6> moments{
        &Nc_, &Nr_, &Ni_, &Ns_, &Ng_, &Nh_};
    const std::array<Field3D*, 6> tendencies{
        &dNc_dt_, &dNr_dt_, &dNi_dt_, &dNs_dt_, &dNg_dt_, &dNh_dt_};

    const auto matches = [&](const Field3D& field)
    { return field.size_r() == nr && field.size_th() == nth && field.size_z() == nz; };
    for (std::size_t species = 0; species < moments.size(); ++species)
        if (!matches(*moments[species]) || !matches(*tendencies[species]))
        {
            error = "number-moment stochastic field shape mismatch";
            return false;
        }
    auto& admitted_multipliers = stochastic_multiplier_workspace_;
    admitted_multipliers.assign(column_multipliers.begin(), column_multipliers.end());

    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            const std::size_t index =
                static_cast<std::size_t>(i) * static_cast<std::size_t>(nth)
                + static_cast<std::size_t>(j);
            const float requested = column_multipliers[index];
            double admitted = requested;
            if (!std::isfinite(admitted) || admitted < 0.0)
            {
                error = "invalid requested number-moment stochastic multiplier";
                return false;
            }
            for (std::size_t species = 0; species < moments.size(); ++species)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
                    const double old_value = (*moments[species])[i][j][k];
                    const double rate = (*tendencies[species])[i][j][k];
                    if (!std::isfinite(old_value) || old_value < 0.0
                        || !std::isfinite(rate))
                    {
                        error = "invalid number-moment stochastic reservoir";
                        return false;
                    }
                    if (rate < 0.0)
                        admitted = std::min(
                            admitted, old_value / (-dt * rate));
                }
            float stored = static_cast<float>(std::max(0.0, admitted));
            if (static_cast<double>(stored)
                    < static_cast<double>(requested))
                stored = std::nextafter(stored, 0.0f);
            admitted_multipliers[index] = stored;
        }

    for (std::size_t species = 0; species < moments.size(); ++species)
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
                    const std::size_t column    = static_cast<std::size_t>(i) * nth + j;
                    const float       rate      = (*tendencies[species])(i, j, k) * admitted_multipliers[column];
                    const double      old_value = (*moments[species])(i, j, k);
                    const double      candidate = old_value + dt * static_cast<double>(rate);
                    if (!std::isfinite(rate) || !std::isfinite(candidate) ||
                        candidate > std::numeric_limits<float>::max() || candidate < -1.0e-5 * std::max(1.0, old_value))
                    {
                        error = "number-moment stochastic update is not representable";
                        return false;
                    }
                }
    std::copy(admitted_multipliers.begin(), admitted_multipliers.end(), column_multipliers.begin());
    for (Field3D* tendency : tendencies)
    {
        #pragma omp parallel for collapse(2)
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
            {
                const float multiplier =
                    column_multipliers[
                        static_cast<std::size_t>(i)
                            * static_cast<std::size_t>(nth)
                        + static_cast<std::size_t>(j)];
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                    (*tendency)[i][j][k] *= multiplier;
            }
    }
    error.clear();
    return true;
}

void MilbrandtScheme::commit_prognostic_update(double dt) noexcept
{
    if (internal_triple_)
    {
        internal_triple_->poisoned = true;
        internal_triple_->candidate_valid = false;
        internal_triple_->prepared = false;
        return;
    }
    const std::array<Field3D*, 6> moments{
        &Nc_, &Nr_, &Ni_, &Ns_, &Ng_, &Nh_};
    const std::array<const Field3D*, 6> tendencies{
        &dNc_dt_, &dNr_dt_, &dNi_dt_, &dNs_dt_, &dNg_dt_, &dNh_dt_};
    if (moments.front()->empty()) return;
    const PhysicalCellDomain domain = physical_cell_domain(
        moments.front()->size_r(), moments.front()->size_th(),
        moments.front()->size_z());
    for (std::size_t species = 0; species < moments.size(); ++species)
    {
        #pragma omp parallel for collapse(2)
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
                    const double candidate =
                        static_cast<double>((*moments[species])[i][j][k])
                        + dt * static_cast<double>(
                            (*tendencies[species])[i][j][k]);
                    (*moments[species])[i][j][k] = static_cast<float>(
                        std::max(0.0, candidate));
                }
    }
}

void MilbrandtScheme::invalidate_transient_state_after_domain_recycling() noexcept
{
    if (internal_triple_)
    {
        internal_triple_->poisoned = true;
        internal_triple_->candidate_valid = false;
        internal_triple_->prepared = false;
        return;
    }
    Field3D* const transient_fields[] = {
        &dNc_dt_, &dNr_dt_, &dNi_dt_, &dNs_dt_, &dNg_dt_, &dNh_dt_,
        &surface_sedimentation_sink_rate_};
    for (Field3D* field : transient_fields)
        if (!field->empty()) field->fill(0.0f);
}

PortableSchemeState MilbrandtScheme::portable_state() const
{
    if (internal_triple_)
    {
        throw std::logic_error(
            "internal Milbrandt triple sedimentation has no portable checkpoint state");
    }
    PortableSchemeState state;
    state.version = 4;
    state.fields = {
        Nc_, Nr_, Ni_, Ns_, Ng_, Nh_,
        dNc_dt_, dNr_dt_, dNi_dt_, dNs_dt_, dNg_dt_, dNh_dt_,
        surface_sedimentation_sink_rate_
    };
    state.integers = {hail_processes_ ? 1ULL : 0ULL};
    state.reals = {
        alpha_r_, alpha_i_, alpha_s_, alpha_g_, alpha_h_,
        c_r_, d_r_, c_i_, d_i_, c_s_, d_s_, c_g_, d_g_, c_h_, d_h_,
        a_r_, b_r_, a_i_, b_i_, a_s_, b_s_, a_g_, b_g_, a_h_, b_h_
    };
    return state;
}

PortableSchemeState MilbrandtScheme::portable_parameter_state() const
{
    if (internal_triple_)
    {
        throw std::logic_error(
            "internal Milbrandt triple sedimentation has no portable parameter state");
    }
    PortableSchemeState state;
    state.version = 4;
    state.integers = {hail_processes_ ? 1ULL : 0ULL};
    state.reals = {
        alpha_r_, alpha_i_, alpha_s_, alpha_g_, alpha_h_,
        c_r_, d_r_, c_i_, d_i_, c_s_, d_s_, c_g_, d_g_, c_h_, d_h_,
        a_r_, b_r_, a_i_, b_i_, a_s_, b_s_, a_g_, b_g_, a_h_, b_h_};
    return state;
}

MicrophysicsParameterIdentity
MilbrandtScheme::parameter_identity() const noexcept
{
    MicrophysicsParameterIdentity identity;
    identity.version = internal_triple_ ? 5 : 4;
    identity.integer_count = 1;
    identity.integers[0] = hail_processes_ ? 1ULL : 0ULL;
    identity.real_count = 25;
    identity.reals = {
        alpha_r_, alpha_i_, alpha_s_, alpha_g_, alpha_h_,
        c_r_, d_r_, c_i_, d_i_, c_s_, d_s_, c_g_, d_g_, c_h_, d_h_,
        a_r_, b_r_, a_i_, b_i_, a_s_, b_s_, a_g_, b_g_, a_h_, b_h_};
    return identity;
}

bool MilbrandtScheme::restore_portable_state(const PortableSchemeState& state,
                                              std::string& error)
{
    if (internal_triple_)
    {
        error = "internal Milbrandt triple sedimentation cannot restore portable state";
        return false;
    }
    const bool legacy_v1 = state.version == 1 && state.reals.size() == 27
        && state.fields.size() == 21 && state.integers.size() == 2;
    const bool legacy_v2 = state.version == 2 && state.reals.size() == 28
        && state.fields.size() == 21 && state.integers.size() == 2;
    const bool legacy_v3 = state.version == 3 && state.reals.size() == 28
        && state.fields.size() == 23 && state.integers.size() == 2;
    const bool current = state.version == 4 && state.reals.size() == 25
        && state.fields.size() == 13 && state.integers.size() == 1;
    if (!legacy_v1 && !legacy_v2 && !legacy_v3 && !current)
    {
        error = "invalid Milbrandt portable-state layout";
        return false;
    }
    const std::uint64_t hail_flag =
        current ? state.integers[0] : state.integers[1];
    const std::uint64_t obsolete_triple_flag =
        current ? 0ULL : state.integers[0];
    const std::size_t first_physical_constant =
        current ? 0 : (legacy_v1 ? 2 : 3);
    const auto valid_physical_constants =
        [&]()
        {
            if (state.reals.size() < first_physical_constant + 25)
                return false;
            const double* value =
                state.reals.data() + first_physical_constant;
            for (std::size_t n = 0; n < 25; ++n)
                if (!std::isfinite(value[n])) return false;
            for (std::size_t n = 0; n < 5; ++n)
                if (value[n] < 0.0 || value[n] > 10.0) return false;
            for (std::size_t n = 5; n < 15; n += 2)
                if (!(value[n] > 0.0) || !(value[n + 1] > 0.0))
                    return false;
            for (std::size_t n = 15; n < 25; n += 2)
                if (!(value[n] > 0.0) || value[n + 1] < 0.0
                    || value[n + 1] > 2.0)
                    return false;
            return true;
        };
    if (hail_flag > 1 || obsolete_triple_flag > 1
        || obsolete_triple_flag != 0
        || !valid_physical_constants())
    {
        error = "invalid Milbrandt portable-state scalar";
        return false;
    }

    const Field3D& reference = state.fields.front();
    const bool initialized = !reference.empty();
    if ((legacy_v1 || legacy_v2) && initialized)
    {
        error = "initialized experimental Milbrandt v1/v2 state used "
                "volumetric number units and cannot be restored into the "
                "production dry-air-mixing-ratio state";
        return false;
    }

    const std::size_t number_count =
        (current || legacy_v3) ? 6 : 5;
    const std::size_t first_number_tendency =
        current ? 6 : ((legacy_v3) ? 11 : 10);
    const std::size_t sink_index =
        current ? 12 : ((legacy_v3) ? 22 : 20);
    for (std::size_t n = 0; n < number_count; ++n)
    {
        const Field3D& moment = state.fields[n];
        const Field3D& tendency =
            state.fields[first_number_tendency + n];
        if (moment.empty() == initialized
            || tendency.empty() == initialized)
        {
            error = "incomplete Milbrandt number-moment history";
            return false;
        }
        if (initialized
            && (moment.size_r() != reference.size_r()
                || moment.size_th() != reference.size_th()
                || moment.size_z() != reference.size_z()
                || tendency.size_r() != reference.size_r()
                || tendency.size_th() != reference.size_th()
                || tendency.size_z() != reference.size_z()))
        {
            error =
                "inconsistent Milbrandt portable-state field dimensions";
            return false;
        }
    }
    if (legacy_v3)
    {
        // v3 carried ten unused reflectivity-moment placeholders. A
        // double-moment checkpoint must not contain hidden active state there.
        for (std::size_t index = 6; index < 11; ++index)
            if (!state.fields[index].empty())
            {
                error = "obsolete Milbrandt triple-moment field is nonempty";
                return false;
            }
        for (std::size_t index = 17; index < 22; ++index)
            if (!state.fields[index].empty())
            {
                error =
                    "obsolete Milbrandt triple-moment tendency is nonempty";
                return false;
            }
    }
    const Field3D& sink = state.fields[sink_index];
    if (!sink.empty() && (!initialized || sink.size_r() != reference.size_r() ||
                          sink.size_th() != reference.size_th() || sink.size_z() != 1))
    {
        error = "invalid Milbrandt surface-sink dimensions";
        return false;
    }

    if (current || legacy_v3)
    {
        Nc_ = state.fields[0];
    }
    else
    {
        Nc_ = {};
    }
    const std::size_t first_legacy_number =
        (current || legacy_v3) ? 1 : 0;
    Nr_ = state.fields[first_legacy_number + 0];
    Ni_ = state.fields[first_legacy_number + 1];
    Ns_ = state.fields[first_legacy_number + 2];
    Ng_ = state.fields[first_legacy_number + 3];
    Nh_ = state.fields[first_legacy_number + 4];
    if (current || legacy_v3)
    {
        dNc_dt_ = state.fields[first_number_tendency];
    }
    else
    {
        dNc_dt_ = {};
    }
    const std::size_t first_legacy_tendency =
        first_number_tendency
        + ((current || legacy_v3) ? 1 : 0);
    dNr_dt_ = state.fields[first_legacy_tendency + 0];
    dNi_dt_ = state.fields[first_legacy_tendency + 1];
    dNs_dt_ = state.fields[first_legacy_tendency + 2];
    dNg_dt_ = state.fields[first_legacy_tendency + 3];
    dNh_dt_ = state.fields[first_legacy_tendency + 4];
    surface_sedimentation_sink_rate_ = sink;

    hail_processes_ = hail_flag != 0;
    std::size_t real = first_physical_constant;
    alpha_r_ = state.reals[real++]; alpha_i_ = state.reals[real++];
    alpha_s_ = state.reals[real++]; alpha_g_ = state.reals[real++]; alpha_h_ = state.reals[real++];
    c_r_ = state.reals[real++]; d_r_ = state.reals[real++];
    c_i_ = state.reals[real++]; d_i_ = state.reals[real++];
    c_s_ = state.reals[real++]; d_s_ = state.reals[real++];
    c_g_ = state.reals[real++]; d_g_ = state.reals[real++];
    c_h_ = state.reals[real++]; d_h_ = state.reals[real++];
    a_r_ = state.reals[real++]; b_r_ = state.reals[real++];
    a_i_ = state.reals[real++]; b_i_ = state.reals[real++];
    a_s_ = state.reals[real++]; b_s_ = state.reals[real++];
    a_g_ = state.reals[real++]; b_g_ = state.reals[real++];
    a_h_ = state.reals[real++]; b_h_ = state.reals[real];
    error.clear();
    return true;
}

/**
 * @brief Computes the tendencies for the Milbrandt scheme.
 */
void MilbrandtScheme::compute_tendencies(
    const Field3D& p,
    const Field3D& theta,
    const Field3D& rho_d,
    const Field3D& qv,
    const Field3D& qc,
    const Field3D& qr,
    const Field3D& qi,
    const Field3D& qs,
    const Field3D& qg,
    const Field3D& qh,
    double dt,
    Field3D& dtheta_dt,
    Field3D& dqv_dt,
    Field3D& dqc_dt,
    Field3D& dqr_dt,
    Field3D& dqi_dt,
    Field3D& dqs_dt,
    Field3D& dqg_dt,
    Field3D& dqh_dt
) 

{
    if (internal_triple_)
    {
        if (internal_triple_->mode == InternalTripleMode::FrozenCompositionOnly ||
            internal_triple_->mode == InternalTripleMode::MeltingOnly ||
            internal_triple_->mode == InternalTripleMode::IceMeltingOnly ||
            internal_triple_->mode == InternalTripleMode::SnowMeltingOnly ||
            internal_triple_->mode == InternalTripleMode::WarmPrimaryOnly ||
            internal_triple_->mode == InternalTripleMode::CloudAdjustmentOnly ||
            internal_triple_->mode == InternalTripleMode::WarmCloudPrimaryOnly ||
            internal_triple_->mode == InternalTripleMode::DynamicWarmCloudPrimaryOnly ||
            internal_triple_->mode == InternalTripleMode::SubcycledWarmCloudPrimaryOnly ||
            internal_triple_->mode == InternalTripleMode::FrozenCollectionPrimaryOnly ||
            internal_triple_->mode == InternalTripleMode::ColdCloudCollectionPrimaryOnly ||
            internal_triple_->mode == InternalTripleMode::ColdCloudDiffusionPrimaryOnly ||
            is_cold_riming_mode(internal_triple_->mode))
        {
            internal_triple_->candidate_valid = false;
            internal_triple_->prepared = false;
            internal_triple_->primary_endpoints.owner = nullptr;
        }
        if (internal_triple_->poisoned ||
            require_native_dispatch() || native_execution_selected())
        {
            throw std::runtime_error(
                "internal Milbrandt triple mode requires its live CPU direct-step owner");
        }
        if (internal_triple_->mode == InternalTripleMode::FrozenCompositionOnly ||
            internal_triple_->mode == InternalTripleMode::MeltingOnly ||
            internal_triple_->mode == InternalTripleMode::IceMeltingOnly ||
            internal_triple_->mode == InternalTripleMode::SnowMeltingOnly ||
            internal_triple_->mode == InternalTripleMode::WarmPrimaryOnly ||
            internal_triple_->mode == InternalTripleMode::CloudAdjustmentOnly ||
            internal_triple_->mode == InternalTripleMode::WarmCloudPrimaryOnly ||
            internal_triple_->mode == InternalTripleMode::DynamicWarmCloudPrimaryOnly ||
            internal_triple_->mode == InternalTripleMode::SubcycledWarmCloudPrimaryOnly ||
            internal_triple_->mode == InternalTripleMode::FrozenCollectionPrimaryOnly ||
            internal_triple_->mode == InternalTripleMode::ColdCloudCollectionPrimaryOnly ||
            internal_triple_->mode == InternalTripleMode::ColdCloudDiffusionPrimaryOnly ||
            is_cold_riming_mode(internal_triple_->mode))
        {
            compute_internal_triple_primary_tendencies(
                p, theta, rho_d, qv, qc,
                {&qr, &qi, &qs, &qg, &qh}, dt,
                {&dtheta_dt, &dqv_dt, &dqc_dt, &dqr_dt,
                 &dqi_dt, &dqs_dt, &dqg_dt, &dqh_dt});
        }
        else if (internal_triple_->mode == InternalTripleMode::FrozenDiffusionOnly)
        {
            compute_internal_triple_frozen_diffusion_tendencies(
                p, theta, rho_d, qv, qc,
                {&qr, &qi, &qs, &qg, &qh}, dt,
                {&dtheta_dt, &dqv_dt, &dqc_dt, &dqr_dt,
                 &dqi_dt, &dqs_dt, &dqg_dt, &dqh_dt});
        }
        else if (internal_triple_->mode == InternalTripleMode::RainFreezingOnly)
        {
            compute_internal_triple_rain_freezing_tendencies(
                p, theta, rho_d, qv, qc,
                {&qr, &qi, &qs, &qg, &qh}, dt,
                {&dtheta_dt, &dqv_dt, &dqc_dt, &dqr_dt,
                 &dqi_dt, &dqs_dt, &dqg_dt, &dqh_dt});
        }
        else if (internal_triple_->mode !=
            InternalTripleMode::SedimentationOnly)
        {
            compute_internal_triple_warm_tendencies(
                p, theta, rho_d, qv, qc,
                {&qr, &qi, &qs, &qg, &qh}, dt,
                {&dtheta_dt, &dqv_dt, &dqc_dt, &dqr_dt,
                 &dqi_dt, &dqs_dt, &dqg_dt, &dqh_dt});
        }
        else
        {
            compute_internal_triple_sedimentation_tendencies(
                rho_d, {&qr, &qi, &qs, &qg, &qh}, dt,
                {&dqr_dt, &dqi_dt, &dqs_dt, &dqg_dt, &dqh_dt},
                dtheta_dt, dqv_dt, dqc_dt);
        }
        return;
    }
    if (require_native_dispatch() || native_execution_selected())
    {
        compute_native_tendencies({&theta, &p, &rho_d, &qv, &qc, &qr, &qi, &qs, &qg, &qh},
                                  dt,
                                  {&dtheta_dt, &dqv_dt, &dqc_dt, &dqr_dt, &dqi_dt, &dqs_dt, &dqg_dt, &dqh_dt});
        return;
    }
    if (!std::isfinite(dt) || dt <= 0.0)
    {
        throw std::invalid_argument(
            "Milbrandt microphysics timestep must be positive and finite");
    }
    int NR = p.size_r();
    if (NR == 0) return;
    int NTH = p.size_th();
    int NZ = p.size_z();

    prepare_prognostic_state(rho_d, qc, qr, qi, qs, qg, qh);

    auto& pool = FieldPool::instance();
    auto temperature_guard = pool.scoped_acquire_for_overwrite(NR, NTH, NZ);
    Field3D& temperature = temperature_guard.field;
    thermodynamics::convert_theta_to_temperature_field(theta, p, temperature);

    init_tendency_fields(NR, NTH, NZ, dtheta_dt, dqv_dt, dqc_dt, dqr_dt,
                         dqi_dt, dqs_dt, dqg_dt, dqh_dt);

    // Saturation adjustment: convert supersaturated vapor to cloud water
    // before processing microphysics. Without this, Milbrandt-Yau only
    // acts on existing hydrometeors and never initiates condensation.
    auto qv_adj_guard = pool.scoped_acquire_copy(qv);
    auto qc_adj_guard = pool.scoped_acquire_copy(qc);
    Field3D& qv_adj = qv_adj_guard.field;
    Field3D& qc_adj = qc_adj_guard.field;
    const PhysicalCellDomain domain = physical_cell_domain(NR, NTH, NZ);
    #pragma omp parallel for collapse(2)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
    {
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double T_old = static_cast<double>(temperature[i][j][k]);
                double T_val = T_old;
                double p_val = static_cast<double>(p[i][j][k]);
                const double qv_old = static_cast<double>(qv[i][j][k]);
                const double qc_old = static_cast<double>(qc[i][j][k]);
                double qv_val = qv_old;
                double qc_val = qc_old;
                T_val = thermodynamics::saturation_adjustment(
                    T_val, p_val, qv_val, qc_val);
                temperature[i][j][k] = static_cast<float>(T_val);
                qv_adj[i][j][k] = static_cast<float>(qv_val);
                qc_adj[i][j][k] = static_cast<float>(qc_val);
                if (dt > 0.0)
                {
                    dqv_dt[i][j][k] += static_cast<float>((qv_val - qv_old) / dt);
                    dqc_dt[i][j][k] += static_cast<float>((qc_val - qc_old) / dt);
                    dtheta_dt[i][j][k] += static_cast<float>((T_val - T_old) / dt);
                }
            }
        }
    }

    dNc_dt_.resize(NR, NTH, NZ, 0.0f);
    dNr_dt_.resize(NR, NTH, NZ, 0.0f);
    dNi_dt_.resize(NR, NTH, NZ, 0.0f);
    dNs_dt_.resize(NR, NTH, NZ, 0.0f);
    dNg_dt_.resize(NR, NTH, NZ, 0.0f);
    dNh_dt_.resize(NR, NTH, NZ, 0.0f);

    // Saturation adjustment also changes the cloud-droplet population.
    // Newly condensed cloud activates the qualified continental background;
    // partial evaporation removes droplets in proportion to cloud mass.
    const auto saturation_adjusted_cloud_number =
        [&](int i, int j, int k)
        {
            const double qc_old = qc[i][j][k];
            const double qc_new = qc_adj[i][j][k];
            const double nc_old = Nc_[i][j][k];
            if (qc_new > qc_old)
            {
                const double activated_m3 = milbrandt_detail::continental_activated_ccn(
                    vertical_velocity_at_scalar(i, j, k), temperature[i][j][k], p[i][j][k]);
                return std::max(
                    nc_old,
                    activated_m3
                        / static_cast<double>(rho_d[i][j][k]));
            }
            if (qc_new < qc_old)
            {
                return qc_new > 0.0
                    ? nc_old * qc_new / std::max(qc_old, 1.0e-30)
                    : 0.0;
            }
            return nc_old;
        };
    #pragma omp parallel for collapse(2)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double nc_old = Nc_[i][j][k];
                const double nc_new =
                    saturation_adjusted_cloud_number(i, j, k);
                dNc_dt_[i][j][k] =
                    static_cast<float>((nc_new - nc_old) / dt);
            }

    compute_warm_rain_processes(temperature, p, rho_d, qv_adj, qc_adj, qr,
                               dqc_dt, dqr_dt, dqv_dt, dtheta_dt,
                               dNc_dt_, dNr_dt_, dt);

    // The operational process order is warm liquid physics followed by the
    // mixed-phase source/sink set. Publish the admitted warm candidate into
    // scratch state before diagnosing any cold rate so a scarce liquid donor
    // cannot be spent independently by both families.
    auto qr_process_guard = pool.scoped_acquire_copy(qr);
    Field3D& qr_process = qr_process_guard.field;
    #pragma omp parallel for collapse(2)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                qv_adj[i][j][k] = static_cast<float>(
                    std::max(
                        0.0,
                        static_cast<double>(qv[i][j][k])
                            + dt * static_cast<double>(
                                dqv_dt[i][j][k])));
                qc_adj[i][j][k] = static_cast<float>(
                    std::max(
                        0.0,
                        static_cast<double>(qc[i][j][k])
                            + dt * static_cast<double>(
                                dqc_dt[i][j][k])));
                qr_process[i][j][k] = static_cast<float>(
                    std::max(
                        0.0,
                        static_cast<double>(qr[i][j][k])
                            + dt * static_cast<double>(
                                dqr_dt[i][j][k])));
                const double original_temperature =
                    thermodynamics::theta_to_temperature(
                        theta[i][j][k], p[i][j][k]);
                temperature[i][j][k] = static_cast<float>(
                    original_temperature
                    + dt * static_cast<double>(
                        dtheta_dt[i][j][k]));
            }

    compute_ice_processes(
        temperature, p, rho_d, qv_adj, qc_adj, qr_process, qi, qs, qg, qh,
        dqc_dt, dqr_dt, dqi_dt, dqs_dt, dqg_dt, dqh_dt,
        dqv_dt, dtheta_dt, dNc_dt_, dNr_dt_, dNi_dt_, dNs_dt_,
        dNg_dt_, dNh_dt_, dt);

    compute_melting_processes(
        temperature, p, rho_d, qv_adj, qi, qs, qg, qh,
        dqi_dt, dqs_dt, dqg_dt, dqh_dt, dqr_dt, dtheta_dt,
        dNr_dt_, dNi_dt_, dNs_dt_, dNg_dt_, dNh_dt_, dt);

    const std::array<const Field3D*, 6> number_state{
        &Nc_, &Nr_, &Ni_, &Ns_, &Ng_, &Nh_};
    const std::array<Field3D*, 6> number_rate{
        &dNc_dt_, &dNr_dt_, &dNi_dt_, &dNs_dt_, &dNg_dt_, &dNh_dt_};

    // Sedimentation starts from the admitted point-process state and replaces
    // each precipitating species' tendency with a conservative, CFL-subcycled
    // column result.
    compute_sedimentation(rho_d, qr, qi, qs, qg, qh,
                         Nr_, Ni_, Ns_, Ng_, Nh_,
                         dqr_dt, dqi_dt, dqs_dt, dqg_dt, dqh_dt,
                         dNr_dt_, dNi_dt_, dNs_dt_, dNg_dt_, dNh_dt_, dt);

    std::size_t invalid_cloud_storage = std::numeric_limits<std::size_t>::max();
#pragma omp         parallel for collapse(2) reduction(min : invalid_cloud_storage)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                std::array<float, 4> rates{dqc_dt[i][j][k], dqv_dt[i][j][k], dtheta_dt[i][j][k], dNc_dt_[i][j][k]};
                if (!milbrandt_detail::finalize_cloud_storage(
                        qc[i][j][k], qc_adj[i][j][k], qv[i][j][k], rho_d[i][j][k], Nc_[i][j][k], dt, rates))
                    invalid_cloud_storage = std::min(invalid_cloud_storage, (std::size_t(i) * NTH + j) * NZ + k);
                else
                {
                    dqc_dt[i][j][k]    = rates[0];
                    dqv_dt[i][j][k]    = rates[1];
                    dtheta_dt[i][j][k] = rates[2];
                    dNc_dt_[i][j][k]   = rates[3];
                }
            }
    if (invalid_cloud_storage != std::numeric_limits<std::size_t>::max())
    {
        const int          i = invalid_cloud_storage / (std::size_t(NTH) * NZ), j = invalid_cloud_storage / NZ % NTH,
                           k = invalid_cloud_storage % NZ;
        std::ostringstream message;
        message << std::setprecision(17) << "Milbrandt cloud tendency exceeds the process storage envelope"
                << ": cell=" << i << ',' << j << ',' << k << ", dt=" << dt << ", cloud=" << qc(i, j, k)
                << ", warm_cloud=" << qc_adj(i, j, k) << ", cloud_rate=" << dqc_dt(i, j, k)
                << ", number=" << Nc_(i, j, k) << ", number_rate=" << dNc_dt_(i, j, k) << ", vapor=" << qv(i, j, k)
                << ", vapor_rate=" << dqv_dt(i, j, k);
        throw std::runtime_error(message.str());
    }

    // Number moments remain untouched until the runtime accepts the complete
    // primary-state update. Limit each collision/sedimentation sink to the
    // locally available concentration so acceptance never relies on a clamp.
    for (std::size_t species = 0; species < number_state.size(); ++species)
    {
        #pragma omp parallel for collapse(2)
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
                    const double minimum_rate = -static_cast<double>(
                        (*number_state[species])[i][j][k]) / dt;
                    (*number_rate[species])[i][j][k] = static_cast<float>(
                        std::max(minimum_rate, static_cast<double>(
                            (*number_rate[species])[i][j][k])));
                }
    }

    #pragma omp parallel for collapse(2)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
    {
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                dtheta_dt[i][j][k] = static_cast<float>(
                    thermodynamics::temperature_tendency_to_theta(
                        static_cast<float>(dtheta_dt[i][j][k]), static_cast<float>(p[i][j][k])
                    )
                );
            }
        }
    }
}

/**
 * @brief Initializes the prognostic number concentrations for the Milbrandt scheme.
 */
void MilbrandtScheme::initialize_prognostic_fields(
    const Field3D& rho_d,
    const Field3D& qc, const Field3D& qr,
    const Field3D& qi, const Field3D& qs,
    const Field3D& qg, const Field3D& qh)
{
    const int NR = rho_d.size_r();
    const int NTH = rho_d.size_th();
    const int NZ = rho_d.size_z();
    Nc_.resize(NR, NTH, NZ, 0.0f);
    Nr_.resize(NR, NTH, NZ, 0.0f);
    Ni_.resize(NR, NTH, NZ, 0.0f);
    Ns_.resize(NR, NTH, NZ, 0.0f);
    Ng_.resize(NR, NTH, NZ, 0.0f);
    Nh_.resize(NR, NTH, NZ, 0.0f);
    for (Field3D* tendency : {&dNc_dt_, &dNr_dt_, &dNi_dt_, &dNs_dt_, &dNg_dt_, &dNh_dt_})
    {
        tendency->resize(NR, NTH, NZ, 0.0f);
    }
    surface_sedimentation_sink_rate_.resize(NR, NTH, 1, 0.0f);

    const auto seed_number = [](double q, double c,
                                double d, double diameter)
    {
        if (!(q > 0.0)) return 0.0;
        return q / (c * std::pow(diameter, d));
    };
    const PhysicalCellDomain domain = physical_cell_domain(NR, NTH, NZ);
    #pragma omp parallel for collapse(2)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double density = rho_d[i][j][k];
                Nc_[i][j][k] = qc[i][j][k] > 0.0f
                    ? static_cast<float>(2.0e8 / density) : 0.0f;
                Nr_[i][j][k] = static_cast<float>(seed_number(
                    qr[i][j][k], c_r_, d_r_, 5.0e-4));
                Ni_[i][j][k] = static_cast<float>(seed_number(
                    qi[i][j][k], c_i_, d_i_, 1.0e-4));
                Ns_[i][j][k] = static_cast<float>(seed_number(
                    qs[i][j][k], c_s_, d_s_, 1.0e-3));
                Ng_[i][j][k] = static_cast<float>(seed_number(
                    qg[i][j][k], c_g_, d_g_, 1.0e-3));
                Nh_[i][j][k] = static_cast<float>(seed_number(
                    qh[i][j][k], c_h_, d_h_, 5.0e-3));
            }

}

/**
 * @brief Computes warm-rain source and sink tendencies.
 */
void MilbrandtScheme::compute_warm_rain_processes(
    const Field3D& temperature,
    const Field3D& p,
    const Field3D& rho,
    const Field3D& qv,
    const Field3D& qc,
    const Field3D& qr,
    Field3D& dqc_dt,
    Field3D& dqr_dt,
    Field3D& dqv_dt,
    Field3D& dtheta_dt,
    Field3D& dNc_dt,
    Field3D& dNr_dt,
    double dt
) 
{
    int NR = temperature.size_r();
    int NTH = temperature.size_th();
    int NZ = temperature.size_z();
    const PhysicalCellDomain domain = physical_cell_domain(NR, NTH, NZ);

    const auto parameters = milbrandt_detail::warm_rain_parameters(alpha_r_, c_r_, d_r_, a_r_, b_r_);
#pragma omp parallel for collapse(2)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const milbrandt_detail::WarmRainCell cell{temperature(i, j, k),
                                                          p(i, j, k),
                                                          rho(i, j, k),
                                                          qv(i, j, k),
                                                          qc(i, j, k),
                                                          qr(i, j, k),
                                                          Nc_(i, j, k),
                                                          Nr_(i, j, k)};
                milbrandt_detail::WarmRainTendencies rates{dqc_dt(i, j, k),
                                                           dqr_dt(i, j, k),
                                                           dqv_dt(i, j, k),
                                                           dtheta_dt(i, j, k),
                                                           dNc_dt(i, j, k),
                                                           dNr_dt(i, j, k)};
                milbrandt_detail::warm_rain_cell(parameters, cell, dt, rates);
                dqc_dt(i, j, k)    = rates.cloud;
                dqr_dt(i, j, k)    = rates.rain;
                dqv_dt(i, j, k)    = rates.vapor;
                dtheta_dt(i, j, k) = rates.temperature;
                dNc_dt(i, j, k)    = rates.cloud_number;
                dNr_dt(i, j, k)    = rates.rain_number;
            }
}

/**
 * @brief Computes the ice processes for the Milbrandt scheme.
 */
void MilbrandtScheme::compute_ice_processes(
    const Field3D& temperature,
    const Field3D& p,
    const Field3D& rho,
    const Field3D& qv,
    const Field3D& qc,
    const Field3D& qr,
    const Field3D& qi,
    const Field3D& qs,
    const Field3D& qg,
    const Field3D& qh,
    Field3D& dqc_dt,
    Field3D& dqr_dt,
    Field3D& dqi_dt,
    Field3D& dqs_dt,
    Field3D& dqg_dt,
    Field3D& dqh_dt,
    Field3D& dqv_dt,
    Field3D& dtheta_dt,
    Field3D& dNc_dt,
    Field3D& dNr_dt,
    Field3D& dNi_dt,
    Field3D& dNs_dt,
    Field3D& dNg_dt,
    Field3D& dNh_dt,
    double dt
) 
{
    const int NR = temperature.size_r();
    const int NTH = temperature.size_th();
    const int NZ = temperature.size_z();
    const PhysicalCellDomain domain = physical_cell_domain(NR, NTH, NZ);

    const milbrandt_detail::ColdParameters parameters{
        a_g_, a_h_, a_i_, a_r_, a_s_, alpha_g_, alpha_h_, alpha_i_, alpha_r_, alpha_s_, b_g_, b_h_, b_i_,
        b_r_, b_s_, c_g_, c_h_, c_i_, c_r_,     c_s_,     d_g_,     d_h_,     d_i_,     d_r_, d_s_, hail_processes_};

    std::exception_ptr failure;
    std::size_t first_failure = std::numeric_limits<std::size_t>::max();
#pragma omp parallel for collapse(2)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const std::array<double, 16> cell{temperature[i][j][k],
                                                  p[i][j][k],
                                                  rho[i][j][k],
                                                  qv[i][j][k],
                                                  qc[i][j][k],
                                                  qr[i][j][k],
                                                  qi[i][j][k],
                                                  qs[i][j][k],
                                                  qg[i][j][k],
                                                  qh[i][j][k],
                                                  Nc_[i][j][k],
                                                  Nr_[i][j][k],
                                                  Ni_[i][j][k],
                                                  Ns_[i][j][k],
                                                  Ng_[i][j][k],
                                                  Nh_[i][j][k]};
                std::array<float, 14>        rates{dqc_dt[i][j][k],
                                                   dqr_dt[i][j][k],
                                                   dqi_dt[i][j][k],
                                                   dqs_dt[i][j][k],
                                                   dqg_dt[i][j][k],
                                                   dqh_dt[i][j][k],
                                                   dqv_dt[i][j][k],
                                                   dtheta_dt[i][j][k],
                                                   dNc_dt[i][j][k],
                                                   dNr_dt[i][j][k],
                                                   dNi_dt[i][j][k],
                                                   dNs_dt[i][j][k],
                                                   dNg_dt[i][j][k],
                                                   dNh_dt[i][j][k]};
                try
                {
                    milbrandt_detail::cold_cell(parameters, cell, dt, rates);
                }
                catch (...)
                {
                    const std::size_t index = (static_cast<std::size_t>(i) * NTH + j) * NZ + k;
#pragma omp critical(nimbus_milbrandt_cold_failure)
                    {
                        if (index < first_failure)
                        {
                            first_failure = index;
                            failure = std::current_exception();
                        }
                    }
                    continue;
                }
                dqc_dt[i][j][k]    = rates[0];
                dqr_dt[i][j][k]    = rates[1];
                dqi_dt[i][j][k]    = rates[2];
                dqs_dt[i][j][k]    = rates[3];
                dqg_dt[i][j][k]    = rates[4];
                dqh_dt[i][j][k]    = rates[5];
                dqv_dt[i][j][k]    = rates[6];
                dtheta_dt[i][j][k] = rates[7];
                dNc_dt[i][j][k]    = rates[8];
                dNr_dt[i][j][k]    = rates[9];
                dNi_dt[i][j][k]    = rates[10];
                dNs_dt[i][j][k]    = rates[11];
                dNg_dt[i][j][k]    = rates[12];
                dNh_dt[i][j][k]    = rates[13];
            }
    if (failure)
    {
        std::rethrow_exception(failure);
    }
}

/**
 * @brief Computes melting exchanges between frozen and liquid hydrometeors.
 */
void MilbrandtScheme::compute_melting_processes(
    const Field3D& temperature,
    const Field3D& p,
    const Field3D& rho,
    const Field3D& qv,
    const Field3D& qi,
    const Field3D& qs,
    const Field3D& qg,
    const Field3D& qh,
    Field3D& dqi_dt,
    Field3D& dqs_dt,
    Field3D& dqg_dt,
    Field3D& dqh_dt,
    Field3D& dqr_dt,
    Field3D& dtheta_dt,
    Field3D& dNr_dt,
    Field3D& dNi_dt,
    Field3D& dNs_dt,
    Field3D& dNg_dt,
    Field3D& dNh_dt,
    double dt
) 
{
    const int NR = temperature.size_r();
    const int NTH = temperature.size_th();
    const int NZ = temperature.size_z();
    const PhysicalCellDomain domain = physical_cell_domain(NR, NTH, NZ);

    const std::array<double, 13> parameters{
        alpha_s_, a_s_, b_s_, alpha_g_, c_g_, d_g_, a_g_, b_g_, alpha_h_, c_h_, d_h_, a_h_, b_h_};

#pragma omp parallel for collapse(2)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const std::array<double, 12> cell{temperature[i][j][k],
                                                  p[i][j][k],
                                                  rho[i][j][k],
                                                  qv[i][j][k],
                                                  qi[i][j][k],
                                                  qs[i][j][k],
                                                  qg[i][j][k],
                                                  qh[i][j][k],
                                                  Ni_[i][j][k],
                                                  Ns_[i][j][k],
                                                  Ng_[i][j][k],
                                                  Nh_[i][j][k]};
                std::array<float, 11>        rates{dqi_dt[i][j][k],
                                                   dqs_dt[i][j][k],
                                                   dqg_dt[i][j][k],
                                                   dqh_dt[i][j][k],
                                                   dqr_dt[i][j][k],
                                                   dtheta_dt[i][j][k],
                                                   dNr_dt[i][j][k],
                                                   dNi_dt[i][j][k],
                                                   dNs_dt[i][j][k],
                                                   dNg_dt[i][j][k],
                                                   dNh_dt[i][j][k]};
                milbrandt_detail::melting_cell(parameters, cell, dt, rates);
                dqi_dt[i][j][k]    = rates[0];
                dqs_dt[i][j][k]    = rates[1];
                dqg_dt[i][j][k]    = rates[2];
                dqh_dt[i][j][k]    = rates[3];
                dqr_dt[i][j][k]    = rates[4];
                dtheta_dt[i][j][k] = rates[5];
                dNr_dt[i][j][k]    = rates[6];
                dNi_dt[i][j][k]    = rates[7];
                dNs_dt[i][j][k]    = rates[8];
                dNg_dt[i][j][k]    = rates[9];
                dNh_dt[i][j][k]    = rates[10];
            }
}

/**
 * @brief Computes the sedimentation for the Milbrandt scheme.
 */
void MilbrandtScheme::compute_sedimentation(
    const Field3D& rho_d,
    const Field3D& qr,
    const Field3D& qi,
    const Field3D& qs,
    const Field3D& qg,
    const Field3D& qh,
    const Field3D& Nr,
    const Field3D& Ni,
    const Field3D& Ns,
    const Field3D& Ng,
    const Field3D& Nh,
    Field3D& dqr_dt,
    Field3D& dqi_dt,
    Field3D& dqs_dt,
    Field3D& dqg_dt,
    Field3D& dqh_dt,
    Field3D& dNr_dt,
    Field3D& dNi_dt,
    Field3D& dNs_dt,
    Field3D& dNg_dt,
    Field3D& dNh_dt,
    double dt
) 
{
    int NR = qr.size_r();
    int NTH = qr.size_th();
    int NZ = qr.size_z();
    const PhysicalCellDomain domain = physical_cell_domain(NR, NTH, NZ);
    surface_sedimentation_sink_rate_.resize(NR, NTH, 1, 0.0f);

    const int levels = domain.k_end - domain.k_begin;
    const std::size_t physical_column_count =
        static_cast<std::size_t>(domain.nx()) * domain.ny();
    const bool parallel_columns = physical_column_count >= 64;
    int worker_count = 1;
#ifdef _OPENMP
    if (parallel_columns)
        worker_count = std::max(
            1, std::min(
                omp_get_max_threads(),
                static_cast<int>(physical_column_count)));
#endif
#ifdef NIMBUS_PHYSICS_LEGACY_ALLOCATION_EXPERIMENT
    // Measurement-only reconstruction of the pre-SCI-124 lifetime. These
    // workspaces are replaceable and omitted from all committed/checkpoint
    // state, so discarding them cannot alter the numerical trajectory.
    std::vector<MilbrandtSedimentationWorkspace>().swap(
        sedimentation_workspaces_);
#endif
    sedimentation_workspaces_.resize(
        static_cast<std::size_t>(worker_count));
    for (MilbrandtSedimentationWorkspace& workspace :
         sedimentation_workspaces_)
    {
        workspace.resize(static_cast<std::size_t>(levels));
    }

    int sedimentation_failed = 0;
    #pragma omp parallel if(parallel_columns) num_threads(worker_count) reduction(|:sedimentation_failed)
    {
        int worker_index = 0;
#ifdef _OPENMP
        worker_index = omp_get_thread_num();
#endif
        // Five hydrometeor categories and every assigned column share the
        // worker's retained arrays serially. After warm-up, sedimentation
        // performs no heap allocation unless NZ or required workers change.
        MilbrandtSedimentationWorkspace& workspace =
            sedimentation_workspaces_[
                static_cast<std::size_t>(worker_index)];

        #pragma omp for collapse(2)
        for (int i = domain.i_begin; i < domain.i_end; ++i)
        {
          for (int j = domain.j_begin; j < domain.j_end; ++j)
          {
              for (int level = 0; level < levels; ++level)
                  workspace.thickness[static_cast<std::size_t>(level)] =
                      vertical_spacing_m(i, j, domain.k_begin + level);
              const auto sediment_species = [&](const Field3D& q_field,
                                                const Field3D& n_field,
                                                Field3D&       dq_field,
                                                Field3D&       dn_field,
                                                double         alpha,
                                                double         c_mass,
                                                double         d_mass,
                                                double         a_fall,
                                                double         b_fall,
                                                double         maximum_speed,
                                                double         maximum_diameter)
              {
                  const std::size_t base         = (static_cast<std::size_t>(i) * NTH + j) * NZ + domain.k_begin;
                  double            surface_loss = 0;
                  const bool        succeeded    = milbrandt_detail::sediment_column(
                      {alpha, c_mass, d_mass, a_fall, b_fall, maximum_speed, maximum_diameter},
                      levels,
                      q_field.data() + base,
                      n_field.data() + base,
                      rho_d.data() + base,
                      workspace.thickness.data(),
                      dt,
                      workspace,
                      dq_field.data() + base,
                      dn_field.data() + base,
                      surface_loss);
                  if (!succeeded)
                      sedimentation_failed = 1;
                  surface_sedimentation_sink_rate_[i][j][0] += static_cast<float>(surface_loss / dt);
              };

              sediment_species(qr, Nr, dqr_dt, dNr_dt, alpha_r_, c_r_, d_r_, a_r_, b_r_, 16.0, 5.0e-3);
              sediment_species(qi, Ni, dqi_dt, dNi_dt, alpha_i_, c_i_, d_i_, a_i_, b_i_, 2.0, 5.0e-3);
              sediment_species(qs, Ns, dqs_dt, dNs_dt, alpha_s_, c_s_, d_s_, a_s_, b_s_, 4.0, 5.0e-3);
              sediment_species(qg, Ng, dqg_dt, dNg_dt, alpha_g_, c_g_, d_g_, a_g_, b_g_, 6.0, 2.0e-3);
              // hail_processes_ controls production, never the physical
              // evolution of hail already present in an initial/restart state.
              sediment_species(qh, Nh, dqh_dt, dNh_dt, alpha_h_, c_h_, d_h_, a_h_, b_h_, 25.0, 80.0e-3);
          }
        }
    }
    if (sedimentation_failed != 0)
    {
        throw std::runtime_error(
            "Milbrandt sedimentation subcycling failed");
    }
}

double MilbrandtScheme::bottom_sedimentation_sink_rate(
    const Field3D& rho_d,
    const Field3D& qr, const Field3D& qs,
    const Field3D& qg, const Field3D& qh,
    int i, int j, double dt_micro) const
{
    if (internal_triple_)
    {
        const auto& internal = *internal_triple_;
        if (!internal.candidate_valid || internal.poisoned ||
            i < 0 || j < 0 || i >= internal.surface_sink_after.size_r() ||
            j >= internal.surface_sink_after.size_th())
        {
            throw std::runtime_error(
                "internal Milbrandt triple sedimentation surface candidate is unavailable");
        }
        return internal.surface_sink_after(i, j, 0);
    }
    (void)rho_d;
    (void)qr;
    (void)qs;
    (void)qg;
    (void)qh;
    (void)dt_micro;
    if (i < 0 || j < 0
        || i >= surface_sedimentation_sink_rate_.size_r()
        || j >= surface_sedimentation_sink_rate_.size_th()
        || surface_sedimentation_sink_rate_.size_z() == 0)
    {
        return 0.0;
    }
    return std::max(
        0.0,
        static_cast<double>(surface_sedimentation_sink_rate_[i][j][0]));
}

} // namespace tmv
