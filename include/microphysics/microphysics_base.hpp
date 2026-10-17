#pragma once

#include <exception>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/field/field3d.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "core/infra/physical_constants.hpp"
#include "core/runtime/portable_scheme_state.hpp"
#include "microphysics/config.hpp"

struct SimulationStateView;
struct ComputeBackendRuntimeState;
namespace nimbus::parity { struct ParityRuntimeState; }

// Microphysics scheme contracts, repair ledgers, and persistent runtime state.

namespace tmv
{

struct MicrophysicsWaterRepairReport;
struct MicrophysicsThermodynamicRepairReport;

/**
 * Scheme-owned scalar that participates in dynamics transport.
 *
 * The storage semantic is explicit because operational bulk schemes commonly
 * retain number concentration in #/m3 while the dynamics transports its
 * equivalent dry-air number mixing ratio.
 */
enum class MicrophysicsScalarStorage
{
    DryAirMixingRatio,
    VolumeConcentration
};

enum class MicrophysicsMassSpecies
{
    None,
    Cloud,
    Rain,
    Ice,
    Snow,
    Graupel,
    Hail
};

struct MicrophysicsPrognosticScalar
{
    const char* name = nullptr;
    Field3D* field = nullptr;
    bool positive_definite = true;
    MicrophysicsScalarStorage storage           = MicrophysicsScalarStorage::DryAirMixingRatio;
    MicrophysicsMassSpecies   mass_species      = MicrophysicsMassSpecies::None;
};

struct MicrophysicsParameterIdentity
{
    static constexpr std::size_t maximum_integers = 4;
    static constexpr std::size_t maximum_reals = 32;

    std::uint32_t version = 0;
    std::size_t integer_count = 0;
    std::size_t real_count = 0;
    std::array<std::uint64_t, maximum_integers> integers{};
    std::array<double, maximum_reals> reals{};

    bool operator==(const MicrophysicsParameterIdentity& other) const noexcept
    {
        if (integer_count > maximum_integers ||
            other.integer_count > maximum_integers ||
            real_count > maximum_reals ||
            other.real_count > maximum_reals)
        {
            return false;
        }
        return version == other.version &&
            integer_count == other.integer_count &&
            real_count == other.real_count &&
            std::equal(integers.begin(), integers.begin() + integer_count,
                       other.integers.begin()) &&
            std::equal(reals.begin(), reals.begin() + real_count,
                       other.reals.begin());
    }
};

/**
 * Fixed-capacity, allocation-free set of scheme-private transport bindings.
 *
 * Six moments are currently sufficient for Milbrandt.  The capacity matches
 * the host/device scalar-transport interface ceiling and leaves room for
 * richer schemes without putting owning containers in the split-step path.
 */
class MicrophysicsPrognosticScalarList
{
public:
    static constexpr std::size_t capacity = 16;
    using Storage =
        std::array<MicrophysicsPrognosticScalar, capacity>;
    using iterator = Storage::iterator;
    using const_iterator = Storage::const_iterator;

    MicrophysicsPrognosticScalarList() = default;
    MicrophysicsPrognosticScalarList(
        std::initializer_list<MicrophysicsPrognosticScalar> scalars)
    {
        if (scalars.size() > capacity)
        {
            throw std::length_error(
                "too many microphysics prognostic transport scalars");
        }
        count_ = scalars.size();
        std::copy(scalars.begin(), scalars.end(), scalars_.begin());
    }

    [[nodiscard]] bool empty() const noexcept { return count_ == 0; }
    [[nodiscard]] std::size_t size() const noexcept { return count_; }
    [[nodiscard]] iterator begin() noexcept { return scalars_.begin(); }
    [[nodiscard]] iterator end() noexcept
    {
        return scalars_.begin() + static_cast<std::ptrdiff_t>(count_);
    }
    [[nodiscard]] const_iterator begin() const noexcept
    {
        return scalars_.begin();
    }
    [[nodiscard]] const_iterator end() const noexcept
    {
        return scalars_.begin()
            + static_cast<std::ptrdiff_t>(count_);
    }
    [[nodiscard]] MicrophysicsPrognosticScalar&
    operator[](std::size_t index) noexcept
    {
        return scalars_[index];
    }
    [[nodiscard]] const MicrophysicsPrognosticScalar&
    operator[](std::size_t index) const noexcept
    {
        return scalars_[index];
    }

private:
    Storage scalars_{};
    std::size_t count_ = 0;
};

class MicrophysicsScheme;

struct MicrophysicsPrimaryEndpointCandidate
{
    const MicrophysicsScheme* owner = nullptr;
    std::uint64_t generation = 0;
    double duration_s = 0.0;
    int nr = 0;
    int nth = 0;
    int nz = 0;
    // Bind rho, pressure, theta, then qv/qc/qr/qi/qs/qg/qh in that order.
    std::array<const Field3D*, 10> source_fields{};
    std::array<std::vector<double>, 7> water_after;
    std::vector<double> theta_increment;
};

struct MicrophysicsAcceptedPrimaryFields
{
    const Field3D& rho_d;
    const Field3D& qv;
    const Field3D& qc;
    const Field3D& qr;
    const Field3D& qi;
    const Field3D& qs;
    const Field3D& qg;
    const Field3D& qh;
    const MicrophysicsWaterRepairReport* water_repairs = nullptr;
    const MicrophysicsThermodynamicRepairReport* thermodynamic_repairs = nullptr;
    const MicrophysicsPrimaryEndpointCandidate* primary_endpoints = nullptr;
};

class MicrophysicsScheme
{
public:
    MicrophysicsScheme() = default;
    MicrophysicsScheme(const MicrophysicsScheme& other)
        : vertical_spacing_m_(other.vertical_spacing_m_),
          vertical_spacing_by_cell_m_(other.vertical_spacing_by_cell_m_),
          spacing_nr_(other.spacing_nr_),
          spacing_nth_(other.spacing_nth_),
          spacing_nz_(other.spacing_nz_),
          physical_domain_(other.physical_domain_),
          domain_nr_(other.domain_nr_),
          domain_nth_(other.domain_nth_),
          domain_nz_(other.domain_nz_),
          has_physical_domain_(other.has_physical_domain_),
          require_native_dispatch_(other.require_native_dispatch_)
    {
        // Live field bindings belong to one SimulationStateView and must be
        // rebound by its orchestration layer after a runtime snapshot.
    }
    MicrophysicsScheme& operator=(const MicrophysicsScheme& other)
    {
        if (this != &other)
        {
            vertical_spacing_m_ = other.vertical_spacing_m_;
            vertical_spacing_by_cell_m_ =
                other.vertical_spacing_by_cell_m_;
            spacing_nr_ = other.spacing_nr_;
            spacing_nth_ = other.spacing_nth_;
            spacing_nz_ = other.spacing_nz_;
            physical_domain_ = other.physical_domain_;
            domain_nr_ = other.domain_nr_;
            domain_nth_ = other.domain_nth_;
            domain_nz_ = other.domain_nz_;
            has_physical_domain_ = other.has_physical_domain_;
            require_native_dispatch_ = other.require_native_dispatch_;
            vertical_velocity_ = nullptr;
            vertical_velocity_face_staggered_ = false;
            lower_boundary_vertical_velocity_.clear();
        }
        return *this;
    }
    MicrophysicsScheme(MicrophysicsScheme&&) noexcept = default;
    MicrophysicsScheme& operator=(MicrophysicsScheme&&) noexcept = default;
    virtual ~MicrophysicsScheme() = default;

    // Deep copy, including scheme-private prognostic history.
    virtual std::unique_ptr<MicrophysicsScheme> clone() const { return nullptr; }

    // Preserve scientific parameters, without evolved state or grid bindings.
    virtual std::unique_ptr<MicrophysicsScheme> clone_parameters() const
    {
        return nullptr;
    }

    // Applies validated runtime configuration after factory construction.
    // Schemes override only the parameters they actually consume.
    virtual void configure(const MicrophysicsConfig&) {}

    virtual PortableSchemeState portable_state() const { return {}; }
    // Field-free immutable scheme identity for hot-path compatibility checks.
    // Schemes with large portable field payloads should override this method.
    virtual PortableSchemeState portable_parameter_state() const
    {
        PortableSchemeState state = portable_state();
        state.fields.clear();
        return state;
    }
    // Exact fixed-capacity parameter identity for allocation-free hot-path
    // compatibility checks. Stateful schemes override every consumed value.
    virtual MicrophysicsParameterIdentity parameter_identity() const noexcept
    {
        return {};
    }
    virtual bool restore_portable_state(const PortableSchemeState& state,
                                        std::string& error)
    {
        if (empty_portable_scheme_state(state)) return true;
        error = "microphysics scheme does not accept portable state";
        return false;
    }

    /**
     * Allocate/seed scheme-owned moments before dynamics transports them.
     * Implementations must preserve an already initialized state of the same
     * shape.  The default is appropriate for schemes with no private moments.
     */
    virtual void prepare_prognostic_state(const Field3D& rho_d,
                                          const Field3D& qc,
                                          const Field3D& qr,
                                          const Field3D& qi,
                                          const Field3D& qs,
                                          const Field3D& qg,
                                          const Field3D& qh)
    {
        (void)rho_d; (void)qc; (void)qr; (void)qi;
        (void)qs; (void)qg; (void)qh;
    }

    /** Return every scheme-owned scalar that dynamics must transport. */
    virtual MicrophysicsPrognosticScalarList
    prognostic_scalars_for_transport()
    {
        return {};
    }

    /**
     * Validate a prepared private-moment update before primary state mutates.
     * A false result rejects the entire microphysics transaction.
     */
    virtual bool validate_prognostic_update(double dt, std::string& error) const
    {
        (void)dt;
        error.clear();
        return true;
    }

    virtual bool prepare_post_primary_private_commit(
        double dt,
        const MicrophysicsAcceptedPrimaryFields& accepted,
        std::string& error)
    {
        (void)dt;
        (void)accepted;
        error.clear();
        return true;
    }

    virtual bool owns_primary_endpoint_commit() const noexcept
    {
        return false;
    }

    virtual const Field3D* protected_primary_endpoint_wind_source() const noexcept
    {
        return nullptr;
    }

    virtual bool prepare_primary_endpoint_commit(
        const SimulationStateView& state,
        double dt,
        MicrophysicsPrimaryEndpointCandidate& candidate,
        std::string& error) const
    {
        (void)state;
        (void)dt;
        (void)candidate;
        error = "microphysics scheme has no primary endpoint owner";
        return false;
    }

    virtual bool validate_primary_endpoint_commit(
        const SimulationStateView& state,
        double dt,
        const MicrophysicsPrimaryEndpointCandidate& candidate,
        std::string& error) const
    {
        (void)state;
        (void)dt;
        (void)candidate;
        error = "microphysics scheme cannot validate primary endpoints";
        return false;
    }

    virtual bool owns_post_primary_private_commit() const noexcept
    {
        return false;
    }

    virtual bool commit_prepared_private_candidate() noexcept
    {
        return false;
    }

    /**
     * Commit the previously validated private-moment update.  This is called
     * only after all primary-state and coupled-source acceptance gates pass.
     */
    virtual void commit_prognostic_update(double dt) noexcept { (void)dt; }

    /**
     * Return true when the scheme publishes process and transport tendencies
     * that require the runtime's shared finite-step column donor admission.
     * The runtime applies one admitted factor to every coupled primary
     * tendency and to the matching surface fallout owner.
     */
    virtual bool requires_runtime_column_donor_admission() const noexcept
    {
        return false;
    }

    /**
     * Admit and apply a common stochastic process multiplier to any
     * scheme-private prepared prognostic tendencies. Implementations may
     * reduce a requested column factor to preserve their reservoir bounds.
     * Primary mass/heat tendencies are scaled by the shared runtime only
     * after this method succeeds.
     */
    virtual bool apply_stochastic_column_multipliers(double              dt,
                                                     const Field3D&      density,
                                                     std::vector<float>& column_multipliers,
                                                     std::string&        error)
    {
        (void)dt;
        (void)density;
        for (float multiplier : column_multipliers)
        {
            if (!std::isfinite(multiplier) || multiplier < 0.0f)
            {
                error = "invalid stochastic microphysics column multiplier";
                return false;
            }
        }
        error.clear();
        return true;
    }

    // Clear scheme-private step products after a domain-generation change.
    // Prognostic moments are shifted separately and must not be modified.
    virtual void invalidate_transient_state_after_domain_recycling() noexcept
    {
    }

    // Supplies live vertical air velocity for activation/nucleation closures.
    // On a C-grid, w[k] and w[k-1] bound scalar cell k.
    void set_vertical_velocity_field(const Field3D&      vertical_velocity,
                                     bool                face_staggered,
                                     std::vector<double> lower_boundary_velocity = {})
    {
        if (!lower_boundary_velocity.empty() &&
            (!face_staggered || vertical_velocity.empty() ||
             lower_boundary_velocity.size() != static_cast<std::size_t>(vertical_velocity.size_r()) *
                                                   static_cast<std::size_t>(vertical_velocity.size_th()) ||
             !std::all_of(lower_boundary_velocity.begin(),
                          lower_boundary_velocity.end(),
                          [](double value) { return std::isfinite(value); })))
        {
            throw std::invalid_argument("microphysics lower-boundary velocity requires finite C-grid columns");
        }
        vertical_velocity_                = &vertical_velocity;
        vertical_velocity_face_staggered_ = face_staggered;
        lower_boundary_vertical_velocity_ = std::move(lower_boundary_velocity);
    }

    void rebind_vertical_velocity_field(const Field3D& vertical_velocity)
    {
        if (!vertical_velocity_ || vertical_velocity.size_r() != vertical_velocity_->size_r() ||
            vertical_velocity.size_th() != vertical_velocity_->size_th() ||
            vertical_velocity.size_z() != vertical_velocity_->size_z())
        {
            throw std::invalid_argument("microphysics velocity rebinding must preserve the prepared grid");
        }
        vertical_velocity_ = &vertical_velocity;
    }

    // Sedimentation is grid-CFL limited and must use the caller's vertical
    // spacing rather than process-global geometry.
    void set_vertical_spacing_m(double spacing_m)
    {
        if (!std::isfinite(spacing_m) || spacing_m <= 0.0)
        {
            throw std::invalid_argument("microphysics vertical spacing must be finite and positive");
        }
        vertical_spacing_m_ = spacing_m;
        vertical_spacing_by_cell_m_.clear();
        spacing_nr_ = spacing_nth_ = spacing_nz_ = 0;
    }

    // Supplies exact physical layer thickness for terrain/nonuniform grids.
    // Values use Field3D row-major ordering: (i*nth + j)*nz + k.
    void set_vertical_spacing_m(int nr, int nth, int nz,
                                std::vector<double> spacing_m)
    {
        const std::size_t expected = static_cast<std::size_t>(nr)
            * static_cast<std::size_t>(nth) * static_cast<std::size_t>(nz);
        if (nr <= 0 || nth <= 0 || nz <= 0 || spacing_m.size() != expected)
            throw std::invalid_argument(
                "microphysics layer-thickness field has the wrong shape");
        for (double spacing : spacing_m)
            if (!std::isfinite(spacing) || spacing <= 0.0)
                throw std::invalid_argument(
                    "microphysics layer thickness must be finite and positive");
        spacing_nr_ = nr;
        spacing_nth_ = nth;
        spacing_nz_ = nz;
        vertical_spacing_by_cell_m_ = std::move(spacing_m);
    }

    // Declares the caller-owned atmospheric cells. Storage outside this
    // half-open range is boundary closure and must not participate in
    // pointwise physics or sedimentation. Direct scheme calls that do not set
    // a domain retain the historical full-field contract.
    void set_physical_cell_domain(int nr, int nth, int nz,
                                  PhysicalCellDomain domain)
    {
        if (nr <= 0 || nth <= 0 || nz <= 0 || domain.empty()
            || domain.i_begin < 0 || domain.i_end > nr
            || domain.j_begin < 0 || domain.j_end > nth
            || domain.k_begin < 0 || domain.k_end > nz)
        {
            throw std::invalid_argument(
                "microphysics physical-cell domain is invalid for the field shape");
        }
        domain_nr_ = nr;
        domain_nth_ = nth;
        domain_nz_ = nz;
        physical_domain_ = domain;
        has_physical_domain_ = true;
    }

    // Qualified accelerator profiles use this to make an operator dispatch
    // failure fatal instead of silently changing the selected execution path.
    void bind_execution_owners(ComputeBackendRuntimeState& compute, nimbus::parity::ParityRuntimeState* parity) noexcept
    {
        compute_runtime_ = &compute;
        parity_runtime_ = parity;
    }
    ComputeBackendRuntimeState* compute_runtime() const noexcept { return compute_runtime_; }
    nimbus::parity::ParityRuntimeState* parity_runtime() const noexcept { return parity_runtime_; }

    void set_require_native_dispatch(bool required) noexcept
    {
        require_native_dispatch_ = required;
    }

    /**
     * @brief Computes thermodynamic and hydrometeor tendencies.
     * @param p Pressure field.
     * @param theta Potential temperature field.
     * @param rho_d Prognostic live dry-air density field (kg_dry/m^3).
     * @param qv Water vapor dry-air mixing ratio field (kg/kg_dry).
     * @param qc Cloud water dry-air mixing ratio field (kg/kg_dry).
     * @param qr Rain dry-air mixing ratio field (kg/kg_dry).
     * @param qi Cloud ice dry-air mixing ratio field (kg/kg_dry).
     * @param qs Snow dry-air mixing ratio field (kg/kg_dry).
     * @param qg Graupel dry-air mixing ratio field (kg/kg_dry).
     * @param qh Hail dry-air mixing ratio field (kg/kg_dry).
     * @param dt Time step in seconds.
     * @param dtheta_dt Potential temperature tendency.
     * @param dqv_dt Water vapor tendency.
     * @param dqc_dt Cloud water tendency.
     * @param dqr_dt Rain tendency.
     * @param dqi_dt Cloud ice tendency.
     * @param dqs_dt Snow tendency.
     * @param dqg_dt Graupel tendency.
     * @param dqh_dt Hail tendency.
     */
    virtual void compute_tendencies(const Field3D& p,
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
                                    Field3D& dqh_dt) = 0;

    /**
     * @brief Computes surface precipitation-rate diagnostics.
     * @param qr Rain dry-air mixing ratio field.
     * @param qs Snow dry-air mixing ratio field.
     * @param qg Graupel dry-air mixing ratio field.
     * @param qh Hail dry-air mixing ratio field.
     * @param precip_rate_rain Output rain rate.
     * @param precip_rate_snow Output snow rate.
     * @param precip_rate_grau Output graupel rate.
     * @param precip_rate_hail Output hail rate.
     */
    virtual void compute_precipitation_rates(const Field3D&,
                                             const Field3D&,
                                             const Field3D&,
                                             const Field3D&,
                                             Field3D&,
                                             Field3D&,
                                             Field3D&,
                                             Field3D&)
    {
    }

    /**
     * @brief Returns the scheme identifier.
     * @return Scheme name.
     */
    /**
     * @brief Mixing-ratio rate (kg/kg/s, >= 0) at which sedimentation
     *        removes precipitation through the bottom face of column
     *        (i, j) - the intentional surface sink. The conservation
     *        gate credits this back so ordinary rain reaching the
     *        ground does not read as scheme non-conservation
     *        (SCIENCE.md M6/finding 10). Must mirror the scheme's
     *        own lowest-physical-cell sedimentation sink exactly, including
     *        its dependence on live dry density. Schemes without
     *        sedimentation return 0.
     */
    virtual double bottom_sedimentation_sink_rate(
        const Field3D& rho_d,
        const Field3D& qr, const Field3D& qs,
        const Field3D& qg, const Field3D& qh,
        int i, int j, double dt_micro) const
    {
        (void)rho_d; (void)qr; (void)qs; (void)qg; (void)qh;
        (void)i; (void)j; (void)dt_micro;
        return 0.0;
    }

    // Stability admission queries independent columns concurrently; implementations must be read-only.
    virtual double explicit_terminal_velocity_mps(
        std::size_t species, double mixing_ratio, double dry_density) const
    {
        return 0.0;
    }

    double maximum_explicit_terminal_velocity_mps(
        double mixing_ratio, double dry_density) const
    {
        double maximum = 0.0;
        for (std::size_t species = 0; species < 4; ++species)
            maximum = std::max(maximum,
                explicit_terminal_velocity_mps(species, mixing_ratio, dry_density));
        return maximum;
    }

    virtual std::string get_scheme_name() const = 0;

    /**
     * @brief Returns the number of prognostic microphysics variables.
     * @return Count of prognostic microphysics variables.
     */
    virtual int get_num_prognostic_vars() const = 0;

    /**
     * @brief Optional one-line caveat about this scheme's scientific
     *        completeness, logged loudly at initialization so results are not
     *        mis-attributed (e.g. a scheme that runs but is single-moment, or
     *        whose number concentrations are not advected). Empty = no caveat.
     */
    virtual std::string capability_caveat() const { return {}; }

protected:
    void require_stable_sedimentation(
        const Field3D& density, const std::array<const Field3D*, 4>& species,
        double timestep) const
    {
        if (!std::isfinite(timestep) || timestep <= 0.0)
            throw std::invalid_argument("sedimentation requires a positive finite timestep");
        for (const auto* field : species)
            if (field != nullptr && (field->size_r() != density.size_r() ||
                field->size_th() != density.size_th() || field->size_z() != density.size_z()))
                throw std::invalid_argument("sedimentation fields must match density");
        const auto domain = physical_cell_domain(
            density.size_r(), density.size_th(), density.size_z());
        std::exception_ptr failure;
        std::size_t first_failed_column = std::numeric_limits<std::size_t>::max();
        const std::size_t columns = static_cast<std::size_t>(domain.i_end - domain.i_begin) *
            static_cast<std::size_t>(domain.j_end - domain.j_begin);
#pragma omp parallel for collapse(2) schedule(static) if(columns >= 64)
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
            {
                try
                {
                    for (int k = domain.k_begin; k < domain.k_end; ++k)
                    {
                        const double rho = density[i][j][k];
                        if (!std::isfinite(rho) || rho <= 0.0)
                            throw std::invalid_argument("sedimentation density must be positive and finite");
                        for (std::size_t index = 0; index < species.size(); ++index)
                        {
                            if (species[index] == nullptr) continue;
                            const double q = (*species[index])[i][j][k];
                            if (!std::isfinite(q) || q < 0.0)
                                throw std::invalid_argument("sedimentation mixing ratio must be nonnegative and finite");
                            const double courant = explicit_terminal_velocity_mps(index, q, rho)
                                * timestep / vertical_spacing_m(i, j, k);
                            if (!std::isfinite(courant) || courant > 0.9)
                                throw std::invalid_argument(
                                    "sedimentation timestep exceeds the physical terminal-velocity CFL limit");
                        }
                    }
                }
                catch (...)
                {
                    const std::size_t column = static_cast<std::size_t>(i) * density.size_th() + j;
#pragma omp critical(nimbus_microphysics_stable_sedimentation_failure)
                    {
                        if (column < first_failed_column)
                        {
                            first_failed_column = column;
                            failure = std::current_exception();
                        }
                    }
                }
            }
        if (failure) std::rethrow_exception(failure);
    }

    PhysicalCellDomain physical_cell_domain(int nr, int nth, int nz) const noexcept
    {
        if (has_physical_domain_ && domain_nr_ == nr && domain_nth_ == nth
            && domain_nz_ == nz)
        {
            return physical_domain_;
        }
        return {0, nr, 0, nth, 0, nz};
    }

    double vertical_spacing_m() const noexcept { return vertical_spacing_m_; }
    bool has_nonuniform_vertical_spacing() const noexcept
    {
        return !vertical_spacing_by_cell_m_.empty();
    }
    bool require_native_dispatch() const noexcept
    {
        return require_native_dispatch_;
    }
    double vertical_spacing_m(int i, int j, int k) const
    {
        if (vertical_spacing_by_cell_m_.empty())
            return vertical_spacing_m_;
        if (i < 0 || i >= spacing_nr_ || j < 0 || j >= spacing_nth_
            || k < 0 || k >= spacing_nz_)
            throw std::out_of_range(
                "microphysics layer-thickness index is outside the field");
        const std::size_t index =
            (static_cast<std::size_t>(i) * static_cast<std::size_t>(spacing_nth_) + static_cast<std::size_t>(j)) *
                static_cast<std::size_t>(spacing_nz_) +
            static_cast<std::size_t>(k);
        return vertical_spacing_by_cell_m_[index];
    }

    const Field3D* vertical_velocity_field() const noexcept
    {
        return vertical_velocity_;
    }
    bool vertical_velocity_is_face_staggered() const noexcept
    {
        return vertical_velocity_face_staggered_;
    }
    const std::vector<double>& lower_boundary_vertical_velocity() const noexcept
    {
        return lower_boundary_vertical_velocity_;
    }

    double vertical_velocity_at_scalar(int i, int j, int k) const noexcept
    {
        if (vertical_velocity_ == nullptr || i < 0 || i >= vertical_velocity_->size_r() || j < 0 ||
            j >= vertical_velocity_->size_th() || k < 0 || k >= vertical_velocity_->size_z())
        {
            return 0.0;
        }
        if (!vertical_velocity_face_staggered_)
        {
            return (*vertical_velocity_)[i][j][k];
        }
        if (k == 0)
        {
            const double lower =
                lower_boundary_vertical_velocity_.empty()
                    ? 0.0
                    : lower_boundary_vertical_velocity_[static_cast<std::size_t>(i) * vertical_velocity_->size_th() +
                                                        j];
            return 0.5 * (lower + static_cast<double>((*vertical_velocity_)[i][j][0]));
        }
        return 0.5 * (static_cast<double>((*vertical_velocity_)[i][j][k]) +
                      static_cast<double>((*vertical_velocity_)[i][j][k - 1]));
    }

    /**
     * @brief Resizes all 8 tendency outputs and clears them in one parallel pass.
     *
     * Call at the top of compute_tendencies() to eliminate the identical
     * 8-line resize block duplicated across every scheme implementation.
     */
    static void init_tendency_fields(int      NR,
                                     int      NTH,
                                     int      NZ,
                                     Field3D& dtheta_dt,
                                     Field3D& dqv_dt,
                                     Field3D& dqc_dt,
                                     Field3D& dqr_dt,
                                     Field3D& dqi_dt,
                                     Field3D& dqs_dt,
                                     Field3D& dqg_dt,
                                     Field3D& dqh_dt)
    {
        dtheta_dt.resize(NR, NTH, NZ);
        dqv_dt.resize(NR, NTH, NZ);
        dqc_dt.resize(NR, NTH, NZ);
        dqr_dt.resize(NR, NTH, NZ);
        dqi_dt.resize(NR, NTH, NZ);
        dqs_dt.resize(NR, NTH, NZ);
        dqg_dt.resize(NR, NTH, NZ);
        dqh_dt.resize(NR, NTH, NZ);

        const std::size_t count = dtheta_dt.size();
        float* const dtheta = dtheta_dt.data();
        float* const dqv = dqv_dt.data();
        float* const dqc = dqc_dt.data();
        float* const dqr = dqr_dt.data();
        float* const dqi = dqi_dt.data();
        float* const dqs = dqs_dt.data();
        float* const dqg = dqg_dt.data();
        float* const dqh = dqh_dt.data();
        #pragma omp parallel for
        for (std::size_t n = 0; n < count; ++n)
        {
            dtheta[n] = 0.0f;
            dqv[n] = 0.0f;
            dqc[n] = 0.0f;
            dqr[n] = 0.0f;
            dqi[n] = 0.0f;
            dqs[n] = 0.0f;
            dqg[n] = 0.0f;
            dqh[n] = 0.0f;
        }
    }

private:
    ComputeBackendRuntimeState* compute_runtime_ = nullptr;
    nimbus::parity::ParityRuntimeState* parity_runtime_ = nullptr;
    double vertical_spacing_m_ = 100.0;
    std::vector<double> vertical_spacing_by_cell_m_;
    int spacing_nr_ = 0;
    int spacing_nth_ = 0;
    int spacing_nz_ = 0;
    PhysicalCellDomain physical_domain_{};
    int domain_nr_ = 0;
    int domain_nth_ = 0;
    int domain_nz_ = 0;
    bool has_physical_domain_ = false;
    bool require_native_dispatch_ = false;
    const Field3D* vertical_velocity_ = nullptr;
    bool vertical_velocity_face_staggered_ = false;
    std::vector<double> lower_boundary_vertical_velocity_;
};

/**
 * @brief Signed water-mass correction introduced by one post-apply repair.
 *
 * Lower-bound corrections are positive (water added back), upper-bound
 * corrections are negative (water removed), and non-finite replacements are
 * measured relative to the last finite pre-apply value when one exists.
 */
struct MicrophysicsWaterFieldRepair
{
    // Difference between the exact fp64 application candidate and the fp32
    // value handed to the bounds writer. This is ordinary storage rounding,
    // not a positivity repair, but it is required to close requested water
    // sources against the value actually committed.
    double storage_rounding_kg = 0.0;
    double lower_bound_correction_kg = 0.0;
    // Sub-ULP cancellation at an exactly drained reservoir is distinguished
    // from a material positivity repair. It remains in the signed mass ledger,
    // but strict mode may admit it within the analytic fp32 bound.
    double lower_bound_roundoff_correction_kg = 0.0;
    double upper_bound_correction_kg = 0.0;
    double nonfinite_correction_kg = 0.0;
    std::size_t lower_bound_count = 0;
    std::size_t lower_bound_roundoff_count = 0;
    std::size_t upper_bound_count = 0;
    std::size_t nonfinite_count = 0;
};

/**
 * @brief Water-repair ledger for the most recent microphysics application.
 *
 * Species order is qv, qc, qr, qi, qs, qg, qh. This ledger is deliberately
 * separate from the raw-tendency conservation gate: it records mass changed
 * by numerical safety repairs after those tendencies have been computed.
 */
struct MicrophysicsWaterRepairReport
{
    static constexpr std::size_t species_count = 7;
    std::array<MicrophysicsWaterFieldRepair, species_count> species{};
    std::size_t nonfinite_tendency_count = 0;

    double lower_bound_correction_kg() const;
    double lower_bound_roundoff_correction_kg() const;
    double upper_bound_correction_kg() const;
    double nonfinite_correction_kg() const;
    double storage_rounding_kg() const;
    double net_correction_kg() const;
    std::size_t lower_bound_count() const;
    std::size_t lower_bound_roundoff_count() const;
    std::size_t upper_bound_count() const;
    std::size_t nonfinite_count() const;
};

/** Raw microphysics total-water conservation result for one application. */
struct MicrophysicsConservationReport
{
    bool active = false;
    double tendency_imbalance_kg = 0.0;
    double baseline_q_kg = 0.0;
    double surface_precipitation_kg = 0.0;
    double relative_imbalance = 0.0;
};

/** Signed heat/water source and theta-repair ledger for one physics apply. */
struct MicrophysicsThermodynamicRepairReport
{
    static constexpr std::uint32_t report_schema_version = 2;
    static constexpr std::uint32_t energy_basis_version = 2;

    double microphysics_heat_j = 0.0;
    double radiation_heat_j = 0.0;
    double boundary_layer_heat_j = 0.0;
    double theta_tendency_limit_correction_j = 0.0;
    double theta_nonfinite_correction_j = 0.0;
    double theta_storage_rounding_j = 0.0;
    double theta_storage_residual_release_j = 0.0;
    double theta_storage_residual_j = 0.0;
    double theta_bound_correction_j = 0.0;

    double microphysics_water_tendency_kg = 0.0;
    // Requested mass conversion in qv,qc,qr,qi,qs,qg,qh order. The aggregate
    // above is retained for compatibility and must equal this array's sum.
    std::array<double, MicrophysicsWaterRepairReport::species_count>
        microphysics_water_tendency_kg_by_species{};
    double boundary_layer_vapor_tendency_kg = 0.0;
    double boundary_layer_cloud_liquid_tendency_kg = 0.0;
    double boundary_layer_cloud_ice_tendency_kg = 0.0;

    // Exact moist-mixture basis-v2 inventory change made by the shared
    // theta/seven-water commit. Density, pressure, velocity, geometry, and
    // TKE are held fixed by this operator. The six reservoir deltas are
    // measured from the actual bounded FP32 values stored by the commit, not
    // reconstructed later from heat or water-mass proxies.
    double moist_dry_internal_change_j = 0.0;
    double moist_water_sensible_change_j = 0.0;
    double moist_phase_reference_change_j = 0.0;
    double moist_carrier_kinetic_change_j = 0.0;
    double moist_gravitational_change_j = 0.0;
    double moist_subgrid_tke_change_j = 0.0;
    double moist_committed_energy_change_j = 0.0;
    double moist_energy_closure_residual_j = 0.0;
    bool moist_energy_complete = false;

    std::size_t theta_tendency_limit_count = 0;
    std::size_t theta_nonfinite_count = 0;
    std::size_t theta_lower_bound_count = 0;
    std::size_t theta_upper_bound_count = 0;

    double requested_heat_j() const
    {
        return microphysics_heat_j + radiation_heat_j + boundary_layer_heat_j;
    }

    double net_theta_repair_j() const
    {
        return theta_tendency_limit_correction_j
             + theta_nonfinite_correction_j
             + theta_bound_correction_j;
    }

    double committed_heat_j() const
    {
        return requested_heat_j() + net_theta_repair_j()
             + theta_storage_rounding_j + theta_storage_residual_j;
    }

    double state_heat_change_j() const
    {
        return requested_heat_j() + net_theta_repair_j()
             + theta_storage_residual_release_j
             + theta_storage_rounding_j;
    }
};

// Reusable tendency storage owned by one simulation runtime.
struct MicrophysicsTendencyFields
{
    Field3D dtheta_dt;
    Field3D dqv_dt;
    Field3D dqc_dt;
    Field3D dqr_dt;
    Field3D dqi_dt;
    Field3D dqs_dt;
    Field3D dqg_dt;
    Field3D dqh_dt;
};

// Replaceable execution storage for one independently advancing runtime.
// These buffers preserve transactional rollback and deterministic report
// ownership without rebuilding full fields or reduction vectors every step.
// They are deliberately omitted from runtime copies and checkpoints.
struct MicrophysicsExecutionScratch
{

    std::vector<int> clamp_count_by_k;
    std::vector<int> clamp_count_by_i;
    std::vector<MicrophysicsWaterRepairReport> water_report_partials;
    std::vector<MicrophysicsThermodynamicRepairReport>
        thermodynamic_report_partials;
    // Fixed physical-x-slab partials for the raw water-conservation gate.
    // The GPU/strict projection and ordinary CPU commit fill these inside
    // traversals they already require, then merge them in the same compensated
    // order as the former standalone diagnostic pass.
    std::vector<std::array<double, 3>> conservation_report_partials;
    // One admitted process-rate factor per horizontal column. This is
    // transient transaction state, not restart state.
    std::vector<float> stochastic_column_multipliers;
    // Transactional surface-fallout publication. These retain capacity and
    // prevent a failed physics application from advancing rainfall history.
    std::vector<float> precipitation_rate_candidate_mm_h;
    std::vector<float> accumulated_rainfall_candidate_mm;
    std::vector<float> accumulated_rainfall_residual_candidate_mm;
};

struct MicrophysicsStepState
{
    std::unique_ptr<MicrophysicsScheme> scheme;
    MicrophysicsTendencyFields tendencies;
    MicrophysicsWaterRepairReport last_water_repair_report;
    MicrophysicsThermodynamicRepairReport last_thermodynamic_repair_report;
    MicrophysicsConservationReport last_conservation_report;
    Field3D theta_storage_residual;

    MicrophysicsStepState() = default;
    MicrophysicsStepState(const MicrophysicsStepState&) = delete;
    MicrophysicsStepState& operator=(const MicrophysicsStepState&) = delete;
    MicrophysicsStepState(MicrophysicsStepState&&) noexcept = default;
    MicrophysicsStepState& operator=(MicrophysicsStepState&&) noexcept = default;

    // Capture writes only the rollback owner. Its field capacities survive
    // warm steps; the live scheme and its field bindings remain untouched.
    void capture_from(const MicrophysicsStepState& source)
    {
        auto captured_scheme = source.scheme ? source.scheme->clone() : nullptr;
        if (source.scheme && !captured_scheme)
            throw std::runtime_error("microphysics scheme does not support runtime snapshots: " +
                source.scheme->get_scheme_name());
        tendencies = source.tendencies;
        theta_storage_residual = source.theta_storage_residual;
        last_water_repair_report = source.last_water_repair_report;
        last_thermodynamic_repair_report = source.last_thermodynamic_repair_report;
        last_conservation_report = source.last_conservation_report;
        scheme = std::move(captured_scheme);
    }

    void restore_to(MicrophysicsStepState& destination) noexcept
    {
        using std::swap;
        swap(*this, destination);
    }
};

struct MicrophysicsSubcycleWorkspace;

struct MicrophysicsRuntimeState : MicrophysicsStepState
{
    MicrophysicsExecutionScratch execution_scratch;
    std::shared_ptr<MicrophysicsSubcycleWorkspace> subcycle_workspace;

    MicrophysicsRuntimeState() = default;
    MicrophysicsRuntimeState(const MicrophysicsRuntimeState& source)
    {
        capture_from(source);
    }
    MicrophysicsRuntimeState& operator=(const MicrophysicsRuntimeState& source)
    {
        if (this != &source)
        {
            MicrophysicsRuntimeState captured(source);
            *this = std::move(captured);
        }
        return *this;
    }
    MicrophysicsRuntimeState(MicrophysicsRuntimeState&&) noexcept = default;
    MicrophysicsRuntimeState& operator=(MicrophysicsRuntimeState&&) noexcept = default;
};

namespace microphysics_constants
{
constexpr double T0 = physical_constants::freezing_temperature_k;
constexpr double L_v = physical_constants::latent_heat_vaporization_jkg;
constexpr double L_f = physical_constants::latent_heat_fusion_jkg;
constexpr double L_s = L_v + L_f;
constexpr double cp = physical_constants::specific_heat_cp_jkgk;
constexpr double R_v = 461.5;
constexpr double R_d = physical_constants::gas_constant_dry_air_jkgk;
constexpr double p0 = physical_constants::reference_pressure_pa;
constexpr double rho_w = 1000.0;
constexpr double rho_i = 917.0;

// Melting-rate coefficients per kelvin of warm excess (SCIENCE.md
// M10). Single-sphere heat conduction: a particle of diameter D and
// bulk density rho_x melting in air at T > T0 receives heat at
// dm/dt = 2 pi D K_a (T - T0) f_vent / L_f, so the mixing-ratio rate is
// dq/dt = q (12 K_a f_vent / (rho_x D^2 L_f)) (T - T0), with air
// conductivity K_a = 0.024 W/(m K). Characteristic (D, rho_x, f_vent):
// snow (2 mm, 100, 2), graupel (5 mm, 400, 5), hail (10 mm, 900, 8).
// Result: snow clears within the radar bright band (~120 s at 2 K of
// warm excess), graupel melts over ~2 km of warm layer, and large hail
// reaches the surface only partially melted. The old constant 1e-3 1/s
// let every frozen species survive kilometers below the melting level.
constexpr double c_melt_snow_per_K = 4.3e-3;
constexpr double c_melt_graupel_per_K = 4.3e-4;
constexpr double c_melt_hail_per_K = 7.7e-5;

// Cloud-droplet freezing (SCIENCE.md M8): immersion freezing ramps
// linearly with supercooling and becomes instantaneous (one step, mass
// limited) at the homogeneous-nucleation threshold T0 - 38 C. The old
// constant rate glaciated -0.5 C cloud in ~17 min while -60 C cloud
// kept liquid for the same ~17 min - both ends unphysical.
constexpr double homogeneous_freezing_dT = 38.0;

// Marshall-Palmer intercepts and bulk densities for the precipitating
// species (LFO83 Table 1 values; hail intercept 4e4 per LFO83). Used by
// the gamma-integrated fall speeds (SCIENCE.md M12) and mirrored
// as literals in the sedimentation shaders.
constexpr double N0_rain = 8.0e6;     // 1/m^4
constexpr double N0_snow = 3.0e6;
constexpr double N0_graupel = 4.0e5;
constexpr double N0_hail = 4.0e4;
constexpr double rho_snow = 100.0;    // kg/m^3 bulk
constexpr double rho_graupel = 400.0;
constexpr double rho_hail = 913.0;

// Bigg (1953) stochastic freezing of supercooled rain (SCIENCE.md
// M9; LFO83 eq 45): per-drop freezing probability B' V_drop
// (exp(A' (T0 - T)) - 1) integrated over the Marshall-Palmer spectrum.
constexpr double bigg_A = 0.66;   // 1/K
constexpr double bigg_B = 100.0;  // 1/(m^3 s)

// Snow-collects-cloud (PSACW) collection efficiency.
constexpr double E_snow_cloud = 1.0;
} // namespace microphysics_constants

} // namespace tmv

#include "microphysics/factory.hpp"
#include "microphysics/runtime.hpp"

using tmv::MicrophysicsScheme;
using tmv::MicrophysicsRuntimeState;
using tmv::create_microphysics_scheme;
using tmv::get_available_microphysics_schemes;
namespace microphysics_constants = tmv::microphysics_constants;
