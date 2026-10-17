#pragma once

#include <memory>
#include <string>
#include <vector>

class BoundaryLayerComputeKernels;

#include "core/infra/physical_constants.hpp"

// Shared configuration and column contracts for PBL schemes.

namespace tmv
{

namespace boundary_layer_constants
{
inline constexpr double kappa = physical_constants::von_karman_constant;
inline constexpr double g     = physical_constants::gravity_ms2;
inline constexpr double cp    = physical_constants::specific_heat_cp_jkgk;
inline constexpr double T0    = physical_constants::freezing_temperature_k;
inline constexpr double pi    = physical_constants::pi;
} // namespace boundary_layer_constants

struct BoundaryLayerConfig
{
    std::string scheme_id            = "slab";
    double      dt_pbl               = 60.0;
    bool        apply_surface_fluxes = true;
    std::string surface_layer_id     = "bulk";
    bool        clear_sky_only       = true;
    bool        enable_tke           = false;
    std::string pbl_top_method       = "ri";
    double      pbl_max_height       = 2000.0;
    double      min_ustar            = 1e-4;
};

struct SurfaceConfig
{
    double z0m        = 0.1;
    double z0h        = 0.01;
    double albedo     = 0.2;
    double emissivity = 0.95;
    // Prescribed surface (skin) state that drives surface fluxes (CM1 tsk0
    // lineage). A sentinel value < 0 means "unset": the boundary-layer coupling
    // then holds the INITIAL surface-air state as the skin (zero flux at t=0;
    // flux develops as the boundary layer evolves away from its initial state).
    // Tsfc is an absolute temperature [K]; the coupling converts it to potential
    // temperature at the surface pressure. qsfc is a mixing ratio [kg/kg].
    double Tsfc = -1.0;
    double qsfc = -1.0;
};

// Per-column prognostic memory for the slab mixed-layer scheme.
struct SlabColumnState
{
    double h_m         = 0.0;
    double theta_m_k   = 0.0;
    double qv_m_kgkg   = 0.0;
    bool   initialized = false;
};

// Reusable per-worker scratch for schemes with auxiliary profiles.
struct BoundaryLayerColumnWorkspace
{
    // Physical column geometry. The runtime fills these in place for each
    // assigned column so flat and mapped-terrain PBL execution share the same
    // allocation-free geometry path after warm-up.
    std::vector<double> z_center;
    std::vector<double> z_interface;
    std::vector<double> layer_thickness;

    // Runtime-only C-grid reconstruction buffers. Column closures consume
    // physical winds at scalar centers; native u/v storage lives on faces.
    std::vector<double> centered_u;
    std::vector<double> centered_v;
    std::vector<double> tke;
    std::vector<double> mixing_length;
    std::vector<double> K_m;
    std::vector<double> K_h;
    std::vector<double> K_e;
    std::vector<double> nonlocal_u_flux;
    std::vector<double> nonlocal_v_flux;
    std::vector<double> nonlocal_theta_flux;
    std::vector<double> nonlocal_qv_flux;
    std::vector<double> cloud_top_diffusivity;
    std::vector<double> liquid_water_potential_temperature;
    std::vector<double> cloud_top_u_tendency;
    std::vector<double> cloud_top_v_tendency;
    std::vector<double> cloud_top_theta_l_tendency;
    std::vector<double> cloud_top_qv_tendency;

    // Validated FP32 publication and conserved-source storage shared by the
    // ordinary cadence path and recycled-window leading-column preparation.
    std::vector<float> stored_dtheta;
    std::vector<float> stored_dqv;
    std::vector<float> stored_dqc;
    std::vector<float> stored_dqi;
    std::vector<float> stored_du;
    std::vector<float> stored_dv;
    std::vector<float> stored_dtke;
    std::vector<float> stored_u_force_n;
    std::vector<float> stored_v_force_n;
    std::vector<float> stored_theta_inventory_rate_kgk_s;
    std::vector<float> stored_vapor_mass_rate_kg_s;
    std::vector<float> stored_cloud_liquid_mass_rate_kg_s;
    std::vector<float> stored_cloud_ice_mass_rate_kg_s;
};

// Non-owning profile over model fp32 or direct-call fp64 data.
class BoundaryLayerProfileView
{
public:
    BoundaryLayerProfileView() = default;
    BoundaryLayerProfileView(std::nullptr_t) noexcept
    {
    }
    BoundaryLayerProfileView(const std::vector<double>& values) noexcept : fp64_(values.data()), size_(values.size())
    {
    }
    BoundaryLayerProfileView(const std::vector<double>* values) noexcept
    {
        *this = values;
    }
    BoundaryLayerProfileView(const float* values, std::size_t size) noexcept : fp32_(values), size_(size)
    {
    }

    BoundaryLayerProfileView& operator=(const std::vector<double>* values) noexcept
    {
        fp64_ = values ? values->data() : nullptr;
        fp32_ = nullptr;
        size_ = values ? values->size() : 0;
        return *this;
    }
    BoundaryLayerProfileView& operator=(std::nullptr_t) noexcept
    {
        fp64_ = nullptr;
        fp32_ = nullptr;
        size_ = 0;
        return *this;
    }

    explicit operator bool() const noexcept
    {
        return fp64_ || fp32_;
    }
    bool operator==(std::nullptr_t) const noexcept
    {
        return !static_cast<bool>(*this);
    }
    bool operator!=(std::nullptr_t) const noexcept
    {
        return static_cast<bool>(*this);
    }
    std::size_t size() const noexcept
    {
        return size_;
    }
    bool empty() const noexcept
    {
        return size_ == 0;
    }
    double operator[](std::size_t index) const noexcept
    {
        return fp64_ ? fp64_[index] : static_cast<double>(fp32_[index]);
    }

    // Preserve the historical pointer-like column syntax while callers migrate
    // to direct value views.
    const BoundaryLayerProfileView& operator*() const noexcept
    {
        return *this;
    }
    const BoundaryLayerProfileView* operator->() const noexcept
    {
        return this;
    }

private:
    const double* fp64_ = nullptr;
    const float*  fp32_ = nullptr;
    std::size_t   size_ = 0;
};

enum class BoundaryLayerRequirements
{
    BASIC    = 1,
    MOISTURE = 2,
    TKE      = 4,
    CLOUDS   = 8
};

struct BoundaryLayerColumnStateView
{
    BoundaryLayerProfileView u;
    BoundaryLayerProfileView v;
    BoundaryLayerProfileView theta;
    BoundaryLayerProfileView qv;
    BoundaryLayerProfileView p;
    BoundaryLayerProfileView rho;
    BoundaryLayerProfileView z_int;

    BoundaryLayerProfileView tke;
    BoundaryLayerProfileView qc;
    BoundaryLayerProfileView qi;
    BoundaryLayerProfileView radiative_theta_tendency;

    double u_sfc     = 0.0;
    double v_sfc     = 0.0;
    double theta_sfc = 288.0;
    double qv_sfc    = 0.01;
    double z_sfc     = 10.0;
    // Physical terrain slopes dz_ground/dx and dz_ground/dy. Surface-layer
    // drag is evaluated on the tangent velocity and returned as generalized
    // force per horizontal area. Both are zero on a flat grid.
    double surface_slope_x = 0.0;
    double surface_slope_y = 0.0;

    // Optional scheme-specific runtime memory. Required by the slab scheme;
    // ignored by stateless/diagnostic column schemes.
    SlabColumnState*              slab_state     = nullptr;
    BoundaryLayerColumnWorkspace* workspace      = nullptr;
    double                        elapsed_time_s = 0.0;
};

struct BoundaryLayerTendencies
{
    std::vector<double> dudt_pbl;
    std::vector<double> dvdt_pbl;
    std::vector<double> dthetadt_pbl;
    std::vector<double> dqvdt_pbl;
    std::vector<double> dqcdt_pbl;
    std::vector<double> dqidt_pbl;
    std::vector<double> dtkedt_pbl;

    void clear_retain_capacity() noexcept
    {
        dudt_pbl.clear();
        dvdt_pbl.clear();
        dthetadt_pbl.clear();
        dqvdt_pbl.clear();
        dqcdt_pbl.clear();
        dqidt_pbl.clear();
        dtkedt_pbl.clear();
    }
};

struct BoundaryLayerDiagnostics
{
    // Runtime surface diagnostics do not consume the full diffusivity profiles.
    // Tests and direct scheme callers retain the historical default.
    bool                include_diffusivity_profiles = true;
    std::vector<double> K_m;
    std::vector<double> K_h;
    std::vector<double> K_e;

    double pbl_height = 0.0;
    double ustar      = 0.0;

    double tau_u = 0.0;
    double tau_v = 0.0;
    double H     = 0.0;
    double E     = 0.0;

    double L        = 0.0;
    double rib_bulk = 0.0;

    double Cd = 0.0;
    double Ch = 0.0;
    double Ce = 0.0;

    bool cloud_top_radiative_mixing_active = false;
    double cloud_top_height = 0.0;
    double cloud_top_cooling_flux_kms = 0.0;
    double cloud_top_velocity_scale = 0.0;

    void reset_retain_capacity(bool include_profiles) noexcept
    {
        include_diffusivity_profiles = include_profiles;
        K_m.clear();
        K_h.clear();
        K_e.clear();
        pbl_height = 0.0;
        ustar      = 0.0;
        tau_u      = 0.0;
        tau_v      = 0.0;
        H          = 0.0;
        E          = 0.0;
        L          = 0.0;
        rib_bulk   = 0.0;
        Cd         = 0.0;
        Ch         = 0.0;
        Ce         = 0.0;
        cloud_top_radiative_mixing_active = false;
        cloud_top_height = 0.0;
        cloud_top_cooling_flux_kms = 0.0;
        cloud_top_velocity_scale = 0.0;
    }
};

class BoundaryLayerSchemeBase
{
public:
    virtual ~BoundaryLayerSchemeBase() = default;
    virtual std::unique_ptr<BoundaryLayerSchemeBase> clone() const
    {
        return nullptr;
    }

    virtual std::string name() const = 0;

    // Bitmask of BoundaryLayerRequirements.
    virtual int required_fields() const = 0;

    virtual void initialize(const BoundaryLayerConfig& cfg) = 0;

    // Compute tendencies for one vertical column. Diagnostics are optional.
    virtual void compute_column(const BoundaryLayerConfig&          cfg,
                                const SurfaceConfig&                sfc,
                                const BoundaryLayerColumnStateView& col,
                                BoundaryLayerTendencies&            tend,
                                BoundaryLayerDiagnostics*           diag_opt = nullptr) = 0;

    virtual bool supports_native_column_batch(const BoundaryLayerConfig& /*cfg*/,
                                              const BoundaryLayerComputeKernels& /*kernels*/) const
    {
        return false;
    }
    virtual bool compute_columns_native(const BoundaryLayerConfig& /*cfg*/,
                                        const SurfaceConfig& /*surface*/,
                                        const BoundaryLayerColumnStateView* /*columns*/,
                                        std::size_t /*column_count*/,
                                        BoundaryLayerTendencies* /*tendencies*/,
                                        BoundaryLayerDiagnostics* /*diagnostics*/,
                                        BoundaryLayerComputeKernels& /*kernels*/)
    {
        return false;
    }
};

} // namespace tmv

#include "boundary_layer/factory.hpp"
#include "boundary_layer/runtime.hpp"

namespace boundary_layer_constants = tmv::boundary_layer_constants;
using tmv::BoundaryLayerColumnStateView;
using tmv::BoundaryLayerColumnWorkspace;
using tmv::BoundaryLayerConfig;
using tmv::BoundaryLayerDiagnostics;
using tmv::BoundaryLayerProfileView;
using tmv::BoundaryLayerRequirements;
using tmv::BoundaryLayerSchemeBase;
using tmv::BoundaryLayerSchemeFactory;
using tmv::BoundaryLayerTendencies;
using tmv::create_boundary_layer_scheme;
using tmv::get_available_boundary_layer_schemes;
using tmv::SlabColumnState;
using tmv::SurfaceConfig;
