#include "catch2/catch.hpp"
#include "chaos/base/correlation_filter.hpp"
#include "chaos/base/earth_correlated_noise.hpp"
#include "chaos/base/perturbation_field.hpp"
#include "chaos/base/random_generator.hpp"
#include "chaos/config_contract.hpp"
#include "chaos/runtime.hpp"
#include "chaos/factory.hpp"
#include "compute/backend_interface.hpp"
#include "core/infra/physical_constants.hpp"
#include "core/orchestration/dynamics/thermodynamic_closure.hpp"
#include "core/runtime/simulation.hpp"
#include "core/runtime/simulation_state.hpp"
#include "core/runtime/simulation_runtime.hpp"
#include "numerics/advection/advection_base.hpp"
#include "numerics/diffusion/diffusion_base.hpp"
#include "numerics/time_stepping/time_stepping_base.hpp"
#include "turbulence/turbulence_base.hpp"
#include "util/log_profile.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <vector>

SimulationStorageState global_simulation_storage{{2, 2, 3,
                                                  1000.0, 1000.0, 1.0, 1000.0}};
int& NR = global_simulation_storage.grid.nr;
int& NTH = global_simulation_storage.grid.nth;
int& NZ = global_simulation_storage.grid.nz;
double& dr = global_simulation_storage.grid.dx;
double& dz = global_simulation_storage.grid.dz;
double& dt = global_simulation_storage.grid.dt;
double& dtheta = global_simulation_storage.grid.dy;
GridGeometry& global_grid_geometry = global_simulation_storage.grid.geometry;
tmv::GridMetrics& tmv::global_grid_metrics = global_simulation_storage.process_grid_metrics;
tmv::RadiationRuntimeState& tmv::global_radiation_runtime_state = global_simulation_storage.runtime.radiation;
tmv::TurbulenceRuntimeState& tmv::global_turbulence_runtime_state = global_simulation_storage.runtime.turbulence;
tmv::BoundaryLayerRuntimeState& tmv::global_boundary_layer_runtime_state = global_simulation_storage.runtime.boundary_layer;
tmv::BoundaryLayerConfig& tmv::global_boundary_layer_config =
    tmv::global_boundary_layer_runtime_state.config;
tmv::chaos::ChaosRuntimeState& tmv::chaos::global_chaos_runtime_state = global_simulation_storage.runtime.chaos;
RayleighDampingRuntimeState& global_rayleigh_runtime_state = global_simulation_storage.runtime.rayleigh;
tmv::MicrophysicsRuntimeState& tmv::global_microphysics_runtime_state = global_simulation_storage.runtime.microphysics;
tmv::AdvectionRuntimeState& tmv::global_advection_runtime_state = global_simulation_storage.runtime.advection;
tmv::DiffusionRuntimeState& tmv::global_diffusion_runtime_state = global_simulation_storage.runtime.diffusion;
tmv::TimeSteppingRuntimeState& tmv::global_time_stepping_runtime_state = global_simulation_storage.runtime.time_stepping;

Field3D& rho = global_simulation_storage.fields.rho;
Field3D& p = global_simulation_storage.fields.p;
Field3D& u = global_simulation_storage.fields.u;
Field3D& w = global_simulation_storage.fields.w;
Field3D& v = global_simulation_storage.fields.v;
Field3D& tracer = global_simulation_storage.fields.tracer;
Field3D& theta = global_simulation_storage.fields.theta;
Field3D& qv = global_simulation_storage.fields.qv;
Field3D& qc = global_simulation_storage.fields.qc;
Field3D& qr = global_simulation_storage.fields.qr;
Field3D& qi = global_simulation_storage.fields.qi;
Field3D& qs = global_simulation_storage.fields.qs;
Field3D& qh = global_simulation_storage.fields.qh;
Field3D& qg = global_simulation_storage.fields.qg;
Field3D& tke = global_simulation_storage.fields.tke;

Field3D& dtheta_dt_rad = global_simulation_storage.tendencies.dtheta_dt_rad;
Field3D& dtheta_dt_pbl = global_simulation_storage.tendencies.dtheta_dt_pbl;
Field3D& dqv_dt_pbl = global_simulation_storage.tendencies.dqv_dt_pbl;
Field3D& du_dt_pbl = global_simulation_storage.tendencies.du_dt_pbl;
Field3D& dv_dt_pbl = global_simulation_storage.tendencies.dv_dt_pbl;
Field3D& dtke_dt_pbl = global_simulation_storage.tendencies.dtke_dt_pbl;

std::vector<float>& sfc_ustar = global_simulation_storage.surface.ustar;
std::vector<float>& sfc_tau_u = global_simulation_storage.surface.tau_u;
std::vector<float>& sfc_tau_v = global_simulation_storage.surface.tau_v;
std::vector<float>& sfc_sensible_heat_flux = global_simulation_storage.surface.sensible_heat_flux;
std::vector<float>& sfc_latent_heat_flux = global_simulation_storage.surface.latent_heat_flux;
std::vector<float>& sfc_moisture_flux = global_simulation_storage.surface.moisture_flux;
std::vector<float>& pbl_height_2d = global_simulation_storage.surface.pbl_height;

std::vector<double>& rho0_base = global_simulation_storage.base.rho0;
std::vector<double>& p0_base = global_simulation_storage.base.p0;
std::vector<double>& qv0_base = global_simulation_storage.base.qv0;
std::vector<double>& theta0_base = global_simulation_storage.base.theta0;
std::vector<double>& u0_base = global_simulation_storage.base.u0;
std::vector<double>& v0_base = global_simulation_storage.base.v0;

double& coriolis_f = global_simulation_storage.coriolis_f;
double& simulation_time = global_simulation_storage.time_s;
LogProfile& global_log_profile =
    (global_simulation_storage.configuration.log_profile = LogProfile::quiet);

bool g_boundary_layer_updated = false;

bool tmv::boundary_layer_updated_this_step()
{
    return g_boundary_layer_updated;
}

bool tmv::boundary_layer_updated_this_step(const SimulationStateView&)
{
    return g_boundary_layer_updated;
}

namespace
{

class FailingEarthControlBackend final : public ComputeBackend
{
public:
    enum class Behavior
    {
        Reject,
        Throw,
        SucceedOnceThenReject,
        SucceedOnceThenThrow,
    };

    explicit FailingEarthControlBackend(Behavior behavior)
        : behavior_(behavior)
    {
    }

    std::string name() const override { return "failing-earth-control"; }
    bool initialize(std::string&) override { return true; }
    void shutdown() override {}
    bool supports_earth_control_evolution() const override { return true; }

    bool dispatch_earth_control_evolution(
        const EarthControlEvolutionDispatch& dispatch) override
    {
        ++dispatch_count_;
        last_rho_ = dispatch.rho;
        const bool first_succeeds =
            behavior_ == Behavior::SucceedOnceThenReject ||
            behavior_ == Behavior::SucceedOnceThenThrow;
        if (first_succeeds && dispatch_count_ == 1)
        {
            std::fill_n(
                dispatch.destination,
                static_cast<std::size_t>(dispatch.control_i_count) *
                    dispatch.control_j_count * dispatch.nz,
                0.0f);
            return true;
        }
        if (behavior_ == Behavior::Throw ||
            behavior_ == Behavior::SucceedOnceThenThrow)
            throw std::runtime_error("injected Earth-control dispatch failure");
        return false;
    }

    int dispatch_count() const noexcept { return dispatch_count_; }
    double last_rho() const noexcept
    {
        return last_rho_;
    }

private:
    Behavior behavior_;
    int dispatch_count_ = 0;
    double last_rho_ = 0.0;
};

void configure_earth_noise(chaos::EarthCorrelatedNoise& noise)
{
    noise.configure(
        1234567ULL, 9, 7001ULL, 7101ULL,
        900.0, 700.0, 200.0, 250.0);
}

bool bitwise_field_equal(const Field3D& lhs, const Field3D& rhs)
{
    return lhs.size_r() == rhs.size_r() &&
        lhs.size_th() == rhs.size_th() && lhs.size_z() == rhs.size_z() &&
        (lhs.size() == 0 ||
         std::memcmp(lhs.data(), rhs.data(), lhs.size() * sizeof(float)) == 0);
}

bool earth_noise_state_equal(
    const chaos::EarthCorrelatedNoise& lhs,
    const chaos::EarthCorrelatedNoise& rhs)
{
    if (lhs.initialized() != rhs.initialized() || lhs.nr() != rhs.nr() ||
        lhs.nth() != rhs.nth() || lhs.nz() != rhs.nz() ||
        lhs.earth_origin_i_cells() != rhs.earth_origin_i_cells() ||
        lhs.earth_origin_j_cells() != rhs.earth_origin_j_cells() ||
        lhs.continuous_offset_x_m() != rhs.continuous_offset_x_m() ||
        lhs.continuous_offset_y_m() != rhs.continuous_offset_y_m() ||
        lhs.control_i_min() != rhs.control_i_min() ||
        lhs.control_j_min() != rhs.control_j_min() ||
        !bitwise_field_equal(lhs.current_controls(), rhs.current_controls()) ||
        !bitwise_field_equal(lhs.previous_controls(), rhs.previous_controls()))
        return false;
    Field3D lhs_current;
    Field3D rhs_current;
    Field3D lhs_previous;
    Field3D rhs_previous;
    lhs.render_current(lhs_current);
    rhs.render_current(rhs_current);
    lhs.render_previous(lhs_previous);
    rhs.render_previous(rhs_previous);
    return bitwise_field_equal(lhs_current, rhs_current) &&
        bitwise_field_equal(lhs_previous, rhs_previous);
}

void initialize_process_globals()
{
    NR = 2;
    NTH = 2;
    NZ = 3;
    dr = 1000.0;
    dz = 1000.0;
    dt = 1.0;
    dtheta = 1000.0;
    global_grid_geometry.initialize(NR, NTH, NZ, dr, dz, dtheta,
                                    CoordinateSystem::Cartesian);
    global_grid_metrics.dx = dr;
    global_grid_metrics.dy = dtheta;
    global_grid_metrics.dz.assign(static_cast<std::size_t>(NZ), dz);
    global_grid_metrics.z_int = {0.0, 1000.0, 2000.0, 3000.0};

    p.resize(NR, NTH, NZ, 100000.0f);
    theta.resize(NR, NTH, NZ, 280.0f);
    rho.resize(NR, NTH, NZ, 1.1f);
    qv.resize(NR, NTH, NZ, 0.005f);
    u.resize(NR, NTH, NZ, 0.0f);
    v.resize(NR, NTH, NZ, 0.0f);
    w.resize(NR, NTH, NZ, 0.0f);
    tracer.resize(NR, NTH, NZ, 0.0f);
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 0.0f);
    qi.resize(NR, NTH, NZ, 0.0f);
    qs.resize(NR, NTH, NZ, 0.0f);
    qh.resize(NR, NTH, NZ, 0.0f);
    qg.resize(NR, NTH, NZ, 0.0f);
    dtheta_dt_rad.resize(NR, NTH, NZ, 0.0f);
    dtheta_dt_pbl.resize(NR, NTH, NZ, 0.0f);
    dqv_dt_pbl.resize(NR, NTH, NZ, 0.0f);
    du_dt_pbl.resize(NR, NTH, NZ, 0.0f);
    dv_dt_pbl.resize(NR, NTH, NZ, 0.0f);
    dtke_dt_pbl.resize(NR, NTH, NZ, 0.0f);
    g_boundary_layer_updated = false;
}

bool level_changed_from(const Field3D& field, int k, float original)
{
    for (int i = 0; i < static_cast<int>(field.size_r()); ++i)
    {
        for (int j = 0; j < static_cast<int>(field.size_th()); ++j)
        {
            if (std::abs(field[i][j][k] - original) > 1.0e-5f)
            {
                return true;
            }
        }
    }
    return false;
}

bool level_matches(const Field3D& field, int k, float expected)
{
    for (int i = 0; i < static_cast<int>(field.size_r()); ++i)
    {
        for (int j = 0; j < static_cast<int>(field.size_th()); ++j)
        {
            if (std::abs(field[i][j][k] - expected) > 1.0e-6f)
            {
                return false;
            }
        }
    }
    return true;
}

bool field_matches(const Field3D& field, float expected)
{
    for (int k = 0; k < static_cast<int>(field.size_z()); ++k)
    {
        if (!level_matches(field, k, expected))
        {
            return false;
        }
    }
    return true;
}

chaos::ChaosConfig local_ic_config()
{
    chaos::ChaosConfig cfg;
    cfg.scheme_id = "initial_conditions";
    cfg.seed = 12345;
    cfg.member_id = 7;
    cfg.filter_id = "recursive_gaussian";
    cfg.Lx = 1.0;
    cfg.Ly = 1.0;
    cfg.apply_to_ic = {"theta", "qv"};
    cfg.sigma_ic["theta"] = 4.0;
    cfg.sigma_ic["qv"] = 0.001;
    cfg.taper_id = "pbl_only";
    cfg.taper_z1 = 50.0;
    cfg.taper_z2 = 100.0;
    cfg.xi_max = 2.0;
    return cfg;
}

chaos::ChaosConfig local_pbl_config()
{
    chaos::ChaosConfig cfg;
    cfg.scheme_id = "boundary_layer";
    cfg.seed = 24680;
    cfg.member_id = 3;
    cfg.dt_chaos = 1.0;
    cfg.apply_to_tendencies = {"pbl"};
    cfg.alpha_tend["pbl"] = 1.0;
    cfg.taper_id = "none";
    cfg.xi_max = 2.0;
    return cfg;
}

class CountingChaosScheme final : public chaos::ChaosScheme
{
public:
    int noise_steps = 0;
    int tendency_applications = 0;
    double last_noise_dt = 0.0;
    bool last_grid_used_terrain = false;

    std::string name() const override { return "counting"; }
    std::unique_ptr<chaos::ChaosScheme> clone() const override
    { return std::make_unique<CountingChaosScheme>(*this); }
    void initialize(const chaos::ChaosConfig&, const GridMetrics&) override {}

    void apply_tendencies(const chaos::ChaosConfig&,
                          const GridMetrics&,
                          const chaos::ChaosStateView&,
                          chaos::ChaosTendencies& tendencies,
                          chaos::ChaosDiagnostics*) override
    {
        ++tendency_applications;
        if (tendencies.du_diff_dt != nullptr)
        {
            for (std::size_t idx = 0; idx < tendencies.du_diff_dt->size(); ++idx)
            {
                tendencies.du_diff_dt->data()[idx] *= 2.0f;
            }
        }
    }

    void step_noise(const chaos::ChaosConfig&,
                    const GridMetrics& grid,
                    double step_dt) override
    {
        ++noise_steps;
        last_noise_dt = step_dt;
        last_grid_used_terrain = grid.terrain_metrics_active
            && grid.terrain_metrics != nullptr;
    }
};

class CoefficientTransportScheme final : public TurbulenceSchemeBase
{
public:
    int rebuilds = 0;
    Field3D nu;
    Field3D kth;
    Field3D kq;

    std::string name() const override { return "smagorinsky"; }
    int required_fields() const override { return 0; }
    void initialize(const TurbulenceConfig&) override {}
    void compute(const TurbulenceConfig&, const GridMetrics&,
                 const TurbulenceStateView&, TurbulenceTendencies&,
                 TurbulenceDiagnostics*) override {}
    const Field3D* realized_eddy_viscosity() const noexcept override
    {
        return &nu;
    }
    bool mutable_realized_transport_fields(
        tmv::TurbulenceTransportFields& fields) noexcept override
    {
        fields.eddy_viscosity = &nu;
        fields.theta_diffusivity = &kth;
        fields.moisture_diffusivity = &kq;
        return true;
    }
    bool rebuild_from_realized_transport(
        const TurbulenceConfig&, const GridMetrics&,
        const TurbulenceStateView&, TurbulenceTendencies& tendencies) override
    {
        ++rebuilds;
        tendencies.dudt_sgs.fill(0.0f);
        tendencies.dvdt_sgs.fill(0.0f);
        tendencies.dwdt_sgs.fill(0.0f);
        tendencies.dthetadt_sgs.fill(0.0f);
        tendencies.dqvdt_sgs.fill(0.0f);
        tendencies.dtkedt_sgs.fill(0.0f);
        // Minimal conservative pair used only to prove that the runtime
        // consumed the perturbed coefficient and rebuilt a flux divergence.
        if (tendencies.dudt_sgs.size() >= 2)
        {
            tendencies.dudt_sgs.data()[0] = nu.data()[0];
            tendencies.dudt_sgs.data()[1] = -nu.data()[0];
        }
        return true;
    }
};

void apply_test_horizontal_correlation(
    Field3D& field,
    chaos::CorrelationFilter& filter,
    double dx,
    double dy
)
{
    std::vector<std::vector<double>> slice(
        static_cast<std::size_t>(field.size_r()),
        std::vector<double>(static_cast<std::size_t>(field.size_th()), 0.0));
    for (int k = 0; k < field.size_z(); ++k)
    {
        for (int i = 0; i < field.size_r(); ++i)
        {
            for (int j = 0; j < field.size_th(); ++j)
            {
                slice[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                    static_cast<double>(field(i, j, k));
            }
        }
        filter.apply_2d(slice, dx, dy);
        for (int i = 0; i < field.size_r(); ++i)
        {
            for (int j = 0; j < field.size_th(); ++j)
            {
                field(i, j, k) = static_cast<float>(
                    slice[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]);
            }
        }
    }
}

Field3D make_correlated_noise(
    chaos::ChaosRNG& rng,
    int nr,
    int nth,
    int nz,
    uint64_t stream_key,
    const std::string& field_name,
    chaos::CorrelationFilter& filter,
    double dx,
    double dy
)
{
    Field3D noise = chaos::generate_white_noise_field3d(
        rng, nr, nth, nz, stream_key, field_name);
    apply_test_horizontal_correlation(noise, filter, dx, dy);
    chaos::renormalize_to_unit_variance(noise);
    return noise;
}

[[maybe_unused]] Field3D make_correlated_update_noise(
    chaos::ChaosRNG& rng,
    int nr,
    int nth,
    int nz,
    uint64_t stream_key,
    uint64_t sequence_key,
    chaos::CorrelationFilter& filter,
    double dx,
    double dy
)
{
    Field3D noise(nr, nth, nz);
    rng.fill_field_normal(noise, stream_key, sequence_key);
    apply_test_horizontal_correlation(noise, filter, dx, dy);
    chaos::renormalize_to_unit_variance(noise);
    return noise;
}

GridMetrics stochastic_test_grid(int nr, int nth, int nz)
{
    (void)nr;
    (void)nth;
    GridMetrics grid;
    grid.dx = 1000.0;
    grid.dy = 750.0;
    grid.dz.assign(static_cast<std::size_t>(nz), 250.0);
    grid.z_int.resize(static_cast<std::size_t>(nz + 1));
    for (int k = 0; k <= nz; ++k)
    {
        grid.z_int[static_cast<std::size_t>(k)] = 250.0 * k;
    }
    return grid;
}

void require_multiplier_matches_noise(
    const Field3D& actual,
    const Field3D& noise,
    double alpha,
    double xi_max
)
{
    REQUIRE(actual.size() == noise.size());
    for (std::size_t idx = 0; idx < actual.size(); ++idx)
    {
        const double bounded = xi_max * std::tanh(
            static_cast<double>(noise.data()[idx]) / xi_max);
        const double multiplier = std::clamp(1.0 + alpha * bounded, 0.0, 2.0);
        REQUIRE(static_cast<double>(actual.data()[idx]) ==
                Approx(multiplier).margin(2.0e-6));
    }
}

} // namespace

TEST_CASE("legacy chaos globals reference the owning runtime",
          "[physics][chaos][state][modularity]")
{
    REQUIRE(&chaos::chaos_scheme == &chaos::global_chaos_runtime_state.scheme);
    REQUIRE(&chaos::global_chaos_config ==
            &chaos::global_chaos_runtime_state.config);
    REQUIRE(&chaos::global_chaos_diagnostics ==
            &chaos::global_chaos_runtime_state.diagnostics);

    initialize_process_globals();
    chaos::ChaosConfig config;
    config.scheme_id = "none";
    initialize_chaos(global_simulation_state(), config);

    REQUIRE(chaos::global_chaos_runtime_state.scheme != nullptr);
    REQUIRE(chaos::global_chaos_runtime_state.config.scheme_id == "none");
    REQUIRE(chaos::global_chaos_runtime_state.noise_elapsed_s == Approx(0.0));
    REQUIRE_FALSE(
        chaos::global_chaos_runtime_state.pbl_base_tendencies_valid);
}

TEST_CASE("canonical chaos targets resolve identically for public defaults",
          "[physics][chaos][config][defaults]")
{
    initialize_process_globals();
    const std::vector<std::string> canonical_ic = {
        "u", "v", "w", "theta",
    };
    const std::vector<std::string> canonical_full_tendencies = {
        "microphysics", "pbl", "diffusion",
    };
    const std::vector<std::string> canonical_pbl = {"pbl"};
    const std::vector<std::string> empty;

    struct ExpectedTargets
    {
        const char* scheme;
        const std::vector<std::string>* initial;
        const std::vector<std::string>* tendencies;
    };
    for (const ExpectedTargets& expected : {
             ExpectedTargets{"none", &empty, &empty},
             ExpectedTargets{"initial_conditions", &canonical_ic, &empty},
             ExpectedTargets{"boundary_layer", &empty, &canonical_pbl},
             ExpectedTargets{
                 "full_stochastic", &canonical_ic,
                 &canonical_full_tendencies},
         })
    {
        CAPTURE(expected.scheme);
        REQUIRE(chaos::canonical_chaos_initial_condition_targets(
                    expected.scheme) == *expected.initial);
        REQUIRE(chaos::canonical_chaos_tendency_targets(expected.scheme) ==
                *expected.tendencies);
        REQUIRE(chaos::chaos_targets_are_canonical_or_implicit(
            expected.scheme, {}, {}));

        chaos::ChaosConfig config;
        config.scheme_id = expected.scheme;
        initialize_chaos(global_simulation_state(), config);
        REQUIRE(chaos::global_chaos_runtime_state.config.apply_to_ic ==
                *expected.initial);
        REQUIRE(chaos::global_chaos_runtime_state.config.apply_to_tendencies ==
                *expected.tendencies);
    }

    const Field3D qv_before = qv;
    apply_chaos_initial_conditions(global_simulation_state());
    REQUIRE(std::any_of(
        u.data(), u.data() + u.size(), [](float value) { return value != 0.0f; }));
    REQUIRE(std::any_of(
        v.data(), v.data() + v.size(), [](float value) { return value != 0.0f; }));
    REQUIRE(std::any_of(
        w.data(), w.data() + w.size(), [](float value) { return value != 0.0f; }));
    REQUIRE(std::any_of(
        theta.data(), theta.data() + theta.size(),
        [](float value) { return value != 280.0f; }));
    REQUIRE(std::memcmp(
        qv.data(), qv_before.data(), qv.size() * sizeof(float)) == 0);

    REQUIRE_FALSE(chaos::chaos_targets_are_canonical_or_implicit(
        "initial_conditions", {"theta"}, {}));
    REQUIRE_FALSE(chaos::chaos_targets_are_canonical_or_implicit(
        "boundary_layer", {}, {"diffusion"}));
    REQUIRE_FALSE(chaos::chaos_targets_are_canonical_or_implicit(
        "full_stochastic", canonical_ic, {"pbl"}));
}

TEST_CASE("Chaos descriptors expose hook-specific field contracts",
          "[physics][chaos][descriptor]")
{
    const auto& descriptors = chaos::get_chaos_scheme_descriptors();
    REQUIRE(descriptors.size() == 4);

    const auto* ic = tmv::find_scheme_descriptor(descriptors, "ic");
    REQUIRE(ic != nullptr);
    REQUIRE(ic->capabilities.declared);
    REQUIRE(std::find(ic->capabilities.produced_state_fields.begin(),
                      ic->capabilities.produced_state_fields.end(),
                      "perturbed_initial_state")
            != ic->capabilities.produced_state_fields.end());
    REQUIRE(ic->capabilities.staggerings
            == std::vector<std::string>{"collocated", "c_grid"});

    const auto* pbl = tmv::find_scheme_descriptor(descriptors, "pbl");
    REQUIRE(pbl != nullptr);
    REQUIRE(std::find(pbl->capabilities.produced_state_fields.begin(),
                      pbl->capabilities.produced_state_fields.end(),
                      "stochastically_scaled_tendencies")
            != pbl->capabilities.produced_state_fields.end());
    REQUIRE(pbl->capabilities.staggerings
            == std::vector<std::string>{"collocated", "c_grid"});

    const auto* full = tmv::find_scheme_descriptor(
        descriptors, "full_stochastic");
    REQUIRE(full != nullptr);
    REQUIRE(full->capabilities.staggerings
            == std::vector<std::string>{"collocated", "c_grid"});

    const auto* none = tmv::find_scheme_descriptor(descriptors, "none");
    REQUIRE(none != nullptr);
    REQUIRE(none->capabilities.staggerings
            == std::vector<std::string>{"collocated", "c_grid"});
}

TEST_CASE("chaos IC theta and moisture perturbations are EOS-closed before step zero")
{
    initialize_process_globals();

    int local_nr = 2;
    int local_nth = 2;
    int local_nz = 3;
    double local_dr = 100.0;
    double local_dz = 100.0;
    double local_dt = 1.0;
    double local_dtheta = 100.0;
    GridGeometry local_geometry;
    local_geometry.initialize(local_nr, local_nth, local_nz,
                              local_dr, local_dz, local_dtheta,
                              CoordinateSystem::Cartesian,
                              StaggerType::CGrid);

    Field3D local_rho(local_nr, local_nth, local_nz, 1.1f);
    Field3D local_p(local_nr, local_nth, local_nz, 100000.0f);
    Field3D local_u(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_w(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_v(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_tracer(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_theta(local_nr, local_nth, local_nz, 300.0f);
    Field3D local_qv(local_nr, local_nth, local_nz, 0.010f);
    Field3D local_qc(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_qr(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_qi(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_qs(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_qh(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_qg(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_tke(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_dtheta_dt_rad(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_dtheta_dt_pbl(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_dqv_dt_pbl(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_du_dt_pbl(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_dv_dt_pbl(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_dtke_dt_pbl(local_nr, local_nth, local_nz, 0.0f);

    std::vector<double> local_rho0(local_nz, 1.1);
    std::vector<double> local_p0(local_nz, 100000.0);
    std::vector<double> local_qv0(local_nz, 0.010);
    std::vector<double> local_theta0(local_nz, 300.0);
    std::vector<double> local_u0(local_nz, 0.0);
    std::vector<double> local_v0(local_nz, 0.0);
    double local_coriolis = 0.0;
    double local_time = 0.0;
    chaos::ChaosRuntimeState local_chaos_runtime;

    SimulationStateView local_state{
        {local_nr, local_nth, local_nz, local_dr, local_dz,
         local_dt, local_dtheta, local_geometry},
        {local_rho, local_p, local_u, local_w, local_v, local_tracer,
         local_theta, local_qv, local_qc, local_qr, local_qi, local_qs,
         local_qh, local_qg, local_tke},
        {local_rho0, local_p0, local_qv0, local_theta0, local_u0, local_v0},
        {local_dtheta_dt_rad, local_dtheta_dt_pbl, local_dqv_dt_pbl,
         local_du_dt_pbl, local_dv_dt_pbl, local_dtke_dt_pbl},
        local_coriolis,
        local_time,
        {}, nullptr, nullptr, nullptr, &local_chaos_runtime};

    const Field3D theta_before_rejected_perturbation = local_theta;
    const Field3D qv_before_rejected_perturbation = local_qv;
    chaos::ChaosConfig invalid_ic = local_ic_config();
    invalid_ic.taper_id = "none";
    invalid_ic.sigma_ic["qv"] = 1.0;
    set_primary_state_repair_policy(false, true);
    initialize_chaos(local_state, invalid_ic);
    REQUIRE_THROWS_AS(
        apply_chaos_initial_conditions(local_state), std::runtime_error);
    REQUIRE(std::memcmp(local_theta.data(),
                        theta_before_rejected_perturbation.data(),
                        local_theta.size() * sizeof(float)) == 0);
    REQUIRE(std::memcmp(local_qv.data(),
                        qv_before_rejected_perturbation.data(),
                        local_qv.size() * sizeof(float)) == 0);
    REQUIRE_FALSE(local_chaos_runtime.diagnostics.negative_moisture_detected);

    set_primary_state_repair_policy(true, false);
    initialize_chaos(local_state, local_ic_config());
    apply_chaos_initial_conditions(local_state);

    REQUIRE(level_changed_from(local_theta, 0, 300.0f));
    REQUIRE(level_changed_from(local_qv, 0, 0.010f));
    REQUIRE(level_matches(local_theta, 1, 300.0f));
    REQUIRE(level_matches(local_theta, 2, 300.0f));
    REQUIRE(level_matches(local_qv, 1, 0.010f));
    REQUIRE(level_matches(local_qv, 2, 0.010f));

    const std::vector<float> pressure_after_chaos(
        local_p.data(), local_p.data() + local_p.size());
    const std::vector<float> qv_after_chaos(
        local_qv.data(), local_qv.data() + local_qv.size());

    // This is the runtime's zero-step boundary: chaos has changed theta and
    // qv, but no dynamics tendency has run. Pressure is the controlled IC
    // variable, so closure must change only rho. Vapor remains an independent
    // buoyancy variable in the active dry-EOS dynamics.
    apply_initial_thermodynamic_closure(local_state);

    constexpr double kappa =
        physical_constants::gas_constant_dry_air_jkgk /
        physical_constants::specific_heat_cp_jkgk;
    constexpr double gamma =
        physical_constants::specific_heat_cp_jkgk /
        (physical_constants::specific_heat_cp_jkgk -
         physical_constants::gas_constant_dry_air_jkgk);
    double max_relative_pressure_residual = 0.0;
    for (std::size_t idx = 0; idx < local_rho.size(); ++idx)
    {
        REQUIRE(local_p.data()[idx] == pressure_after_chaos[idx]);
        REQUIRE(local_qv.data()[idx] == qv_after_chaos[idx]);

        const double pressure = static_cast<double>(local_p.data()[idx]);
        const double potential_temperature =
            static_cast<double>(local_theta.data()[idx]);
        const double temperature = potential_temperature * std::pow(
            pressure / physical_constants::reference_pressure_pa, kappa);
        const double expected_density = pressure /
            (physical_constants::gas_constant_dry_air_jkgk * temperature);
        REQUIRE(static_cast<double>(local_rho.data()[idx]) ==
                Approx(expected_density).epsilon(2.0e-6));

        const double pressure_from_rho_theta =
            physical_constants::reference_pressure_pa * std::pow(
                static_cast<double>(local_rho.data()[idx]) *
                    physical_constants::gas_constant_dry_air_jkgk *
                    potential_temperature /
                    physical_constants::reference_pressure_pa,
                gamma);
        max_relative_pressure_residual = std::max(
            max_relative_pressure_residual,
            std::abs(pressure_from_rho_theta - pressure) / pressure);
    }
    REQUIRE(max_relative_pressure_residual < 2.0e-6);
    REQUIRE(field_matches(theta, 280.0f));

    const std::vector<float> density_before_invalid_closure(
        local_rho.data(), local_rho.data() + local_rho.size());
    local_p[0][0][0] = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS_AS(
        apply_initial_thermodynamic_closure(local_state), std::runtime_error);
    for (std::size_t idx = 0; idx < local_rho.size(); ++idx)
    {
        REQUIRE(local_rho.data()[idx] == density_before_invalid_closure[idx]);
    }
}

TEST_CASE("chaos PBL tendency dispatch uses explicit SimulationStateView")
{
    initialize_process_globals();

    du_dt_pbl.resize(NR, NTH, NZ, 90.0f);
    dv_dt_pbl.resize(NR, NTH, NZ, 91.0f);
    dtheta_dt_pbl.resize(NR, NTH, NZ, 92.0f);
    dqv_dt_pbl.resize(NR, NTH, NZ, 93.0f);

    int local_nr = 2;
    int local_nth = 2;
    int local_nz = 3;
    double local_dr = 100.0;
    double local_dz = 100.0;
    double local_dt = 1.0;
    double local_dtheta = 100.0;
    GridGeometry local_geometry;
    local_geometry.initialize(local_nr, local_nth, local_nz,
                              local_dr, local_dz, local_dtheta,
                              CoordinateSystem::Cartesian);

    Field3D local_rho(local_nr, local_nth, local_nz, 1.1f);
    Field3D local_p(local_nr, local_nth, local_nz, 100000.0f);
    Field3D local_u(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_w(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_v(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_tracer(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_theta(local_nr, local_nth, local_nz, 300.0f);
    Field3D local_qv(local_nr, local_nth, local_nz, 0.010f);
    Field3D local_qc(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_qr(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_qi(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_qs(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_qh(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_qg(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_tke(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_dtheta_dt_rad(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_dtheta_dt_pbl(local_nr, local_nth, local_nz, 3.0f);
    Field3D local_dqv_dt_pbl(local_nr, local_nth, local_nz, 4.0f);
    Field3D local_du_dt_pbl(local_nr, local_nth, local_nz, 1.0f);
    Field3D local_dv_dt_pbl(local_nr, local_nth, local_nz, 2.0f);
    Field3D local_dtke_dt_pbl(local_nr, local_nth, local_nz, 0.0f);

    std::vector<double> local_rho0(local_nz, 1.1);
    std::vector<double> local_p0(local_nz, 100000.0);
    std::vector<double> local_qv0(local_nz, 0.010);
    std::vector<double> local_theta0(local_nz, 300.0);
    std::vector<double> local_u0(local_nz, 0.0);
    std::vector<double> local_v0(local_nz, 0.0);
    double local_coriolis = 0.0;
    double local_time = 0.0;
    chaos::ChaosRuntimeState local_chaos_runtime;

    SimulationStateView local_state{
        {local_nr, local_nth, local_nz, local_dr, local_dz,
         local_dt, local_dtheta, local_geometry},
        {local_rho, local_p, local_u, local_w, local_v, local_tracer,
         local_theta, local_qv, local_qc, local_qr, local_qi, local_qs,
         local_qh, local_qg, local_tke},
        {local_rho0, local_p0, local_qv0, local_theta0, local_u0, local_v0},
        {local_dtheta_dt_rad, local_dtheta_dt_pbl, local_dqv_dt_pbl,
         local_du_dt_pbl, local_dv_dt_pbl, local_dtke_dt_pbl},
        local_coriolis,
        local_time,
        {}, nullptr, nullptr, nullptr, &local_chaos_runtime};

    tmv::BoundaryLayerRuntimeState local_pbl_runtime;
    local_state.boundary_layer_runtime = &local_pbl_runtime;
    initialize_chaos(local_state, local_pbl_config());
    g_boundary_layer_updated = true;
    apply_chaos_tendencies(local_state);
    g_boundary_layer_updated = false;

    REQUIRE_FALSE(field_matches(local_du_dt_pbl, 1.0f));
    REQUIRE_FALSE(field_matches(local_dv_dt_pbl, 2.0f));
    REQUIRE_FALSE(field_matches(local_dtheta_dt_pbl, 3.0f));
    REQUIRE_FALSE(field_matches(local_dqv_dt_pbl, 4.0f));
    REQUIRE(field_matches(du_dt_pbl, 90.0f));
    REQUIRE(field_matches(dv_dt_pbl, 91.0f));
    REQUIRE(field_matches(dtheta_dt_pbl, 92.0f));
    REQUIRE(field_matches(dqv_dt_pbl, 93.0f));
}

TEST_CASE("production PBL chaos hook perturbs held conserved sources without compounding",
          "[chaos][pbl][runtime][conservation]")
{
    initialize_process_globals();
    auto& pbl_runtime = global_simulation_storage.runtime.boundary_layer;
    pbl_runtime.held_u_force_n.resize(NR, NTH, NZ, 2.0f);
    pbl_runtime.held_v_force_n.resize(NR, NTH, NZ, -3.0f);
    pbl_runtime.held_theta_inventory_rate_kgk_s.resize(
        NR, NTH, NZ, 5.0f);
    pbl_runtime.held_vapor_mass_rate_kg_s.resize(NR, NTH, NZ, -7.0f);
    dtke_dt_pbl.resize(NR, NTH, NZ, 11.0f);
    const std::size_t surface_size =
        static_cast<std::size_t>(NR) * static_cast<std::size_t>(NTH);
    sfc_ustar.assign(surface_size, 4.0f);
    sfc_tau_u.assign(surface_size, -13.0f);
    sfc_tau_v.assign(surface_size, 17.0f);
    sfc_sensible_heat_flux.assign(surface_size, 19.0f);
    sfc_latent_heat_flux.assign(surface_size, 23.0f);
    sfc_moisture_flux.assign(surface_size, 29.0f);

    chaos::ChaosConfig cfg;
    cfg.scheme_id = "full_stochastic";
    cfg.seed = 818181ULL;
    cfg.member_id = 2;
    cfg.dt_chaos = 1.0;
    cfg.Lx = 1000.0;
    cfg.Ly = 1000.0;
    cfg.taper_id = "none";
    cfg.apply_to_tendencies = {"pbl"};
    cfg.alpha_tend["pbl"] = 0.3;
    initialize_chaos(global_simulation_state(), cfg);

    g_boundary_layer_updated = true;
    apply_chaos_tendencies(global_simulation_state());
    g_boundary_layer_updated = false;
    REQUIRE(chaos::global_chaos_runtime_state.pbl_base_tendencies_valid);
    const Field3D first_u_force = pbl_runtime.held_u_force_n;
    const Field3D first_vapor_rate = pbl_runtime.held_vapor_mass_rate_kg_s;
    const Field3D first_tke = dtke_dt_pbl;
    const std::vector<float> first_tau_u = sfc_tau_u;
    const std::vector<float> first_moisture = sfc_moisture_flux;

    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
        {
            const double multiplier =
                static_cast<double>(first_u_force[i][j][0]) / 2.0;
            for (int k = 0; k < NZ; ++k)
            {
                REQUIRE(first_u_force[i][j][k] ==
                        Approx(2.0 * multiplier));
                REQUIRE(first_vapor_rate[i][j][k] ==
                        Approx(-7.0 * multiplier));
                REQUIRE(first_tke[i][j][k] ==
                        Approx(11.0 * multiplier));
            }
            const std::size_t index =
                static_cast<std::size_t>(i) * static_cast<std::size_t>(NTH) +
                static_cast<std::size_t>(j);
            REQUIRE(first_tau_u[index] == Approx(-13.0 * multiplier));
            REQUIRE(first_moisture[index] == Approx(29.0 * multiplier));
        }

    // A second application at the same stochastic epoch must restore the
    // deterministic closure publication before applying the factor again.
    apply_chaos_tendencies(global_simulation_state());
    for (std::size_t index = 0; index < first_u_force.size(); ++index)
    {
        REQUIRE(pbl_runtime.held_u_force_n.data()[index] ==
                first_u_force.data()[index]);
        REQUIRE(pbl_runtime.held_vapor_mass_rate_kg_s.data()[index] ==
                first_vapor_rate.data()[index]);
        REQUIRE(dtke_dt_pbl.data()[index] == first_tke.data()[index]);
    }
    REQUIRE(sfc_tau_u == first_tau_u);
    REQUIRE(sfc_moisture_flux == first_moisture);
}

TEST_CASE("ChaosRNG normal streams are independent of odd-sized call history")
{
    // Nine values deliberately leave a cached spare in implementations that
    // reuse one std::normal_distribution across independently seeded streams.
    Field3D unrelated_odd_field(3, 3, 1);
    Field3D after_unrelated_stream(3, 3, 1);
    Field3D directly_addressed(3, 3, 1);

    chaos::ChaosRNG with_history(24681357ULL, 5);
    with_history.fill_field_normal(
        unrelated_odd_field, 7001ULL, 3ULL, "unrelated");
    with_history.fill_field_normal(
        after_unrelated_stream, 8001ULL, 9ULL, "target");

    chaos::ChaosRNG direct(24681357ULL, 5);
    direct.fill_field_normal(
        directly_addressed, 8001ULL, 9ULL, "target");

    REQUIRE(after_unrelated_stream.size() == directly_addressed.size());
    for (std::size_t idx = 0; idx < directly_addressed.size(); ++idx)
    {
        REQUIRE(after_unrelated_stream.data()[idx] ==
                directly_addressed.data()[idx]);
    }

    const auto odd_prefix = with_history.normal_stream(9, 9001ULL, 1ULL);
    (void)odd_prefix;
    const auto after_prefix = with_history.normal_stream(9, 9002ULL, 2ULL);
    chaos::ChaosRNG direct_vector(24681357ULL, 5);
    const auto direct_values = direct_vector.normal_stream(9, 9002ULL, 2ULL);
    REQUIRE(after_prefix == direct_values);
    REQUIRE(with_history.normal(9100ULL, 4ULL) ==
            direct_vector.normal(9100ULL, 4ULL));
}

TEST_CASE("ChaosRNG process and timestep keys cannot alias adjacent blocks")
{
    Field3D microphysics_step_one(3, 3, 1);
    Field3D pbl_step_zero(3, 3, 1);
    chaos::ChaosRNG rng(97531ULL, 2);

    // A flattened `stream_key + timestep` maps both of these tuples to 4102.
    rng.fill_field_normal(microphysics_step_one, 4101ULL, 1ULL);
    rng.fill_field_normal(pbl_step_zero, 4102ULL, 0ULL);

    bool differs = false;
    for (std::size_t idx = 0; idx < microphysics_step_one.size(); ++idx)
    {
        differs = differs ||
            microphysics_step_one.data()[idx] != pbl_step_zero.data()[idx];
    }
    REQUIRE(differs);

    Field3D repeated_microphysics(3, 3, 1);
    chaos::ChaosRNG repeated_rng(97531ULL, 2);
    repeated_rng.fill_field_normal(
        repeated_microphysics, 4101ULL, 1ULL);
    for (std::size_t idx = 0; idx < repeated_microphysics.size(); ++idx)
    {
        REQUIRE(repeated_microphysics.data()[idx] ==
                microphysics_step_one.data()[idx]);
    }
}

TEST_CASE("chaos noise cadence is independent of fresh SGS tendency cadence")
{
    initialize_process_globals();
    chaos::ChaosConfig cfg;
    cfg.scheme_id = "full_stochastic";
    cfg.dt_chaos = 2.5;
    cfg.apply_to_tendencies = {"diffusion"};
    initialize_chaos(global_simulation_state(), cfg);

    auto counting = std::make_unique<CountingChaosScheme>();
    CountingChaosScheme* const observed = counting.get();
    chaos::chaos_scheme = std::move(counting);

    step_chaos_noise(global_simulation_state(), 1.0);
    step_chaos_noise(global_simulation_state(), 1.0);
    REQUIRE(observed->noise_steps == 0);
    step_chaos_noise(global_simulation_state(), 0.5);
    REQUIRE(observed->noise_steps == 1);
    REQUIRE(observed->last_noise_dt == Approx(2.5));
    step_chaos_noise(global_simulation_state(), 5.0);
    REQUIRE(observed->noise_steps == 3);

    TurbulenceTendencies tendencies;
    tendencies.dudt_sgs.resize(NR, NTH, NZ, 3.0f);
    tendencies.resident_source_unchanged = true;
    apply_chaos_to_turbulence_tendencies(
        global_simulation_state(), tendencies);
    REQUIRE_FALSE(tendencies.resident_source_unchanged);
    REQUIRE(field_matches(tendencies.dudt_sgs, 6.0f));

    // A later SGS refresh supplies a new deterministic buffer. It receives
    // the current multiplier exactly once even though no noise update is due.
    tendencies.dudt_sgs.fill(3.0f);
    apply_chaos_to_turbulence_tendencies(
        global_simulation_state(), tendencies);
    REQUIRE(field_matches(tendencies.dudt_sgs, 6.0f));
    REQUIRE(observed->tendency_applications == 2);
}

TEST_CASE("stochastic SGS diffusion perturbs coefficients before divergence",
          "[chaos][turbulence][conservative][coefficients]")
{
    initialize_process_globals();
    chaos::ChaosConfig cfg;
    cfg.scheme_id = "full_stochastic";
    cfg.seed = 808080ULL;
    cfg.member_id = 3;
    cfg.apply_to_tendencies = {"diffusion"};
    cfg.alpha_tend["diffusion"] = 0.25;
    cfg.taper_id = "none";
    cfg.xi_max = 2.0;
    initialize_chaos(global_simulation_state(), cfg);

    auto transport = std::make_unique<CoefficientTransportScheme>();
    CoefficientTransportScheme* observed = transport.get();
    observed->nu.resize(NR, NTH, NZ, 1.0f);
    observed->kth.resize(NR, NTH, NZ, 2.0f);
    observed->kq.resize(NR, NTH, NZ, 3.0f);
    auto& turbulence = global_simulation_storage.runtime.turbulence;
    turbulence.scheme = std::move(transport);
    turbulence.config.scheme_id = "smagorinsky";
    auto& tendencies = turbulence.held_tendencies;
    tendencies.dudt_sgs.resize(NR, NTH, NZ, 4.0f);
    tendencies.dvdt_sgs.resize(NR, NTH, NZ, 4.0f);
    tendencies.dwdt_sgs.resize(NR, NTH, NZ, 4.0f);
    tendencies.dthetadt_sgs.resize(NR, NTH, NZ, 4.0f);
    tendencies.dqvdt_sgs.resize(NR, NTH, NZ, 4.0f);
    tendencies.dtkedt_sgs.resize(NR, NTH, NZ, 4.0f);
    turbulence.realized_transport_tendency_current = true;

    apply_chaos_to_turbulence_tendencies(
        global_simulation_state(), tendencies);

    REQUIRE(observed->rebuilds == 1);
    REQUIRE(turbulence.realized_transport_tendency_current);
    REQUIRE(turbulence.stochastic_transport_applied_to_current);
    REQUIRE_FALSE(tendencies.resident_source_unchanged);
    bool coefficient_changed = false;
    for (std::size_t index = 0; index < observed->nu.size(); ++index)
    {
        coefficient_changed = coefficient_changed
            || observed->nu.data()[index] != 1.0f;
        REQUIRE(observed->nu.data()[index] >= 0.0f);
        REQUIRE(observed->kth.data()[index] >= 0.0f);
        REQUIRE(observed->kq.data()[index] >= 0.0f);
    }
    REQUIRE(coefficient_changed);
    double momentum_sum = 0.0;
    for (std::size_t index = 0; index < tendencies.dudt_sgs.size(); ++index)
    {
        momentum_sum += tendencies.dudt_sgs.data()[index];
        REQUIRE(tendencies.dthetadt_sgs.data()[index] == 0.0f);
        REQUIRE(tendencies.dqvdt_sgs.data()[index] == 0.0f);
    }
    REQUIRE(momentum_sum == Approx(0.0).margin(1.0e-7));
    REQUIRE_THROWS_WITH(
        apply_chaos_to_turbulence_tendencies(
            global_simulation_state(), tendencies),
        Catch::Contains("already applied"));
    REQUIRE(observed->rebuilds == 1);
}

TEST_CASE("terrain chaos uses mapped physical geometry and local AGL taper",
          "[chaos][runtime][terrain][state][geometry]")
{
    constexpr int nr = 4, nth = 4, nz = 3;
    SimulationStorageState storage{{nr, nth, nz, 100.0, 80.0, 1.0, 120.0}};
    storage.grid.geometry.initialize(
        nr, nth, nz, 100.0, 80.0, 120.0,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    auto& terrain = storage.runtime.terrain;
    terrain.config.scheme_id = "bell";
    terrain.metrics.z.resize(nr, nth, nz, 0.0);
    terrain.metrics.z_face.resize(nr, nth, nz + 1, 0.0);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
        {
            double lower = 4.0 * i;
            terrain.metrics.z_face(i, j, 0) = lower;
            for (int k = 0; k < nz; ++k)
            {
                const double upper = lower + 55.0 + i + k;
                terrain.metrics.z(i, j, k) = 0.5 * (lower + upper);
                terrain.metrics.z_face(i, j, k + 1) = upper;
                lower = upper;
            }
        }

    auto& runtime = storage.runtime.chaos;
    runtime.config.scheme_id = "full_stochastic";
    runtime.config.dt_chaos = 1.0;
    auto counting = std::make_unique<CountingChaosScheme>();
    CountingChaosScheme* observed = counting.get();
    runtime.scheme = std::move(counting);
    runtime.noise_elapsed_s = 0.25;
    SimulationStateView state = make_simulation_state_view(storage);

    step_chaos_noise(state, 0.75);
    REQUIRE(runtime.noise_elapsed_s == Approx(0.0));
    REQUIRE(observed->noise_steps == 1);
    REQUIRE(observed->last_noise_dt == Approx(1.0));
    REQUIRE(observed->last_grid_used_terrain);
    REQUIRE(runtime.grid_metrics_scratch.terrain_metrics == &terrain.metrics);

    Field3D tapered(nr, nth, nz, 1.0f);
    chaos::apply_vertical_taper(
        tapered, runtime.grid_metrics_scratch, "cosine", 60.0, 100.0);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
        {
            REQUIRE(tapered(i, j, 0) == 1.0f);
            REQUIRE(tapered(i, j, 1) > 0.0f);
            REQUIRE(tapered(i, j, 1) < 1.0f);
            REQUIRE(tapered(i, j, 2) == 0.0f);
        }

    const Field3D unchanged = tapered;
    terrain.metrics.z(2, 2, 1) =
        std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_WITH(
        chaos::apply_vertical_taper(
            tapered, runtime.grid_metrics_scratch, "cosine", 60.0, 100.0),
        Catch::Contains("positive finite above-ground heights"));
    REQUIRE(tapered.size() == unchanged.size());
    REQUIRE(std::equal(
        tapered.data(), tapered.data() + tapered.size(), unchanged.data()));
}

TEST_CASE("A-grid initial-condition chaos preserves every closure plane",
          "[chaos][runtime][agrid][physical-domain][qualified]")
{
    constexpr int n = 5;
    NR = n;
    NTH = n;
    NZ = n;
    dr = 100.0;
    dtheta = 120.0;
    dz = 80.0;
    global_grid_geometry.initialize(
        n, n, n, dr, dz, dtheta, CoordinateSystem::Cartesian,
        StaggerType::Collocated);
    global_grid_metrics = {};
    global_grid_metrics.dx = dr;
    global_grid_metrics.dy = dtheta;
    global_grid_metrics.dz.assign(n, dz);
    global_grid_metrics.z_int = {0.0, 80.0, 160.0, 240.0, 320.0, 400.0};
    global_grid_metrics.staggering = StaggerType::Collocated;

    p.resize(n, n, n, 100000.0f);
    rho.resize(n, n, n, 1.0f);
    theta.resize(n, n, n, 280.0f);
    qv.resize(n, n, n, 0.005f);
    u.resize(n, n, n, 0.0f);
    v.resize(n, n, n, 0.0f);
    w.resize(n, n, n, 0.0f);
    qc.resize(n, n, n, 0.0f);
    qr.resize(n, n, n, 0.0f);
    qi.resize(n, n, n, 0.0f);
    qs.resize(n, n, n, 0.0f);
    qg.resize(n, n, n, 0.0f);
    qh.resize(n, n, n, 0.0f);

    for (const std::string& scheme_id : {
             std::string("initial_conditions"),
             std::string("full_stochastic")})
    {
        DYNAMIC_SECTION("scheme=" << scheme_id)
        {
            std::fill(theta.data(), theta.data() + theta.size(), 280.0f);
            chaos::ChaosConfig cfg;
            cfg.scheme_id = scheme_id;
            cfg.seed = 918273;
            cfg.member_id = 4;
            cfg.Lx = 200.0;
            cfg.Ly = 240.0;
            cfg.apply_to_ic = {"theta"};
            cfg.sigma_ic["theta"] = 0.5;
            cfg.taper_id = "none";
            initialize_chaos(global_simulation_state(), cfg);
            apply_chaos_initial_conditions(global_simulation_state());

            bool interior_changed = false;
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    for (int k = 0; k < n; ++k)
                    {
                        const bool physical = i > 0 && i < n - 1
                            && j > 0 && j < n - 1
                            && k > 0 && k < n - 1;
                        if (physical)
                            interior_changed = interior_changed
                                || theta(i, j, k) != 280.0f;
                        else
                            REQUIRE(theta(i, j, k) == 280.0f);
                    }
            REQUIRE(interior_changed);
        }
    }
}

TEST_CASE("A-grid evolving chaos scales only physical process ownership",
          "[chaos][runtime][agrid][tendencies][closure]")
{
    constexpr int n = 7;
    GridMetrics grid = stochastic_test_grid(n, n, n);
    grid.staggering = StaggerType::Collocated;
    const PhysicalCellDomain domain = physical_scalar_domain(
        n, n, n, StaggerType::Collocated);

    for (const std::string& scheme_id : {
             std::string("boundary_layer"),
             std::string("full_stochastic")})
    {
        DYNAMIC_SECTION("scheme=" << scheme_id)
        {
            chaos::ChaosConfig cfg;
            cfg.scheme_id = scheme_id;
            cfg.seed = 783421;
            cfg.member_id = 11;
            cfg.Lx = 250.0;
            cfg.Ly = 250.0;
            cfg.taper_id = "none";
            cfg.xi_max = 2.0;
            cfg.alpha_tend["pbl"] = 0.35;
            cfg.alpha_tend["microphysics"] = 0.3;
            cfg.alpha_tend["diffusion"] = 0.25;
            auto scheme = chaos::create_chaos_scheme(scheme_id);
            scheme->initialize(cfg, grid);

            Field3D pbl_force(n, n, n, 1.0f);
            Field3D microphysics(n, n, n, 2.0f);
            Field3D diffusion(n, n, n, 3.0f);
            std::vector<float> tau(
                static_cast<std::size_t>(n) * n, 4.0f);
            chaos::ChaosTendencies tendencies;
            tendencies.pbl_u_force_n = &pbl_force;
            tendencies.pbl_tau_u = &tau;
            if (scheme_id == "full_stochastic")
            {
                tendencies.dtheta_micro_dt = &microphysics;
                tendencies.dtheta_diff_dt = &diffusion;
            }
            chaos::ChaosStateView state_view;
            state_view.grid = &grid;
            scheme->apply_tendencies(
                cfg, grid, state_view, tendencies, nullptr);

            bool pbl_changed = false;
            bool micro_changed = false;
            bool diffusion_changed = false;
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    for (int k = 0; k < n; ++k)
                    {
                        const bool physical =
                            i >= domain.i_begin && i < domain.i_end &&
                            j >= domain.j_begin && j < domain.j_end &&
                            k >= domain.k_begin && k < domain.k_end;
                        if (physical)
                        {
                            pbl_changed = pbl_changed ||
                                pbl_force(i, j, k) != 1.0f;
                            micro_changed = micro_changed ||
                                microphysics(i, j, k) != 2.0f;
                            diffusion_changed = diffusion_changed ||
                                diffusion(i, j, k) != 3.0f;
                        }
                        else
                        {
                            REQUIRE(pbl_force(i, j, k) == 1.0f);
                            REQUIRE(microphysics(i, j, k) == 2.0f);
                            REQUIRE(diffusion(i, j, k) == 3.0f);
                        }
                    }
            REQUIRE(pbl_changed);
            if (scheme_id == "full_stochastic")
            {
                REQUIRE(micro_changed);
                REQUIRE(diffusion_changed);
            }
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                {
                    const bool physical = i >= domain.i_begin &&
                        i < domain.i_end && j >= domain.j_begin &&
                        j < domain.j_end;
                    const float value = tau[
                        static_cast<std::size_t>(i) * n +
                        static_cast<std::size_t>(j)];
                    if (!physical) REQUIRE(value == 4.0f);
                }
        }
    }
}

TEST_CASE("interleaved chaos states own schemes and stochastic clocks",
          "[chaos][runtime][state][sci020]")
{
    initialize_process_globals();
    chaos::ChaosRuntimeState local_runtime;
    SimulationStateView local_state = global_simulation_state();
    local_state.chaos_runtime = &local_runtime;

    chaos::ChaosConfig local_cfg;
    local_cfg.scheme_id = "full_stochastic";
    local_cfg.dt_chaos = 2.0;
    initialize_chaos(local_state, local_cfg);
    auto local_counting = std::make_unique<CountingChaosScheme>();
    CountingChaosScheme* const local_observed = local_counting.get();
    local_runtime.scheme = std::move(local_counting);

    chaos::ChaosConfig global_cfg = local_cfg;
    global_cfg.dt_chaos = 5.0;
    initialize_chaos(global_simulation_state(), global_cfg);
    auto global_counting = std::make_unique<CountingChaosScheme>();
    CountingChaosScheme* const global_observed = global_counting.get();
    chaos::chaos_scheme = std::move(global_counting);

    step_chaos_noise(local_state, 1.0);
    step_chaos_noise(global_simulation_state(), 2.0);
    REQUIRE(local_observed->noise_steps == 0);
    REQUIRE(global_observed->noise_steps == 0);

    step_chaos_noise(local_state, 1.0);
    REQUIRE(local_observed->noise_steps == 1);
    REQUIRE(global_observed->noise_steps == 0);

    step_chaos_noise(global_simulation_state(), 3.0);
    REQUIRE(global_observed->noise_steps == 1);
    REQUIRE(local_runtime.config.dt_chaos == Approx(2.0));
    REQUIRE(local_runtime.noise_elapsed_s == Approx(0.0));
}

TEST_CASE("caller-owned chaos geometry reuses scratch and snapshots omit it",
          "[chaos][runtime][state][allocation]")
{
    constexpr int nr = 3, nth = 4, nz = 5;
    SimulationStorageState storage{
        {nr, nth, nz, 150.0, 75.0, 1.0, 200.0}};
    storage.grid.geometry.initialize(
        nr, nth, nz, 150.0, 75.0, 200.0,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    auto& runtime = storage.runtime.chaos;
    runtime.config.scheme_id = "counting";
    runtime.config.dt_chaos = 1.0;
    runtime.scheme = std::make_unique<CountingChaosScheme>();
    SimulationStateView state = make_simulation_state_view(storage);

    step_chaos_noise(state, 1.0);
    const double* dz_data = runtime.grid_metrics_scratch.dz.data();
    const double* z_int_data = runtime.grid_metrics_scratch.z_int.data();
    REQUIRE(dz_data != nullptr);
    REQUIRE(z_int_data != nullptr);
    REQUIRE(runtime.grid_metrics_scratch.dz.size()
            == static_cast<std::size_t>(nz));
    REQUIRE(runtime.grid_metrics_scratch.z_int.size()
            == static_cast<std::size_t>(nz + 1));

    step_chaos_noise(state, 1.0);
    REQUIRE(runtime.grid_metrics_scratch.dz.data() == dz_data);
    REQUIRE(runtime.grid_metrics_scratch.z_int.data() == z_int_data);

    runtime.scheme.reset();
    const chaos::ChaosRuntimeState snapshot(runtime);
    REQUIRE(snapshot.grid_metrics_scratch.dz.empty());
    REQUIRE(snapshot.grid_metrics_scratch.z_int.empty());
}

TEST_CASE("chaos portable state restores exact stochastic continuation",
          "[chaos][runtime][restart][mod03]")
{
    initialize_process_globals();
    chaos::ChaosRuntimeState runtime;
    SimulationStateView state = global_simulation_state();
    state.chaos_runtime = &runtime;

    chaos::ChaosConfig cfg;
    cfg.scheme_id = "full_stochastic";
    cfg.seed = 8675309;
    cfg.member_id = 4;
    cfg.dt_chaos = 1.0;
    cfg.tau_t = 8.0;
    cfg.Lx = 1000.0;
    cfg.Ly = 1000.0;
    cfg.filter_id = "recursive_gaussian";
    cfg.apply_to_tendencies = {"microphysics"};
    cfg.alpha_tend["microphysics"] = 0.5;
    initialize_chaos(state, cfg);

    step_chaos_noise(state, 1.0);
    const chaos::ChaosRuntimeState snapshot(runtime);
    const PortableSchemeState portable = runtime.scheme->portable_state();

    auto make_tendency = [] { return Field3D(NR, NTH, NZ, 1.0f); };
    Field3D theta_a = make_tendency();
    Field3D qv_a = make_tendency();
    Field3D qc_a = make_tendency();
    Field3D qr_a = make_tendency();
    Field3D qi_a = make_tendency();
    Field3D qs_a = make_tendency();
    Field3D qg_a = make_tendency();
    Field3D qh_a = make_tendency();

    step_chaos_noise(state, 1.0);
    apply_chaos_to_microphysics_tendencies(
        state, theta_a, qv_a, qc_a, qr_a, qi_a, qs_a, qg_a, qh_a);

    runtime = chaos::ChaosRuntimeState{};
    runtime.config = snapshot.config;
    runtime.noise_elapsed_s = snapshot.noise_elapsed_s;
    runtime.scheme = chaos::create_chaos_scheme(cfg.scheme_id);
    GridMetrics restore_grid;
    restore_grid.dx = dr;
    restore_grid.dy = dtheta;
    restore_grid.dz.assign(NZ, dz);
    restore_grid.z_int.resize(static_cast<std::size_t>(NZ + 1));
    for (int k = 0; k <= NZ; ++k)
        restore_grid.z_int[static_cast<std::size_t>(k)] = k * dz;
    runtime.scheme->initialize(runtime.config, restore_grid);
    std::string restore_error;
    REQUIRE(runtime.scheme->restore_portable_state(portable, restore_error));
    Field3D theta_b = make_tendency();
    Field3D qv_b = make_tendency();
    Field3D qc_b = make_tendency();
    Field3D qr_b = make_tendency();
    Field3D qi_b = make_tendency();
    Field3D qs_b = make_tendency();
    Field3D qg_b = make_tendency();
    Field3D qh_b = make_tendency();
    step_chaos_noise(state, 1.0);
    apply_chaos_to_microphysics_tendencies(
        state, theta_b, qv_b, qc_b, qr_b, qi_b, qs_b, qg_b, qh_b);

    const Field3D* first[] = {
        &theta_a, &qv_a, &qc_a, &qr_a, &qi_a, &qs_a, &qg_a, &qh_a};
    const Field3D* restarted[] = {
        &theta_b, &qv_b, &qc_b, &qr_b, &qi_b, &qs_b, &qg_b, &qh_b};
    for (std::size_t field = 0; field < std::size(first); ++field)
    {
        REQUIRE(first[field]->size() == restarted[field]->size());
        for (std::size_t idx = 0; idx < first[field]->size(); ++idx)
        {
            REQUIRE(first[field]->data()[idx] ==
                    restarted[field]->data()[idx]);
        }
    }
    REQUIRE(runtime.noise_elapsed_s == Approx(0.0));
}

TEST_CASE("correlated innovations retain the requested AR1 temporal coefficient")
{
    constexpr int nr = 20;
    constexpr int nth = 18;
    constexpr int nz = 1;
    constexpr double requested_rho = 0.72;
    constexpr int realizations = 256;
    constexpr double dx = 1000.0;
    constexpr double dy = 750.0;

    chaos::ChaosRNG rng(987654ULL, 11);
    auto filter = chaos::create_correlation_filter(
        "recursive_gaussian", 3500.0, 2800.0);
    Field3D previous = make_correlated_noise(
        rng, nr, nth, nz, 8001ULL, "previous", *filter, dx, dy);

    double mean_regression_coefficient = 0.0;
    double mean_realized_variance = 0.0;
    for (int sample = 0; sample < realizations; ++sample)
    {
        Field3D innovation = make_correlated_noise(
            rng,
            nr,
            nth,
            nz,
            9000ULL + static_cast<uint64_t>(sample),
            "innovation",
            *filter,
            dx,
            dy);
        Field3D current(nr, nth, nz);
        chaos::combine_ar1_3d(
            current, previous, innovation, requested_rho);

        double covariance = 0.0;
        for (std::size_t idx = 0; idx < current.size(); ++idx)
        {
            covariance += static_cast<double>(previous.data()[idx]) *
                          static_cast<double>(current.data()[idx]);
        }
        covariance /= static_cast<double>(current.size());
        mean_regression_coefficient += covariance;

        const auto stats = chaos::compute_field_statistics(current);
        mean_realized_variance += std::get<1>(stats);
    }

    mean_regression_coefficient /= static_cast<double>(realizations);
    mean_realized_variance /= static_cast<double>(realizations);
    REQUIRE(mean_regression_coefficient == Approx(requested_rho).margin(0.03));
    REQUIRE(mean_realized_variance == Approx(1.0).margin(0.05));
}

TEST_CASE("chaos correlation filters reject semantic substitutions",
          "[chaos][filter][config-truth]")
{
    REQUIRE_THROWS_AS(
        chaos::create_correlation_filter("unknown_filter", 1000.0, 1000.0),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        chaos::create_correlation_filter("recursive_gaussian", 0.0, 1000.0),
        std::invalid_argument);
}

TEST_CASE("Earth-correlated controls are path-independent across translation",
          "[chaos][earth_coordinates][ar1][restart][D5]")
{
    constexpr int nr = 31;
    constexpr int nth = 27;
    constexpr int nz = 3;
    constexpr int shift_i = 5;
    constexpr int shift_j = -4;
    constexpr double rho = 0.83;
    constexpr std::uint64_t epochs = 7;

    chaos::EarthCorrelatedNoise moved;
    moved.configure(
        1234567ULL, 9, 7001ULL, 7101ULL,
        900.0, 700.0, 200.0, 250.0);
    moved.initialize_domain(nr, nth, nz);
    for (std::uint64_t epoch = 0; epoch < epochs; ++epoch)
        moved.advance(rho, epoch);
    Field3D before;
    moved.render_current(before);
    moved.translate(shift_i, shift_j, epochs, rho);
    Field3D after;
    Field3D after_previous;
    moved.render_current(after);
    moved.render_previous(after_previous);

    chaos::EarthCorrelatedNoise direct;
    direct.configure(
        1234567ULL, 9, 7001ULL, 7101ULL,
        900.0, 700.0, 200.0, 250.0);
    std::string error;
    REQUIRE(direct.restore_uninitialized_origin(
        shift_i, shift_j, error));
    direct.initialize_domain(nr, nth, nz);
    for (std::uint64_t epoch = 0; epoch < epochs; ++epoch)
        direct.advance(rho, epoch);
    Field3D direct_current;
    Field3D direct_previous;
    direct.render_current(direct_current);
    direct.render_previous(direct_previous);

    REQUIRE(after.size() == direct_current.size());
    for (std::size_t index = 0; index < after.size(); ++index)
    {
        REQUIRE(after.data()[index] == direct_current.data()[index]);
        REQUIRE(after_previous.data()[index]
                == direct_previous.data()[index]);
    }
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
        {
            const int old_i = i + shift_i;
            const int old_j = j + shift_j;
            if (old_i < 0 || old_i >= nr || old_j < 0 || old_j >= nth)
                continue;
            for (int k = 0; k < nz; ++k)
                REQUIRE(after(i, j, k) == before(old_i, old_j, k));
        }
    REQUIRE(std::string(chaos::EarthCorrelatedNoise::algorithm_id()) ==
            "earth_control_lattice_cubic_ar1_v1");
}

TEST_CASE("Earth-correlated controls reconstruct continuous moving origins",
          "[chaos][earth_coordinates][continuous-motion][restart]")
{
    constexpr int nr = 12;
    constexpr int nth = 10;
    constexpr int nz = 3;
    constexpr double rho = 0.83;
    constexpr std::uint64_t epoch = 4;
    chaos::EarthCorrelatedNoise noise;
    configure_earth_noise(noise);
    noise.initialize_domain(nr, nth, nz);
    for (std::uint64_t step = 0; step < epoch; ++step)
        noise.advance(rho, step);

    Field3D initial;
    noise.render_current(initial);
    noise.set_continuous_earth_origin(50.0, -62.5, epoch, rho);
    Field3D fractional;
    noise.render_current(fractional);
    REQUIRE_FALSE(bitwise_field_equal(initial, fractional));
    REQUIRE(noise.earth_origin_i_cells() == 0);
    REQUIRE(noise.earth_origin_j_cells() == -1);
    REQUIRE(noise.continuous_offset_x_m() == Approx(50.0));
    REQUIRE(noise.continuous_offset_y_m() == Approx(187.5));

    chaos::EarthCorrelatedNoise restored;
    configure_earth_noise(restored);
    std::string error;
    REQUIRE(restored.restore(
        noise.current_controls(), noise.previous_controls(), nr, nth, nz,
        noise.earth_origin_i_cells(), noise.earth_origin_j_cells(),
        noise.control_i_min(), noise.control_j_min(),
        noise.continuous_offset_x_m(), noise.continuous_offset_y_m(), error));
    Field3D restored_field;
    restored.render_current(restored_field);
    REQUIRE(bitwise_field_equal(fractional, restored_field));

    chaos::EarthCorrelatedNoise direct;
    configure_earth_noise(direct);
    direct.initialize_domain(nr, nth, nz);
    for (std::uint64_t step = 0; step < epoch; ++step)
        direct.advance(rho, step);
    direct.set_continuous_earth_origin(1050.0, -62.5, epoch, rho);
    noise.set_continuous_earth_origin(1050.0, -62.5, epoch, rho);
    REQUIRE(earth_noise_state_equal(noise, direct));

    const chaos::EarthCorrelatedNoise before_invalid = noise;
    REQUIRE_THROWS_AS(
        noise.set_continuous_earth_origin(
            std::numeric_limits<double>::quiet_NaN(), 0.0, epoch, rho),
        std::invalid_argument);
    REQUIRE(earth_noise_state_equal(noise, before_invalid));
}

TEST_CASE("native Earth-control dispatch preserves near-unit model correlation",
          "[chaos][earth_coordinates][native][transaction]")
{
    chaos::EarthCorrelatedNoise noise;
    configure_earth_noise(noise);
    noise.initialize_domain(17, 15, 3);
    const chaos::EarthCorrelatedNoise before = noise;
    FailingEarthControlBackend backend(FailingEarthControlBackend::Behavior::Reject);
    noise.configure_native_evolution(&backend, true);
    for (double rho : {1.0 - 1e-8, 1.0 - 1e-10, std::nextafter(1.0, 0.0)})
    {
        REQUIRE_THROWS_AS(noise.advance(rho, 0), std::runtime_error);
        REQUIRE(backend.last_rho() == rho);
        REQUIRE(earth_noise_state_equal(noise, before));
    }
}

TEST_CASE("required native Earth-control failures preserve complete live state",
          "[chaos][earth_coordinates][native][transaction]")
{
    constexpr double rho = 0.83;
    constexpr std::uint64_t failed_epoch = 3;
    for (const auto behavior : {
             FailingEarthControlBackend::Behavior::Reject,
             FailingEarthControlBackend::Behavior::Throw})
    {
        chaos::EarthCorrelatedNoise noise;
        configure_earth_noise(noise);
        noise.initialize_domain(17, 15, 3);
        noise.advance(rho, 0);
        noise.advance(rho, 1);
        noise.advance(rho, 2);
        const chaos::EarthCorrelatedNoise before = noise;

        FailingEarthControlBackend backend(behavior);
        noise.configure_native_evolution(&backend, true);
        if (behavior == FailingEarthControlBackend::Behavior::Reject)
            REQUIRE_THROWS_WITH(
                noise.advance(rho, failed_epoch),
                "native Earth-control advance dispatch failed");
        else
            REQUIRE_THROWS_WITH(
                noise.advance(rho, failed_epoch),
                "injected Earth-control dispatch failure");
        REQUIRE(earth_noise_state_equal(noise, before));

        noise.configure_native_evolution(nullptr, false);
        noise.advance(rho, failed_epoch);
        chaos::EarthCorrelatedNoise expected = before;
        expected.advance(rho, failed_epoch);
        REQUIRE(earth_noise_state_equal(noise, expected));
    }
}

TEST_CASE("native initialization and reconstruction failures are strongly atomic",
          "[chaos][earth_coordinates][native][transaction]")
{
    chaos::EarthCorrelatedNoise source;
    configure_earth_noise(source);
    source.initialize_domain(19, 17, 3);
    for (std::uint64_t epoch = 0; epoch < 4; ++epoch)
        source.advance(0.81, epoch);

    for (const auto behavior : {
             FailingEarthControlBackend::Behavior::SucceedOnceThenReject,
             FailingEarthControlBackend::Behavior::SucceedOnceThenThrow})
    {
        chaos::EarthCorrelatedNoise destination;
        configure_earth_noise(destination);
        destination.initialize_domain(11, 9, 3);
        const chaos::EarthCorrelatedNoise before = destination;
        FailingEarthControlBackend backend(behavior);
        destination.configure_native_evolution(&backend, true);
        if (behavior == FailingEarthControlBackend::Behavior::SucceedOnceThenReject)
            REQUIRE_THROWS_WITH(
                destination.initialize_domain_from(
                    source, 4, -3, 13, 12, 3, 4, 4, 0.81),
                "native nested Earth-control reconstruction dispatch failed");
        else
            REQUIRE_THROWS_WITH(
                destination.initialize_domain_from(
                    source, 4, -3, 13, 12, 3, 4, 4, 0.81),
                "injected Earth-control dispatch failure");
        REQUIRE(backend.dispatch_count() == 2);
        REQUIRE(earth_noise_state_equal(destination, before));
    }

    for (const auto behavior : {
             FailingEarthControlBackend::Behavior::Reject,
             FailingEarthControlBackend::Behavior::Throw})
    {
        chaos::EarthCorrelatedNoise destination;
        configure_earth_noise(destination);
        destination.initialize_domain(11, 9, 3);
        const chaos::EarthCorrelatedNoise before = destination;
        FailingEarthControlBackend backend(behavior);
        destination.configure_native_evolution(&backend, true);
        if (behavior == FailingEarthControlBackend::Behavior::Reject)
            REQUIRE_THROWS_WITH(
                destination.initialize_domain(13, 12, 3),
                "native Earth-control initialization dispatch failed");
        else
            REQUIRE_THROWS_WITH(
                destination.initialize_domain(13, 12, 3),
                "injected Earth-control dispatch failure");
        REQUIRE(earth_noise_state_equal(destination, before));
    }
}

TEST_CASE("native translation failures preserve live and prepared windows",
          "[chaos][earth_coordinates][native][translation][transaction]")
{
    chaos::EarthCorrelatedNoise noise;
    configure_earth_noise(noise);
    noise.initialize_domain(21, 18, 3);
    for (std::uint64_t epoch = 0; epoch < 4; ++epoch)
        noise.advance(0.79, epoch);

    for (const auto behavior : {
             FailingEarthControlBackend::Behavior::SucceedOnceThenReject,
             FailingEarthControlBackend::Behavior::SucceedOnceThenThrow})
    {
        chaos::EarthCorrelatedNoise candidate = noise;
        const chaos::EarthCorrelatedNoise before = candidate;
        FailingEarthControlBackend backend(behavior);
        candidate.configure_native_evolution(&backend, true);
        if (behavior == FailingEarthControlBackend::Behavior::SucceedOnceThenReject)
            REQUIRE_THROWS_WITH(
                candidate.translate(5, -4, 4, 0.79),
                "native prepared Earth-control reconstruction dispatch failed");
        else
            REQUIRE_THROWS_WITH(
                candidate.translate(5, -4, 4, 0.79),
                "injected Earth-control dispatch failure");
        REQUIRE(earth_noise_state_equal(candidate, before));

        chaos::EarthCorrelatedNoise workspace;
        configure_earth_noise(workspace);
        workspace.initialize_domain(7, 6, 3);
        const chaos::EarthCorrelatedNoise workspace_before = workspace;
        backend = FailingEarthControlBackend(behavior);
        candidate.configure_native_evolution(&backend, true);
        if (behavior == FailingEarthControlBackend::Behavior::SucceedOnceThenReject)
            REQUIRE_THROWS(candidate.prepare_translated_window(
                workspace, 5, -4, 4, 0.79));
        else
            REQUIRE_THROWS_WITH(
                candidate.prepare_translated_window(
                    workspace, 5, -4, 4, 0.79),
                "injected Earth-control dispatch failure");
        REQUIRE(earth_noise_state_equal(workspace, workspace_before));
    }
}

TEST_CASE("Earth-control restore and lifecycle operations preserve valid state",
          "[chaos][earth_coordinates][restart][transaction]")
{
    chaos::EarthCorrelatedNoise noise;
    configure_earth_noise(noise);
    noise.initialize_domain(17, 15, 3);
    noise.advance(0.82, 0);
    const chaos::EarthCorrelatedNoise before = noise;

    Field3D malformed = noise.current_controls();
    malformed.data()[0] = std::numeric_limits<float>::quiet_NaN();
    std::string error;
    REQUIRE_FALSE(noise.restore(
        malformed, noise.previous_controls(), noise.nr(), noise.nth(), noise.nz(),
        noise.earth_origin_i_cells(), noise.earth_origin_j_cells(),
        noise.control_i_min(), noise.control_j_min(), error));
    REQUIRE(earth_noise_state_equal(noise, before));

    REQUIRE_FALSE(noise.restore_inferred_control_bounds(
        noise.current_controls(), noise.previous_controls(), noise.nr(), noise.nth(),
        noise.nz(), std::numeric_limits<std::int64_t>::max(),
        noise.earth_origin_j_cells(), error));
    REQUIRE(earth_noise_state_equal(noise, before));

    REQUIRE_THROWS_WITH(
        configure_earth_noise(noise),
        "initialized Earth-correlated chaos cannot be reconfigured");
    REQUIRE(earth_noise_state_equal(noise, before));

    noise.initialize_domain(17, 15, 3);
    chaos::EarthCorrelatedNoise fresh;
    configure_earth_noise(fresh);
    fresh.initialize_domain(17, 15, 3);
    REQUIRE(earth_noise_state_equal(noise, fresh));
}

TEST_CASE("Earth-control lattice realizes unit variance and configured e-folding",
          "[chaos][earth_coordinates][statistics]")
{
    constexpr int nr = 40;
    constexpr int nth = 3;
    constexpr int nz = 384;
    constexpr int lag_cells = 8;
    chaos::EarthCorrelatedNoise noise;
    noise.configure(
        998877ULL, 3, 7201ULL, 7301ULL,
        800.0, 600.0, 100.0, 100.0);
    noise.initialize_domain(nr, nth, nz);
    Field3D field;
    noise.render_current(field);

    double sum_a = 0.0;
    double sum_b = 0.0;
    double sum_aa = 0.0;
    double sum_bb = 0.0;
    double sum_ab = 0.0;
    std::size_t count = 0;
    for (int i = 2; i + lag_cells < nr - 2; ++i)
        for (int k = 0; k < nz; ++k)
        {
            const double a = field(i, 1, k);
            const double b = field(i + lag_cells, 1, k);
            sum_a += a;
            sum_b += b;
            sum_aa += a * a;
            sum_bb += b * b;
            sum_ab += a * b;
            ++count;
        }
    const double inverse_count = 1.0 / static_cast<double>(count);
    const double mean_a = sum_a * inverse_count;
    const double mean_b = sum_b * inverse_count;
    const double variance_a = sum_aa * inverse_count - mean_a * mean_a;
    const double variance_b = sum_bb * inverse_count - mean_b * mean_b;
    const double covariance = sum_ab * inverse_count - mean_a * mean_b;
    const double correlation = covariance / std::sqrt(variance_a * variance_b);
    REQUIRE(variance_a == Approx(1.0).margin(0.06));
    REQUIRE(variance_b == Approx(1.0).margin(0.06));
    REQUIRE(correlation == Approx(std::exp(-0.5)).margin(0.06));
}

TEST_CASE("boundary-layer Earth controls retain requested AR1 history")
{
    constexpr int nr = 48;
    constexpr int nth = 44;
    constexpr int nz = 8;
    constexpr double alpha = 0.08;
    constexpr double target_rho = 0.65;

    GridMetrics grid = stochastic_test_grid(nr, nth, nz);
    chaos::ChaosConfig cfg;
    cfg.scheme_id = "boundary_layer";
    cfg.seed = 112233ULL;
    cfg.member_id = 4;
    cfg.filter_id = "recursive_gaussian";
    cfg.Lx = 1200.0;
    cfg.Ly = 1000.0;
    cfg.tau_t = 100.0;
    cfg.xi_max = 20.0;
    cfg.taper_id = "none";
    cfg.alpha_tend["pbl"] = alpha;

    auto scheme = chaos::create_chaos_scheme(cfg.scheme_id);
    scheme->initialize(cfg, grid);
    chaos::ChaosStateView state_view;
    state_view.grid = &grid;

    Field3D initial_tendency(nr, nth, nz, 1.0f);
    chaos::ChaosTendencies initial_tendencies;
    initial_tendencies.dtheta_pbl_dt = &initial_tendency;
    scheme->apply_tendencies(
        cfg, grid, state_view, initial_tendencies, nullptr);
    const PortableSchemeState initial = scheme->portable_state();
    REQUIRE(initial.version == 3);
    REQUIRE(initial.fields.size() == 4);

    const double update_dt = -cfg.tau_t * std::log(target_rho);
    scheme->step_noise(cfg, grid, update_dt);
    Field3D updated_tendency(nr, nth, nz, 1.0f);
    chaos::ChaosTendencies updated_tendencies;
    updated_tendencies.dtheta_pbl_dt = &updated_tendency;
    scheme->apply_tendencies(
        cfg, grid, state_view, updated_tendencies, nullptr);
    const PortableSchemeState updated = scheme->portable_state();
    REQUIRE(updated.integers[0] == 1);
    REQUIRE(updated.fields[1].size() == initial.fields[0].size());
    for (std::size_t index = 0; index < updated.fields[1].size(); ++index)
        REQUIRE(updated.fields[1].data()[index]
                == initial.fields[0].data()[index]);
    double cross = 0.0;
    double previous_square = 0.0;
    for (std::size_t index = 0; index < updated.fields[2].size(); ++index)
    {
        cross += static_cast<double>(updated.fields[2].data()[index])
            * static_cast<double>(updated.fields[3].data()[index]);
        previous_square +=
            static_cast<double>(updated.fields[3].data()[index])
            * static_cast<double>(updated.fields[3].data()[index]);
    }
    REQUIRE(cross / previous_square == Approx(target_rho).margin(0.035));
    require_multiplier_matches_noise(
        updated_tendency, updated.fields[0], alpha, cfg.xi_max);
}

TEST_CASE("full-stochastic Earth controls preserve exact old/current epochs")
{
    // An odd-sized field also guards the per-stream normal-distribution
    // contract through the actual three-block scheme initialization.
    constexpr int nr = 42;
    constexpr int nth = 38;
    constexpr int nz = 6;
    constexpr double alpha = 0.07;
    constexpr double target_rho = 0.6;

    GridMetrics grid = stochastic_test_grid(nr, nth, nz);
    chaos::ChaosConfig cfg;
    cfg.scheme_id = "full_stochastic";
    cfg.seed = 445566ULL;
    cfg.member_id = 9;
    cfg.filter_id = "recursive_gaussian";
    cfg.Lx = 1100.0;
    cfg.Ly = 900.0;
    cfg.tau_t = 80.0;
    cfg.xi_max = 20.0;
    cfg.taper_id = "none";
    cfg.alpha_tend["microphysics"] = alpha;

    auto scheme = chaos::create_chaos_scheme(cfg.scheme_id);
    scheme->initialize(cfg, grid);
    chaos::ChaosStateView state_view;
    state_view.grid = &grid;

    Field3D initial_tendency(nr, nth, nz, 1.0f);
    chaos::ChaosTendencies initial_tendencies;
    initial_tendencies.dtheta_micro_dt = &initial_tendency;
    scheme->apply_tendencies(
        cfg, grid, state_view, initial_tendencies, nullptr);

    const double update_dt = -cfg.tau_t * std::log(target_rho);
    scheme->step_noise(cfg, grid, update_dt);
    Field3D updated_tendency(nr, nth, nz, 1.0f);
    chaos::ChaosTendencies updated_tendencies;
    updated_tendencies.dtheta_micro_dt = &updated_tendency;
    scheme->apply_tendencies(
        cfg, grid, state_view, updated_tendencies, nullptr);
    const PortableSchemeState first_update = scheme->portable_state();
    REQUIRE(first_update.version == 4);
    REQUIRE(first_update.integers.size() == 16);
    REQUIRE(first_update.fields.size() == 12);
    require_multiplier_matches_noise(
        updated_tendency, first_update.fields[0], alpha, cfg.xi_max);

    scheme->step_noise(cfg, grid, update_dt);
    Field3D second_updated_tendency(nr, nth, nz, 1.0f);
    chaos::ChaosTendencies second_updated_tendencies;
    second_updated_tendencies.dtheta_micro_dt = &second_updated_tendency;
    scheme->apply_tendencies(
        cfg, grid, state_view, second_updated_tendencies, nullptr);
    const PortableSchemeState second_update = scheme->portable_state();
    REQUIRE(second_update.integers[0] == 2);
    for (std::size_t index = 0; index < first_update.fields[0].size(); ++index)
        REQUIRE(second_update.fields[3].data()[index]
                == first_update.fields[0].data()[index]);
    for (std::size_t index = 0; index < first_update.fields[6].size(); ++index)
        REQUIRE(second_update.fields[7].data()[index]
                == first_update.fields[6].data()[index]);
    require_multiplier_matches_noise(
        second_updated_tendency,
        second_update.fields[0],
        alpha,
        cfg.xi_max);
}

TEST_CASE("full-stochastic PBL uncertainty scales conserved columns and boundary fluxes together",
          "[chaos][pbl][conservation]")
{
    constexpr int nr = 7;
    constexpr int nth = 6;
    constexpr int nz = 4;
    constexpr double alpha = 0.35;
    GridMetrics grid = stochastic_test_grid(nr, nth, nz);
    chaos::ChaosConfig cfg;
    cfg.scheme_id = "full_stochastic";
    cfg.seed = 99117ULL;
    cfg.member_id = 3;
    cfg.filter_id = "recursive_gaussian";
    cfg.Lx = 900.0;
    cfg.Ly = 700.0;
    cfg.xi_max = 2.0;
    cfg.taper_id = "none";
    cfg.alpha_tend["pbl"] = alpha;

    auto scheme = chaos::create_chaos_scheme(cfg.scheme_id);
    scheme->initialize(cfg, grid);
    chaos::ChaosStateView state_view;
    state_view.grid = &grid;

    Field3D u_force(nr, nth, nz, 2.0f);
    Field3D v_force(nr, nth, nz, -3.0f);
    Field3D theta_rate(nr, nth, nz, 5.0f);
    Field3D vapor_rate(nr, nth, nz, -7.0f);
    Field3D tke_tendency(nr, nth, nz, 11.0f);
    const std::size_t surface_size =
        static_cast<std::size_t>(nr) * static_cast<std::size_t>(nth);
    std::vector<float> ustar(surface_size, 4.0f);
    std::vector<float> tau_u(surface_size, -13.0f);
    std::vector<float> tau_v(surface_size, 17.0f);
    std::vector<float> sensible(surface_size, 19.0f);
    std::vector<float> latent(surface_size, 23.0f);
    std::vector<float> moisture(surface_size, 29.0f);

    chaos::ChaosTendencies tendencies;
    tendencies.pbl_u_force_n = &u_force;
    tendencies.pbl_v_force_n = &v_force;
    tendencies.pbl_theta_inventory_rate_kgk_s = &theta_rate;
    tendencies.pbl_vapor_mass_rate_kg_s = &vapor_rate;
    tendencies.pbl_tke_tendency = &tke_tendency;
    tendencies.pbl_ustar = &ustar;
    tendencies.pbl_tau_u = &tau_u;
    tendencies.pbl_tau_v = &tau_v;
    tendencies.pbl_sensible_heat_flux = &sensible;
    tendencies.pbl_latent_heat_flux = &latent;
    tendencies.pbl_moisture_flux = &moisture;
    scheme->apply_tendencies(cfg, grid, state_view, tendencies, nullptr);

    const PortableSchemeState portable = scheme->portable_state();
    REQUIRE(portable.fields.size() == 12);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
        {
            const double multiplier =
                static_cast<double>(u_force[i][j][0]) / 2.0;
            REQUIRE(multiplier >= 0.0);
            REQUIRE(multiplier <= 2.0);
            for (int k = 0; k < nz; ++k)
            {
                REQUIRE(u_force[i][j][k] == Approx(2.0 * multiplier));
                REQUIRE(v_force[i][j][k] == Approx(-3.0 * multiplier));
                REQUIRE(theta_rate[i][j][k] == Approx(5.0 * multiplier));
                REQUIRE(vapor_rate[i][j][k] == Approx(-7.0 * multiplier));
                REQUIRE(tke_tendency[i][j][k] == Approx(11.0 * multiplier));
            }
            const std::size_t index =
                static_cast<std::size_t>(i) * static_cast<std::size_t>(nth) +
                static_cast<std::size_t>(j);
            REQUIRE(tau_u[index] == Approx(-13.0 * multiplier));
            REQUIRE(tau_v[index] == Approx(17.0 * multiplier));
            REQUIRE(sensible[index] == Approx(19.0 * multiplier));
            REQUIRE(latent[index] == Approx(23.0 * multiplier));
            REQUIRE(moisture[index] == Approx(29.0 * multiplier));
            REQUIRE(ustar[index] == Approx(4.0 * std::sqrt(multiplier)));
        }
}

TEST_CASE("full-stochastic restart retains every Earth lattice after repeated D5 moves",
          "[chaos][earth_coordinates][restart][D5][SCI-280]")
{
    constexpr int nr = 18;
    constexpr int nth = 18;
    constexpr int nz = 6;
    GridMetrics grid = stochastic_test_grid(nr, nth, nz);
    grid.dy = 1000.0;

    chaos::ChaosConfig cfg;
    cfg.scheme_id = "full_stochastic";
    cfg.seed = 233ULL;
    cfg.member_id = 0;
    cfg.dt_chaos = 60.0;
    cfg.tau_t = 21600.0;
    cfg.Lx = 50000.0;
    cfg.Ly = 50000.0;
    cfg.filter_id = "recursive_gaussian";

    auto source = chaos::create_chaos_scheme(cfg.scheme_id);
    source->initialize(cfg, grid);
    chaos::ChaosStateView state_view;
    state_view.grid = &grid;
    Field3D tendency(nr, nth, nz, 1.0f);
    chaos::ChaosTendencies tendencies;
    tendencies.dtheta_micro_dt = &tendency;
    source->apply_tendencies(cfg, grid, state_view, tendencies, nullptr);

    std::string error;
    const PortableSchemeState initial_v3 = source->portable_state();
    REQUIRE(initial_v3.integers[3] == initial_v3.integers[7]);
    REQUIRE(initial_v3.integers[3] == initial_v3.integers[11]);
    REQUIRE(initial_v3.integers[4] == initial_v3.integers[8]);
    REQUIRE(initial_v3.integers[4] == initial_v3.integers[12]);
    PortableSchemeState legacy_v2 = initial_v3;
    legacy_v2.version = 2;
    legacy_v2.integers = {
        initial_v3.integers[0], initial_v3.integers[1],
        initial_v3.integers[2], initial_v3.integers[3],
        initial_v3.integers[4], initial_v3.integers[13],
        initial_v3.integers[14], initial_v3.integers[15]};
    legacy_v2.reals = {initial_v3.reals[0]};
    auto legacy_restored = chaos::create_chaos_scheme(cfg.scheme_id);
    legacy_restored->initialize(cfg, grid);
    REQUIRE(legacy_restored->restore_portable_state(legacy_v2, error));

    for (int shift_i : {3, 2, 2, 2})
        REQUIRE(source->translate_horizontal_stochastic_state(
            shift_i, 0, cfg, grid, error));
    const PortableSchemeState checkpoint = source->portable_state();
    REQUIRE(checkpoint.version == 4);
    REQUIRE(checkpoint.integers.size() == 16);
    REQUIRE(checkpoint.integers[1] == checkpoint.integers[5]);
    REQUIRE(checkpoint.integers[1] == checkpoint.integers[9]);
    REQUIRE(checkpoint.integers[2] == checkpoint.integers[6]);
    REQUIRE(checkpoint.integers[2] == checkpoint.integers[10]);
    REQUIRE((checkpoint.integers[3] != checkpoint.integers[7]
             || checkpoint.integers[3] != checkpoint.integers[11]
             || checkpoint.integers[4] != checkpoint.integers[8]
             || checkpoint.integers[4] != checkpoint.integers[12]));

    PortableSchemeState moved_legacy_v2 = checkpoint;
    moved_legacy_v2.version = 2;
    moved_legacy_v2.integers = {
        checkpoint.integers[0], checkpoint.integers[1],
        checkpoint.integers[2], checkpoint.integers[3],
        checkpoint.integers[4], checkpoint.integers[13],
        checkpoint.integers[14], checkpoint.integers[15]};
    moved_legacy_v2.reals = {checkpoint.reals[0]};
    auto moved_legacy_restored = chaos::create_chaos_scheme(cfg.scheme_id);
    moved_legacy_restored->initialize(cfg, grid);
    REQUIRE(moved_legacy_restored->restore_portable_state(
        moved_legacy_v2, error));
    const PortableSchemeState promoted_legacy =
        moved_legacy_restored->portable_state();
    REQUIRE(promoted_legacy.version == 4);
    for (std::size_t field = 0; field < checkpoint.fields.size(); ++field)
        for (std::size_t index = 0; index < checkpoint.fields[field].size(); ++index)
            REQUIRE(promoted_legacy.fields[field].data()[index]
                    == checkpoint.fields[field].data()[index]);

    auto restored = chaos::create_chaos_scheme(cfg.scheme_id);
    restored->initialize(cfg, grid);
    REQUIRE(restored->restore_portable_state(checkpoint, error));
    const PortableSchemeState exact = restored->portable_state();
    REQUIRE(exact.version == checkpoint.version);
    REQUIRE(exact.integers == checkpoint.integers);
    REQUIRE(exact.reals == checkpoint.reals);
    REQUIRE(exact.fields.size() == checkpoint.fields.size());
    for (std::size_t field = 0; field < exact.fields.size(); ++field)
    {
        REQUIRE(exact.fields[field].size() == checkpoint.fields[field].size());
        for (std::size_t index = 0; index < exact.fields[field].size(); ++index)
            REQUIRE(exact.fields[field].data()[index]
                    == checkpoint.fields[field].data()[index]);
    }

    source->step_noise(cfg, grid, cfg.dt_chaos);
    restored->step_noise(cfg, grid, cfg.dt_chaos);
    const PortableSchemeState continued = source->portable_state();
    const PortableSchemeState restarted = restored->portable_state();
    REQUIRE(restarted.integers == continued.integers);
    REQUIRE(restarted.reals == continued.reals);
    for (std::size_t field = 0; field < continued.fields.size(); ++field)
        for (std::size_t index = 0; index < continued.fields[field].size(); ++index)
            REQUIRE(restarted.fields[field].data()[index]
                    == continued.fields[field].data()[index]);
}

TEST_CASE("production full-stochastic microphysics publishes one multiplier per sedimenting column",
          "[chaos][microphysics][conservation]")
{
    constexpr int nr = 7;
    constexpr int nth = 5;
    constexpr int nz = 4;
    constexpr double alpha = 0.45;
    GridMetrics grid = stochastic_test_grid(nr, nth, nz);
    chaos::ChaosConfig cfg;
    cfg.scheme_id = "full_stochastic";
    cfg.seed = 233991ULL;
    cfg.member_id = 2;
    cfg.Lx = 900.0;
    cfg.Ly = 700.0;
    cfg.taper_id = "none";
    cfg.alpha_tend["microphysics"] = alpha;

    auto scheme = chaos::create_chaos_scheme(cfg.scheme_id);
    scheme->initialize(cfg, grid);
    chaos::ChaosStateView state_view;
    state_view.grid = &grid;
    Field3D theta_tendency(nr, nth, nz, 3.0f);
    Field3D vapor_tendency(nr, nth, nz, -2.0f);
    std::vector<float> multipliers;
    chaos::ChaosTendencies tendencies;
    tendencies.dtheta_micro_dt = &theta_tendency;
    tendencies.dqv_micro_dt = &vapor_tendency;
    tendencies.microphysics_column_multipliers = &multipliers;
    scheme->apply_tendencies(cfg, grid, state_view, tendencies, nullptr);

    const PortableSchemeState portable = scheme->portable_state();
    REQUIRE(portable.fields.size() == 12);
    REQUIRE(multipliers.size()
            == static_cast<std::size_t>(nr * nth));
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
        {
            const double raw =
                static_cast<double>(portable.fields[0][i][j][0]);
            const double bounded = cfg.xi_max
                * std::tanh(raw / cfg.xi_max);
            const double expected = std::clamp(
                1.0 + alpha * bounded, 0.0, 2.0);
            REQUIRE(multipliers[static_cast<std::size_t>(i * nth + j)]
                    == Approx(expected));
            for (int k = 0; k < nz; ++k)
            {
                // The chaos layer publishes the process control; the
                // microphysics transaction admits and applies it once.
                REQUIRE(theta_tendency[i][j][k] == 3.0f);
                REQUIRE(vapor_tendency[i][j][k] == -2.0f);
            }
        }
}

TEST_CASE("boundary-layer-only chaos uses the same conservative PBL source contract",
          "[chaos][pbl][conservation]")
{
    constexpr int nr = 5, nth = 4, nz = 3;
    GridMetrics grid = stochastic_test_grid(nr, nth, nz);
    chaos::ChaosConfig cfg;
    cfg.scheme_id = "boundary_layer";
    cfg.seed = 44119ULL;
    cfg.member_id = 5;
    cfg.Lx = 800.0;
    cfg.Ly = 600.0;
    cfg.taper_id = "none";
    cfg.alpha_tend["pbl"] = 0.4;
    auto scheme = chaos::create_chaos_scheme(cfg.scheme_id);
    scheme->initialize(cfg, grid);

    Field3D force(nr, nth, nz, 2.0f);
    Field3D theta_rate(nr, nth, nz, 3.0f);
    std::vector<float> tau(static_cast<std::size_t>(nr * nth), 5.0f);
    std::vector<float> ustar(static_cast<std::size_t>(nr * nth), 7.0f);
    chaos::ChaosTendencies tendencies;
    tendencies.pbl_u_force_n = &force;
    tendencies.pbl_theta_inventory_rate_kgk_s = &theta_rate;
    tendencies.pbl_tau_u = &tau;
    tendencies.pbl_ustar = &ustar;
    chaos::ChaosStateView state_view;
    state_view.grid = &grid;
    scheme->apply_tendencies(cfg, grid, state_view, tendencies, nullptr);

    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
        {
            const double multiplier = force[i][j][0] / 2.0;
            for (int k = 0; k < nz; ++k)
            {
                REQUIRE(force[i][j][k] == Approx(2.0 * multiplier));
                REQUIRE(theta_rate[i][j][k] == Approx(3.0 * multiplier));
            }
            const std::size_t index =
                static_cast<std::size_t>(i * nth + j);
            REQUIRE(tau[index] == Approx(5.0 * multiplier));
            REQUIRE(ustar[index] == Approx(7.0 * std::sqrt(multiplier)));
        }
}

TEST_CASE("evolving chaos schemes exchange prepared Earth windows reversibly",
          "[chaos][translation][transaction]")
{
    constexpr int nr = 12;
    constexpr int nth = 10;
    constexpr int nz = 4;
    GridMetrics grid = stochastic_test_grid(nr, nth, nz);
    const auto portable_equal = [](const PortableSchemeState& lhs,
                                   const PortableSchemeState& rhs) {
        if (lhs.version != rhs.version || lhs.integers != rhs.integers ||
            lhs.reals != rhs.reals || lhs.fields.size() != rhs.fields.size())
            return false;
        for (std::size_t field = 0; field < lhs.fields.size(); ++field)
        {
            if (lhs.fields[field].size_r() != rhs.fields[field].size_r() ||
                lhs.fields[field].size_th() != rhs.fields[field].size_th() ||
                lhs.fields[field].size_z() != rhs.fields[field].size_z() ||
                std::memcmp(lhs.fields[field].data(),
                            rhs.fields[field].data(),
                            lhs.fields[field].size() * sizeof(float)) != 0)
                return false;
        }
        return true;
    };

    for (const std::string& scheme_name :
         {std::string("boundary_layer"), std::string("full_stochastic")})
    {
        chaos::ChaosConfig cfg;
        cfg.scheme_id = scheme_name;
        cfg.seed = 71923ULL;
        cfg.member_id = 9;
        cfg.dt_chaos = 2.0;
        cfg.tau_t = 18.0;
        cfg.Lx = 1800.0;
        cfg.Ly = 1400.0;
        cfg.taper_id = "none";
        cfg.alpha_tend["pbl"] = 0.2;

        auto scheme = chaos::create_chaos_scheme(scheme_name);
        scheme->initialize(cfg, grid);
        Field3D tendency(nr, nth, nz, 1.0f);
        chaos::ChaosStateView state_view;
        state_view.grid = &grid;
        chaos::ChaosTendencies tendencies;
        if (scheme_name == "boundary_layer")
            tendencies.dtheta_pbl_dt = &tendency;
        else
            tendencies.dtheta_micro_dt = &tendency;
        scheme->apply_tendencies(
            cfg, grid, state_view, tendencies, nullptr);
        scheme->step_noise(cfg, grid, 2.0);

        const PortableSchemeState before = scheme->portable_state();
        auto expected = scheme->clone();
        REQUIRE(expected);
        std::string error;
        REQUIRE(expected->translate_horizontal_stochastic_state(
            2, -1, cfg, grid, error));
        const PortableSchemeState expected_after = expected->portable_state();

        std::unique_ptr<chaos::ChaosScheme> workspace;
        chaos::HorizontalStochasticStatePhase phase;
        REQUIRE(scheme->prepare_horizontal_stochastic_state_translation(
            2, -1, cfg, grid, workspace, phase, error));
        REQUIRE(workspace);
        auto* const workspace_identity = workspace.get();
        REQUIRE(portable_equal(scheme->portable_state(), before));
        chaos::HorizontalStochasticStorageIdentity workspace_storage;
        REQUIRE(workspace->horizontal_stochastic_storage_identity(
            workspace_storage, error));
        REQUIRE(workspace_storage.process_count ==
                (scheme_name == "boundary_layer" ? 1 : 3));
        for (std::size_t process = 0;
             process < workspace_storage.process_count; ++process)
        {
            REQUIRE(workspace_storage.current_controls[process] != nullptr);
            REQUIRE(workspace_storage.previous_controls[process] != nullptr);
            REQUIRE(workspace_storage.current_rendered[process] == nullptr);
            REQUIRE(workspace_storage.previous_rendered[process] == nullptr);
        }

        scheme->exchange_prepared_horizontal_stochastic_state_translation(
            workspace);
        REQUIRE(portable_equal(scheme->portable_state(), expected_after));
        scheme->exchange_prepared_horizontal_stochastic_state_translation(
            workspace);
        REQUIRE(portable_equal(scheme->portable_state(), before));

        REQUIRE(scheme->prepare_horizontal_stochastic_state_translation(
            2, -1, cfg, grid, workspace, phase, error));
        REQUIRE(workspace.get() == workspace_identity);
        scheme->exchange_prepared_horizontal_stochastic_state_translation(
            workspace);
        REQUIRE(portable_equal(scheme->portable_state(), expected_after));
    }
}

namespace
{
class TransactionalNoiseProbe final : public chaos::ChaosScheme
{
public:
    int epochs = 0;
    int fail_epoch = 2;
    double origin = 0.0;
    bool fail_origin = false;
    std::string name() const override { return "full_stochastic"; }
    std::unique_ptr<chaos::ChaosScheme> clone() const override
    { return std::make_unique<TransactionalNoiseProbe>(*this); }
    void initialize(const chaos::ChaosConfig&, const GridMetrics&) override {}
    void step_noise(const chaos::ChaosConfig&, const GridMetrics&, double) override
    {
        if (++epochs == fail_epoch) throw std::runtime_error("noise epoch failure");
    }
    bool set_continuous_horizontal_earth_origin(double x, double,
        const chaos::ChaosConfig&, const GridMetrics&, std::string& error) override
    {
        origin = x;
        if (fail_origin) { error = "origin failure"; return false; }
        return true;
    }
};
}

TEST_CASE("chaos interval failure restores all epochs origin and cadence",
          "[chaos][runtime][atomicity]")
{
    initialize_process_globals();
    auto state = global_simulation_state();
    auto& runtime = *state.chaos_runtime;
    runtime.scheme = std::make_unique<TransactionalNoiseProbe>();
    runtime.config.dt_chaos = 0.1;
    runtime.config.native_generation_required = false;
    runtime.noise_elapsed_s = 0.0;
    const auto observed = [&]() -> TransactionalNoiseProbe& {
        return dynamic_cast<TransactionalNoiseProbe&>(*runtime.scheme);
    };
    SECTION("a later failed epoch restores earlier successful epochs")
    {
        REQUIRE_THROWS(step_chaos_noise(state, 0.25));
        REQUIRE(observed().epochs == 0);
        REQUIRE(runtime.noise_elapsed_s == 0.0);
        observed().fail_epoch = 0;
        step_chaos_noise(state, 0.25);
        REQUIRE(observed().epochs == 2);
        REQUIRE(runtime.noise_elapsed_s == Approx(0.05));
    }
    SECTION("origin publication is part of the same transaction")
    {
        state.domain_frame->velocity_ms.u_ms = 5.0;
        state.time_s = 2.0;
        observed().fail_origin = true;
        REQUIRE_THROWS(step_chaos_noise(state, 0.025));
        REQUIRE(observed().origin == 0.0);
        REQUIRE(runtime.noise_elapsed_s == 0.0);
        observed().fail_origin = false;
        step_chaos_noise(state, 0.025);
        REQUIRE(observed().origin == 10.0);
        REQUIRE(runtime.noise_elapsed_s == Approx(0.025));
    }
    SECTION("missing native ownership rejects before changing the clock")
    {
        runtime.config.native_generation_required = true;
        REQUIRE_THROWS(step_chaos_noise(state, 0.25));
        REQUIRE(observed().epochs == 0);
        REQUIRE(runtime.noise_elapsed_s == 0.0);
    }
    SECTION("an impractical epoch count rejects before doing work")
    {
        REQUIRE_THROWS(step_chaos_noise(state, 1.0e10));
        REQUIRE(observed().epochs == 0);
        REQUIRE(runtime.noise_elapsed_s == 0.0);
    }
    SECTION("decimal substeps leave a nonnegative canonical cadence remainder")
    {
        observed().fail_epoch = 0;
        runtime.config.dt_chaos = 1.0;
        for (int step = 0; step < 100; ++step)
        {
            step_chaos_noise(state, 0.1);
            REQUIRE(runtime.noise_elapsed_s >= 0.0);
            REQUIRE(runtime.noise_elapsed_s < 1.0);
            REQUIRE(observed().epochs == (step + 1) / 10);
            if ((step + 1) % 10 == 0)
            {
                REQUIRE(runtime.noise_elapsed_s == 0.0);
            }
        }
    }
}
