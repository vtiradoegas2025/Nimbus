#include "microphysics/runtime.hpp"
// Regression tests for explicit-state microphysics stepping and ownership.

#include "catch2/catch.hpp"
#include "compute/parity_check.hpp"
#include "compute/backend_runtime.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "core/field/field_inventory.hpp"
#include "core/runtime/runtime_config.hpp"
#include "core/runtime/simulation.hpp"
#include "core/runtime/simulation_state.hpp"
#include "diagnostics/deterministic_reduction.hpp"
#include "microphysics/microphysics_base.hpp"
#include "numerics/advection/advection.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

bool test_stochastic_column_control_active = false;
float test_stochastic_column_multiplier = 1.0f;
bool test_snapshot_reprojection_failure = false;

class DeclinedApplyBackend final : public ComputeBackend, public MicrophysicsComputeKernels
{
public:
    std::string name() const override { return "declined_apply_test"; }
    bool initialize(std::string&) override { return true; }
    void shutdown() override {}
    const MicrophysicsComputeKernels* microphysics_kernels() const override { return this; }
    MicrophysicsComputeKernels* microphysics_kernels() override { return this; }
    bool supports(MicrophysicsKernelOp op) const override
    {
        return op == MicrophysicsKernelOp::MicrophysicsApply;
    }
};

class StubMicrophysicsScheme : public MicrophysicsScheme
{
public:
    std::unique_ptr<MicrophysicsScheme> clone() const override
    { return std::make_unique<StubMicrophysicsScheme>(*this); }
    void compute_tendencies(const Field3D& p,
                            const Field3D& /*theta*/,
                            const Field3D& /*rho_d*/,
                            const Field3D& /*qv*/,
                            const Field3D& /*qc*/,
                            const Field3D& /*qr*/,
                            const Field3D& /*qi*/,
                            const Field3D& /*qs*/,
                            const Field3D& /*qg*/,
                            const Field3D& /*qh*/,
                            double /*dt*/,
                            Field3D& dtheta_dt,
                            Field3D& dqv_dt,
                            Field3D& dqc_dt,
                            Field3D& dqr_dt,
                            Field3D& dqi_dt,
                            Field3D& dqs_dt,
                            Field3D& dqg_dt,
                            Field3D& dqh_dt) override
    {
        const int nr = p.size_r();
        const int nth = p.size_th();
        const int nz = p.size_z();
        dtheta_dt.resize(nr, nth, nz, 2.0f);
        dqv_dt.resize(nr, nth, nz, -0.001f);
        dqc_dt.resize(nr, nth, nz, 0.0005f);
        dqr_dt.resize(nr, nth, nz, 0.0f);
        dqi_dt.resize(nr, nth, nz, 0.0f);
        dqs_dt.resize(nr, nth, nz, 0.0f);
        dqg_dt.resize(nr, nth, nz, 0.0f);
        dqh_dt.resize(nr, nth, nz, 0.0f);
    }

    std::string get_scheme_name() const override { return "stub"; }
    int get_num_prognostic_vars() const override { return 0; }
};

class ThrowingMicrophysicsScheme final : public MicrophysicsScheme
{
public:
    std::unique_ptr<MicrophysicsScheme> clone() const override
    { return std::make_unique<ThrowingMicrophysicsScheme>(*this); }
    void compute_tendencies(
        const Field3D&, const Field3D&, const Field3D&, const Field3D&,
        const Field3D&, const Field3D&, const Field3D&, const Field3D&,
        const Field3D&, const Field3D&, double,
        Field3D&, Field3D&, Field3D&, Field3D&,
        Field3D&, Field3D&, Field3D&, Field3D&) override
    {
        throw std::runtime_error("injected tendency failure");
    }

    std::string get_scheme_name() const override { return "throwing"; }
    int get_num_prognostic_vars() const override { return 0; }
};

class WrongShapeMicrophysicsScheme final : public StubMicrophysicsScheme
{
public:
    std::unique_ptr<MicrophysicsScheme> clone() const override
    { return std::make_unique<WrongShapeMicrophysicsScheme>(*this); }
    void compute_tendencies(
        const Field3D& p, const Field3D& theta_in, const Field3D& rho_in,
        const Field3D& qv_in, const Field3D& qc_in, const Field3D& qr_in,
        const Field3D& qi_in, const Field3D& qs_in, const Field3D& qg_in,
        const Field3D& qh_in, double dt_in,
        Field3D& dtheta_dt, Field3D& dqv_dt, Field3D& dqc_dt,
        Field3D& dqr_dt, Field3D& dqi_dt, Field3D& dqs_dt,
        Field3D& dqg_dt, Field3D& dqh_dt) override
    {
        StubMicrophysicsScheme::compute_tendencies(
            p, theta_in, rho_in, qv_in, qc_in, qr_in, qi_in, qs_in,
            qg_in, qh_in, dt_in, dtheta_dt, dqv_dt, dqc_dt, dqr_dt,
            dqi_dt, dqs_dt, dqg_dt, dqh_dt);
        dqv_dt.resize(2, 1, 1, 0.0f);
    }

    std::string get_scheme_name() const override { return "wrong_shape"; }
};

class RejectingPrivatePrognosticScheme final
    : public StubMicrophysicsScheme
{
public:
    std::unique_ptr<MicrophysicsScheme> clone() const override
    { return std::make_unique<RejectingPrivatePrognosticScheme>(*this); }
    bool validate_prognostic_update(
        double, std::string& error) const override
    {
        error = "injected private-moment rejection";
        return false;
    }

    void commit_prognostic_update(double) noexcept override
    {
        committed = true;
    }

    std::string get_scheme_name() const override
    {
        return "rejecting_private";
    }

    bool committed = false;
};

struct PostPrimaryProbe
{
    bool saw_accepted_primary = false;
    float accepted_qv = 0.0f;
    float accepted_qc = 0.0f;
    const tmv::MicrophysicsWaterRepairReport* water_repairs = nullptr;
    const tmv::MicrophysicsThermodynamicRepairReport* thermodynamic_repairs = nullptr;
    std::size_t lower_bound_repairs = 0;
    std::size_t lower_bound_roundoff_repairs = 0;
    std::size_t upper_bound_repairs = 0;
    std::size_t nonfinite_repairs = 0;
    std::size_t thermal_bound_repairs = 0;
    std::size_t thermal_nonfinite_repairs = 0;
    int commits = 0;
};

class PostPrimaryCandidateScheme final : public MicrophysicsScheme
{
public:
    enum class Mode
    {
        RejectPrepare,
        MissingToken,
        Commit
    };

    PostPrimaryCandidateScheme(Mode mode, PostPrimaryProbe& probe)
        : mode_(mode), probe_(&probe)
    {
    }

    std::unique_ptr<MicrophysicsScheme> clone() const override
    {
        return std::make_unique<PostPrimaryCandidateScheme>(*this);
    }

    void compute_tendencies(
        const Field3D& pressure, const Field3D&, const Field3D&,
        const Field3D&, const Field3D&, const Field3D&,
        const Field3D&, const Field3D&, const Field3D&,
        const Field3D&, double,
        Field3D& dtheta_dt, Field3D& dqv_dt,
        Field3D& dqc_dt, Field3D& dqr_dt,
        Field3D& dqi_dt, Field3D& dqs_dt,
        Field3D& dqg_dt, Field3D& dqh_dt) override
    {
        const int nr = pressure.size_r();
        const int nth = pressure.size_th();
        const int nz = pressure.size_z();
        dtheta_dt.resize(nr, nth, nz, 0.0f);
        dqv_dt.resize(nr, nth, nz, -0.001f);
        dqc_dt.resize(nr, nth, nz, 0.001f);
        dqr_dt.resize(nr, nth, nz, 0.0f);
        dqi_dt.resize(nr, nth, nz, 0.0f);
        dqs_dt.resize(nr, nth, nz, 0.0f);
        dqg_dt.resize(nr, nth, nz, 0.0f);
        dqh_dt.resize(nr, nth, nz, 0.0f);
    }

    bool prepare_post_primary_private_commit(
        double dt,
        const tmv::MicrophysicsAcceptedPrimaryFields& accepted,
        std::string& error) override
    {
        probe_->saw_accepted_primary = true;
        probe_->accepted_qv = accepted.qv(0, 0, 0);
        probe_->accepted_qc = accepted.qc(0, 0, 0);
        probe_->water_repairs = accepted.water_repairs;
        probe_->thermodynamic_repairs = accepted.thermodynamic_repairs;
        if (accepted.water_repairs)
        {
            probe_->lower_bound_repairs =
                accepted.water_repairs->lower_bound_count();
            probe_->lower_bound_roundoff_repairs =
                accepted.water_repairs->lower_bound_roundoff_count();
            probe_->upper_bound_repairs =
                accepted.water_repairs->upper_bound_count();
            probe_->nonfinite_repairs =
                accepted.water_repairs->nonfinite_count();
        }
        if (accepted.thermodynamic_repairs)
        {
            probe_->thermal_bound_repairs =
                accepted.thermodynamic_repairs->theta_tendency_limit_count +
                accepted.thermodynamic_repairs->theta_lower_bound_count +
                accepted.thermodynamic_repairs->theta_upper_bound_count;
            probe_->thermal_nonfinite_repairs =
                accepted.thermodynamic_repairs->theta_nonfinite_count;
        }
        if (mode_ == Mode::RejectPrepare)
        {
            error = "injected post-primary rejection";
            return false;
        }
        if (mode_ == Mode::Commit)
        {
            private_candidate_.resize(1, 1, 1,
                static_cast<float>(42.0 + dt));
            prepared_ = true;
        }
        error.clear();
        return true;
    }

    bool owns_post_primary_private_commit() const noexcept override
    {
        return true;
    }

    bool commit_prepared_private_candidate() noexcept override
    {
        if (!prepared_)
        {
            return false;
        }
        private_state_ = std::move(private_candidate_);
        prepared_ = false;
        ++probe_->commits;
        return true;
    }

    std::string get_scheme_name() const override
    {
        return "post_primary_candidate";
    }

    int get_num_prognostic_vars() const override
    {
        return 1;
    }

    const Field3D& private_state() const noexcept
    {
        return private_state_;
    }

private:
    Mode mode_;
    PostPrimaryProbe* probe_ = nullptr;
    Field3D private_state_;
    Field3D private_candidate_;
    bool prepared_ = false;
};

class WaterTransferScheme final : public MicrophysicsScheme
{
public:
    std::unique_ptr<MicrophysicsScheme> clone() const override
    { return std::make_unique<WaterTransferScheme>(*this); }
    WaterTransferScheme(float dqv_dt_value, float dqc_dt_value)
        : rates_{{dqv_dt_value, dqc_dt_value, 0, 0, 0, 0, 0}}
    {
    }
    explicit WaterTransferScheme(std::array<float, 7> rates)
        : rates_(rates)
    {
    }

    void compute_tendencies(const Field3D& p,
                            const Field3D&, const Field3D&, const Field3D&,
                            const Field3D&,
                            const Field3D&, const Field3D&, const Field3D&,
                            const Field3D&, const Field3D&, double,
                            Field3D& dtheta_dt, Field3D& dqv_dt,
                            Field3D& dqc_dt, Field3D& dqr_dt,
                            Field3D& dqi_dt, Field3D& dqs_dt,
                            Field3D& dqg_dt, Field3D& dqh_dt) override
    {
        const int nr = p.size_r();
        const int nth = p.size_th();
        const int nz = p.size_z();
        dtheta_dt.resize(nr, nth, nz, 0.0f);
        const std::array<Field3D*, 7> water{{
            &dqv_dt, &dqc_dt, &dqr_dt, &dqi_dt, &dqs_dt, &dqg_dt, &dqh_dt}};
        for (std::size_t species = 0; species < water.size(); ++species)
            water[species]->resize(nr, nth, nz, rates_[species]);
    }

    std::string get_scheme_name() const override { return "water_transfer"; }
    int get_num_prognostic_vars() const override { return 0; }

private:
    std::array<float, 7> rates_;
};

class ThetaSourceScheme final : public MicrophysicsScheme
{
public:
    std::unique_ptr<MicrophysicsScheme> clone() const override
    { return std::make_unique<ThetaSourceScheme>(*this); }
    explicit ThetaSourceScheme(float dtheta_dt_value)
        : dtheta_dt_value_(dtheta_dt_value)
    {
    }

    void compute_tendencies(
        const Field3D& p, const Field3D&, const Field3D&, const Field3D&,
        const Field3D&, const Field3D&, const Field3D&, const Field3D&,
        const Field3D&, const Field3D&, double,
        Field3D& dtheta_dt, Field3D& dqv_dt, Field3D& dqc_dt,
        Field3D& dqr_dt, Field3D& dqi_dt, Field3D& dqs_dt,
        Field3D& dqg_dt, Field3D& dqh_dt) override
    {
        const int nr = p.size_r();
        const int nth = p.size_th();
        const int nz = p.size_z();
        dtheta_dt.resize(nr, nth, nz, dtheta_dt_value_);
        dqv_dt.resize(nr, nth, nz, 0.0f);
        dqc_dt.resize(nr, nth, nz, 0.0f);
        dqr_dt.resize(nr, nth, nz, 0.0f);
        dqi_dt.resize(nr, nth, nz, 0.0f);
        dqs_dt.resize(nr, nth, nz, 0.0f);
        dqg_dt.resize(nr, nth, nz, 0.0f);
        dqh_dt.resize(nr, nth, nz, 0.0f);
    }

    std::string get_scheme_name() const override { return "theta_source"; }
    int get_num_prognostic_vars() const override { return 0; }

private:
    float dtheta_dt_value_;
};

class LinearFalloutScheme final : public MicrophysicsScheme
{
public:
    int calls = 0;
    int fail_on_call = 0;
    std::unique_ptr<MicrophysicsScheme> clone() const override
    { return std::make_unique<LinearFalloutScheme>(*this); }
    double explicit_terminal_velocity_mps(std::size_t species, double, double) const override
    { return species == 0 ? 0.4 : 0.0; }
    void compute_tendencies(
        const Field3D& p, const Field3D&, const Field3D&, const Field3D&,
        const Field3D&, const Field3D& rain, const Field3D&, const Field3D&,
        const Field3D&, const Field3D&, double,
        Field3D& heat, Field3D& vapor, Field3D& cloud, Field3D& fallout,
        Field3D& ice, Field3D& snow, Field3D& graupel, Field3D& hail) override
    {
        if (++calls == fail_on_call) throw std::runtime_error("late sedimentation failure");
        for (auto* field : {&heat, &vapor, &cloud, &fallout, &ice, &snow, &graupel, &hail})
            field->resize(p.size_r(), p.size_th(), p.size_z(), 0.0f);
        for (std::size_t index = 0; index < rain.size(); ++index)
            fallout.data()[index] = -0.1f * rain.data()[index];
    }
    double bottom_sedimentation_sink_rate(const Field3D&, const Field3D& rain,
        const Field3D&, const Field3D&, const Field3D&, int i, int j, double) const override
    { return 0.1f * rain[i][j][0]; }
    std::string get_scheme_name() const override { return "linear_fallout"; }
    int get_num_prognostic_vars() const override { return 1; }
};

class SurfaceFalloutScheme final : public MicrophysicsScheme
{
public:
    std::unique_ptr<MicrophysicsScheme> clone() const override
    { return std::make_unique<SurfaceFalloutScheme>(*this); }
    explicit SurfaceFalloutScheme(float sink_rate_kgkg_s)
        : sink_rate_kgkg_s_(sink_rate_kgkg_s)
    {
    }

    void compute_tendencies(
        const Field3D& p, const Field3D&, const Field3D&, const Field3D&,
        const Field3D&, const Field3D&, const Field3D&, const Field3D&,
        const Field3D&, const Field3D&, double,
        Field3D& dtheta_dt, Field3D& dqv_dt, Field3D& dqc_dt,
        Field3D& dqr_dt, Field3D& dqi_dt, Field3D& dqs_dt,
        Field3D& dqg_dt, Field3D& dqh_dt) override
    {
        const int nr = p.size_r();
        const int nth = p.size_th();
        const int nz = p.size_z();
        dtheta_dt.resize(nr, nth, nz, 0.0f);
        dqv_dt.resize(nr, nth, nz, 0.0f);
        dqc_dt.resize(nr, nth, nz, 0.0f);
        dqr_dt.resize(nr, nth, nz, -sink_rate_kgkg_s_);
        dqi_dt.resize(nr, nth, nz, 0.0f);
        dqs_dt.resize(nr, nth, nz, 0.0f);
        dqg_dt.resize(nr, nth, nz, 0.0f);
        dqh_dt.resize(nr, nth, nz, 0.0f);
    }

    double bottom_sedimentation_sink_rate(
        const Field3D&, const Field3D&, const Field3D&,
        const Field3D&, const Field3D&, int, int, double) const override
    {
        return sink_rate_kgkg_s_;
    }

    std::string get_scheme_name() const override { return "surface_fallout"; }
    int get_num_prognostic_vars() const override { return 1; }

private:
    float sink_rate_kgkg_s_;
};

class CombinedProcessFalloutScheme final : public MicrophysicsScheme
{
public:
    std::unique_ptr<MicrophysicsScheme> clone() const override
    { return std::make_unique<CombinedProcessFalloutScheme>(*this); }
    void compute_tendencies(
        const Field3D& p, const Field3D&, const Field3D&, const Field3D&,
        const Field3D&, const Field3D&, const Field3D&, const Field3D&,
        const Field3D&, const Field3D&, double,
        Field3D& dtheta_dt, Field3D& dqv_dt, Field3D& dqc_dt,
        Field3D& dqr_dt, Field3D& dqi_dt, Field3D& dqs_dt,
        Field3D& dqg_dt, Field3D& dqh_dt) override
    {
        const int nr = p.size_r();
        const int nth = p.size_th();
        const int nz = p.size_z();
        dtheta_dt.resize(nr, nth, nz, 0.0f);
        dqv_dt.resize(nr, nth, nz, 6.0e-4f);
        dqc_dt.resize(nr, nth, nz, 0.0f);
        dqr_dt.resize(nr, nth, nz, -1.2e-3f);
        dqi_dt.resize(nr, nth, nz, 0.0f);
        dqs_dt.resize(nr, nth, nz, 0.0f);
        dqg_dt.resize(nr, nth, nz, 0.0f);
        dqh_dt.resize(nr, nth, nz, 0.0f);
    }

    double bottom_sedimentation_sink_rate(
        const Field3D&, const Field3D&, const Field3D&,
        const Field3D&, const Field3D&, int, int, double) const override
    {
        return 6.0e-4;
    }

    bool requires_runtime_column_donor_admission() const noexcept override
    {
        return true;
    }

    std::string get_scheme_name() const override
    {
        return "combined_process_fallout";
    }

    int get_num_prognostic_vars() const override { return 1; }
};

void initialize_single_cell_state(float qv_value, float qc_value)
{
    test_stochastic_column_control_active = false;
    test_stochastic_column_multiplier = 1.0f;
    global_gpu_apply_microphysics = false;
    global_validation_policy = {};
    global_validation_policy.mode = tmv::GuardMode::Sanitize;

    NR = 1;
    NTH = 1;
    NZ = 1;
    dr = 2.0;
    dtheta = 3.0;
    dz = 4.0;
    dt = 1.0;
    simulation_time = 0.0;
    global_grid_geometry.initialize(NR, NTH, NZ, dr, dz, dtheta,
                                    CoordinateSystem::Cartesian);

    rho.resize(NR, NTH, NZ, 2.0f);
    p.resize(NR, NTH, NZ, 90000.0f);
    u.resize(NR, NTH, NZ, 0.0f);
    v.resize(NR, NTH, NZ, 0.0f);
    w.resize(NR, NTH, NZ, 0.0f);
    tracer.resize(NR, NTH, NZ, 0.0f);
    theta.resize(NR, NTH, NZ, 300.0f);
    qv.resize(NR, NTH, NZ, qv_value);
    qc.resize(NR, NTH, NZ, qc_value);
    qr.resize(NR, NTH, NZ, 0.0f);
    qi.resize(NR, NTH, NZ, 0.0f);
    qs.resize(NR, NTH, NZ, 0.0f);
    qg.resize(NR, NTH, NZ, 0.0f);
    qh.resize(NR, NTH, NZ, 0.0f);
    tke.resize(NR, NTH, NZ, 0.0f);

    dtheta_dt_rad.resize(NR, NTH, NZ, 0.0f);
    dtheta_dt_pbl.resize(NR, NTH, NZ, 0.0f);
    dqv_dt_pbl.resize(NR, NTH, NZ, 0.0f);
    dqc_dt_pbl.resize(NR, NTH, NZ, 0.0f);
    dqi_dt_pbl.resize(NR, NTH, NZ, 0.0f);
    du_dt_pbl.resize(NR, NTH, NZ, 0.0f);
    dv_dt_pbl.resize(NR, NTH, NZ, 0.0f);
    dtke_dt_pbl.resize(NR, NTH, NZ, 0.0f);

    global_radiation_runtime_state.held_atmospheric_power_w = 0.0;
    global_radiation_runtime_state.held_cell_power_w.release_storage();
    global_radiation_runtime_state.last_expected_atmospheric_energy_j = 0.0;
    global_radiation_runtime_state.last_applied_radiation_heat_j = 0.0;
    global_radiation_runtime_state.accumulated_expected_atmospheric_energy_j = 0.0;
    global_radiation_runtime_state.accumulated_applied_radiation_heat_j = 0.0;
    global_radiation_runtime_state.applied_step_count = 0;
    global_radiation_runtime_state.config.scheme_id = "none";
    global_simulation_storage.runtime.boundary_layer.config.clear_sky_only = true;
    global_microphysics_runtime_state.theta_storage_residual = {};
    global_simulation_storage.surface.precipitation_rate_mm_h.assign(1, 0.0f);
    global_simulation_storage.surface.accumulated_rainfall_mm.assign(1, 0.0f);
    global_simulation_storage.surface.accumulated_rainfall_residual_mm.assign(
        1, 0.0f);
    global_simulation_storage.surface.accumulated_rainfall_last_update_s = 0.0;

    rho0_base.assign(1, 2.0);
    p0_base.assign(1, 90000.0);
    qv0_base.assign(1, static_cast<double>(qv_value));
    theta0_base.assign(1, 300.0);
    u0_base.assign(1, 0.0);
    v0_base.assign(1, 0.0);
}

void initialize_threaded_ledger_state(int nr = 13, int nth = 9, int nz = 7)
{
    initialize_single_cell_state(0.010f, 0.001f);

    NR = nr;
    NTH = nth;
    NZ = nz;
    dr = 500.0;
    dtheta = 700.0;
    dz = 250.0;
    global_grid_geometry.initialize(
        NR, NTH, NZ, dr, dz, dtheta,
        CoordinateSystem::Cartesian, StaggerType::Collocated);
    const std::size_t surface_size =
        static_cast<std::size_t>(NR) * static_cast<std::size_t>(NTH);
    global_simulation_storage.surface.precipitation_rate_mm_h.assign(
        surface_size, 0.0f);
    global_simulation_storage.surface.accumulated_rainfall_mm.assign(
        surface_size, 0.0f);
    global_simulation_storage.surface.accumulated_rainfall_residual_mm.assign(
        surface_size, 0.0f);

    const auto resize = [](Field3D& field, float value)
    {
        field.resize(NR, NTH, NZ, value);
    };
    resize(rho, 1.0f);
    resize(p, 90000.0f);
    resize(u, 0.0f);
    resize(v, 0.0f);
    resize(w, 0.0f);
    resize(tracer, 0.0f);
    resize(theta, 300.0f);
    resize(qv, 0.010f);
    resize(qc, 0.001f);
    resize(qr, 0.0f);
    resize(qi, 0.0f);
    resize(qs, 0.0f);
    resize(qg, 0.0f);
    resize(qh, 0.0f);
    resize(tke, 0.0f);
    resize(dtheta_dt_rad, 0.0f);
    resize(dtheta_dt_pbl, 0.0f);
    resize(dqv_dt_pbl, 0.0f);
    resize(dqc_dt_pbl, 0.0f);
    resize(dqi_dt_pbl, 0.0f);
    resize(du_dt_pbl, 0.0f);
    resize(dv_dt_pbl, 0.0f);
    resize(dtke_dt_pbl, 0.0f);

    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                const float pattern = static_cast<float>(
                    (19 * i + 11 * j + 5 * k) % 31);
                rho[i][j][k] = 0.72f + 0.013f * pattern;
                p[i][j][k] = 72000.0f + 317.0f * pattern;
                theta[i][j][k] =
                    278.0f + 1.7f * static_cast<float>(k)
                    + 0.021f * pattern;
                qv[i][j][k] = 0.004f + 0.00017f * pattern;
                qc[i][j][k] =
                    0.00003f * static_cast<float>((i + 2 * j + k) % 13);
                u[i][j][k] = -8.0f + 0.37f * pattern;
                v[i][j][k] = 3.0f - 0.19f * pattern;
                w[i][j][k] = -1.0f + 0.08f * pattern;
                tke[i][j][k] = 0.02f * pattern;
            }

    rho0_base.assign(static_cast<std::size_t>(NZ), 1.0);
    p0_base.assign(static_cast<std::size_t>(NZ), 90000.0);
    qv0_base.assign(static_cast<std::size_t>(NZ), 0.010);
    theta0_base.assign(static_cast<std::size_t>(NZ), 300.0);
    u0_base.assign(static_cast<std::size_t>(NZ), 0.0);
    v0_base.assign(static_cast<std::size_t>(NZ), 0.0);
    microphysics_scheme = std::make_unique<StubMicrophysicsScheme>();
}

std::vector<double> thermodynamic_report_values(
    const tmv::MicrophysicsThermodynamicRepairReport& report)
{
    std::vector<double> values{
        report.microphysics_heat_j,
        report.radiation_heat_j,
        report.boundary_layer_heat_j,
        report.theta_tendency_limit_correction_j,
        report.theta_nonfinite_correction_j,
        report.theta_storage_rounding_j,
        report.theta_storage_residual_release_j,
        report.theta_storage_residual_j,
        report.theta_bound_correction_j,
        report.microphysics_water_tendency_kg};
    values.insert(
        values.end(),
        report.microphysics_water_tendency_kg_by_species.begin(),
        report.microphysics_water_tendency_kg_by_species.end());
    const std::array<double, 11> tail{{
        report.boundary_layer_vapor_tendency_kg,
        report.boundary_layer_cloud_liquid_tendency_kg,
        report.boundary_layer_cloud_ice_tendency_kg,
        report.moist_dry_internal_change_j,
        report.moist_water_sensible_change_j,
        report.moist_phase_reference_change_j,
        report.moist_carrier_kinetic_change_j,
        report.moist_gravitational_change_j,
        report.moist_subgrid_tke_change_j,
        report.moist_committed_energy_change_j,
        report.moist_energy_closure_residual_j}};
    values.insert(values.end(), tail.begin(), tail.end());
    return values;
}

bool all_equal(const Field3D& field, float expected)
{
    for (int i = 0; i < static_cast<int>(field.size_r()); ++i)
    {
        for (int j = 0; j < static_cast<int>(field.size_th()); ++j)
        {
            for (int k = 0; k < static_cast<int>(field.size_z()); ++k)
            {
                if (std::abs(field[i][j][k] - expected) > 1.0e-6f)
                {
                    return false;
                }
            }
        }
    }
    return true;
}

} // namespace

void tmv::initialize_microphysics(const std::string&)
{
}

void tmv::prepare_boundary_layer_tendencies_for_application(const SimulationStateView& state)
{
    if (!test_snapshot_reprojection_failure) return;
    for (auto* field : tmv::fields::pointers(state.fields, tmv::fields::microphysics_state))
        field->fill(-19.0f);
    for (auto* field : tmv::fields::pointers(state.tendencies, tmv::fields::physics_tendencies))
        if (field) field->fill(-23.0f);
    throw std::runtime_error("injected snapshot reprojection failure");
}

bool step_diagnostics_enabled()
{
    return false;
}

void tmv::chaos::apply_chaos_to_microphysics_tendencies(const SimulationStateView&,
                                                        Field3D&, Field3D&, Field3D&,
                                                        Field3D&, Field3D&, Field3D&,
                                                        Field3D&, Field3D&)
{
}

bool tmv::chaos::apply_chaos_to_microphysics_tendencies(
    const SimulationStateView& state,
    Field3D&, Field3D&, Field3D&, Field3D&,
    Field3D&, Field3D&, Field3D&, Field3D&,
    std::vector<float>& column_multipliers)
{
    column_multipliers.assign(
        static_cast<std::size_t>(state.grid.nr)
            * static_cast<std::size_t>(state.grid.nth),
        test_stochastic_column_multiplier);
    return test_stochastic_column_control_active;
}

bool dispatch_microphysics_apply_backend(
    ComputeBackendRuntimeState&,
    float*, float*, float*, float*, float*, float*, float*, float*,
    const float*, const float*, const float*, const float*,
    const float*, const float*, const float*, const float*,
    const float*, const float*, const float*,
    int, int, int,
    int, int, int, int,
    float, float,
    float, float, float,
    float, float,
    unsigned int*, int)
{
    return false;
}

// This unit owns the host-orchestration path. Native dispatch is exercised by
// the backend parity suites, so no device is active in this isolated fixture.
const ComputeBackend* active_compute_backend()
{
    return nullptr;
}

namespace nimbus::parity
{
bool sample(const char*)
{
    return false;
}

bool check_field(const char*, const char*, const float*, const float*,
                 int, int, int, Tier, ComparisonDomain, int)
{
    return true;
}
} // namespace nimbus::parity

TEST_CASE("legacy microphysics state uses the owning runtime",
          "[physics][microphysics][state][modularity]")
{
    REQUIRE(&microphysics_scheme ==
            &global_microphysics_runtime_state.scheme);
    REQUIRE(&last_microphysics_water_repair_report() ==
            &global_microphysics_runtime_state.last_water_repair_report);
    REQUIRE(&last_microphysics_thermodynamic_repair_report() ==
            &global_microphysics_runtime_state.last_thermodynamic_repair_report);
}

TEST_CASE("microphysics transaction and report scratch survives warm steps but not copies",
          "[physics][microphysics][allocation][transaction]")
{
    initialize_single_cell_state(0.010f, 0.001f);
    microphysics_scheme = std::make_unique<StubMicrophysicsScheme>();

    step_microphysics(global_simulation_state(), 1.0);
    auto& scratch = global_microphysics_runtime_state.execution_scratch;
    const auto* transaction = global_microphysics_runtime_state.subcycle_workspace.get();
    const int* clamp_k = scratch.clamp_count_by_k.data();
    const int* clamp_i = scratch.clamp_count_by_i.data();
    const auto* water_partials = scratch.water_report_partials.data();
    const auto* thermodynamic_partials =
        scratch.thermodynamic_report_partials.data();
    const auto* conservation_partials =
        scratch.conservation_report_partials.data();
    const float* precipitation_candidate =
        scratch.precipitation_rate_candidate_mm_h.data();
    const float* rainfall_candidate =
        scratch.accumulated_rainfall_candidate_mm.data();
    const float* rainfall_residual_candidate =
        scratch.accumulated_rainfall_residual_candidate_mm.data();

    step_microphysics(global_simulation_state(), 1.0);
    REQUIRE(global_microphysics_runtime_state.subcycle_workspace.get() == transaction);
    REQUIRE(scratch.clamp_count_by_k.data() == clamp_k);
    REQUIRE(scratch.clamp_count_by_i.data() == clamp_i);
    REQUIRE(scratch.water_report_partials.data() == water_partials);
    REQUIRE(scratch.thermodynamic_report_partials.data()
            == thermodynamic_partials);
    REQUIRE(scratch.conservation_report_partials.data()
            == conservation_partials);
    REQUIRE(scratch.precipitation_rate_candidate_mm_h.data()
            == precipitation_candidate);
    REQUIRE(scratch.accumulated_rainfall_candidate_mm.data()
            == rainfall_candidate);
    REQUIRE(scratch.accumulated_rainfall_residual_candidate_mm.data()
            == rainfall_residual_candidate);

    MicrophysicsRuntimeState source;
    source.execution_scratch.clamp_count_by_k.assign(2, 1);
    source.execution_scratch.water_report_partials.resize(2);
    source.execution_scratch.conservation_report_partials.resize(2);
    source.execution_scratch.precipitation_rate_candidate_mm_h.resize(2);
    const MicrophysicsRuntimeState copy(source);
    REQUIRE(copy.execution_scratch.clamp_count_by_k.empty());
    REQUIRE(copy.execution_scratch.water_report_partials.empty());
    REQUIRE(copy.execution_scratch.conservation_report_partials.empty());
    REQUIRE(copy.execution_scratch.precipitation_rate_candidate_mm_h.empty());
}

TEST_CASE("Microphysics snapshots restore all field owners across warm captures and shape changes",
          "[physics][microphysics][transaction][snapshot][thread]")
{
#ifdef _OPENMP
    struct RestoreWorkers
    {
        int count = omp_get_max_threads(), dynamic = omp_get_dynamic();
        ~RestoreWorkers() { omp_set_num_threads(count); omp_set_dynamic(dynamic); }
    } restore_workers;
    omp_set_dynamic(0);
    omp_set_num_threads(GENERATE(1, 4));
#endif
    const int optional_fields = GENERATE(0, 1, 2);
    struct RestoreInjection
    {
        ~RestoreInjection() { test_snapshot_reprojection_failure = false; }
    } restore_injection;
    global_microphysics_runtime_state.subcycle_workspace.reset();
    for (const auto shape : {std::array<int, 3>{129, 128, 128},
                             std::array<int, 3>{128, 129, 128},
                             std::array<int, 3>{7, 5, 3}})
    {
        initialize_threaded_ledger_state(shape[0], shape[1], shape[2]);
        theta.fill(300.0f);
        auto state = global_simulation_state();
        if (optional_fields == 0)
        {
            state.tendencies.dqc_dt_pbl = nullptr;
            state.tendencies.dqi_dt_pbl = nullptr;
        }
        else if (optional_fields == 2)
        {
            state.tendencies.dqc_dt_pbl = &state.fields.qv;
            state.tendencies.dqi_dt_pbl = &state.tendencies.dqv_dt_pbl;
        }
        const auto primary = tmv::fields::pointers(state.fields, tmv::fields::microphysics_state);
        const auto coupling = tmv::fields::pointers(state.tendencies, tmv::fields::physics_tendencies);
        std::array<Field3D, 16> before;
        for (std::size_t index = 0; index < primary.size(); ++index)
            before[index] = *primary[index];
        for (std::size_t index = 0; index < coupling.size(); ++index)
            if (coupling[index]) before[primary.size() + index] = *coupling[index];
        test_snapshot_reprojection_failure = true;
        for (int capture = 0; capture < 3; ++capture)
        {
            std::exception_ptr failure;
            const auto step = [&]
            {
                try { step_microphysics(state, 0.1); }
                catch (...) { failure = std::current_exception(); }
            };
            if (capture == 2)
            {
                #pragma omp parallel num_threads(2)
                {
                    #pragma omp single
                    step();
                }
            }
            else step();
            REQUIRE(failure);
            REQUIRE_THROWS_WITH(std::rethrow_exception(failure), "injected snapshot reprojection failure");
            const auto equal = [](const Field3D& value, const Field3D& reference)
            {
                return value.size_r() == reference.size_r() && value.size_th() == reference.size_th() &&
                    value.size_z() == reference.size_z() &&
                    std::memcmp(value.data(), reference.data(), value.size() * sizeof(float)) == 0;
            };
            for (std::size_t index = 0; index < primary.size(); ++index)
                CHECK(equal(*primary[index], before[index]));
            for (std::size_t index = 0; index < coupling.size(); ++index)
                if (coupling[index]) CHECK(equal(*coupling[index], before[primary.size() + index]));
        }
    }
}

TEST_CASE("Microphysics snapshot allocation failure leaves live owners untouched",
          "[physics][microphysics][transaction][snapshot][allocation]")
{
    class FailedClone final : public StubMicrophysicsScheme
    {
        std::unique_ptr<MicrophysicsScheme> clone() const override { throw std::bad_alloc(); }
    };
    initialize_threaded_ledger_state();
    microphysics_scheme = std::make_unique<FailedClone>();
    auto state = global_simulation_state();
    const auto primary = tmv::fields::pointers(state.fields, tmv::fields::microphysics_state);
    std::array<Field3D, 8> before;
    std::array<const float*, 8> owners;
    for (std::size_t index = 0; index < primary.size(); ++index)
    {
        before[index] = *primary[index];
        owners[index] = primary[index]->data();
    }
    auto* scheme = microphysics_scheme.get();
    REQUIRE_THROWS_AS(step_microphysics(state, 0.1), std::bad_alloc);
    CHECK(microphysics_scheme.get() == scheme);
    for (std::size_t index = 0; index < primary.size(); ++index)
    {
        CHECK(primary[index]->data() == owners[index]);
        CHECK(std::memcmp(primary[index]->data(), before[index].data(), before[index].size() * sizeof(float)) == 0);
    }
}

TEST_CASE("checkpointed thermodynamic ledger is bit-exact across CPU thread counts",
          "[physics][microphysics][ledger][thread][determinism]")
{
#ifdef _OPENMP
    const int previous_dynamic = omp_get_dynamic();
    const int previous_threads = omp_get_max_threads();
    const bool previous_deterministic =
        tmv::diagnostics::deterministic_reductions_enabled();
    omp_set_dynamic(0);
    tmv::diagnostics::set_deterministic_reductions(false);

    initialize_threaded_ledger_state();
    omp_set_num_threads(1);
    step_microphysics(global_simulation_state(), 1.0);
    const tmv::MicrophysicsThermodynamicRepairReport serial =
        last_microphysics_thermodynamic_repair_report();
    const tmv::MicrophysicsConservationReport serial_conservation =
        global_microphysics_runtime_state.last_conservation_report;
    const std::vector<double> serial_values =
        thermodynamic_report_values(serial);

    initialize_threaded_ledger_state();
    omp_set_num_threads(4);
    step_microphysics(global_simulation_state(), 1.0);
    const tmv::MicrophysicsThermodynamicRepairReport parallel =
        last_microphysics_thermodynamic_repair_report();
    const tmv::MicrophysicsConservationReport parallel_conservation =
        global_microphysics_runtime_state.last_conservation_report;
    const std::vector<double> parallel_values =
        thermodynamic_report_values(parallel);

    omp_set_num_threads(previous_threads);
    omp_set_dynamic(previous_dynamic);
    tmv::diagnostics::set_deterministic_reductions(previous_deterministic);

    REQUIRE(serial_values == parallel_values);
    REQUIRE(serial.moist_energy_complete == parallel.moist_energy_complete);
    REQUIRE(serial.theta_tendency_limit_count
            == parallel.theta_tendency_limit_count);
    REQUIRE(serial.theta_nonfinite_count == parallel.theta_nonfinite_count);
    REQUIRE(serial.theta_lower_bound_count
            == parallel.theta_lower_bound_count);
    REQUIRE(serial.theta_upper_bound_count
            == parallel.theta_upper_bound_count);
    REQUIRE(serial_conservation.active == parallel_conservation.active);
    REQUIRE(serial_conservation.tendency_imbalance_kg
            == parallel_conservation.tendency_imbalance_kg);
    REQUIRE(serial_conservation.baseline_q_kg
            == parallel_conservation.baseline_q_kg);
    REQUIRE(serial_conservation.surface_precipitation_kg
            == parallel_conservation.surface_precipitation_kg);
    REQUIRE(serial_conservation.relative_imbalance
            == parallel_conservation.relative_imbalance);
#else
    SUCCEED("OpenMP is unavailable in this build");
#endif
}

TEST_CASE("Parallel sedimentation admission preserves the first invalid column without mutation",
          "[physics][microphysics][subcycling][thread][atomicity]")
{
#ifdef _OPENMP
    struct RestoreWorkers
    {
        int threads = omp_get_max_threads();
        int dynamic = omp_get_dynamic();
        ~RestoreWorkers() { omp_set_num_threads(threads); omp_set_dynamic(dynamic); }
    } restore;
    omp_set_dynamic(0);
    omp_set_num_threads(GENERATE(1, 4));
#endif
    const bool density_first = GENERATE(false, true);
    initialize_threaded_ledger_state();
    microphysics_scheme = std::make_unique<LinearFalloutScheme>();
    if (density_first)
    {
        rho[1][1][NZ / 2] = -1.0f;
        qv[NR - 2][NTH - 2][NZ / 2] = -1.0f;
    }
    else
    {
        qv[1][1][NZ / 2] = -1.0f;
        rho[NR - 2][NTH - 2][NZ / 2] = -1.0f;
    }
    const std::vector<float> before_rho(rho.data(), rho.data() + rho.size());
    const std::vector<float> before_qv(qv.data(), qv.data() + qv.size());
    const double before_time = simulation_time;
    REQUIRE_THROWS_WITH(step_microphysics(global_simulation_state(), 1.0),
        Catch::Matchers::Contains(density_first ? "sedimentation density" : "sedimentation input"));
    REQUIRE(std::memcmp(rho.data(), before_rho.data(), rho.size() * sizeof(float)) == 0);
    REQUIRE(std::memcmp(qv.data(), before_qv.data(), qv.size() * sizeof(float)) == 0);
    REQUIRE(simulation_time == before_time);
    REQUIRE(dynamic_cast<LinearFalloutScheme&>(*microphysics_scheme).calls == 0);
}

TEST_CASE("Sedimentation and donor admission leave identical endpoints across worker counts",
          "[physics][microphysics][subcycling][donor][thread][determinism]")
{
#ifdef _OPENMP
    struct RestoreWorkers
    {
        int threads = omp_get_max_threads();
        int dynamic = omp_get_dynamic();
        ~RestoreWorkers() { omp_set_num_threads(threads); omp_set_dynamic(dynamic); }
    } restore;
    omp_set_dynamic(0);
    const bool column_admission = GENERATE(false, true);
    std::vector<std::vector<float>> serial;
    std::vector<float> serial_multipliers;
    std::vector<double> serial_report;
    for (int workers : {1, 4})
    {
        omp_set_num_threads(workers);
        initialize_threaded_ledger_state();
        if (column_admission)
        {
            microphysics_scheme = std::make_unique<WaterTransferScheme>(-0.02f, 0.02f);
            test_stochastic_column_control_active = true;
            test_stochastic_column_multiplier = 2.0f;
        }
        else
        {
            microphysics_scheme = std::make_unique<LinearFalloutScheme>();
            qr.fill(0.001f);
        }
        step_microphysics(global_simulation_state(), 1.0);
        const std::array<const Field3D*, 10> fields{&rho, &p, &theta, &qv, &qc, &qr, &qi, &qs, &qg, &qh};
        if (workers == 1)
        {
            for (const auto* field : fields) serial.emplace_back(field->data(), field->data() + field->size());
            serial_multipliers = global_microphysics_runtime_state.execution_scratch.stochastic_column_multipliers;
            serial_report = thermodynamic_report_values(last_microphysics_thermodynamic_repair_report());
        }
        else
        {
            for (std::size_t index = 0; index < fields.size(); ++index)
                REQUIRE(std::memcmp(fields[index]->data(), serial[index].data(), fields[index]->size() * sizeof(float)) == 0);
            REQUIRE(global_microphysics_runtime_state.execution_scratch.stochastic_column_multipliers == serial_multipliers);
            REQUIRE(thermodynamic_report_values(last_microphysics_thermodynamic_repair_report()) == serial_report);
        }
    }
#else
    SUCCEED("Serial build retains the same sedimentation admission path");
#endif
}

TEST_CASE("surface rainfall integrates exact scheme fallout every physics step",
          "[physics][microphysics][precipitation][surface]")
{
    initialize_single_cell_state(0.010f, 0.0f);
    qr[0][0][0] = 0.010f;
    microphysics_scheme = std::make_unique<SurfaceFalloutScheme>(0.001f);

    step_microphysics(global_simulation_state(), 2.0);
    auto& surface = global_simulation_storage.surface;
    REQUIRE(surface.precipitation_rate_mm_h.size() == 1);
    REQUIRE(surface.accumulated_rainfall_mm.size() == 1);
    REQUIRE(surface.precipitation_rate_mm_h[0] == Approx(28.8f));
    REQUIRE(surface.accumulated_rainfall_mm[0] == Approx(0.016f));
    REQUIRE(surface.accumulated_rainfall_last_update_s == Approx(2.0));

    simulation_time = 2.0;
    step_microphysics(global_simulation_state(), 2.0);
    REQUIRE(surface.precipitation_rate_mm_h[0] == Approx(28.8f));
    REQUIRE(surface.accumulated_rainfall_mm[0] == Approx(0.032f));
    REQUIRE(surface.accumulated_rainfall_last_update_s == Approx(4.0));
    surface.accumulated_rainfall_mm[0] = 100.0f;
    surface.accumulated_rainfall_residual_mm[0] = 0.0f;
    microphysics_scheme = std::make_unique<SurfaceFalloutScheme>(1.0e-9f);
    simulation_time = 4.0;
    step_microphysics(global_simulation_state(), 1.0);
    const double compensated_total =
        static_cast<double>(surface.accumulated_rainfall_mm[0])
        + static_cast<double>(surface.accumulated_rainfall_residual_mm[0]);
    REQUIRE(compensated_total == Approx(100.0 + 8.0e-9).margin(1.0e-11));
    REQUIRE(std::string(
                tmv::surface_precipitation_accumulation_algorithm_id()) ==
            "exact_bottom_sedimentation_fallout_accumulation_v1");
}

TEST_CASE("microphysics step fails closed when no scheme is initialized",
          "[physics][microphysics][configuration]")
{
    microphysics_scheme.reset();
    REQUIRE_THROWS_AS(
        step_microphysics(global_simulation_state(), 1.0),
        std::runtime_error);
}

TEST_CASE("exploratory execution fails closed on a scheme exception",
          "[physics][microphysics][execution][exploratory]")
{
    initialize_single_cell_state(0.010f, 0.001f);
    global_validation_policy.mode = tmv::GuardMode::Sanitize;
    microphysics_scheme = std::make_unique<ThrowingMicrophysicsScheme>();

    REQUIRE_THROWS_WITH(step_microphysics(global_simulation_state(), 1.0),
                        "injected tendency failure");
    REQUIRE(qv[0][0][0] == Approx(0.010f));
    REQUIRE(qc[0][0][0] == Approx(0.001f));
}

TEST_CASE("private-prognostic rejection preserves the primary transaction",
          "[physics][microphysics][transaction][moments]")
{
    initialize_single_cell_state(0.010f, 0.001f);
    auto rejecting =
        std::make_unique<RejectingPrivatePrognosticScheme>();
    microphysics_scheme = std::move(rejecting);

    REQUIRE_THROWS_WITH(
        step_microphysics(global_simulation_state(), 1.0),
        "microphysics private-prognostic update rejected: "
        "injected private-moment rejection");
    REQUIRE(qv[0][0][0] == Approx(0.010f));
    REQUIRE(qc[0][0][0] == Approx(0.001f));
    REQUIRE(theta[0][0][0] == Approx(300.0f));
    REQUIRE_FALSE(dynamic_cast<RejectingPrivatePrognosticScheme&>(*microphysics_scheme).committed);
}

TEST_CASE("post-primary private preparation failure rolls accepted fields back",
          "[physics][microphysics][transaction][post_primary]")
{
    for (const auto mode : {
             PostPrimaryCandidateScheme::Mode::RejectPrepare,
             PostPrimaryCandidateScheme::Mode::MissingToken})
    {
        initialize_single_cell_state(0.010f, 0.001f);
        const Field3D qv_before = qv;
        const Field3D qc_before = qc;
        const Field3D theta_before = theta;
        const auto precipitation_rate_before =
            global_simulation_storage.surface.precipitation_rate_mm_h;
        const auto accumulated_before =
            global_simulation_storage.surface.accumulated_rainfall_mm;
        const auto residual_before =
            global_simulation_storage.surface.accumulated_rainfall_residual_mm;
        const double precipitation_time_before =
            global_simulation_storage.surface.accumulated_rainfall_last_update_s;
        PostPrimaryProbe probe;
        microphysics_scheme =
            std::make_unique<PostPrimaryCandidateScheme>(mode, probe);

        if (mode == PostPrimaryCandidateScheme::Mode::RejectPrepare)
        {
            REQUIRE_THROWS_WITH(
                step_microphysics(global_simulation_state(), 1.0),
                "post-primary microphysics private candidate rejected: "
                "injected post-primary rejection");
        }
        else
        {
            REQUIRE_THROWS_WITH(
                step_microphysics(global_simulation_state(), 1.0),
                "post-primary microphysics private candidate was not prepared");
        }
        CHECK(probe.saw_accepted_primary);
        CHECK(probe.water_repairs ==
              &global_microphysics_runtime_state.last_water_repair_report);
        CHECK(probe.thermodynamic_repairs ==
              &global_microphysics_runtime_state.last_thermodynamic_repair_report);
        CHECK(probe.lower_bound_repairs == 0);
        CHECK(probe.lower_bound_roundoff_repairs == 0);
        CHECK(probe.upper_bound_repairs == 0);
        CHECK(probe.nonfinite_repairs == 0);
        CHECK(probe.thermal_bound_repairs == 0);
        CHECK(probe.thermal_nonfinite_repairs == 0);
        CHECK(probe.accepted_qv < qv_before(0, 0, 0));
        CHECK(probe.accepted_qc > qc_before(0, 0, 0));
        CHECK(probe.commits == 0);
        CHECK(std::memcmp(qv.data(), qv_before.data(),
                          qv.size() * sizeof(float)) == 0);
        CHECK(std::memcmp(qc.data(), qc_before.data(),
                          qc.size() * sizeof(float)) == 0);
        CHECK(std::memcmp(theta.data(), theta_before.data(),
                          theta.size() * sizeof(float)) == 0);
        CHECK(global_simulation_storage.surface.precipitation_rate_mm_h ==
              precipitation_rate_before);
        CHECK(global_simulation_storage.surface.accumulated_rainfall_mm ==
              accumulated_before);
        CHECK(global_simulation_storage.surface.accumulated_rainfall_residual_mm ==
              residual_before);
        CHECK(global_simulation_storage.surface.accumulated_rainfall_last_update_s ==
              precipitation_time_before);
        CHECK(dynamic_cast<PostPrimaryCandidateScheme&>(
            *microphysics_scheme).private_state().empty());
    }
}

TEST_CASE("prepared post-primary private candidate commits once after accepted fields",
          "[physics][microphysics][transaction][post_primary]")
{
    initialize_single_cell_state(0.010f, 0.001f);
    PostPrimaryProbe probe;
    microphysics_scheme = std::make_unique<PostPrimaryCandidateScheme>(
        PostPrimaryCandidateScheme::Mode::Commit, probe);
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1.0));
    CHECK(probe.saw_accepted_primary);
    CHECK(probe.water_repairs ==
          &global_microphysics_runtime_state.last_water_repair_report);
    CHECK(probe.thermodynamic_repairs ==
          &global_microphysics_runtime_state.last_thermodynamic_repair_report);
    CHECK(probe.lower_bound_repairs == 0);
    CHECK(probe.upper_bound_repairs == 0);
    CHECK(probe.nonfinite_repairs == 0);
    CHECK(probe.thermal_bound_repairs == 0);
    CHECK(probe.thermal_nonfinite_repairs == 0);
    CHECK(probe.accepted_qv == qv(0, 0, 0));
    CHECK(probe.accepted_qc == qc(0, 0, 0));
    CHECK(probe.commits == 1);
    const auto& private_state = dynamic_cast<PostPrimaryCandidateScheme&>(
        *microphysics_scheme).private_state();
    REQUIRE(private_state.size() == 1);
    CHECK(private_state(0, 0, 0) == 43.0f);
}

TEST_CASE("qualified execution terminates on a scheme exception",
          "[physics][microphysics][execution][qualified]")
{
    initialize_single_cell_state(0.010f, 0.001f);
    global_validation_policy.mode = tmv::GuardMode::Strict;
    microphysics_scheme = std::make_unique<ThrowingMicrophysicsScheme>();

    REQUIRE_THROWS_WITH(
        step_microphysics(global_simulation_state(), 1.0),
        "injected tendency failure");
    REQUIRE(qv[0][0][0] == Approx(0.010f));
    REQUIRE(qc[0][0][0] == Approx(0.001f));
    global_validation_policy = {};
}

TEST_CASE("microphysics rejects a scheme tendency with the wrong shape",
          "[physics][microphysics][execution]")
{
    initialize_single_cell_state(0.010f, 0.001f);
    microphysics_scheme = std::make_unique<WrongShapeMicrophysicsScheme>();

    REQUIRE_THROWS_WITH(
        step_microphysics(global_simulation_state(), 1.0),
        "microphysics scheme returned a tendency with the wrong shape");
    REQUIRE(qv[0][0][0] == Approx(0.010f));
    REQUIRE(qc[0][0][0] == Approx(0.001f));
}

TEST_CASE("microphysics runtime step uses explicit SimulationStateView")
{
    global_gpu_apply_microphysics = false;
    // Deliberately give the process-global runtime a different scheme. The
    // explicit state must dispatch through its own scheme and scratch/ledger.
    microphysics_scheme = std::make_unique<StubMicrophysicsScheme>();
    MicrophysicsRuntimeState local_microphysics_runtime;
    local_microphysics_runtime.scheme =
        std::make_unique<WaterTransferScheme>(-0.001f, 0.0005f);

    NR = 2;
    NTH = 2;
    NZ = 2;
    dr = 100.0;
    dtheta = 100.0;
    dz = 50.0;
    dt = 1.0;
    global_grid_geometry.initialize(NR, NTH, NZ, dr, dz, dtheta,
                                    CoordinateSystem::Cartesian);

    p.resize(NR, NTH, NZ, 90000.0f);
    theta.resize(NR, NTH, NZ, 280.0f);
    qv.resize(NR, NTH, NZ, 0.003f);
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 0.0f);
    qi.resize(NR, NTH, NZ, 0.0f);
    qs.resize(NR, NTH, NZ, 0.0f);
    qg.resize(NR, NTH, NZ, 0.0f);
    qh.resize(NR, NTH, NZ, 0.0f);

    dtheta_dt_rad.resize(NR, NTH, NZ, 0.0f);
    dtheta_dt_pbl.resize(NR, NTH, NZ, 0.0f);
    dqv_dt_pbl.resize(NR, NTH, NZ, 0.0f);

    int local_nr = 2;
    int local_nth = 2;
    int local_nz = 2;
    double local_dr = 200.0;
    double local_dtheta = 200.0;
    double local_dz = 100.0;
    double local_dt = 1.0;
    GridGeometry local_geometry;
    local_geometry.initialize(local_nr, local_nth, local_nz,
                              local_dr, local_dz, local_dtheta,
                              CoordinateSystem::Cartesian);

    Field3D local_rho(local_nr, local_nth, local_nz, 1.0f);
    Field3D local_p(local_nr, local_nth, local_nz, 90000.0f);
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

    std::vector<double> local_rho0(local_nz, 1.0);
    std::vector<double> local_p0(local_nz, 90000.0);
    std::vector<double> local_qv0(local_nz, 0.010);
    std::vector<double> local_theta0(local_nz, 300.0);
    std::vector<double> local_u0(local_nz, 0.0);
    std::vector<double> local_v0(local_nz, 0.0);
    double local_coriolis = 0.0;
    double local_time = 0.0;
    AdvectionRuntimeState local_advection_runtime;

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
        {}, nullptr, nullptr, nullptr, nullptr, nullptr,
        &local_microphysics_runtime, &local_advection_runtime};

    SimulationRuntimeState local_owners;
    RuntimeConfigurationState local_configuration;
    tmv::domain::HorizontalDomainFrame local_frame;
    local_owners.bind(local_state);
    local_state.configuration = &local_configuration;
    local_state.domain_frame = &local_frame;
    local_state.microphysics_runtime = &local_microphysics_runtime;
    local_state.advection_runtime = &local_advection_runtime;
    step_microphysics(local_state, 2.0);

    REQUIRE(all_equal(local_theta, 300.0f));
    REQUIRE(all_equal(local_qv, 0.008f));
    REQUIRE(all_equal(local_qc, 0.001f));
    REQUIRE(all_equal(theta, 280.0f));
    REQUIRE(all_equal(qv, 0.003f));
    REQUIRE(all_equal(qc, 0.0f));
    REQUIRE(last_microphysics_water_repair_report(local_state)
                .net_correction_kg() == Approx(0.0));
    REQUIRE(local_advection_runtime.activity_current);
    REQUIRE(local_advection_runtime.hydrometeor_active[0]);
    REQUIRE_FALSE(local_advection_runtime.hydrometeor_active[1]);
    REQUIRE(local_advection_runtime.nr == local_nr);
    REQUIRE(local_advection_runtime.nth == local_nth);
    REQUIRE(local_advection_runtime.nz == local_nz);
}

TEST_CASE("microphysics water ledger records lower-bound mass addition")
{
    initialize_single_cell_state(0.001f, 0.0f);
    // Repair accounting must use the live density, matching the global water
    // budget, rather than silently substituting the base-state density.
    rho[0][0][0] = 3.0f;
    microphysics_scheme =
        std::make_unique<WaterTransferScheme>(-0.002f, 0.002f);

    step_microphysics(global_simulation_state(), 1.0);

    const auto& report = last_microphysics_water_repair_report();
    REQUIRE(qv[0][0][0] == Approx(0.0f));
    REQUIRE(qc[0][0][0] == Approx(0.002f));
    REQUIRE(report.lower_bound_count() == 1);
    REQUIRE(report.upper_bound_count() == 0);
    REQUIRE(report.lower_bound_correction_kg() == Approx(0.072).margin(1.0e-6));
    REQUIRE(report.net_correction_kg() == Approx(0.072).margin(1.0e-6));
}

TEST_CASE("microphysics water ledger records upper-bound mass removal")
{
    initialize_single_cell_state(0.049f, 0.010f);
    microphysics_scheme =
        std::make_unique<WaterTransferScheme>(0.002f, -0.002f);

    step_microphysics(global_simulation_state(), 1.0);

    const auto& report = last_microphysics_water_repair_report();
    REQUIRE(qv[0][0][0] == Approx(qv_max_kgkg));
    REQUIRE(qc[0][0][0] == Approx(0.008f));
    REQUIRE(report.lower_bound_count() == 0);
    REQUIRE(report.upper_bound_count() == 1);
    REQUIRE(report.upper_bound_correction_kg() == Approx(-0.048).margin(1.0e-6));
    REQUIRE(report.net_correction_kg() == Approx(-0.048).margin(1.0e-6));
}

TEST_CASE("conservative microphysics update has a zero water-repair ledger")
{
    initialize_single_cell_state(0.010f, 0.005f);
    microphysics_scheme =
        std::make_unique<WaterTransferScheme>(-0.001f, 0.001f);

    step_microphysics(global_simulation_state(), 1.0);

    const auto& report = last_microphysics_water_repair_report();
    REQUIRE(qv[0][0][0] + qc[0][0][0] == Approx(0.015f));
    REQUIRE(report.lower_bound_count() == 0);
    REQUIRE(report.upper_bound_count() == 0);
    REQUIRE(report.nonfinite_count() == 0);
    REQUIRE(report.net_correction_kg() == Approx(0.0));
}

TEST_CASE("stochastic microphysics admits one conservative factor against the full column reservoir",
          "[physics][microphysics][chaos][conservation]")
{
    initialize_single_cell_state(0.001f, 0.0f);
    global_validation_policy.mode = tmv::GuardMode::Strict;
    global_validation_policy.fail_on = tmv::GuardFailOn::Bounds;
    microphysics_scheme =
        std::make_unique<WaterTransferScheme>(-0.002f, 0.002f);
    test_stochastic_column_control_active = true;
    test_stochastic_column_multiplier = 2.0f;

    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1.0));
    REQUIRE(qv[0][0][0] >= 0.0f);
    REQUIRE(qv[0][0][0] < 1.0e-8f);
    REQUIRE(qc[0][0][0] == Approx(0.001f).margin(2.0e-7));
    REQUIRE(qv[0][0][0] + qc[0][0][0]
            == Approx(0.001f).margin(2.0e-7));
    REQUIRE(last_microphysics_water_repair_report().net_correction_kg()
            == Approx(0.0));
}

TEST_CASE("stochastic sedimentation scales the column divergence and surface fallout together",
          "[physics][microphysics][chaos][sedimentation][conservation]")
{
    initialize_single_cell_state(0.010f, 0.0f);
    qr[0][0][0] = 0.010f;
    microphysics_scheme = std::make_unique<SurfaceFalloutScheme>(0.001f);
    test_stochastic_column_control_active = true;
    test_stochastic_column_multiplier = 0.5f;

    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1.0));
    REQUIRE(qr[0][0][0] == Approx(0.0095f));
    REQUIRE(global_simulation_storage.surface.precipitation_rate_mm_h[0]
            == Approx(14.4f));
    REQUIRE(global_microphysics_runtime_state.last_conservation_report
                .tendency_imbalance_kg
            == Approx(0.0).margin(1.0e-9));
    REQUIRE(last_microphysics_water_repair_report().net_correction_kg()
            == Approx(0.0));
}

TEST_CASE("runtime donor admission bounds combined process and fallout without chaos",
          "[physics][microphysics][donor][sedimentation][conservation]")
{
    initialize_single_cell_state(0.010f, 0.0f);
    qr[0][0][0] = 0.001f;
    global_validation_policy.mode = tmv::GuardMode::Strict;
    global_validation_policy.fail_on = tmv::GuardFailOn::Bounds;
    microphysics_scheme = std::make_unique<CombinedProcessFalloutScheme>();

    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1.0));
    REQUIRE(qr[0][0][0] >= 0.0f);
    REQUIRE(qr[0][0][0] < 1.0e-8f);
    REQUIRE(qv[0][0][0] == Approx(0.0105f).margin(2.0e-7));
    REQUIRE(global_simulation_storage.surface.precipitation_rate_mm_h[0]
            == Approx(14.4f).margin(2.0e-3));
    REQUIRE(global_microphysics_runtime_state.last_conservation_report
                .tendency_imbalance_kg
            == Approx(0.0).margin(1.0e-8));
    REQUIRE(last_microphysics_water_repair_report().net_correction_kg()
            == Approx(0.0));

    global_validation_policy = {};
}

TEST_CASE("rejected nonfinite replacement preserves the accepted water ledger")
{
    initialize_single_cell_state(0.010f, 0.0f);
    microphysics_scheme = std::make_unique<WaterTransferScheme>(
        std::numeric_limits<float>::max(), 0.0f);

    REQUIRE_THROWS_WITH(
        step_microphysics(global_simulation_state(), 2.0),
        Catch::Contains("non-finite tendency or state repair"));

    const auto& report = last_microphysics_water_repair_report();
    REQUIRE(qv[0][0][0] == Approx(0.010f));
    REQUIRE(report.nonfinite_count() == 0);
    REQUIRE(report.nonfinite_correction_kg() == Approx(0.0).margin(1.0e-6));
    REQUIRE(report.net_correction_kg() == Approx(0.0).margin(1.0e-6));
}

TEST_CASE("strict bounds validation rejects a microphysics water repair")
{
    initialize_single_cell_state(0.001f, 0.0f);
    global_validation_policy.mode = tmv::GuardMode::Strict;
    global_validation_policy.fail_on = tmv::GuardFailOn::Bounds;
    microphysics_scheme =
        std::make_unique<WaterTransferScheme>(-0.002f, 0.002f);

    REQUIRE_THROWS_AS(
        step_microphysics(global_simulation_state(), 1.0), std::runtime_error);
    REQUIRE(last_microphysics_water_repair_report().lower_bound_count() == 0);
    REQUIRE(qv[0][0][0] == Approx(0.001f));
    REQUIRE(qc[0][0][0] == Approx(0.0f));

    global_validation_policy = {};
}

TEST_CASE("strict bounds admits only ledgered fp32 reservoir cancellation")
{
    initialize_single_cell_state(0.001f, 0.0f);
    global_validation_policy.mode = tmv::GuardMode::Strict;
    global_validation_policy.fail_on = tmv::GuardFailOn::Bounds;
    const float exact_drain = -qv[0][0][0];
    const float rounded_drain = std::nextafter(
        exact_drain, -std::numeric_limits<float>::infinity());
    microphysics_scheme = std::make_unique<WaterTransferScheme>(
        rounded_drain, -rounded_drain);

    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1.0));
    const auto& report = last_microphysics_water_repair_report();
    CHECK(qv[0][0][0] == 0.0f);
    CHECK(report.lower_bound_count() == 1);
    CHECK(report.lower_bound_roundoff_count() == 1);
    CHECK(report.lower_bound_roundoff_correction_kg()
          == Approx(report.lower_bound_correction_kg()));
    CHECK(report.lower_bound_correction_kg() > 0.0);

    global_validation_policy = {};
}

TEST_CASE("declined device apply records each water storage rounding once",
          "[physics][microphysics][ledger][declined-device-apply]")
{
    auto& compute = global_simulation_storage.runtime.compute;
    struct RestoreBackend
    {
        ComputeBackendRuntimeState& runtime;
        std::unique_ptr<ComputeBackend> previous;
        ~RestoreBackend() { runtime.backend = std::move(previous); }
    } restore{compute, std::move(compute.backend)};
    compute.backend = std::make_unique<DeclinedApplyBackend>();
    for (const auto mode : {tmv::GuardMode::Strict, tmv::GuardMode::Warn, tmv::GuardMode::Sanitize})
    {
        initialize_single_cell_state(0.0101234f, 0.00054321f);
        global_validation_policy.mode = mode;
        global_gpu_apply_microphysics = true;
        constexpr float rate = 0.00000011f;
        constexpr double interval = 0.25;
        const float vapor_before = qv[0][0][0];
        const float cloud_before = qc[0][0][0];
        microphysics_scheme = std::make_unique<WaterTransferScheme>(-rate, rate);
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), interval));
        const auto& report = last_microphysics_water_repair_report();
        const double mass = 2.0 * 2.0 * 3.0 * 4.0;
        const double vapor_rounding = (static_cast<double>(qv[0][0][0]) -
            (static_cast<double>(vapor_before) - static_cast<double>(rate) * interval)) * mass;
        const double cloud_rounding = (static_cast<double>(qc[0][0][0]) -
            (static_cast<double>(cloud_before) + static_cast<double>(rate) * interval)) * mass;
        REQUIRE(vapor_rounding != 0.0);
        REQUIRE(cloud_rounding != 0.0);
        CHECK(report.species[0].storage_rounding_kg == Approx(vapor_rounding).epsilon(1.0e-12));
        CHECK(report.species[1].storage_rounding_kg == Approx(cloud_rounding).epsilon(1.0e-12));
        CHECK(report.lower_bound_count() == 0);
        CHECK(report.nonfinite_count() == 0);
    }
    global_gpu_apply_microphysics = false;
    global_validation_policy = {};
}

TEST_CASE("strict microphysics conservation gate is active without optional diagnostics",
          "[physics][microphysics][conservation]")
{
    initialize_single_cell_state(0.010f, 0.0f);
    global_validation_policy.mode = tmv::GuardMode::Strict;
    microphysics_scheme =
        std::make_unique<WaterTransferScheme>(-0.001f, 0.0f);

    REQUIRE_THROWS_WITH(
        step_microphysics(global_simulation_state(), 1.0),
        Catch::Contains("non-conservative microphysics tendencies"));
    REQUIRE(qv[0][0][0] == Approx(0.010f));
    REQUIRE(qc[0][0][0] == Approx(0.0f));

    global_validation_policy = {};
}

TEST_CASE("strict shared projection closes all seven species with cloud coupling",
          "[physics][microphysics][projection][ledger][thread]")
{
#ifdef _OPENMP
    struct RestoreWorkers
    {
        int count = omp_get_max_threads(), dynamic = omp_get_dynamic();
        ~RestoreWorkers() { omp_set_num_threads(count); omp_set_dynamic(dynamic); }
    } restore;
    omp_set_dynamic(0);
#endif
    std::array<double, 7> serial_rounding{};
    std::vector<double> serial_heat;
    constexpr float rate = 0x1.23456p-22f;
    constexpr double interval = 0.375;
    const std::array<float, 7> rates{{-6 * rate, rate, rate, rate, rate, rate, rate}};
    for (const int workers : {1, 4})
    {
#ifdef _OPENMP
        omp_set_num_threads(workers);
#endif
        initialize_threaded_ledger_state();
        global_validation_policy.mode = tmv::GuardMode::Strict;
        global_validation_policy.fail_on = tmv::GuardFailOn::Bounds;
        global_simulation_storage.runtime.boundary_layer.config.clear_sky_only = false;
        microphysics_scheme = std::make_unique<WaterTransferScheme>(rates);
        const std::array<Field3D*, 7> water{{&qv, &qc, &qr, &qi, &qs, &qg, &qh}};
        for (std::size_t index = 0; index < qv.size(); ++index)
        {
            dqv_dt_pbl.data()[index] = -2 * rate;
            dqc_dt_pbl.data()[index] = rate;
            dqi_dt_pbl.data()[index] = rate;
        }
        std::array<Field3D, 7> before;
        for (std::size_t s = 0; s < water.size(); ++s) before[s] = *water[s];
        REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), interval));
        const auto& report = last_microphysics_water_repair_report();
        CHECK(report.net_correction_kg() == 0.0);
        CHECK(report.lower_bound_count() == 0);
        CHECK(report.nonfinite_count() == 0);
        for (std::size_t s = 0; s < water.size(); ++s)
        {
            const double pbl_rate = s == 0 ? -2 * rate : (s == 1 || s == 3 ? rate : 0);
            double rounding = 0.0;
            bool exact = true;
            const auto domain = physical_scalar_domain(global_simulation_state().grid);
            for (int i = 0; i < NR; ++i)
                for (int j = 0; j < NTH; ++j)
                    for (int k = 0; k < NZ; ++k)
                    {
                        const bool physical = i >= domain.i_begin && i < domain.i_end
                            && j >= domain.j_begin && j < domain.j_end
                            && k >= domain.k_begin && k < domain.k_end;
                        const double candidate = static_cast<double>(before[s][i][j][k])
                            + (physical ? (static_cast<double>(rates[s]) + pbl_rate) * interval : 0.0);
                        exact = exact && (*water[s])[i][j][k] == static_cast<float>(candidate);
                        if (physical)
                            rounding += (static_cast<double>(static_cast<float>(candidate)) - candidate)
                                * (static_cast<double>(rho[i][j][k]) * (dr * dtheta * dz));
                    }
            CHECK(exact);
            CHECK(report.species[s].storage_rounding_kg == Approx(rounding).margin(1.0e-12));
            if (workers == 1) serial_rounding[s] = report.species[s].storage_rounding_kg;
            else CHECK(report.species[s].storage_rounding_kg == serial_rounding[s]);
        }
        CHECK(report.species[0].storage_rounding_kg != 0.0);
        const auto heat = thermodynamic_report_values(last_microphysics_thermodynamic_repair_report());
        if (workers == 1) serial_heat = heat;
        else CHECK(heat == serial_heat);
    }
    global_validation_policy = {};
}

TEST_CASE("strict projection preserves water rejection priority and accepted state",
          "[physics][microphysics][projection][atomicity]")
{
    const auto species = GENERATE(0u, 1u, 2u, 3u, 4u, 5u, 6u);
    initialize_threaded_ledger_state();
    global_validation_policy.mode = tmv::GuardMode::Strict;
    global_validation_policy.fail_on = tmv::GuardFailOn::Bounds;
    microphysics_scheme = std::make_unique<WaterTransferScheme>(0.0f, 0.0f);
    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1.0));
    const auto accepted_heat = thermodynamic_report_values(last_microphysics_thermodynamic_repair_report());
    const auto accepted_conservation = global_microphysics_runtime_state.last_conservation_report;
    const std::array<Field3D*, 8> primary{{&theta, &qv, &qc, &qr, &qi, &qs, &qg, &qh}};
    std::array<Field3D, 8> before;
    for (std::size_t s = 0; s < primary.size(); ++s) before[s] = *primary[s];
    std::array<float, 7> rates{};
    rates[species] = -0.02f;
    rates[(species + 1) % 7] = 0.02f;
    microphysics_scheme = std::make_unique<WaterTransferScheme>(rates);
    dtheta_dt_pbl[1][1][1] = 100.0f;
    REQUIRE_THROWS_WITH(step_microphysics(global_simulation_state(), 1.0),
                       Catch::Contains("strict validation rejected microphysics water repair"));
    for (std::size_t s = 0; s < primary.size(); ++s)
        CHECK(std::memcmp(primary[s]->data(), before[s].data(), before[s].size() * sizeof(float)) == 0);
    CHECK(thermodynamic_report_values(last_microphysics_thermodynamic_repair_report()) == accepted_heat);
    CHECK(last_microphysics_water_repair_report().net_correction_kg() == 0.0);
    CHECK(global_microphysics_runtime_state.last_conservation_report.baseline_q_kg
          == accepted_conservation.baseline_q_kg);
    CHECK(simulation_time == 0.0);
    global_validation_policy = {};
}

TEST_CASE("measure strict microphysics projection traversal", "[.microphysics-projection-benchmark]")
{
    const auto setting = [](const char* name, int fallback)
    {
        const char* value = std::getenv(name);
        return value ? std::stoi(value) : fallback;
    };
    const int workers = setting("NIMBUS_MP_BENCHMARK_THREADS", 1);
    const int horizontal = setting("NIMBUS_MP_BENCHMARK_NX", 128);
    REQUIRE(workers >= 1);
    REQUIRE(workers <= 64);
    REQUIRE(horizontal >= 32);
    REQUIRE(horizontal <= 640);
#ifdef _OPENMP
    struct RestoreWorkers
    {
        int count = omp_get_max_threads(), dynamic = omp_get_dynamic();
        ~RestoreWorkers() { omp_set_num_threads(count); omp_set_dynamic(dynamic); }
    } restore;
    omp_set_dynamic(0);
    omp_set_num_threads(workers);
#else
    REQUIRE(workers == 1);
#endif
    initialize_threaded_ledger_state(horizontal, horizontal, 180);
    std::fill(theta.data(), theta.data() + theta.size(), 300.0f);
    global_validation_policy.mode = tmv::GuardMode::Strict;
    global_validation_policy.fail_on = tmv::GuardFailOn::Both;
    constexpr float rate = 0x1.23456p-22f;
    microphysics_scheme = std::make_unique<WaterTransferScheme>(
        std::array<float, 7>{{-6 * rate, rate, rate, rate, rate, rate, rate}});
    struct RestoreTiming
    {
        bool enabled = global_perf_timing_enabled;
        ~RestoreTiming() { global_perf_timing_enabled = enabled; }
    } restore_timing;
    global_perf_timing_enabled = true;
    std::array<double, 6> projection_ms{}, transaction_ms{}, step_ms{};
    double first_transaction_ms = 0.0;
    global_microphysics_runtime_state.subcycle_workspace.reset();
    for (int iteration = -2; iteration < 6; ++iteration)
    {
        tmv::reset_microphysics_perf_stats();
        const auto start = std::chrono::steady_clock::now();
        step_microphysics(global_simulation_state(), 0.375);
        const double elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        if (iteration == -2)
            first_transaction_ms = 1000 * tmv::microphysics_perf_snapshot().transaction_snapshot_s;
        if (iteration >= 0)
        {
            projection_ms[iteration] = 1000 * tmv::microphysics_perf_snapshot().projection_s;
            transaction_ms[iteration] = 1000 * tmv::microphysics_perf_snapshot().transaction_snapshot_s;
            step_ms[iteration] = elapsed;
        }
    }
    std::uint64_t fingerprint = 14695981039346656037ull;
    const auto hash_bytes = [&fingerprint](const void* data, std::size_t bytes)
    {
        const auto* values = static_cast<const unsigned char*>(data);
        for (std::size_t index = 0; index < bytes; ++index)
            fingerprint = (fingerprint ^ values[index]) * 1099511628211ull;
    };
    for (const auto* field : {&theta, &qv, &qc, &qr, &qi, &qs, &qg, &qh})
        hash_bytes(field->data(), field->size() * sizeof(float));
    const auto heat = thermodynamic_report_values(last_microphysics_thermodynamic_repair_report());
    hash_bytes(heat.data(), heat.size() * sizeof(double));
    for (const auto& species : last_microphysics_water_repair_report().species)
        hash_bytes(&species.storage_rounding_kg, sizeof(double));
    REQUIRE(last_microphysics_water_repair_report().net_correction_kg() == 0.0);
    std::cout << std::setprecision(17)
              << "NIMBUS_MP_BENCHMARK {\"threads\":" << workers
              << ",\"shape\":[" << horizontal << ',' << horizontal << ",180]"
              << ",\"fingerprint\":\"" << std::hex << fingerprint << std::dec
              << "\",\"projection_ms\":[";
    for (std::size_t index = 0; index < projection_ms.size(); ++index)
        std::cout << (index ? "," : "") << projection_ms[index];
    std::cout << "],\"step_ms\":[";
    for (std::size_t index = 0; index < step_ms.size(); ++index)
        std::cout << (index ? "," : "") << step_ms[index];
    std::cout << "],\"transaction_ms\":[";
    for (std::size_t index = 0; index < transaction_ms.size(); ++index)
        std::cout << (index ? "," : "") << transaction_ms[index];
    std::cout << "],\"first_transaction_ms\":" << first_transaction_ms << "}" << std::endl;
    global_validation_policy = {};
}

TEST_CASE("thermodynamic ledger attributes physics sources and closes heat")
{
    initialize_single_cell_state(0.010f, 0.0f);
    global_radiation_runtime_state.config.scheme_id = "simple_grey";
    microphysics_scheme = std::make_unique<StubMicrophysicsScheme>();
    dtheta_dt_rad[0][0][0] = 3.0f;
    dtheta_dt_pbl[0][0][0] = -1.0f;
    dqv_dt_pbl[0][0][0] = 0.00025f;
    const double exner = std::pow(0.9, 287.0 / 1004.0);
    const double heat_factor = 2.0 * 1004.0 * exner * (2.0 * 3.0 * 4.0);
    global_radiation_runtime_state.held_cell_power_w.resize(
        1, 1, 1, static_cast<float>(3.0 * heat_factor));
    global_radiation_runtime_state.held_atmospheric_power_w =
        static_cast<double>(
            global_radiation_runtime_state.held_cell_power_w[0][0][0]);

    step_microphysics(global_simulation_state(), 2.0);

    const auto& report = last_microphysics_thermodynamic_repair_report();
    REQUIRE(theta[0][0][0] == Approx(308.0f));
    REQUIRE(report.moist_energy_complete);
    const double dry_mass = 2.0 * (2.0 * 3.0 * 4.0);
    const double cv_d =
        physical_constants::specific_heat_cp_jkgk
        - physical_constants::gas_constant_dry_air_jkgk;
    const double cv_v =
        physical_constants::specific_heat_water_vapor_cp_jkgk
        - physical_constants::gas_constant_water_vapor_jkgk;
    const double c_l =
        physical_constants::specific_heat_liquid_water_jkgk;
    const double t_ref = physical_constants::freezing_temperature_k;
    const double temperature_old = 300.0 * exner;
    const double temperature_new =
        static_cast<double>(theta[0][0][0]) * exner;
    REQUIRE(report.moist_dry_internal_change_j
            == Approx(
                   dry_mass * cv_d
                   * (temperature_new - temperature_old))
                   .epsilon(1.0e-12));
    REQUIRE(report.moist_water_sensible_change_j
            == Approx(
                   dry_mass
                   * ((temperature_new - t_ref)
                          * (static_cast<double>(qv[0][0][0]) * cv_v
                             + static_cast<double>(qc[0][0][0]) * c_l)
                      - (temperature_old - t_ref)
                          * (static_cast<double>(0.010f) * cv_v)))
                   .epsilon(1.0e-12));
    REQUIRE(report.moist_phase_reference_change_j
            == Approx(
                   dry_mass
                   * (static_cast<double>(qv[0][0][0])
                      - static_cast<double>(0.010f))
                   * (physical_constants::latent_heat_vaporization_jkg
                      - physical_constants::gas_constant_water_vapor_jkgk
                          * t_ref))
                   .epsilon(1.0e-12));
    REQUIRE(report.moist_carrier_kinetic_change_j == Approx(0.0));
    REQUIRE(report.moist_subgrid_tke_change_j == Approx(0.0));
    REQUIRE(report.moist_energy_closure_residual_j
            == report.moist_committed_energy_change_j
                - (report.moist_dry_internal_change_j
                   + report.moist_water_sensible_change_j
                   + report.moist_phase_reference_change_j
                   + report.moist_carrier_kinetic_change_j
                   + report.moist_gravitational_change_j
                   + report.moist_subgrid_tke_change_j));
    REQUIRE(report.radiation_heat_j
            == Approx(2.0 * global_radiation_runtime_state
                              .held_atmospheric_power_w)
                   .epsilon(1.0e-12));
    REQUIRE(report.boundary_layer_heat_j
            == Approx(-0.5 * report.microphysics_heat_j).epsilon(1.0e-12));
    REQUIRE(report.committed_heat_j()
            == Approx(report.requested_heat_j()).epsilon(1.0e-12));
    REQUIRE(report.net_theta_repair_j() == Approx(0.0));
    REQUIRE(report.microphysics_water_tendency_kg
            == Approx(-0.048).margin(1.0e-6));
    REQUIRE(report.microphysics_water_tendency_kg_by_species[0]
            == Approx(-0.096).margin(1.0e-6));
    REQUIRE(report.microphysics_water_tendency_kg_by_species[1]
            == Approx(0.048).margin(1.0e-6));
    for (std::size_t species = 2;
         species < report.microphysics_water_tendency_kg_by_species.size();
         ++species)
    {
        REQUIRE(report.microphysics_water_tendency_kg_by_species[species]
                == Approx(0.0));
    }
    REQUIRE(report.boundary_layer_vapor_tendency_kg
            == Approx(0.024).margin(1.0e-6));
    REQUIRE(global_radiation_runtime_state.applied_step_count == 1);
    REQUIRE(global_radiation_runtime_state.last_expected_atmospheric_energy_j
            == Approx(2.0 * global_radiation_runtime_state
                              .held_atmospheric_power_w)
                   .epsilon(1.0e-12));
    REQUIRE(global_radiation_runtime_state.last_applied_radiation_heat_j
            == Approx(global_radiation_runtime_state
                          .last_expected_atmospheric_energy_j)
                   .epsilon(1.0e-12));
    REQUIRE(global_microphysics_runtime_state.theta_storage_residual.size() == 1);
    REQUIRE(std::abs(global_microphysics_runtime_state.theta_storage_residual[0][0][0])
            <= 0.5 * std::numeric_limits<float>::epsilon() * theta[0][0][0]);
}

TEST_CASE("cloud-aware boundary-layer condensate coupling is conservative and atomic",
          "[physics][microphysics][boundary_layer][cloud]")
{
    initialize_single_cell_state(0.010f, 0.001f);
    microphysics_scheme = std::make_unique<ThetaSourceScheme>(0.0f);
    global_simulation_storage.runtime.boundary_layer.config.clear_sky_only = false;
    dqv_dt_pbl[0][0][0] = 0.0001f;
    dqc_dt_pbl[0][0][0] = -0.0002f;
    dqi_dt_pbl[0][0][0] = 0.0001f;

    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 2.0));
    REQUIRE(qv[0][0][0] == Approx(0.0102f));
    REQUIRE(qc[0][0][0] == Approx(0.0006f));
    REQUIRE(qi[0][0][0] == Approx(0.0002f));
    const auto& report = last_microphysics_thermodynamic_repair_report();
    REQUIRE(report.boundary_layer_vapor_tendency_kg
            == Approx(0.0096).margin(1.0e-8));
    REQUIRE(report.boundary_layer_cloud_liquid_tendency_kg
            == Approx(-0.0192).margin(1.0e-8));
    REQUIRE(report.boundary_layer_cloud_ice_tendency_kg
            == Approx(0.0096).margin(1.0e-8));
    REQUIRE(report.boundary_layer_vapor_tendency_kg
                + report.boundary_layer_cloud_liquid_tendency_kg
                + report.boundary_layer_cloud_ice_tendency_kg
            == Approx(0.0).margin(1.0e-12));

    const Field3D qv_before = qv;
    const Field3D qc_before = qc;
    const Field3D qi_before = qi;
    dqc_dt_pbl[0][0][0] = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS_AS(
        step_microphysics(global_simulation_state(), 1.0),
        std::invalid_argument);
    REQUIRE(qv[0][0][0] == qv_before[0][0][0]);
    REQUIRE(qc[0][0][0] == qc_before[0][0][0]);
    REQUIRE(qi[0][0][0] == qi_before[0][0][0]);
}

TEST_CASE("thermodynamic ledger records limiter and bound heat corrections")
{
    initialize_single_cell_state(0.010f, 0.0f);
    theta[0][0][0] = 680.0f;
    microphysics_scheme = std::make_unique<ThetaSourceScheme>(100.0f);

    step_microphysics(global_simulation_state(), 1.0);

    const auto& report = last_microphysics_thermodynamic_repair_report();
    REQUIRE(theta[0][0][0] == Approx(theta_max_k));
    REQUIRE(report.theta_tendency_limit_count == 1);
    REQUIRE(report.theta_upper_bound_count == 1);
    REQUIRE(report.theta_tendency_limit_correction_j < 0.0);
    REQUIRE(report.theta_bound_correction_j < 0.0);
    REQUIRE(report.committed_heat_j() < report.requested_heat_j());
}

TEST_CASE("theta storage rounding closes heat without a false bounds repair")
{
    initialize_single_cell_state(0.010f, 0.0f);
    global_validation_policy.mode = tmv::GuardMode::Strict;
    global_validation_policy.fail_on = tmv::GuardFailOn::Bounds;
    microphysics_scheme = std::make_unique<ThetaSourceScheme>(0.1f);

    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1.0));

    const auto& report = last_microphysics_thermodynamic_repair_report();
    const double heat_factor = report.microphysics_heat_j
        / static_cast<double>(0.1f);
    const double committed_from_state =
        (static_cast<double>(theta[0][0][0]) - 300.0) * heat_factor;
    REQUIRE(report.theta_lower_bound_count == 0);
    REQUIRE(report.theta_upper_bound_count == 0);
    REQUIRE(report.theta_storage_rounding_j != Approx(0.0));
    REQUIRE(report.committed_heat_j()
            == Approx(committed_from_state).epsilon(1.0e-12));

    global_validation_policy = {};
}

TEST_CASE("sub-ulp radiation increments survive compensated theta storage",
          "[physics][radiation][coupling][precision][RA-005]")
{
    initialize_single_cell_state(0.0f, 0.0f);
    microphysics_scheme = std::make_unique<ThetaSourceScheme>(0.0f);
    global_radiation_runtime_state.config.scheme_id = "simple_grey";
    dtheta_dt_rad[0][0][0] = 1.0e-6f;

    constexpr int steps = 100;
    for (int step = 0; step < steps; ++step)
    {
        const double pressure = static_cast<double>(p[0][0][0]);
        const double density = static_cast<double>(rho[0][0][0]);
        const double exner = std::pow(pressure / 100000.0, 287.0 / 1004.0);
        global_radiation_runtime_state.held_atmospheric_power_w =
            density * 1004.0 * exner * (2.0 * 3.0 * 4.0) * 1.0e-6;
        global_radiation_runtime_state.held_cell_power_w.resize(
            1, 1, 1,
            static_cast<float>(
                global_radiation_runtime_state.held_atmospheric_power_w));
        global_radiation_runtime_state.held_atmospheric_power_w =
            static_cast<double>(
                global_radiation_runtime_state.held_cell_power_w[0][0][0]);
        step_microphysics(global_simulation_state(), 1.0);
    }

    const double represented_theta = static_cast<double>(theta[0][0][0])
        + static_cast<double>(
            global_microphysics_runtime_state.theta_storage_residual[0][0][0]);
    REQUIRE(std::abs(static_cast<double>(theta[0][0][0]) - 300.0) > 5.0e-5);
    REQUIRE(represented_theta
            == Approx(300.0 + steps * 1.0e-6).margin(2.0e-7));
    REQUIRE(global_radiation_runtime_state.applied_step_count == steps);
}

TEST_CASE("strict validation rejects theta repair transactionally")
{
    initialize_single_cell_state(0.010f, 0.0f);
    global_validation_policy.mode = tmv::GuardMode::Strict;
    global_validation_policy.fail_on = tmv::GuardFailOn::Bounds;
    microphysics_scheme = std::make_unique<ThetaSourceScheme>(100.0f);
    global_simulation_storage.surface.precipitation_rate_mm_h[0] = 7.0f;
    global_simulation_storage.surface.accumulated_rainfall_mm[0] = 9.0f;
    global_simulation_storage.surface.accumulated_rainfall_residual_mm[0] =
        -1.0e-8f;

    REQUIRE_THROWS_AS(
        step_microphysics(global_simulation_state(), 1.0),
        std::runtime_error);
    REQUIRE(theta[0][0][0] == Approx(300.0f));
    REQUIRE(global_simulation_storage.surface.precipitation_rate_mm_h[0] ==
            7.0f);
    REQUIRE(global_simulation_storage.surface.accumulated_rainfall_mm[0] ==
            9.0f);
    REQUIRE(global_simulation_storage.surface
                .accumulated_rainfall_residual_mm[0] == -1.0e-8f);
    REQUIRE(global_simulation_storage.surface
                .accumulated_rainfall_last_update_s == 0.0);
    REQUIRE(last_microphysics_thermodynamic_repair_report()
                .theta_tendency_limit_count == 0);

    global_validation_policy = {};
}

TEST_CASE("strict nonfinite theta rejection preserves the accepted ledger")
{
    initialize_single_cell_state(0.010f, 0.0f);
    global_validation_policy.mode = tmv::GuardMode::Strict;
    global_validation_policy.fail_on = tmv::GuardFailOn::NonFinite;
    microphysics_scheme = std::make_unique<ThetaSourceScheme>(
        std::numeric_limits<float>::infinity());

    REQUIRE_THROWS_AS(
        step_microphysics(global_simulation_state(), 1.0),
        std::runtime_error);
    REQUIRE(theta[0][0][0] == Approx(300.0f));
    REQUIRE(last_microphysics_thermodynamic_repair_report()
                .theta_nonfinite_count == 0);
    REQUIRE(last_microphysics_water_repair_report()
                .nonfinite_tendency_count == 0);

    global_validation_policy = {};
}

TEST_CASE("normal CPU apply rejects nonfinite tendencies transactionally",
          "[microphysics][cpu011]")
{
    initialize_single_cell_state(0.010f, 0.0f);
    global_validation_policy.mode = tmv::GuardMode::Sanitize;
    microphysics_scheme = std::make_unique<WaterTransferScheme>(
        std::numeric_limits<float>::quiet_NaN(), 0.0f);

    REQUIRE_THROWS_WITH(
        step_microphysics(global_simulation_state(), 1.0),
        Catch::Contains("non-finite tendency or state repair"));

    REQUIRE(qv[0][0][0] == Approx(0.010f));
    REQUIRE(qc[0][0][0] == Approx(0.0f));
    REQUIRE(last_microphysics_water_repair_report()
                .nonfinite_tendency_count == 0);
    REQUIRE(last_microphysics_water_repair_report().nonfinite_count() == 0);
}

TEST_CASE("microphysics rejects invalid timestep instead of skipping physics",
          "[physics][microphysics][config-truth]")
{
    initialize_single_cell_state(0.010f, 0.0f);
    microphysics_scheme = std::make_unique<StubMicrophysicsScheme>();
    REQUIRE_THROWS_AS(
        step_microphysics(global_simulation_state(), 0.0),
        std::invalid_argument);
}

TEST_CASE("microphysics rejects invalid live thermodynamic state without base fallbacks",
          "[physics][microphysics][config-truth]")
{
    initialize_single_cell_state(0.010f, 0.0f);
    microphysics_scheme = std::make_unique<StubMicrophysicsScheme>();
    rho[0][0][0] = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS_AS(
        step_microphysics(global_simulation_state(), 1.0),
        std::invalid_argument);
    REQUIRE(theta[0][0][0] == Approx(300.0f));
    REQUIRE(qv[0][0][0] == Approx(0.010f));

    rho[0][0][0] = 2.0f;
    p[0][0][0] = 0.0f;
    REQUIRE_THROWS_AS(
        step_microphysics(global_simulation_state(), 1.0),
        std::invalid_argument);
    REQUIRE(theta[0][0][0] == Approx(300.0f));
}

TEST_CASE("microphysics input validation preserves fail-first category order",
          "[physics][microphysics][validation][simd]")
{
    initialize_single_cell_state(0.010f, 0.0f);
    microphysics_scheme = std::make_unique<StubMicrophysicsScheme>();
    p[0][0][0] = 0.0f;
    qv[0][0][0] = -0.001f;
    dtheta_dt_rad[0][0][0] =
        std::numeric_limits<float>::quiet_NaN();

    REQUIRE_THROWS_WITH(
        step_microphysics(global_simulation_state(), 1.0),
        "microphysics received invalid pressure, density, or potential temperature");

    p[0][0][0] = 90000.0f;
    REQUIRE_THROWS_WITH(
        step_microphysics(global_simulation_state(), 1.0),
        "microphysics received water outside its physical state bounds");

    qv[0][0][0] = 0.010f;
    REQUIRE_THROWS_WITH(
        step_microphysics(global_simulation_state(), 1.0),
        "microphysics received a non-finite coupled tendency");

    initialize_threaded_ledger_state();
    microphysics_scheme = std::make_unique<StubMicrophysicsScheme>();
    dtheta_dt_rad[1][1][1] =
        std::numeric_limits<float>::quiet_NaN();
    p[1][1][2] = 0.0f;
    REQUIRE_THROWS_WITH(
        step_microphysics(global_simulation_state(), 1.0),
        "microphysics received a non-finite coupled tendency");

    dtheta_dt_rad[1][1][1] = 0.0f;
    p[1][1][2] = 90000.0f;
    qv[1][1][1] = -0.001f;
    theta[1][1][2] = 0.0f;
    REQUIRE_THROWS_WITH(
        step_microphysics(global_simulation_state(), 1.0),
        "microphysics received water outside its physical state bounds");
}

TEST_CASE("Large microphysics input scans preserve serial error order and primary state",
          "[physics][microphysics][validation][thread]")
{
#ifdef _OPENMP
    struct RestoreWorkers
    {
        int count = omp_get_max_threads(), dynamic = omp_get_dynamic();
        ~RestoreWorkers() { omp_set_num_threads(count); omp_set_dynamic(dynamic); }
    } restore;
    omp_set_dynamic(0);
    omp_set_num_threads(GENERATE(1, 4));
#endif
    initialize_threaded_ledger_state(96, 32, 32);
    p[NR - 2][NTH - 2][NZ - 2] = 0;
    qv[1][1][1] = -.001f;
    const auto pressure_before = p;
    const auto vapor_before = qv;
    const auto before_time = simulation_time;
    REQUIRE_THROWS_WITH(step_microphysics(global_simulation_state(), 1.0),
                       "microphysics received water outside its physical state bounds");
    CHECK(std::memcmp(p.data(), pressure_before.data(), p.size() * sizeof(float)) == 0);
    CHECK(std::memcmp(qv.data(), vapor_before.data(), qv.size() * sizeof(float)) == 0);
    CHECK(simulation_time == before_time);
    qv[1][1][1] = .01f;
    dtheta_dt_rad[1][1][1] = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS_WITH(step_microphysics(global_simulation_state(), 1.0),
                       "microphysics received a non-finite coupled tendency");
    dtheta_dt_rad[1][1][1] = 0;
    REQUIRE_THROWS_WITH(step_microphysics(global_simulation_state(), 1.0),
                       "microphysics received invalid pressure, density, or potential temperature");
}

TEST_CASE("microphysics leaves C-grid closure columns outside physical ledgers and updates",
          "[physics][microphysics][ownership]")
{
    initialize_single_cell_state(0.010f, 0.0f);
    NR = 4;
    NTH = 4;
    NZ = 3;
    global_grid_geometry.initialize(
        NR, NTH, NZ, dr, dz, dtheta,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    global_simulation_storage.surface.precipitation_rate_mm_h.assign(
        static_cast<std::size_t>(NR * NTH), 0.0f);
    global_simulation_storage.surface.accumulated_rainfall_mm.assign(
        static_cast<std::size_t>(NR * NTH), 0.0f);
    global_simulation_storage.surface.accumulated_rainfall_residual_mm.assign(
        static_cast<std::size_t>(NR * NTH), 0.0f);
    const auto resize = [](Field3D& field, float value)
    {
        field.resize(NR, NTH, NZ, value);
    };
    resize(rho, 2.0f);
    resize(p, 90000.0f);
    resize(u, 0.0f);
    resize(v, 0.0f);
    resize(w, 0.0f);
    resize(theta, 300.0f);
    resize(qv, 0.010f);
    resize(qc, 0.0f);
    resize(qr, 0.0f);
    resize(qi, 0.0f);
    resize(qs, 0.0f);
    resize(qg, 0.0f);
    resize(qh, 0.0f);
    resize(dtheta_dt_rad, 0.0f);
    resize(dtheta_dt_pbl, 0.0f);
    resize(dqv_dt_pbl, 0.0f);
    microphysics_scheme = std::make_unique<StubMicrophysicsScheme>();

    // Poison only an i-closure column. It has no atmospheric volume and must
    // neither be repaired nor contaminate the physical-column transaction.
    for (int j = 0; j < NTH; ++j)
        for (int k = 0; k < NZ; ++k)
            theta[0][j][k] = std::numeric_limits<float>::quiet_NaN();

    REQUIRE_NOTHROW(step_microphysics(global_simulation_state(), 1.0));
    REQUIRE(std::isnan(theta[0][1][1]));
    REQUIRE(theta[1][1][1] == Approx(302.0f));
    REQUIRE(qv[1][1][1] == Approx(0.009f));
    REQUIRE(qc[1][1][1] == Approx(0.0005f));
    REQUIRE(last_microphysics_water_repair_report().nonfinite_count() == 0);
    REQUIRE(last_microphysics_thermodynamic_repair_report()
                .theta_nonfinite_count == 0);
}

TEST_CASE("Sedimentation subcycles preserve independent water inventory and the caller clock",
          "[physics][microphysics][subcycling][conservation]")
{
    initialize_single_cell_state(0.01f, 0.0f);
    qr.fill(0.001f);
    microphysics_scheme = std::make_unique<LinearFalloutScheme>();
    tmv::reset_microphysics_perf_stats();
    step_microphysics(global_simulation_state(), 20.0);
    REQUIRE(tmv::microphysics_perf_snapshot().sedimentation_substeps > 1);
    REQUIRE(simulation_time == 0.0);
    REQUIRE(qr[0][0][0] >= 0.0f);
    const auto& surface = global_simulation_storage.surface;
    const double retained = rho[0][0][0] * dz * qr[0][0][0];
    const double fallout = static_cast<double>(surface.accumulated_rainfall_mm[0]) +
        surface.accumulated_rainfall_residual_mm[0];
    REQUIRE(retained + fallout == Approx(2.0 * 4.0 * 0.001).epsilon(2e-6));
    REQUIRE(surface.precipitation_rate_mm_h[0] == Approx(fallout * 3600.0 / 20.0).epsilon(2e-6));
    REQUIRE(surface.accumulated_rainfall_last_update_s == 20.0);
    REQUIRE(global_radiation_runtime_state.applied_step_count == 1);
}

TEST_CASE("A rejected sedimentation substep restores the complete interval",
          "[physics][microphysics][subcycling][transaction]")
{
    const double interval = GENERATE(0.1, 20.0);
    initialize_single_cell_state(0.01f, 0.0f);
    qr.fill(0.001f);
    auto scheme = std::make_unique<LinearFalloutScheme>();
    scheme->fail_on_call = interval < 1.0 ? 1 : 2;
    microphysics_scheme = std::move(scheme);
    global_microphysics_runtime_state.last_thermodynamic_repair_report.microphysics_heat_j = 17.0;
    REQUIRE_THROWS_WITH(step_microphysics(global_simulation_state(), interval),
        Catch::Matchers::Contains("late sedimentation failure"));
    REQUIRE(qr[0][0][0] == 0.001f);
    REQUIRE(simulation_time == 0.0);
    REQUIRE(global_simulation_storage.surface.accumulated_rainfall_mm[0] == 0.0f);
    REQUIRE(global_simulation_storage.surface.accumulated_rainfall_last_update_s == 0.0);
    REQUIRE(global_radiation_runtime_state.applied_step_count == 0);
    REQUIRE(global_microphysics_runtime_state.last_thermodynamic_repair_report.microphysics_heat_j == 17.0);
    REQUIRE(dynamic_cast<LinearFalloutScheme&>(*microphysics_scheme).calls == 0);
}

TEST_CASE("Fallout converges to the independent exponential column solution",
          "[physics][microphysics][subcycling][convergence]")
{
    std::array<double, 3> error{};
    for (int refinement = 0; refinement < 3; ++refinement)
    {
        initialize_single_cell_state(0.01f, 0.0f);
        qr.fill(0.001f);
        microphysics_scheme = std::make_unique<LinearFalloutScheme>();
        const double step = 1.0 / (1 << refinement);
        for (int index = 0; index < 10 * (1 << refinement); ++index)
        {
            simulation_time = index * step;
            step_microphysics(global_simulation_state(), step);
        }
        error[refinement] = std::abs(qr[0][0][0] - 0.001 * std::exp(-1.0));
    }
    REQUIRE(error[0] / error[1] > 1.8);
    REQUIRE(error[1] / error[2] > 1.8);
}
