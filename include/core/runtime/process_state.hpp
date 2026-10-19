#pragma once

#include <vector>

#include "core/field/field3d.hpp"
#include "core/infra/nested_grid.hpp"
#include "core/runtime/grid_state.hpp"

// Compatibility bindings for the process-owned simulation instance. New
// orchestration should pass SimulationStateView; legacy kernels include this
// narrow adapter instead of the cross-module simulation umbrella.

namespace tmv
{
class CartesianCGridTransportContext;
struct AdvectionRuntimeState;
}

extern std::vector<double>& rho0_base;
extern std::vector<double>& p0_base;
extern std::vector<double>& qv0_base;
extern std::vector<double>& theta0_base;
extern std::vector<double>& u0_base;
extern std::vector<double>& v0_base;
extern double& coriolis_f;
extern double& simulation_time;

extern Field3D& rho;
extern Field3D& p;
extern Field3D& u;
extern Field3D& w;
extern Field3D& v;
extern Field3D& tracer;
extern Field3D& theta;
extern Field3D& qv;
extern Field3D& qc;
extern Field3D& qr;
extern Field3D& qi;
extern Field3D& qs;
extern Field3D& qh;
extern Field3D& qg;
extern Field3D& tke;

extern NestedGridConfig& nested_config;
extern Field3D& nest_rho;
extern Field3D& nest_p;
extern Field3D& nest_u;
extern Field3D& nest_w;
extern Field3D& nest_v;
extern Field3D& nest_theta;
extern Field3D& nest_qv;
extern Field3D& nest_qc;
extern Field3D& nest_qr;
extern Field3D& nest_qi;
extern Field3D& nest_qs;
extern Field3D& nest_qh;
extern Field3D& nest_qg;
extern Field3D& nest_tracer;

extern Field3D& radar_reflectivity;
extern Field3D& vorticity_r;
extern Field3D& vorticity_theta;
extern Field3D& vorticity_z;
extern Field3D& stretching_term;
extern Field3D& tilting_term;
extern Field3D& baroclinic_term;
extern Field3D& angular_momentum;
extern Field3D& angular_momentum_tendency;
extern Field3D& p_prime;
extern Field3D& dynamic_pressure;
extern Field3D& buoyancy_pressure;

extern Field3D& dtheta_dt_rad;
extern Field3D& dtheta_dt_pbl;
extern Field3D& dqv_dt_pbl;
extern Field3D& dqc_dt_pbl;
extern Field3D& dqi_dt_pbl;
extern Field3D& du_dt_pbl;
extern Field3D& dv_dt_pbl;
extern Field3D& dtke_dt_pbl;

extern std::vector<float>& sfc_ustar;
extern std::vector<float>& sfc_tau_u;
extern std::vector<float>& sfc_tau_v;
extern std::vector<float>& sfc_sensible_heat_flux;
extern std::vector<float>& sfc_latent_heat_flux;
extern std::vector<float>& sfc_moisture_flux;
extern std::vector<float>& pbl_height_2d;

void initialize();
void resize_fields();
void advect_tracer(
    double dt_advect,
    const tmv::CartesianCGridTransportContext* context = nullptr,
    tmv::AdvectionRuntimeState* runtime_state = nullptr);
void advect_thermodynamics(
    double dt_advect,
    const tmv::CartesianCGridTransportContext* context = nullptr,
    tmv::AdvectionRuntimeState* runtime_state = nullptr);
