#pragma once

#include "core/field/field3d.hpp"
#include "numerics/advection/cartesian_interface_flux_register.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "terrain/terrain_base.hpp"

namespace tmv
{

enum class CartesianMomentumComponent : std::size_t
{
    X = 0,
    Y = 1,
    Z = 2,
};

struct CartesianMomentumInterfaceSide
{
    std::array<std::vector<double>, 3> component;
};

struct CartesianMomentumInterfaceFluxes
{
    CartesianMomentumInterfaceSide west;
    CartesianMomentumInterfaceSide east;
    CartesianMomentumInterfaceSide south;
    CartesianMomentumInterfaceSide north;
    std::array<double, 3> source_impulse{};
    std::array<double, 3> storage_impulse{};
    double accumulated_dt_s = 0.0;
    std::uint64_t contribution_count = 0;
};

namespace cartesian_cgrid_momentum_flux
{

double rho_x(const Field3D& rho, int i, int j, int k);
double rho_y(const Field3D& rho, int i, int j, int k);
double rho_z(const Field3D& rho, int i, int j, int k);
double x_normal_mass(const Field3D& rho, const Field3D& u,
                     int face, int j, int k, double frame_u_ms);
double y_normal_mass(const Field3D& rho, const Field3D& v,
                     int i, int face, int k, double frame_v_ms);
double z_normal_mass(const Field3D& rho, const Field3D& w,
                     int i, int j, int face);
double x_flux_x(const Field3D& rho, const Field3D& u,
                int face, int j, int k, double frame_u_ms);
double x_flux_y(const Field3D& rho, const Field3D& u, const Field3D& v,
                int face, int j, int k, double frame_u_ms);
double x_flux_z(const Field3D& rho, const Field3D& u, const Field3D& w,
                int face, int j, int k, double frame_u_ms);
double y_flux_x(const Field3D& rho, const Field3D& u, const Field3D& v,
                int i, int face, int k, double frame_v_ms);
double y_flux_y(const Field3D& rho, const Field3D& v,
                int i, int face, int k, double frame_v_ms);
double y_flux_z(const Field3D& rho, const Field3D& v, const Field3D& w,
                int i, int face, int k, double frame_v_ms);
double z_flux_x(const Field3D& rho, const Field3D& u, const Field3D& w,
                int i, int j, int face);
double z_flux_y(const Field3D& rho, const Field3D& v, const Field3D& w,
                int i, int j, int face);
double z_flux_z(const Field3D& rho, const Field3D& w,
                int i, int j, int face);

} // namespace cartesian_cgrid_momentum_flux

class CartesianCGridMomentumFluxRegister
{
public:
    void configure(int nr, int nth, int nz,
                   CartesianInterfaceFootprint footprint,
                   int x_face_tangential_ratio = 1,
                   int y_face_tangential_ratio = 1);
    void reset_interval() noexcept;
    void stage_advective(const Field3D& rho, const Field3D& u,
                         const Field3D& v, const Field3D& w,
                         double dt_s, double dx_m, double dy_m, double dz_m,
                         double frame_u_ms = 0.0, double frame_v_ms = 0.0);
    void stage_native_fluxes(const float* pairs, std::size_t count, double dt_s,
        double dx_m, double dy_m, double dz_m, bool pressure);
    void stage_pressure(const Field3D& p, double dt_s,
                        double dx_m, double dy_m, double dz_m);
    void stage_advective_mapped(
        const Field3D& rho, const Field3D& u,
        const Field3D& v, const Field3D& w,
        const TerrainMetrics3D& metrics, double dt_s,
        double frame_u_ms = 0.0, double frame_v_ms = 0.0);
    void stage_pressure_mapped(
        const Field3D& p, const TerrainMetrics3D& metrics, double dt_s);
    void stage_mapped_tendency_closure(
        const Field3D& rho, const Field3D& du_dt,
        const Field3D& dv_dt, const Field3D& dw_dt,
        const TerrainMetrics3D& metrics, double dt_s);
    void stage_source_impulse(std::array<double, 3> impulse);
    void stage_pressure_source_impulse(std::array<double, 3> impulse);
    [[nodiscard]] std::array<double, 3> native_dual_inventory(const Field3D& rho,
                                                              const Field3D& u,
                                                              const Field3D& v,
                                                              const Field3D& w,
                                                              double         dx_m,
                                                              double         dy_m,
                                                              double         dz_m) const;
    [[nodiscard]] std::array<double, 3> native_dual_inventory_mapped(const Field3D&          rho,
                                                                     const Field3D&          u,
                                                                     const Field3D&          v,
                                                                     const Field3D&          w,
                                                                     const TerrainMetrics3D& metrics) const;
    [[nodiscard]] std::array<double, 3> native_dual_inventory_buffers(const Field3D&              rho,
                                                                      std::array<const float*, 3> velocity,
                                                                      std::size_t                 cells,
                                                                      double                      dx_m,
                                                                      double                      dy_m,
                                                                      double                      dz_m,
                                                                      const TerrainMetrics3D* metrics = nullptr) const;
    [[nodiscard]] std::array<double, 3> native_dual_tendency_impulse(const Field3D& rho,
                                                                     const Field3D& du_dt,
                                                                     const Field3D& dv_dt,
                                                                     const Field3D& dw_dt,
                                                                     double         dt_s,
                                                                     double         dx_m,
                                                                     double         dy_m,
                                                                     double         dz_m) const;
    [[nodiscard]] std::array<double, 3>
    native_dual_tendency_impulse_mapped(
        const Field3D& rho, const Field3D& du_dt,
        const Field3D& dv_dt, const Field3D& dw_dt,
        double dt_s, const TerrainMetrics3D& metrics) const;
    [[nodiscard]] std::array<double, 3> native_dual_density_tendency_impulse(
        const Field3D& rho, const Field3D& drho_dt,
        const Field3D& u, const Field3D& v, const Field3D& w,
        double dt_s, double dx_m, double dy_m, double dz_m) const;
    [[nodiscard]] std::array<double, 3>
    native_dual_density_tendency_impulse_mapped(
        const Field3D& rho, const Field3D& drho_dt,
        const Field3D& u, const Field3D& v, const Field3D& w,
        double dt_s, const TerrainMetrics3D& metrics) const;
    void accumulate_operator_source_impulse(
        const std::array<double, 3>& impulse);
    [[nodiscard]] std::array<double, 3>
    last_committed_operator_impulse() const;
    [[nodiscard]] std::array<double, 3>
    pending_committed_operator_impulse() const;
    void accumulate_operator_storage_impulse(
        const std::array<double, 3>& impulse);
    void accumulate_mapped_endpoint_storage_impulse(
        const std::array<double, 3>& impulse);
    void accumulate_boundary_source_impulse(
        const std::array<double, 3>& impulse);
    void attribute_operator_commit(
        const std::array<double, 3>& requested_impulse,
        const std::array<double, 3>& committed_impulse);
    void attribute_continuity_commit(
        const std::array<double, 3>& requested_impulse,
        const std::array<double, 3>& committed_impulse);
    void validate_staged_commit();
    void commit_staged() noexcept;

    [[nodiscard]] bool interval_empty() const noexcept
    {
        for (std::size_t component = 0; component < 3; ++component)
            if (advective_.source_impulse[component] != 0 || pressure_.source_impulse[component] != 0 ||
                advective_.storage_impulse[component] != 0 || pressure_.storage_impulse[component] != 0)
                return false;
        return !staged_ready_ && pending_committed_kind_ == StageKind::None &&
               advective_.contribution_count == 0 && pressure_.contribution_count == 0 &&
               advective_.accumulated_dt_s == 0 && pressure_.accumulated_dt_s == 0;
    }
    [[nodiscard]] bool configured() const noexcept;
    [[nodiscard]] const CartesianInterfaceFootprint& footprint() const noexcept;
    [[nodiscard]] const CartesianMomentumInterfaceFluxes&
    advective_fluxes() const noexcept;
    [[nodiscard]] const CartesianMomentumInterfaceFluxes&
    pressure_fluxes() const noexcept;

private:
    enum class StageKind
    {
        None,
        Advective,
        Pressure,
    };

    static void allocate(CartesianMomentumInterfaceFluxes& fluxes,
                         std::size_t x_count, std::size_t y_count);
    static void clear(CartesianMomentumInterfaceFluxes& fluxes) noexcept;
    static void add(CartesianMomentumInterfaceSide& side,
                    std::size_t component, std::size_t bin, double value);
    static void add_side(CartesianMomentumInterfaceSide& destination,
                         const CartesianMomentumInterfaceSide& source) noexcept;
    void require_field(const Field3D& field, double dt_s,
                       double dx_m, double dy_m, double dz_m) const;
    void require_fields(const Field3D& rho, const Field3D& u,
                        const Field3D& v, const Field3D& w,
                        double dt_s, double dx_m, double dy_m, double dz_m) const;
    void finish_stage(double dt_s, StageKind kind);

    bool configured_ = false;
    bool staged_ready_ = false;
    bool staged_validated_ = false;
    StageKind staged_kind_ = StageKind::None;
    int nr_ = 0;
    int nth_ = 0;
    int nz_ = 0;
    int x_ratio_ = 1;
    int y_ratio_ = 1;
    CartesianInterfaceFootprint footprint_{};
    CartesianMomentumInterfaceFluxes advective_{};
    CartesianMomentumInterfaceFluxes pressure_{};
    CartesianMomentumInterfaceFluxes staged_{};
    StageKind last_committed_kind_ = StageKind::None;
    std::array<double, 3> last_committed_operator_impulse_{};
    StageKind pending_committed_kind_ = StageKind::None;
    std::array<double, 3> pending_committed_operator_impulse_{};
};

} // namespace tmv

using tmv::CartesianCGridMomentumFluxRegister;
using tmv::CartesianMomentumComponent;
using tmv::CartesianMomentumInterfaceFluxes;
