#pragma once

#include <cstddef>
#include <deque>
#include <string>
#include <vector>

#include "core/field/field3d.hpp"
#include "core/infra/coordinate_system.hpp"
#include "core/infra/domain_motion.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "terrain/terrain_base.hpp"

/**
 * @file trajectories.hpp
 * @brief Online Lagrangian parcel-trajectory integration.
 *
 * Advances a set of seeded air parcels through the evolving wind field, one
 * model timestep at a time (RK2), sampling fields along each parcel's path.
 * This is the foundation of trajectory / vorticity-budget analysis — the
 * standard tornadogenesis-mechanism diagnostic — and replaces the prior
 * Eulerian placeholder "trajectory" fields.
 *
 * Integration runs in physical Cartesian space (x, y, z in metres). Velocity
 * is sampled with per-component trilinear interpolation that accounts for
 * Arakawa C-grid staggering (u on x faces, v on y faces, w on z faces) through
 * the shared scalar-center/face transforms — no full de-stagger pass per step.
 * Terrain-following runs invert each column's physical scalar/interface
 * heights and enforce the mapped lower-boundary velocity, so paths and their
 * online vorticity budgets remain physical rather than computational-space
 * trajectories.
 *
 * Scope: flat Cartesian A- and C-grids plus terrain-following Cartesian C-grid.
 */

namespace tmv
{

/// One sample of a parcel's state at a single time.
struct TrajectorySample
{
    float t;     ///< Simulation time (s).
    float x;     ///< Earth-relative physical x position (m).
    float y;     ///< Earth-relative physical y position (m).
    float z;     ///< Height (m).
    float w;     ///< Vertical velocity sampled at the parcel (m/s).
    float zeta;  ///< Vertical vorticity sampled at the parcel (1/s).
    /// Vertical-vorticity budget terms (1/s^2), the tornadogenesis-mechanism
    /// diagnostic. Computed fresh at the parcel each step on Cartesian A- and
    /// C-grids; NaN only where the online budget equation is unsupported.
    float tilting;     ///< Tilting of horizontal vorticity into the vertical.
    float stretching;  ///< Vertical stretching (zeta * dw/dz).
    float baroclinic;  ///< Solenoidal (baroclinic) generation.
};

/// Number of float columns per sample in the serialized output, and their order.
inline constexpr int kTrajectoryVarCount = 9;  // t,x,y,z,w,zeta,tilting,stretching,baroclinic
inline constexpr std::size_t kMaximumTrajectoryParcels = 1ULL << 20;
inline constexpr std::size_t kMaximumTrajectorySamples = 1ULL << 28;

/// A single tracked air parcel.
struct Parcel
{
    int id = 0;
    bool active = true;
    double x = 0.0;  ///< Current grid-relative physical position (m).
    double y = 0.0;
    double z = 0.0;
    std::deque<TrajectorySample> path;
};

/// Runtime configuration for the trajectory set (built from the YAML config).
struct TrajectoryConfig
{
    bool enabled = false;
    /// Seed-grid counts along each axis (total parcels = nx*ny*nz).
    int nx = 5;
    int ny = 5;
    int nz = 3;
    /// Simulation time (s) at which parcels are released.
    double seed_time_s = 0.0;
    /// Zero preserves the legacy every-model-step recording behavior. A
    /// positive value records on the first completed step at or beyond each
    /// interval while parcel integration still runs on every model step.
    double sample_interval_s = 0.0;
    /// Maximum retained samples per parcel, including the seed sample.
    int max_samples_per_parcel = 16384;
    /// Seed-box half-extents (m) about the seed centre.
    double half_x_m = 5000.0;
    double half_y_m = 5000.0;
    double half_z_m = 1500.0;
    /// Explicit Earth-relative seed centre (m). NaN components auto-centre on
    /// the developing low-level mesocyclone (strongest low-level |vorticity_z|).
    double center_x_m = 0.0 / 0.0;
    double center_y_m = 0.0 / 0.0;
    double center_z_m = 0.0 / 0.0;
};

/// Durable parcel state owned by a portable simulation checkpoint. Geometry
/// and terrain pointers remain transient and are rebound by configure() before
/// this state is restored.
struct TrajectoryCheckpointState
{
    TrajectoryConfig configuration;
    bool seeded = false;
    /// Next cadence boundary. NaN when every-step recording is selected.
    double next_sample_time_s = 0.0 / 0.0;
    std::vector<Parcel> parcels;
};

/**
 * @brief A set of Lagrangian parcels advanced together through the flow.
 *
 * Lifecycle: configure() once at startup, seed() when the run reaches the
 * seed time, advance() every model step thereafter, write() at run end.
 */
class TrajectorySet
{
public:
    void configure(const TrajectoryConfig& cfg,
                   CoordinateSystem coord,
                   bool staggered,
                   int nr, int nth, int nz,
                   double dr, double dtheta, double dz,
                   tmv::domain::HorizontalDomainFrame frame =
                       tmv::domain::HorizontalDomainFrame::stationary(),
                   const TerrainMetrics3D* terrain_metrics = nullptr,
                   bool earth_fixed_terrain = false);

    bool enabled() const { return cfg_.enabled; }
    bool seeded() const { return seeded_; }
    bool terrain_aware() const { return terrain_metrics_ != nullptr; }
    double seed_time_s() const { return cfg_.seed_time_s; }
    const TrajectoryConfig& configuration() const { return cfg_; }
    std::size_t count() const { return parcels_.size(); }
    std::size_t total_samples() const;
    double next_sample_time_s() const { return next_sample_time_s_; }
    /// Read-only access to the parcels (for inspection / tests).
    const std::vector<Parcel>& parcels() const { return parcels_; }

    /// Admit decoded checkpoint state after configure() has rebound the target
    /// branch's geometry. Validation is fail-closed and leaves this set
    /// unchanged on failure.
    bool restore_checkpoint_state(TrajectoryCheckpointState state,
                                  double checkpoint_time_s,
                                  std::string& error);

    /// Release parcels in a box about the seed centre (auto-centred on the
    /// low-level mesocyclone — strongest low-level |vort_z| — when the configured
    /// centre is NaN). Records each parcel's initial sample, including the fresh
    /// vorticity budget, at `sim_time_s`.
    void seed(double sim_time_s,
              const Field3D& u, const Field3D& v, const Field3D& w,
              const Field3D& rho, const Field3D& pres,
              const Field3D& vort_z);

    /// Advance in a wind field held constant over `dt` (RK2), then sample the path columns
    /// at the new position: w, the vertical vorticity, and the vorticity budget
    /// (tilting/stretching/baroclinic). The budget is computed online from the
    /// prognostic fields (u, v, w, rho, pres), so it is current every step rather
    /// than frozen at the last output frame. Parcels that leave the horizontal
    /// domain are deactivated; z is clamped between the local physical terrain
    /// surface and model lid.
    void advance(double sim_time_s, double dt,
                 const Field3D& u, const Field3D& v, const Field3D& w,
                 const Field3D& rho, const Field3D& pres,
                 const Field3D& vort_z);

    // Evolving winds require both time endpoints. Only midpoint samples are
    // retained, so temporary storage scales with parcels, not grid cells.
    void prepare_advance(double start_time_s, double dt,
                         const Field3D& u, const Field3D& v, const Field3D& w);
    void finish_advance(double end_time_s, double dt,
                        const Field3D& u, const Field3D& v, const Field3D& w,
                        const Field3D& rho, const Field3D& pres,
                        const Field3D& vort_z);

    /** Validate a no-allocation grid-frame translation before recycling commit. */
    void preflight_domain_recycling(
        double displacement_x_m,
        double displacement_y_m,
        const tmv::domain::HorizontalDomainFrame& post_shift_frame) const;

    /**
     * Translate parcel working coordinates into the recycled grid frame.
     * Recorded path samples remain Earth-relative and byte-identical. Parcels
     * retired with the old swath become inactive.
     */
    void commit_domain_recycling(
        double displacement_x_m,
        double displacement_y_m,
        const tmv::domain::HorizontalDomainFrame& post_shift_frame) noexcept;

    /// Serialize all parcel paths to a single (n_parcels, max_samples,
    /// kTrajectoryVarCount) float32 .npy, NaN-padded. Column order matches
    /// TrajectorySample (t, x, y, z, w, zeta, tilting, stretching, baroclinic).
    /// Returns false if nothing to write.
    bool write(const std::string& path) const;

private:
    TrajectoryConfig cfg_;
    CoordinateSystem coord_ = CoordinateSystem::Cartesian;
    bool staggered_ = false;
    int nr_ = 0;
    int nth_ = 0;
    int nz_ = 0;
    PhysicalCellDomain domain_{};
    double dr_ = 1.0;      ///< x-spacing (m).
    double dtheta_ = 1.0;  ///< y-spacing (m).
    double dz_ = 1.0;
    const TerrainMetrics3D* terrain_metrics_ = nullptr;
    tmv::domain::HorizontalVelocity surface_velocity_ms_{};
    bool seeded_ = false;
    double next_sample_time_s_ = 0.0 / 0.0;
    tmv::domain::HorizontalDomainFrame frame_{};
    std::vector<Parcel> parcels_;
    struct Midpoint
    {
        double x, y, z, vx, vy, vz;
    };
    std::vector<Midpoint> midpoints_;
    double prepared_end_time_s_ = 0.0 / 0.0;
    double prepared_dt_ = 0.0;

    void retain_sample(Parcel& parcel, const TrajectorySample& sample);

    /// Cartesian velocity (dx/dt, dy/dt, dz/dt) sampled at physical (x, y, z).
    void velocity_at(double x, double y, double z,
                     const Field3D& u, const Field3D& v, const Field3D& w,
                     double& vx, double& vy, double& vz) const;

    /// Trilinear sample of a cell-centred scalar field at physical (x, y, z).
    float scalar_at(double x, double y, double z, const Field3D& f) const;

    /// Fill a sample's field-derived columns (w, zeta, and the vorticity budget)
    /// at physical (x, y, z). On Cartesian A- and C-grids the vorticity + budget
    /// are computed fresh from the prognostic fields via shared kernels and
    /// trilinearly interpolated from surrounding scalar centres, so they reflect
    /// this step without nearest-cell jumps. On unsupported arrangements zeta is
    /// sampled from the (output-cadence) vort_z field and the budget terms are
    /// set to NaN.
    void sample_fields(double x, double y, double z,
                       const Field3D& u, const Field3D& v, const Field3D& w,
                       const Field3D& rho, const Field3D& pres,
                       const Field3D& vort_z, TrajectorySample& s) const;

    /// True when (x, y, z) is inside the horizontal domain (z is clamped, not
    /// tested, by the caller).
    bool in_horizontal_domain(double x, double y) const;

    double terrain_height_at(double x, double y, int level,
                             bool interface) const;
    double scalar_vertical_fractional_index(double x, double y,
                                            double z) const;
    double upper_face_vertical_fractional_index(double x, double y,
                                                double z) const;
    double terrain_ground_vertical_velocity_at(double x, double y,
                                               const Field3D& u,
                                               const Field3D& v) const;
    double local_z_lower(double x, double y) const;
    double local_z_upper(double x, double y) const;
    double clamp_vertical(double x, double y, double z) const;

    /// Domain extents in metres along each model axis.
    double x_lower() const { return static_cast<double>(domain_.i_begin) * dr_; }
    double x_upper() const { return static_cast<double>(domain_.i_end) * dr_; }
    double y_lower() const { return static_cast<double>(domain_.j_begin) * dtheta_; }
    double y_upper() const { return static_cast<double>(domain_.j_end) * dtheta_; }
    double z_lower() const { return static_cast<double>(domain_.k_begin) * dz_; }
    double z_upper() const { return static_cast<double>(domain_.k_end) * dz_; }
};

}  // namespace tmv
