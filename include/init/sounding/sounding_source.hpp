#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace tmv::init 
{

/**
 * @file sounding_source.hpp
 * @brief Abstract producer of an initial-condition vertical column.
 *
 * A Sounding is the per-level base state the dynamics consumes when it
 * broadcasts the column into the 3D field. Concrete sources convert from
 * their native representation to this struct:
 *
 *   - ParametricCAPESoundingSource : procedural multi-layer profile from
 *     a CAPE target plus surface theta/qv plus tropopause height.
 *   - FileSoundingSource (next step): reads a SHARPY-style HDF5/NetCDF and
 *     re-integrates pressure hydrostatically on the model grid.
 *   - parametric_targets, analytical, etc. : drop-in additions in
 *     src/init/sounding/sources/.
 *
 * The runtime constructs a source via make_sounding_source(...) and calls
 * build() exactly once during initialize(). The returned column is then
 * broadcast through the existing coordinate / staggering dispatch.
 */

/**
 * @brief Per-level vertical base-state column.
 *
 * All arrays are length z_m.size() (one entry per model level). The column
 * must be self-consistent on return from any SoundingSource::build():
 *
 *   p_pa[k]      = hydrostatic integration of T_k[k] (no per-level imbalance
 *                  so the dynamics sees zero spurious vertical-velocity at t=0)
 *   rho_kgm3[k]  = p_pa[k] / (R_d * T_k[k])
 *   theta_k[k]   = T_k[k] * (p0 / p_pa[k])^(R_d / cp)
 *   qv_kgkg[k]   <= 0.95 * qvsat(T_k[k], p_pa[k])
 *
 * Sources that read from files must re-integrate pressure rather than copy
 * the file's pressure column verbatim, otherwise sub-percent imbalances
 * between the file's T and p seed a startup transient.
 */
struct Sounding
{
    std::vector<double> z_m;
    std::vector<double> T_k;
    std::vector<double> theta_k;
    std::vector<double> qv_kgkg;
    std::vector<double> p_pa;
    std::vector<double> rho_kgm3;

    /**
     * Optional Cartesian wind columns. Empty means "no wind override" — the
     * runtime falls back to the parametric WK 3-point hodograph that the
     * existing apply_*_wind_initialization() helpers consume. Sources that
     * carry their own winds (file-based readings, future hodograph sources)
     * fill these with the same length as z_m, in m/s, in the Cartesian
     * (u_x, u_y) basis. Coordinate-specific projection happens in the
     * runtime, not the source.
     */
    std::vector<double> u_ms;
    std::vector<double> v_ms;

    std::size_t size() const { return z_m.size(); }

    bool has_winds() const
    {
        return !u_ms.empty() && u_ms.size() == z_m.size()
            && !v_ms.empty() && v_ms.size() == z_m.size();
    }

    /// Throws std::logic_error on the first invariant violation. Thresholds
    /// are loose enough to tolerate float roundoff in the hydrostatic
    /// integration but tight enough to catch a missing or wrong column.
    void verify_self_consistent() const;
};

/**
 * @brief Abstract producer of a vertical sounding column.
 *
 * Implementations live in src/init/sounding/sources/ and are registered
 * with the factory in src/init/sounding/factory.cpp. To add a new source:
 *
 *   1. Add a concrete subclass with a build() that returns a self-consistent
 *      Sounding for the supplied vertical grid.
 *   2. Add a factory case keyed on the YAML 'environment.sounding.type'
 *      string (or whatever shorthand makes sense).
 *   3. Add a unit test under tests/init/.
 *
 * No other code needs to change.
 */
class SoundingSource
{
public:
    virtual ~SoundingSource() = default;

    /**
     * @brief Builds a self-consistent base-state column on the model grid.
     *
     * @param z_m  Model heights (m), monotonically increasing, length NZ.
     * @param dz   Vertical spacing (m). Used by sources that integrate
     *             hydrostatically with a constant step. Sources that work
     *             with non-uniform grids may compute z_m[k+1] - z_m[k]
     *             directly and ignore this argument.
     */
    virtual Sounding build(const std::vector<double>& z_m, double dz) const = 0;

    /// Short identifier for logging (e.g. "parametric_cape", "file/sharpy").
    virtual std::string describe() const = 0;
};

}  // namespace tmv::init
