#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace tmv::init
{

/**
 * @file hodograph_source.hpp
 * @brief Abstract producer of an initial-condition wind column.
 *
 * A WindColumn is the per-level Cartesian wind (u_x, u_y) the runtime
 * broadcasts into the 3D field, with coordinate-aware projection applied
 * downstream (cylindrical projects onto (r, theta); cartesian stores
 * directly). Concrete sources convert from their native representation:
 *
 *   - WKParamHodograph : 3-point linear interpolation between sfc / 1 km /
 *     6 km, constant above 6 km. Today's default.
 *   - ZeroHodograph    : u = v = 0 at every level. Hydrostatic-test recipe.
 *   - (future) FileHodograph, StormRelativeHodograph, ...
 *
 * The runtime constructs a source via make_hodograph_source(...) and calls
 * build(z_m) once during initialize() unless the active SoundingSource
 * already supplied winds (in which case those win and the hodograph source
 * is unused — see HodographSourceConfig::Type::Auto).
 */

/**
 * @brief Per-level Cartesian wind column. Length matches the model heights
 *        passed to HodographSource::build(z_m). u_ms and v_ms are u_x and
 *        u_y respectively, in m/s.
 */
struct WindColumn
{
    std::vector<double> u_ms;
    std::vector<double> v_ms;

    std::size_t size() const { return u_ms.size(); }

    bool is_consistent_with(const std::vector<double>& z_m) const
    {
        return u_ms.size() == z_m.size() && v_ms.size() == z_m.size();
    }
};

class HodographSource
{
public:
    virtual ~HodographSource() = default;

    /// Produces a wind column on the supplied vertical grid. Length of
    /// returned u_ms / v_ms must equal z_m.size().
    virtual WindColumn build(const std::vector<double>& z_m) const = 0;

    /// Short identifier for logging (e.g. "wk_param", "zero", "file/sharpy").
    virtual std::string describe() const = 0;
};

}  // namespace tmv::init
