#pragma once

#include "core/infra/coordinate_system.hpp"
#include "core/runtime/grid_state.hpp"

#include <algorithm>

/**
 * Half-open ownership range for prognostic scalar control volumes.
 *
 * Storage planes outside this range are boundary-closure state. They may be
 * required by a stencil or published for restart fidelity, but they carry no
 * atmospheric volume and must never enter a physical inventory.
 */
struct PhysicalCellDomain
{
    int i_begin = 0;
    int i_end = 0;
    int j_begin = 0;
    int j_end = 0;
    int k_begin = 0;
    int k_end = 0;

    [[nodiscard]] int nx() const noexcept { return i_end - i_begin; }
    [[nodiscard]] int ny() const noexcept { return j_end - j_begin; }
    [[nodiscard]] int nz() const noexcept { return k_end - k_begin; }
    [[nodiscard]] bool empty() const noexcept
    {
        return nx() <= 0 || ny() <= 0 || nz() <= 0;
    }
};

inline PhysicalCellDomain physical_scalar_domain(
    int nr, int nth, int nz, StaggerType staggering)
{
    const int i_halo = nr >= 3 ? 1 : 0;
    const int j_halo = nth >= 3 ? 1 : 0;
    const int k_halo = staggering == StaggerType::Collocated && nz >= 3 ? 1 : 0;
    return {
        i_halo, std::max(i_halo, nr - i_halo),
        j_halo, std::max(j_halo, nth - j_halo),
        k_halo, std::max(k_halo, nz - k_halo),
    };
}

inline PhysicalCellDomain physical_scalar_domain(const GridStateView& grid)
{
    return physical_scalar_domain(
        grid.nr, grid.nth, grid.nz,
        grid.geometry.staggered ? StaggerType::CGrid
                                : StaggerType::Collocated);
}
