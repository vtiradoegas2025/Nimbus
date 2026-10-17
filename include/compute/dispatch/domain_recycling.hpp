#pragma once

/**
 * @file domain_recycling.hpp
 * @brief Domain-recycling compute dispatch contracts.
 */

/**
 * One exact homogeneous whole-cell field shift.
 *
 * Destination cells inside the retained rectangle read
 * `source(i + shift_i_cells, j + shift_j_cells, k)`; every other stored cell
 * receives `leading_profile[k]` or `leading_field(i,j,k)`. Exactly one
 * leading source is supplied. The caller supplies the canonical
 * stagger-aware source/destination rectangles produced by the recycle planner.
 */
struct HomogeneousDomainFieldShiftDispatch
{
    const float* source = nullptr;
    const float* leading_profile = nullptr;
    float* destination = nullptr;
    int nr = 0;
    int nth = 0;
    int nz = 0;
    int shift_i_cells = 0;
    int shift_j_cells = 0;
    int destination_i_begin = 0;
    int destination_i_end = 0;
    int destination_j_begin = 0;
    int destination_j_end = 0;
    int source_i_begin = 0;
    int source_i_end = 0;
    int source_j_begin = 0;
    int source_j_end = 0;
    const float* leading_field = nullptr;
};

/**
 * One atomic homogeneous recycling device batch.
 *
 * Every field must describe the same grid and signed movement, but may carry
 * its own stagger-aware rectangles and leading profile. Stable resident IDs
 * bind the authoritative device allocations. No destination may change unless
 * the complete command buffer reaches its fence.
 */
struct HomogeneousDomainFieldShiftBatchDispatch
{
    const HomogeneousDomainFieldShiftDispatch* fields = nullptr;
    const char* const* resident_ids = nullptr;
    int count = 0;
};
