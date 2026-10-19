#pragma once

#include "core/field/field_snapshot.hpp"
#include "diagnostics/field_contract.hpp"

#include <cstddef>
#include <cstdint>

namespace nimbus::output
{

enum class ZfpFieldStorage : std::uint8_t
{
    Compressed,
    LosslessNpy,
    InvalidShape,
};

inline ZfpFieldStorage classify_zfp_field_storage(const FieldSnapshotEntry& entry)
{
    if (entry.dim0 <= 0)
    {
        return ZfpFieldStorage::InvalidShape;
    }
    const int dim1 = entry.is_3d
        ? entry.dim1 : (entry.dim1 > 0 ? entry.dim1 : entry.dim2);
    const int dim2 = entry.is_3d ? entry.dim2 : 1;
    if (dim1 <= 0 || dim2 <= 0)
    {
        return ZfpFieldStorage::InvalidShape;
    }
    const std::size_t d0 = static_cast<std::size_t>(entry.dim0);
    const std::size_t d1 = static_cast<std::size_t>(dim1);
    const std::size_t d2 = static_cast<std::size_t>(dim2);
    const std::size_t size = entry.values().size();
    if (d0 > size || d1 > size / d0 || d2 > size / (d0 * d1) ||
        d0 * d1 * d2 != size)
    {
        return ZfpFieldStorage::InvalidShape;
    }
    const auto* contract = tmv::find_field_contract(entry.name);
    if (contract != nullptr && contract->allows_missing_values)
    {
        return ZfpFieldStorage::LosslessNpy;
    }
    return ZfpFieldStorage::Compressed;
}

} // namespace nimbus::output
