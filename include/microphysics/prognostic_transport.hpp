#pragma once

#include "microphysics/microphysics_base.hpp"

namespace tmv
{
inline void validate_microphysics_scalar_storage(MicrophysicsScalarStorage storage)
{
    if (storage != MicrophysicsScalarStorage::DryAirMixingRatio &&
        storage != MicrophysicsScalarStorage::VolumeConcentration)
        throw std::invalid_argument("unknown microphysics scalar storage units");
}

inline double microphysics_scalar_mass_density(
    double value, double density, MicrophysicsScalarStorage storage) noexcept
{
    return storage == MicrophysicsScalarStorage::VolumeConcentration ? value : value * density;
}

inline double microphysics_scalar_from_mass_density(
    double value, double density, MicrophysicsScalarStorage storage) noexcept
{
    return storage == MicrophysicsScalarStorage::VolumeConcentration ? value : value / density;
}

inline void microphysics_scalar_to_mixing_ratio(
    const MicrophysicsPrognosticScalar& scalar, const Field3D& density, Field3D& destination)
{
    validate_microphysics_scalar_storage(scalar.storage);
    if (scalar.field == nullptr || scalar.field->size_r() != density.size_r() ||
        scalar.field->size_th() != density.size_th() || scalar.field->size_z() != density.size_z())
        throw std::invalid_argument("microphysics scalar conversion requires matching fields");
    destination.resize(density.size_r(), density.size_th(), density.size_z());
    for (std::size_t index = 0; index < density.size(); ++index)
    {
        const double rho = density.data()[index];
        const double value = scalar.field->data()[index];
        const float converted = static_cast<float>(
            scalar.storage == MicrophysicsScalarStorage::VolumeConcentration ? value / rho : value);
        if (!std::isfinite(rho) || rho <= 0.0 || !std::isfinite(converted) ||
            (scalar.positive_definite && converted < 0.0f))
            throw std::invalid_argument("microphysics scalar conversion produced invalid state");
        destination.data()[index] = converted;
    }
}
}
