#pragma once

#include "microphysics/microphysics_base.hpp"
#include "numerics/advection/cartesian_interface_flux_register.hpp"
#include "numerics/advection/scalar_boundary_report.hpp"
#include <string>
#include <vector>

namespace tmv
{
struct PrivateScalarInterfaceCapture
{
    std::string                    name;
    CartesianInterfaceFluxRegister flux;
    ScalarBoundaryTransportReport  boundary;
    double                         committed_physics = 0;
};

struct CartesianPrivateInterfaceCapture
{
    std::vector<PrivateScalarInterfaceCapture> fields;

    void require_layout(const MicrophysicsPrognosticScalarList& scalars) const
    {
        if (fields.size() != scalars.size())
            throw std::invalid_argument("private interface capture layout changed");
        for (std::size_t i = 0; i < fields.size(); ++i)
            if (!scalars[i].name || fields[i].name != scalars[i].name || !scalars[i].field ||
                !scalars[i].positive_definite || scalars[i].storage != MicrophysicsScalarStorage::DryAirMixingRatio)
                throw std::invalid_argument("private interface capture requires stable positive dry-specific carriers");
    }

    void configure(int                                     nr,
                   int                                     nth,
                   int                                     nz,
                   CartesianInterfaceFootprint             footprint,
                   int                                     ratio,
                   const MicrophysicsPrognosticScalarList& scalars)
    {
        CartesianPrivateInterfaceCapture candidate;
        for (const auto& scalar : scalars)
        {
            if (!scalar.name || !*scalar.name || std::char_traits<char>::length(scalar.name) >= 64)
                throw std::invalid_argument("private interface capture name is invalid");
            for (const auto& prior : candidate.fields)
                if (prior.name == scalar.name)
                    throw std::invalid_argument("private interface capture names must be unique");
            PrivateScalarInterfaceCapture field;
            field.name = scalar.name;
            field.flux.configure(nr, nth, nz, footprint, ratio, ratio);
            candidate.fields.push_back(std::move(field));
        }
        candidate.require_layout(scalars);
        *this = std::move(candidate);
    }

    void reset_interval() noexcept
    {
        for (auto& field : fields)
        {
            field.flux.reset_interval();
            field.boundary          = {};
            field.committed_physics = 0;
        }
    }

    bool interval_empty() const noexcept
    {
        for (const auto& field : fields)
            if (!field.flux.interval_empty() || field.boundary.complete || field.committed_physics != 0)
                return false;
        return true;
    }
};
} // namespace tmv
