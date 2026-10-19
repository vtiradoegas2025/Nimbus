#pragma once

#include "core/field/field3d.hpp"

#include <array>
#include <cstddef>
#include <string>

namespace tmv::milbrandt_detail
{

struct TripleMomentMassLaw
{
    double coefficient = 0.0;
    double exponent = 0.0;
};

using TripleMomentFieldRefs = std::array<const Field3D*, 5>;
using TripleMomentMassLaws = std::array<TripleMomentMassLaw, 5>;

bool gamma_sixth_moment_per_kg(double density,
                              double mass_mixing_ratio,
                              double number_per_kg,
                              double alpha,
                              TripleMomentMassLaw mass_law,
                              float& result) noexcept;

class TripleMomentCandidateState
{
public:
    bool prepare(const Field3D& density,
                 const TripleMomentFieldRefs& mass,
                 const TripleMomentFieldRefs& number,
                 const TripleMomentMassLaws& mass_laws,
                 const std::array<double, 5>& initial_alpha,
                 std::string& error);

    bool commit(double dt,
                const Field3D& density,
                const TripleMomentFieldRefs& mass_after,
                const TripleMomentFieldRefs& number_after,
                const TripleMomentMassLaws& mass_laws,
                std::string& error);

    const Field3D& moment(std::size_t species) const;
    Field3D& tendency(std::size_t species);

private:
    std::array<Field3D, 5> moments_;
    std::array<Field3D, 5> tendencies_;
};

} // namespace tmv::milbrandt_detail
