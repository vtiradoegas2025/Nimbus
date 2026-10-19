#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

#include "dynamics/final_boundary_energy_report.hpp"

struct CartesianCGridAcousticEnergyReportV1;

namespace tmv
{

struct BoundaryFieldWriteSet
{
    const char* const* field_ids = nullptr;
    std::size_t count = 0;
    bool complete = false;
};

class BoundaryConditionScheme
{
public:
    virtual ~BoundaryConditionScheme() = default;

    // Full closure for momentum, thermodynamic, and moisture fields.
    virtual void apply_full() = 0;

    virtual void apply_rk_stage() { apply_full(); }

    // Completed-step closure with exact basis-v2 physical-inventory
    // attribution. This is distinct from the N+1 acoustic closures because it
    // occurs after the EOS and sponge commits.
    virtual FinalBoundaryEnergyReportV1 apply_full_attributed() = 0;

    // Names every prognostic volume that apply_full_attributed() may write.
    // An implementation that cannot provide a complete set retains the
    // conservative full-residency invalidation at the orchestration boundary.
    virtual BoundaryFieldWriteSet full_attributed_write_set() const noexcept
    {
        return {};
    }

    // Acoustic closure for u, v, w, rho, and p. C-grid radiative faces
    // advance their Klemp-Wilhelmson boundary state by dt_small; collocated
    // closure owns halo storage only.
    virtual void apply_acoustic(double dt_small) = 0;

    // Attributed closure must be implemented by the owning layout. The
    // default rejects before mutation rather than fabricating energy evidence.
    virtual void apply_acoustic_attributed(
        double /*dt_small*/,
        CartesianCGridAcousticEnergyReportV1& /*report*/)
    {
        throw std::runtime_error(
            "this boundary scheme does not implement acoustic energy attribution");
    }

    // Seed time-integrated radiative state after slow tendencies and before
    // acoustic substeps. Earlier seeding leaves the face state one step stale.
    virtual void seed_acoustic_radiation() {}

    // Rebuild every closure owner after a whole-cell domain recycle. This is
    // deliberately distinct from ordinary completed-step closure: advected
    // scalar ghosts were replaced by the shift and must be closed once before
    // the next reconstruction stencil. Radiative history is then reseeded
    // from the rebuilt physical faces rather than the retired domain.
    virtual void rebuild_after_domain_recycling()
    {
        apply_full();
        seed_acoustic_radiation();
    }

    virtual std::string get_scheme_name() const = 0;
};

} // namespace tmv

using tmv::BoundaryConditionScheme;
