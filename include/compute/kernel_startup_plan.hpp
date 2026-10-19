#pragma once

#include "compute/kernel_manifest.hpp"

#include <array>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace nimbus::compute
{

enum class KernelStartupOwner : std::uint8_t
{
    FlatCartesianCgridDynamics,
    CartesianScalarTransport,
    CartesianExplicitDiffusion,
    Count
};

inline std::initializer_list<KernelId> startup_owner_kernels(KernelStartupOwner owner)
{
    static const std::initializer_list<KernelId> dynamics  = {KernelId::AcousticMomentumCartesianCgrid,
                                                              KernelId::AcousticPressureCartesianCgrid,
                                                              KernelId::AcousticPressureMassFluxCartesianCgrid,
                                                              KernelId::AcousticBcFaceCartesianCgrid,
                                                              KernelId::CartesianCgridSlowTendencies,
                                                              KernelId::SlowApply,
                                                              KernelId::StateRelaxationCartesian};
    static const std::initializer_list<KernelId> transport = {KernelId::AdvectConservativeMassFluxCartesianCgrid,
                                                              KernelId::AdvectConservativeScalarFluxCartesianCgrid,
                                                              KernelId::AdvectConservativeUpdateCartesianCgrid,
                                                              KernelId::AdvectPdLimiterCartesianCgrid,
                                                              KernelId::AdvectX,
                                                              KernelId::AdvectY,
                                                              KernelId::TvdVerticalFlux};
    static const std::initializer_list<KernelId> diffusion = {KernelId::Diffusion};
    switch (owner)
    {
    case KernelStartupOwner::FlatCartesianCgridDynamics:
        return dynamics;
    case KernelStartupOwner::CartesianScalarTransport:
        return transport;
    case KernelStartupOwner::CartesianExplicitDiffusion:
        return diffusion;
    case KernelStartupOwner::Count:
        break;
    }
    return {};
}

class KernelStartupPlan
{
public:
    static std::shared_ptr<const KernelStartupPlan> make(const std::string&                     profile,
                                                         const std::vector<KernelStartupOwner>& owners,
                                                         const std::vector<KernelId>&           kernels,
                                                         std::string&                           error)
    {
        error.clear();
        if (profile.empty() || profile.size() > 128 || owners.empty() ||
            owners.size() > static_cast<std::size_t>(KernelStartupOwner::Count) || kernels.empty() ||
            kernels.size() > kernel_count())
        {
            error = "invalid bounded kernel startup plan";
            return {};
        }
        auto candidate = std::shared_ptr<KernelStartupPlan>(new KernelStartupPlan(profile));
        for (const auto owner : owners)
        {
            const auto index = static_cast<std::size_t>(owner);
            if (index >= candidate->owners_.size() || candidate->owners_[index])
            {
                error = "invalid or duplicate kernel startup owner";
                return {};
            }
            candidate->owners_[index] = true;
        }
        for (const auto kernel : kernels)
        {
            const auto index = static_cast<std::size_t>(kernel);
            if (index >= candidate->kernels_.size() || candidate->kernels_[index])
            {
                error = "invalid or duplicate startup kernel";
                return {};
            }
            candidate->kernels_[index] = true;
            ++candidate->kernel_count_;
        }
        if (!candidate->validate_dependencies(error))
        {
            return {};
        }
        return candidate;
    }

    static std::shared_ptr<const KernelStartupPlan> make(const std::string&                     profile,
                                                         const std::vector<KernelStartupOwner>& owners,
                                                         const std::vector<KernelId>&           kernels,
                                                         const KernelManifest&                  manifest,
                                                         std::string&                           error)
    {
        auto plan = make(profile, owners, kernels, error);
        if (!plan || !plan->validate(manifest, error))
        {
            return {};
        }
        return plan;
    }

    bool contains(KernelId kernel) const noexcept
    {
        const auto index = static_cast<std::size_t>(kernel);
        return index < kernels_.size() && kernels_[index];
    }

    std::size_t size() const noexcept
    {
        return kernel_count_;
    }

    const std::string& profile() const noexcept
    {
        return profile_;
    }

    bool validate(const KernelManifest& manifest, std::string& error) const
    {
        error.clear();
        if (!validate_dependencies(error))
        {
            return false;
        }
        for (std::size_t index = 0; index < kernels_.size(); ++index)
        {
            if (kernels_[index] && (static_cast<std::size_t>(manifest[index].id) != index ||
                                    manifest[index].availability != KernelAvailability::Active))
            {
                error = "required startup kernel is unavailable";
                return false;
            }
        }
        return true;
    }

    std::string cache_identity() const
    {
        std::string identity = "kernel-startup-v1:" + profile_ + ":";
        for (const bool owner : owners_)
        {
            identity += owner ? '1' : '0';
        }
        identity += ':';
        for (const bool kernel : kernels_)
        {
            identity += kernel ? '1' : '0';
        }
        return identity;
    }

private:
    explicit KernelStartupPlan(std::string profile) : profile_(std::move(profile))
    {
    }

    bool validate_dependencies(std::string& error) const
    {
        for (std::size_t owner = 0; owner < owners_.size(); ++owner)
        {
            if (!owners_[owner])
            {
                continue;
            }
            for (const auto kernel : startup_owner_kernels(static_cast<KernelStartupOwner>(owner)))
            {
                if (!contains(kernel))
                {
                    error = "incomplete kernel startup owner dependencies";
                    return false;
                }
            }
        }
        return true;
    }

    std::string                                                           profile_;
    std::array<bool, static_cast<std::size_t>(KernelStartupOwner::Count)> owners_{};
    std::array<bool, kernel_count()>                                      kernels_{};
    std::size_t                                                           kernel_count_ = 0;
};

template <class Create, class Destroy, class Observe>
bool prepare_startup_pipelines(const KernelStartupPlan* plan,
                               const KernelManifest&    manifest,
                               Create&&                 create,
                               Destroy&&                destroy,
                               Observe&&                observe,
                               std::string&             error)
{
    error.clear();
    const char* manifest_error = nullptr;
    if (!validate_kernel_manifest(manifest, &manifest_error))
    {
        error = std::string("invalid compute kernel manifest: ") +
                (manifest_error != nullptr ? manifest_error : "unknown error");
        return false;
    }
    if (plan != nullptr && !plan->validate(manifest, error))
    {
        return false;
    }
    std::array<bool, kernel_count()> created{};
    std::size_t                      ordinal = 0;
    for (const auto& kernel : manifest)
    {
        if (kernel.availability != KernelAvailability::Active || (plan != nullptr && !plan->contains(kernel.id)))
        {
            continue;
        }
        std::string pipeline_error;
        const bool  success = create(kernel, pipeline_error);
        observe(kernel, success, ++ordinal, pipeline_error);
        if (success)
        {
            created[static_cast<std::size_t>(kernel.id)] = true;
            continue;
        }
        if (plan != nullptr || kernel.startup_required)
        {
            for (const auto& previous : manifest)
            {
                if (created[static_cast<std::size_t>(previous.id)])
                {
                    destroy(previous);
                }
            }
            error = std::string(kernel.operation_id) + ": " + pipeline_error;
            return false;
        }
    }
    return true;
}

} // namespace nimbus::compute
