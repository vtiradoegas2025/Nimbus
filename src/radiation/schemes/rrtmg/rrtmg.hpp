#pragma once

#include "radiation/radiation_base.hpp"

#include <memory>
#include <string>

namespace tmv::rrtmg
{
struct LongwaveCoefficients;
struct ShortwaveCoefficients;
} // namespace tmv::rrtmg

namespace tmv
{

class RrtmgScheme final : public RadiationSchemeBase
{
public:
    RrtmgScheme();

    std::unique_ptr<RadiationSchemeBase> clone() const override;
    std::string                          name() const override;
    int                                  required_fields() const override;
    void                                 initialize(const RadiationConfig& config) override;
    void                                 compute_column(const RadiationConfig&          config,
                                                        const RadiationColumnStateView& column,
                                                        RadiationColumnTendencies&      tendencies,
                                                        RadiationColumnFluxes*          fluxes = nullptr) override;
    bool compute_mapped_shortwave_optics(
        const RadiationConfig& config,
        const RadiationColumnStateView& column,
        MappedShortwaveColumnOptics& optics) override;
    bool supports_native_column_batch(const RadiationConfig&         config,
                                      const RadiationComputeKernels& kernels) const override;
    bool compute_columns_native(const RadiationConfig&          config,
                                const RadiationColumnStateView* columns,
                                std::size_t                     column_count,
                                RadiationColumnTendencies*      tendencies,
                                RadiationComputeKernels&        kernels) override;

    bool compute_columns_native_diagnosed(const RadiationConfig&,const RadiationColumnStateView*,std::size_t,
        RadiationColumnTendencies*,RadiationComputeKernels&,RadiationColumnFluxes*)override;
private:
    void validate_compute_config(const RadiationConfig& config) const;

    struct Coefficients
    {
        std::shared_ptr<const rrtmg::LongwaveCoefficients>  longwave;
        std::shared_ptr<const rrtmg::ShortwaveCoefficients> shortwave;
    };
    std::shared_ptr<const Coefficients> coefficients_;
    RadiationConfig                     initialized_config_;
    bool                                initialized_ = false;
};

} // namespace tmv
