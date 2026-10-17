#pragma once
#include "radiation/base/radiative_transfer.hpp"
#include "radiation/radiation_base.hpp"

namespace tmv
{

struct SimpleGreyAbsorptionCoefficients
{
    double longwave_vapor_m2kg;
    double longwave_liquid_m2kg;
    double longwave_ice_m2kg;
    double shortwave_vapor_m2kg;
    double shortwave_liquid_m2kg;
    double shortwave_ice_m2kg;
};

class SimpleGreyScheme : public RadiationSchemeBase 
{
private:
    double tau_lw_ref_;
    double tau_sw_ref_;
    double n_lw_;
    double n_sw_;
    std::string optical_model_;
    SimpleGreyAbsorptionCoefficients absorption_coefficients_;

public:
    std::unique_ptr<RadiationSchemeBase> clone() const override
    {
        return std::make_unique<SimpleGreyScheme>(*this);
    }

    SimpleGreyScheme();

    std::string name() const override { return "simple_grey"; }
    int required_fields() const override;

    void initialize(const RadiationConfig& cfg) override;

    void compute_column(
        const RadiationConfig& cfg,
        const RadiationColumnStateView& col,
        RadiationColumnTendencies& tend,
        RadiationColumnFluxes* fluxes_opt = nullptr) override;

    bool supports_native_column_batch(const RadiationConfig&,const RadiationComputeKernels&)const override;
    bool compute_columns_native(const RadiationConfig&,const RadiationColumnStateView*,std::size_t,RadiationColumnTendencies*,RadiationComputeKernels&)override;
    bool compute_columns_native_diagnosed(const RadiationConfig&,const RadiationColumnStateView*,std::size_t,RadiationColumnTendencies*,RadiationComputeKernels&,RadiationColumnFluxes*)override;
private:
    void validate_column(const RadiationConfig&,const RadiationColumnStateView&)const;
    double compute_lw_heating(
        const RadiationConfig& cfg,
        const RadiationColumnStateView& col,
        std::vector<double>& dTdt_lw,
        RadiationColumnFluxes* fluxes_opt,
        RadiationColumnWorkspace& workspace);

    double compute_sw_heating(
        const RadiationConfig& cfg,
        const RadiationColumnStateView& col,
        std::vector<double>& dTdt_sw,
        RadiationColumnFluxes* fluxes_opt,
        RadiationColumnWorkspace& workspace);
};

const char* simple_grey_absorbing_water_cloud_algorithm_id() noexcept;
const SimpleGreyAbsorptionCoefficients&
simple_grey_absorbing_water_cloud_coefficients() noexcept;

void compute_simple_grey_shortwave_layer_optical_depths(
    const RadiationConfig& config,
    const std::vector<double>& pressure_interface_pa,
    const std::vector<double>& layer_thickness_m,
    const std::vector<double>& density_kgm3,
    const std::vector<double>* vapor_mixing_ratio,
    const std::vector<double>* cloud_liquid_mixing_ratio,
    const std::vector<double>* cloud_ice_mixing_ratio,
    std::vector<double>& optical_depth_profile,
    std::vector<double>& layer_optical_depth);

} // namespace tmv

using tmv::SimpleGreyScheme;
