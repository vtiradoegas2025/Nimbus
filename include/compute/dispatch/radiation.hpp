#pragma once

/**
 * @file radiation.hpp
 * @brief Radiation compute dispatch contracts.
 */

#include "compute/dispatch/simple_grey.hpp"

enum class RadiationKernelOp
{
    RrtmgLongwaveTransfer,
    RrtmgShortwaveTransfer,
    RrtmgLongwaveTransferBatch,
    RrtmgShortwaveTransferBatch,
    SimpleGrey
};

// Flattened, equal-height column batches. Every column uses the same layer
// count and band-width table. Column-major inputs are packed contiguously;
// outputs are [column][interface]. Backends publish the complete batch only
// after every column passes result validation.
struct RrtmgLongwaveTransferBatchDispatch
{
    const float* absorption_optical_depth = nullptr; // columns * layers * 140
    const float* planck_fraction = nullptr;          // columns * layers * 140
    const float* layer_planck = nullptr;             // columns * layers * 16
    const float* interface_planck = nullptr;         // columns * (layers + 1) * 16
    const float* surface_planck = nullptr;           // columns * 16
    const float* band_width_cm = nullptr;            // shared 16
    const float* cloud_absorption_optical_depth = nullptr; // optional columns * layers * 16
    const float* surface_emissivity = nullptr;       // columns
    const float* precipitable_water_cm = nullptr;    // columns
    float* upward_wm2 = nullptr;                     // columns * (layers + 1)
    float* downward_wm2 = nullptr;                   // columns * (layers + 1)
    int columns = 0;
    int layers = 0;
};

struct RrtmgShortwaveTransferBatchDispatch
{
    const float* absorption_optical_depth = nullptr; // columns * layers * 112
    const float* rayleigh_optical_depth = nullptr;   // columns * layers * 112
    const float* solar_source_fraction = nullptr;    // columns * 112
    const float* cloud_optical_depth = nullptr;      // optional columns * layers * 14
    const float* cloud_single_scattering_albedo = nullptr;
    const float* cloud_asymmetry = nullptr;
    const float* cosine_zenith = nullptr;             // columns
    const float* solar_irradiance_wm2 = nullptr;      // columns
    const float* surface_albedo = nullptr;            // columns
    float* upward_wm2 = nullptr;                      // columns * (layers + 1)
    float* downward_wm2 = nullptr;                    // columns * (layers + 1)
    int columns = 0;
    int layers = 0;
};

struct RrtmgShortwaveTransferDispatch
{
    const float* absorption_optical_depth = nullptr; // layers * 112
    const float* rayleigh_optical_depth = nullptr;   // layers * 112
    const float* solar_source_fraction = nullptr;    // 112
    const float* cloud_optical_depth = nullptr;      // optional layers * 14
    const float* cloud_single_scattering_albedo = nullptr;
    const float* cloud_asymmetry = nullptr;
    float* upward_wm2 = nullptr;                      // layers + 1
    float* downward_wm2 = nullptr;                    // layers + 1
    int layers = 0;
    float cosine_zenith = 0.0f;
    float solar_irradiance_wm2 = 0.0f;
    float surface_albedo = 0.0f;
};

// Prepared, single-column RRTMG longwave spectral transfer. Inputs use the
// canonical 140 reduced g-points and 16 bands. Backends publish both flux
// profiles together only after all spectral results pass validation.
struct RrtmgLongwaveTransferDispatch
{
    const float* absorption_optical_depth = nullptr; // layers * 140
    const float* planck_fraction = nullptr;          // layers * 140
    const float* layer_planck = nullptr;             // layers * 16
    const float* interface_planck = nullptr;         // (layers + 1) * 16
    const float* surface_planck = nullptr;           // 16
    const float* band_width_cm = nullptr;             // 16
    const float* cloud_absorption_optical_depth = nullptr; // optional layers * 16
    float* upward_wm2 = nullptr;                      // layers + 1
    float* downward_wm2 = nullptr;                    // layers + 1
    int layers = 0;
    float surface_emissivity = 1.0f;
    float precipitable_water_cm = 0.0f;
};

class RadiationComputeKernels
{
public:
    virtual ~RadiationComputeKernels() = default;
    virtual bool supports(RadiationKernelOp /*op*/) const { return false; }
    virtual bool dispatch_simple_grey(const SimpleGreyDispatch&)
    {
        return false;
    }
    virtual bool dispatch_rrtmg_longwave_transfer(
        const RrtmgLongwaveTransferDispatch& /*dispatch*/)
    {
        return false;
    }
    virtual bool dispatch_rrtmg_shortwave_transfer(
        const RrtmgShortwaveTransferDispatch& /*dispatch*/)
    {
        return false;
    }
    virtual bool dispatch_rrtmg_longwave_transfer_batch(
        const RrtmgLongwaveTransferBatchDispatch& /*dispatch*/)
    {
        return false;
    }
    virtual bool dispatch_rrtmg_shortwave_transfer_batch(
        const RrtmgShortwaveTransferBatchDispatch& /*dispatch*/)
    {
        return false;
    }
};
