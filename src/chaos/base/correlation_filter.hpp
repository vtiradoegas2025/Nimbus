/**
 * @file correlation_filter.hpp
 * @brief Declarations for the chaos module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the chaos runtime and scheme implementations.
 * This file is part of the src/chaos subsystem.
 */

#pragma once
#include <vector>
#include <complex>
#include <string>
#include <memory>

namespace tmv::chaos
{


/**
 * @brief Base class for spatial correlation filters
 */
class CorrelationFilter 
{
public:
    virtual ~CorrelationFilter() = default;
    virtual std::unique_ptr<CorrelationFilter> clone() const = 0;

    /**
     * @brief Apply correlation filter to 2D field
     * @param field Input/output field (modified in-place)
     * @param dx Grid spacing in x-direction
     * @param dy Grid spacing in y-direction
     */
    virtual     void apply_2d(
        std::vector<std::vector<double>>& field,
        double dx,
        double dy
    ) = 0;

    /**
     * @brief Apply correlation filter to 3D field (horizontal correlation only)
     * @param field Input/output field (modified in-place)
     * @param dx Grid spacing in x-direction
     * @param dy Grid spacing in y-direction
     */
    virtual void apply_3d(
        std::vector<std::vector<std::vector<float>>>& field,
        double dx,
        double dy
    ) 
    {
        if (field.empty() || field[0].empty() || field[0][0].empty()) 
        {
            return;
        }

        size_t NR = field.size();
        size_t NTH = field[0].size();
        size_t NZ = field[0][0].size();

        for (size_t k = 0; k < NZ; ++k) 
        {
            std::vector<std::vector<double>> horizontal_slice(NR, std::vector<double>(NTH));

            for (size_t i = 0; i < NR; ++i) 
            {
                for (size_t j = 0; j < NTH; ++j) 
                {
                    horizontal_slice[i][j] = static_cast<double>(field[i][j][k]);
                }
            }

            apply_2d(horizontal_slice, dx, dy);

            for (size_t i = 0; i < NR; ++i) 
            {
                for (size_t j = 0; j < NTH; ++j) 
                {
                    field[i][j][k] = static_cast<float>(horizontal_slice[i][j]);
                }
            }
        }
    }

    /**
     * @brief Get filter name
     */
    virtual std::string name() const = 0;
};

/**
 * @brief Spectral Gaussian correlation filter
 *
 * Applies isotropic/anisotropic Gaussian filtering in 2D spectral space:
 * G(kx, ky) = exp(-0.5 * ((Lx*kx)^2 + (Ly*ky)^2)).
 */
class SpectralGaussianFilter : public CorrelationFilter
{
public:
    /**
     * @brief Constructs spectral Gaussian filter with horizontal length scales.
     */
    SpectralGaussianFilter(double Lx, double Ly);
    std::unique_ptr<CorrelationFilter> clone() const override
    {
        return std::make_unique<SpectralGaussianFilter>(*this);
    }
    void apply_2d(std::vector<std::vector<double>>& field, double dx, double dy) override;
    std::string name() const override { return "spectral_gaussian"; }

private:
    double Lx_, Ly_;
    size_t nr_fft_ = 0;
    size_t nth_fft_ = 0;
    std::vector<std::complex<double>> spectral_workspace_;
    std::vector<std::complex<double>> row_workspace_;
    std::vector<std::complex<double>> col_workspace_;

    /**
     * @brief Resizes FFT workspaces to match active padded dimensions.
     */
    void ensure_workspace(size_t nr_fft, size_t nth_fft);
};

/**
 * @brief Recursive filter approximation (future implementation)
 *
 * Simpler and faster than spectral filter but approximate.
 * Uses successive passes of 1D filters.
 */
class RecursiveGaussianFilter : public CorrelationFilter 
{
public:
    /**
     * @brief Constructs recursive Gaussian filter with target length scales.
     */
    RecursiveGaussianFilter(double Lx, double Ly);
    std::unique_ptr<CorrelationFilter> clone() const override
    {
        return std::make_unique<RecursiveGaussianFilter>(*this);
    }
    void apply_2d(std::vector<std::vector<double>>& field, double dx, double dy) override;
    std::string name() const override { return "recursive_gaussian"; }

private:
    double Lx_, Ly_;
};


/**
 * @brief Create correlation filter instance
 * @param filter_id Filter type ("spectral_gaussian", "recursive_gaussian")
 * @param Lx Correlation length x
 * @param Ly Correlation length y
 * @return Unique pointer to correlation filter
 */
std::unique_ptr<CorrelationFilter> create_correlation_filter(
    const std::string& filter_id,
    double Lx,
    double Ly
);

} // namespace tmv::chaos

#ifndef NIMBUS_CHAOS_NAMESPACE_ALIAS
#define NIMBUS_CHAOS_NAMESPACE_ALIAS
namespace chaos = tmv::chaos;
#endif
