/**
 * @file correlation_filter.cpp
 * @brief Implementation for the chaos module.
 *
 * Provides executable logic for the chaos runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/chaos subsystem.
 */

#include "correlation_filter.hpp"
#include "util/log.hpp"

#include <cmath>
#include <algorithm>
#include <complex>
#include <stdexcept>
#include <vector>

namespace tmv::chaos
{

namespace
{

constexpr double pi = 3.14159265358979323846;

inline bool is_power_of_two(size_t n)
{
    return n > 0 && (n & (n - 1)) == 0;
}

size_t next_power_of_two(size_t n)
{
    if (n <= 1)
    {
        return 1;
    }
    if (is_power_of_two(n))
    {
        return n;
    }
    size_t p = 1;
    while (p < n)
    {
        p <<= 1;
    }
    return p;
}

void fft_1d(std::vector<std::complex<double>>& values, bool inverse)
{
    const size_t n = values.size();
    if (n <= 1)
    {
        return;
    }

    for (size_t i = 1, j = 0; i < n; ++i)
    {
        size_t bit = n >> 1;
        while (j & bit)
        {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j)
        {
            std::swap(values[i], values[j]);
        }
    }

    for (size_t len = 2; len <= n; len <<= 1)
    {
        const double angle = 2.0 * pi / static_cast<double>(len) * (inverse ? 1.0 : -1.0);
        const std::complex<double> w_len(std::cos(angle), std::sin(angle));
        for (size_t i = 0; i < n; i += len)
        {
            std::complex<double> w(1.0, 0.0);
            for (size_t j = 0; j < len / 2; ++j)
            {
                const std::complex<double> u = values[i + j];
                const std::complex<double> v = values[i + j + len / 2] * w;
                values[i + j] = u + v;
                values[i + j + len / 2] = u - v;
                w *= w_len;
            }
        }
    }

    if (inverse)
    {
        const double inv_n = 1.0 / static_cast<double>(n);
        for (std::complex<double>& value : values)
        {
            value *= inv_n;
        }
    }
}

void fft_2d(
    std::vector<std::complex<double>>& field,
    size_t nr,
    size_t nth,
    bool inverse,
    std::vector<std::complex<double>>& row_workspace,
    std::vector<std::complex<double>>& col_workspace
)
{
    if (nr == 0 || nth == 0)
    {
        return;
    }
    row_workspace.resize(nth);
    col_workspace.resize(nr);

    for (size_t i = 0; i < nr; ++i)
    {
        for (size_t j = 0; j < nth; ++j)
        {
            row_workspace[j] = field[i * nth + j];
        }
        fft_1d(row_workspace, inverse);
        for (size_t j = 0; j < nth; ++j)
        {
            field[i * nth + j] = row_workspace[j];
        }
    }

    for (size_t j = 0; j < nth; ++j)
    {
        for (size_t i = 0; i < nr; ++i)
        {
            col_workspace[i] = field[i * nth + j];
        }
        fft_1d(col_workspace, inverse);
        for (size_t i = 0; i < nr; ++i)
        {
            field[i * nth + j] = col_workspace[i];
        }
    }
}

}


SpectralGaussianFilter::SpectralGaussianFilter(double Lx, double Ly)
    : Lx_(Lx), Ly_(Ly)
{
    if (!std::isfinite(Lx_) || Lx_ <= 0.0 || !std::isfinite(Ly_) || Ly_ <= 0.0)
    {
        throw std::invalid_argument(
            "Spectral Gaussian correlation lengths must be positive and finite");
    }
}

void SpectralGaussianFilter::ensure_workspace(size_t nr_fft, size_t nth_fft)
{
    if (nr_fft_ != nr_fft || nth_fft_ != nth_fft)
    {
        nr_fft_ = nr_fft;
        nth_fft_ = nth_fft;
        spectral_workspace_.assign(nr_fft_ * nth_fft_, std::complex<double>(0.0, 0.0));
        row_workspace_.assign(nth_fft_, std::complex<double>(0.0, 0.0));
        col_workspace_.assign(nr_fft_, std::complex<double>(0.0, 0.0));
    }
    else
    {
        std::fill(spectral_workspace_.begin(), spectral_workspace_.end(), std::complex<double>(0.0, 0.0));
    }
}

/**
 * @brief Applies the 2D spectral Gaussian correlation filter.
 */
void SpectralGaussianFilter::apply_2d(
    std::vector<std::vector<double>>& field,
    double dx,
    double dy
)
{
    if (field.empty() || field[0].empty())
    {
        return;
    }

    const size_t nr = field.size();
    const size_t nth = field[0].size();
    for (size_t i = 1; i < nr; ++i)
    {
        if (field[i].size() != nth)
        {
            throw std::invalid_argument("Correlation-filter field must be rectangular");
        }
    }

    if (!std::isfinite(dx) || dx <= 0.0 || !std::isfinite(dy) || dy <= 0.0)
    {
        throw std::invalid_argument("Correlation-filter grid spacing must be positive and finite");
    }
    const double dx_eff = dx;
    const double dy_eff = dy;
    const size_t nr_fft = next_power_of_two(nr);
    const size_t nth_fft = next_power_of_two(nth);

    ensure_workspace(nr_fft, nth_fft);
    auto& spectral = spectral_workspace_;
    for (size_t i = 0; i < nr; ++i)
    {
        for (size_t j = 0; j < nth; ++j)
        {
            spectral[i * nth_fft + j] = std::complex<double>(field[i][j], 0.0);
        }
    }

    fft_2d(spectral, nr_fft, nth_fft, false, row_workspace_, col_workspace_);

    const double domain_x = static_cast<double>(nr_fft) * dx_eff;
    const double domain_y = static_cast<double>(nth_fft) * dy_eff;
    for (size_t i = 0; i < nr_fft; ++i)
    {
        const int ki = (i <= nr_fft / 2) ? static_cast<int>(i) : static_cast<int>(i) - static_cast<int>(nr_fft);
        const double kx = 2.0 * pi * static_cast<double>(ki) / domain_x;
        const double lxkx = Lx_ * kx;
        for (size_t j = 0; j < nth_fft; ++j)
        {
            const int kj = (j <= nth_fft / 2) ? static_cast<int>(j) : static_cast<int>(j) - static_cast<int>(nth_fft);
            const double ky = 2.0 * pi * static_cast<double>(kj) / domain_y;
            const double lyky = Ly_ * ky;
            const double gain = std::exp(-0.5 * (lxkx * lxkx + lyky * lyky));
            spectral[i * nth_fft + j] *= gain;
        }
    }

    fft_2d(spectral, nr_fft, nth_fft, true, row_workspace_, col_workspace_);

    for (size_t i = 0; i < nr; ++i)
    {
        for (size_t j = 0; j < nth; ++j)
        {
            field[i][j] = spectral[i * nth_fft + j].real();
        }
    }
}


RecursiveGaussianFilter::RecursiveGaussianFilter(double Lx, double Ly)
    : Lx_(Lx), Ly_(Ly)
{
    if (!std::isfinite(Lx_) || Lx_ <= 0.0 || !std::isfinite(Ly_) || Ly_ <= 0.0)
    {
        throw std::invalid_argument(
            "Recursive Gaussian correlation lengths must be positive and finite");
    }
}

/**
 * @brief Applies the 2D recursive Gaussian filter.
 */
void RecursiveGaussianFilter::apply_2d(
    std::vector<std::vector<double>>& field,
    double dx,
    double dy
) 

{
    if (field.empty() || field[0].empty())
    {
        return;
    }

    const size_t nr = field.size();
    const size_t nth = field[0].size();
    for (size_t i = 1; i < nr; ++i)
    {
        if (field[i].size() != nth)
        {
            throw std::invalid_argument("Correlation-filter field must be rectangular");
        }
    }

    if (!std::isfinite(dx) || dx <= 0.0 || !std::isfinite(dy) || dy <= 0.0)
    {
        throw std::invalid_argument("Correlation-filter grid spacing must be positive and finite");
    }
    const double dx_eff = dx;
    const double dy_eff = dy;
    const double wx = std::exp(-dx_eff / Lx_);
    const double wy = std::exp(-dy_eff / Ly_);
    const int passes = 3;

    for (int pass = 0; pass < passes; ++pass)
    {
        for (size_t j = 0; j < nth; ++j)
        {
            for (size_t i = 1; i < nr; ++i)
            {
                field[i][j] = wx * field[i - 1][j] + (1.0 - wx) * field[i][j];
            }
            for (size_t i = nr - 1; i > 0; --i)
            {
                field[i - 1][j] = wx * field[i][j] + (1.0 - wx) * field[i - 1][j];
            }
        }

        for (size_t i = 0; i < nr; ++i)
        {
            std::vector<double>& row = field[i];
            for (size_t j = 1; j < nth; ++j)
            {
                row[j] = wy * row[j - 1] + (1.0 - wy) * row[j];
            }
            for (size_t j = nth - 1; j > 0; --j)
            {
                row[j - 1] = wy * row[j] + (1.0 - wy) * row[j - 1];
            }
        }
    }
}


std::unique_ptr<CorrelationFilter> create_correlation_filter(
    const std::string& filter_id,
    double Lx,
    double Ly
) 
{
    if (filter_id == "spectral_gaussian")
    {
        return std::make_unique<SpectralGaussianFilter>(Lx, Ly);
    }
    if (filter_id == "recursive_gaussian")
    {
        return std::make_unique<RecursiveGaussianFilter>(Lx, Ly);
    }
    throw std::invalid_argument(
        "Unknown correlation filter '" + filter_id +
        "'. Supported filters: spectral_gaussian, recursive_gaussian");
}

}
