#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace tmv::diagnostics
{

struct PressureProjectionGrid
{
    int                 nx         = 0;
    int                 ny         = 0;
    int                 nz         = 0;
    double              dx         = 0.0;
    double              dy         = 0.0;
    bool                periodic_x = false;
    bool                periodic_y = false;
    std::vector<double> height;
    std::vector<double> volume;
    std::vector<double> density;
    std::vector<double> reference_density;
};

struct PressureProjectionReport
{
    int    iterations           = 0;
    double relative_residual    = 0.0;
    double removed_rhs_mean     = 0.0;
    double volume_mean_pressure = 0.0;
};

// The adjoint gradient projection solves div(rho0/rho grad p) = div(rho0 F).
// Nonperiodic boundaries impose the forcing-consistent normal momentum condition;
// the pressure gauge is its physical-volume mean. Stored pressure is in Pa.
class PressureProjection
{
public:
    explicit PressureProjection(const PressureProjectionGrid& grid);

    std::vector<double> solve(const std::array<std::vector<double>, 3>& acceleration,
                              PressureProjectionReport&                 report,
                              double                                    relative_tolerance = 1.0e-10,
                              int                                       maximum_iterations = 4096) const;

private:
    struct Row
    {
        std::size_t left                 = 0;
        std::size_t right                = 0;
        std::size_t begin                = 0;
        std::size_t end                  = 0;
        int         component            = 0;
        double      reference_mass_left  = 0.0;
        double      reference_mass_right = 0.0;
        double      weight               = 0.0;
    };

    void apply(const std::vector<double>& input, std::vector<double>& output) const;

    std::vector<Row>         rows_;
    std::vector<std::size_t> indices_;
    std::vector<double>      coefficients_;
    std::vector<double>      diagonal_;
    std::vector<double>      volume_;
    double                   normalization_ = 1.0;
};

} // namespace tmv::diagnostics
