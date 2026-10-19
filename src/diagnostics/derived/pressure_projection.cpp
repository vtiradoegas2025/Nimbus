#include "diagnostics/pressure_projection.hpp"
#include "diagnostics/deterministic_reduction.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace tmv::diagnostics
{
namespace
{

double dot(const std::vector<double>& a, const std::vector<double>& b)
{
    CompensatedSum sum;
    for (std::size_t i = 0; i < a.size(); ++i)
        sum.add(a[i] * b[i]);
    if (!std::isfinite(sum.value()))
        throw std::overflow_error("pressure projection dot product overflowed");
    return sum.value();
}

double remove_mean(std::vector<double>& values)
{
    CompensatedSum sum;
    for (double value : values)
        sum.add(value);
    const double mean = sum.value() / static_cast<double>(values.size());
    if (!std::isfinite(mean))
        throw std::overflow_error("pressure projection mean overflowed");
    for (double& value : values)
        value -= mean;
    return mean;
}

} // namespace

PressureProjection::PressureProjection(const PressureProjectionGrid& grid)
{
    if (grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0 || !std::isfinite(grid.dx) || grid.dx <= 0.0 ||
        !std::isfinite(grid.dy) || grid.dy <= 0.0)
        throw std::invalid_argument("pressure projection requires positive Cartesian dimensions and spacing");
    const auto nx = static_cast<std::size_t>(grid.nx);
    const auto ny = static_cast<std::size_t>(grid.ny);
    const auto nz = static_cast<std::size_t>(grid.nz);
    if (nx > std::numeric_limits<std::size_t>::max() / ny || nx * ny > std::numeric_limits<std::size_t>::max() / nz)
        throw std::length_error("pressure projection dimensions overflow");
    const std::size_t count = nx * ny * nz;
    for (const auto* field : {&grid.height, &grid.volume, &grid.density, &grid.reference_density})
        if (field->size() != count)
            throw std::invalid_argument("pressure projection geometry or density shape differs");
    const auto index = [&](int i, int j, int k)
    { return (static_cast<std::size_t>(i) * ny + static_cast<std::size_t>(j)) * nz + static_cast<std::size_t>(k); };
    for (std::size_t n = 0; n < count; ++n)
    {
        if (!std::isfinite(grid.height[n]) || !std::isfinite(grid.volume[n]) || grid.volume[n] <= 0.0 ||
            !std::isfinite(grid.density[n]) || grid.density[n] <= 0.0 || !std::isfinite(grid.reference_density[n]) ||
            grid.reference_density[n] <= 0.0 || (n % nz != 0 && grid.height[n] <= grid.height[n - 1]))
            throw std::invalid_argument("pressure projection needs finite positive measures and increasing columns");
    }
    volume_ = grid.volume;
    diagonal_.assign(count, 0.0);

    struct Term
    {
        std::size_t index       = 0;
        double      coefficient = 0.0;
    };
    const auto append = [&](int i, int j, int k, int ii, int jj, int kk, int component)
    {
        const std::size_t   left  = index(i, j, k);
        const std::size_t   right = index(ii, jj, kk);
        std::array<Term, 8> terms{};
        std::size_t         used = 0;
        const auto          add  = [&](std::size_t n, double coefficient)
        {
            if (!std::isfinite(coefficient))
                throw std::overflow_error("pressure projection gradient is not representable");
            for (std::size_t slot = 0; slot < used; ++slot)
                if (terms[slot].index == n)
                {
                    terms[slot].coefficient += coefficient;
                    return;
                }
            if (used == terms.size())
                throw std::logic_error("pressure projection exceeded its gradient stencil");
            terms[used++] = {n, coefficient};
        };
        const auto vertical_derivative = [&](int x, int y, int z, double factor)
        {
            if (grid.nz == 1 || factor == 0.0)
                return;
            if (grid.nz == 2)
            {
                const double scale = factor / (grid.height[index(x, y, 1)] - grid.height[index(x, y, 0)]);
                add(index(x, y, 0), -scale);
                add(index(x, y, 1), scale);
                return;
            }
            const int    first      = std::max(0, std::min(z - 1, grid.nz - 3));
            const double evaluation = grid.height[index(x, y, z)];
            for (int point = 0; point < 3; ++point)
            {
                const auto   n = index(x, y, first + point);
                const double a = grid.height[n];
                const double b = grid.height[index(x, y, first + (point + 1) % 3)];
                const double c = grid.height[index(x, y, first + (point + 2) % 3)];
                add(n, factor * ((evaluation - b) + (evaluation - c)) / ((a - b) * (a - c)));
            }
        };
        const double spacing = component == 0   ? grid.dx
                               : component == 1 ? grid.dy
                                                : grid.height[right] - grid.height[left];
        add(left, -1.0 / spacing);
        add(right, 1.0 / spacing);
        if (component != 2)
        {
            const double factor = -.5 * (grid.height[right] - grid.height[left]) / spacing;
            vertical_derivative(i, j, k, factor);
            vertical_derivative(ii, jj, kk, factor);
        }
        Row row;
        row.left                 = left;
        row.right                = right;
        row.component            = component;
        row.reference_mass_left  = .5 * grid.volume[left] * grid.reference_density[left];
        row.reference_mass_right = .5 * grid.volume[right] * grid.reference_density[right];
        const double face_density =
            (grid.volume[left] * grid.density[left] + grid.volume[right] * grid.density[right]) /
            (grid.volume[left] + grid.volume[right]);
        row.weight = (row.reference_mass_left + row.reference_mass_right) / face_density;
        if (!std::isfinite(row.weight) || row.weight <= 0.0)
            throw std::overflow_error("pressure projection face weight is not representable");
        row.begin    = indices_.size();
        double pivot = 0.0;
        for (std::size_t slot = 0; slot < used; ++slot)
        {
            const auto& term = terms[slot];
            if (term.index == left || term.coefficient == 0.0)
                continue;
            indices_.push_back(term.index);
            coefficients_.push_back(term.coefficient);
            pivot -= term.coefficient;
            diagonal_[term.index] += row.weight * term.coefficient * term.coefficient;
        }
        diagonal_[left] += row.weight * pivot * pivot;
        row.end = indices_.size();
        rows_.push_back(row);
    };
    for (int i = 0; i < grid.nx; ++i)
        for (int j = 0; j < grid.ny; ++j)
            for (int k = 0; k < grid.nz; ++k)
            {
                if (grid.nx > 1 && (i + 1 < grid.nx || grid.periodic_x))
                    append(i, j, k, (i + 1) % grid.nx, j, k, 0);
                if (grid.ny > 1 && (j + 1 < grid.ny || grid.periodic_y))
                    append(i, j, k, i, (j + 1) % grid.ny, k, 1);
                if (k + 1 < grid.nz)
                    append(i, j, k, i, j, k + 1, 2);
            }
    if (count == 1)
        return;
    normalization_ = *std::max_element(diagonal_.begin(), diagonal_.end());
    if (!std::isfinite(normalization_) || normalization_ <= 0.0)
        throw std::overflow_error("pressure projection diagonal is not representable");
    for (double& diagonal : diagonal_)
    {
        diagonal /= normalization_;
        if (!std::isfinite(diagonal) || diagonal <= 0.0)
            throw std::overflow_error("pressure projection preconditioner is not representable");
    }
    for (auto& row : rows_)
        row.weight /= normalization_;
}

void PressureProjection::apply(const std::vector<double>& input, std::vector<double>& output) const
{
    std::fill(output.begin(), output.end(), 0.0);
    for (const auto& row : rows_)
    {
        double gradient = 0.0;
        for (std::size_t slot = row.begin; slot < row.end; ++slot)
            gradient += coefficients_[slot] * (input[indices_[slot]] - input[row.left]);
        const double flux = row.weight * gradient;
        for (std::size_t slot = row.begin; slot < row.end; ++slot)
        {
            const double value = coefficients_[slot] * flux;
            output[indices_[slot]] += value;
            output[row.left] -= value;
        }
    }
    remove_mean(output);
}

std::vector<double> PressureProjection::solve(const std::array<std::vector<double>, 3>& acceleration,
                                              PressureProjectionReport&                 report,
                                              double                                    relative_tolerance,
                                              int                                       maximum_iterations) const
{
    const std::size_t count = volume_.size();
    if (!std::isfinite(relative_tolerance) || relative_tolerance <= 0.0 || relative_tolerance >= 1.0 ||
        maximum_iterations < 1)
        throw std::invalid_argument("pressure projection solve controls are invalid");
    for (const auto& field : acceleration)
        if (field.size() != count ||
            !std::all_of(field.begin(), field.end(), [](double value) { return std::isfinite(value); }))
            throw std::invalid_argument("pressure projection requires complete finite forcing");
    std::vector<double> rhs(count, 0.0), x(count, 0.0), residual(count), direction(count), image(count),
        preconditioned(count);
    for (const auto& row : rows_)
    {
        const auto&  force = acceleration[static_cast<std::size_t>(row.component)];
        const double value = row.reference_mass_left * force[row.left] + row.reference_mass_right * force[row.right];
        for (std::size_t slot = row.begin; slot < row.end; ++slot)
        {
            const double contribution = coefficients_[slot] * value;
            rhs[indices_[slot]] += contribution;
            rhs[row.left] -= contribution;
        }
    }
    double scale = 0.0;
    for (double value : rhs)
    {
        if (!std::isfinite(value))
            throw std::overflow_error("pressure projection forcing overflowed");
        scale = std::max(scale, std::abs(value));
    }
    PressureProjectionReport candidate;
    if (scale == 0.0)
    {
        report = candidate;
        return x;
    }
    for (double& value : rhs)
        value /= scale;
    candidate.removed_rhs_mean = remove_mean(rhs) * scale;
    const double initial       = dot(rhs, rhs);
    if (initial == 0.0)
    {
        report = candidate;
        return x;
    }
    residual                = rhs;
    const auto precondition = [&]()
    {
        for (std::size_t n = 0; n < count; ++n)
            preconditioned[n] = residual[n] / diagonal_[n];
        remove_mean(preconditioned);
    };
    precondition();
    direction        = preconditioned;
    double previous  = dot(residual, preconditioned);
    bool   converged = false;
    for (int iteration = 0; iteration < maximum_iterations; ++iteration)
    {
        apply(direction, image);
        const double denominator = dot(direction, image);
        if (!std::isfinite(previous) || previous <= 0.0 || denominator <= 0.0)
            throw std::runtime_error("pressure projection conjugate gradient lost positive definiteness");
        const double alpha = previous / denominator;
        for (std::size_t n = 0; n < count; ++n)
        {
            x[n] += alpha * direction[n];
            residual[n] -= alpha * image[n];
        }
        remove_mean(x);
        remove_mean(residual);
        candidate.iterations = iteration + 1;
        double     norm      = dot(residual, residual);
        const bool restart   = candidate.iterations % 64 == 0;
        if (restart || norm <= initial * relative_tolerance * relative_tolerance)
        {
            apply(x, image);
            for (std::size_t n = 0; n < count; ++n)
                residual[n] = rhs[n] - image[n];
            remove_mean(residual);
            norm = dot(residual, residual);
            if (norm <= initial * relative_tolerance * relative_tolerance)
            {
                candidate.relative_residual = std::sqrt(norm / initial);
                converged                   = true;
                break;
            }
        }
        precondition();
        const double next = dot(residual, preconditioned);
        const double beta = restart ? 0.0 : next / previous;
        for (std::size_t n = 0; n < count; ++n)
            direction[n] = preconditioned[n] + beta * direction[n];
        remove_mean(direction);
        previous = next;
    }
    if (!converged)
        throw std::runtime_error("pressure projection did not converge within " + std::to_string(maximum_iterations) +
                                 " iterations");
    const double pressure_scale = scale / normalization_;
    if (!std::isfinite(pressure_scale) || pressure_scale == 0.0)
        throw std::overflow_error("pressure projection scale is not representable");
    for (double& value : x)
    {
        value *= pressure_scale;
        if (!std::isfinite(value))
            throw std::overflow_error("pressure projection solution is not representable");
    }
    const double   volume_scale = *std::max_element(volume_.begin(), volume_.end());
    CompensatedSum measure, mean;
    for (std::size_t n = 0; n < count; ++n)
    {
        const double weight = volume_[n] / volume_scale;
        measure.add(weight);
        mean.add(weight * x[n]);
    }
    const double gauge = mean.value() / measure.value();
    if (!std::isfinite(gauge))
        throw std::overflow_error("pressure projection gauge is not representable");
    CompensatedSum residual_mean;
    for (std::size_t n = 0; n < count; ++n)
    {
        x[n] -= gauge;
        if (!std::isfinite(x[n]))
            throw std::overflow_error("pressure projection gauge-shifted solution is not representable");
        residual_mean.add((volume_[n] / volume_scale) * x[n]);
    }
    candidate.volume_mean_pressure = residual_mean.value() / measure.value();
    if (!std::isfinite(candidate.volume_mean_pressure))
        throw std::overflow_error("pressure projection final mean is not representable");
    report                         = candidate;
    return x;
}

} // namespace tmv::diagnostics
