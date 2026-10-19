#pragma once
#include "compute/implicit_diffusion_contract.hpp"

namespace nimbus::compute
{
struct ImplicitDiffusionSolveReport
{
    int    iterations               = 0;
    double initial_residual_squared = 0, final_residual_squared = 0;
};

inline bool solve_implicit_diffusion(DiffusionComputeKernels&      kernels,
                                     const DiffusionDispatch&      target,
                                     float                         dt,
                                     ImplicitDiffusionSolveReport& report,
                                     int                           maximum_iterations = 4096)
{
    std::size_t n = 0, c = 0;
    if (!kernels.supports(DiffusionKernelOp::Implicit) || maximum_iterations < 1 ||
        !implicit_contract::dimensions(target.nx, target.ny, target.nz, n, c) || !target.field || !target.rho ||
        !target.dz || !target.tendency || !std::isfinite(dt) || dt <= 0)
        return false;
    for (const float* input : {target.field, target.rho, target.vertical_K})
        if (input != nullptr && implicit_contract::overlap(target.tendency, n, input, n))
            return false;
    if (implicit_contract::overlap(target.tendency, n, target.dz, static_cast<std::size_t>(target.nz)))
        return false;
    std::vector<float> rhs(n), diagonal(n), image(n), x(n), r(n), p(n), nx(n), nr(n), np(n), dot(c), rz(c), rr(c);
    ImplicitDiffusionStencilDispatch stencil;
    stencil.diffusion           = target;
    stencil.diffusion.tendency  = image.data();
    stencil.dt                  = dt;
    stencil.diagonal            = diagonal.data();
    stencil.dot                 = dot.data();
    stencil.preconditioned_norm = rz.data();
    stencil.norm                = rr.data();
    stencil.field_scale         = 0;
    stencil.density_scale       = 0;
    stencil.thickness_scale     = 0;
    for (int i = 0; i < target.nx; ++i)
        for (int j = 0; j < target.ny; ++j)
            for (int k = 0; k < target.nz; ++k)
            {
                const auto index = (static_cast<std::size_t>(i) * target.ny + j) * target.nz + k;
                if (implicit_contract::physical(target, i, j, k))
                {
                    if (!std::isfinite(target.field[index]))
                        return false;
                    stencil.field_scale = std::max(stencil.field_scale, std::abs(target.field[index]));
                }
                if (implicit_contract::physical(target, i, j, k, true))
                {
                    if (!std::isfinite(target.rho[index]))
                        return false;
                    stencil.density_scale = std::max(stencil.density_scale, target.rho[index]);
                }
            }
    for (int k = 0; k < target.nz; ++k)
        stencil.thickness_scale = std::max(stencil.thickness_scale, target.dz[k]);
    if (stencil.field_scale == 0)
        stencil.field_scale = 1;
    stencil.form_rhs = true;
    if (!kernels.dispatch_implicit_stencil(stencil))
        return false;
    rhs.swap(image);
    auto sum = [](const std::vector<float>& v)
    {
        double result = 0;
        for (float x : v)
            result += x;
        return result;
    };
    const double initial = sum(rr);
    if (!std::isfinite(initial) || initial < 0)
        return false;
    const double tolerance   = initial * 1e-12;
    auto         vector_step = [&](ImplicitDiffusionVectorStage stage, double alpha = 0, double beta = 0)
    {
        ImplicitDiffusionVectorDispatch d;
        d.original            = target.field;
        d.rhs                 = rhs.data();
        d.diagonal            = diagonal.data();
        d.solution            = x.data();
        d.residual            = r.data();
        d.direction           = p.data();
        d.image               = image.data();
        d.next_solution       = nx.data();
        d.next_residual       = nr.data();
        d.next_direction      = np.data();
        d.preconditioned_norm = rz.data();
        d.norm                = rr.data();
        d.nx                  = target.nx;
        d.ny                  = target.ny;
        d.nz                  = target.nz;
        d.stage               = stage;
        d.alpha               = static_cast<float>(alpha);
        d.beta                = static_cast<float>(beta);
        d.dt                  = dt;
        d.field_scale         = stencil.field_scale;
        if (!kernels.dispatch_implicit_vector(d))
            return false;
        x.swap(nx);
        r.swap(nr);
        p.swap(np);
        return true;
    };
    auto apply = [&](const std::vector<float>& input)
    {
        stencil.form_rhs           = false;
        stencil.diffusion.field    = input.data();
        stencil.diffusion.tendency = image.data();
        return kernels.dispatch_implicit_stencil(stencil);
    };
    if (!vector_step(ImplicitDiffusionVectorStage::Initialize))
        return false;
    double residual = initial, rho = sum(rz);
    int    iteration = 0;
    while (residual > tolerance && iteration < maximum_iterations)
    {
        if (!apply(p))
            return false;
        double denominator = sum(dot);
        if (!std::isfinite(denominator) || denominator <= 0 || !std::isfinite(rho) || rho <= 0)
            return false;
        if (!vector_step(ImplicitDiffusionVectorStage::Update, rho / denominator))
            return false;
        residual              = sum(rr);
        const double next_rho = sum(rz);
        ++iteration;
        if (!std::isfinite(residual) || residual < 0)
            return false;
        if (residual <= tolerance || iteration % 32 == 0)
        {
            if (!apply(x) || !vector_step(ImplicitDiffusionVectorStage::Restart))
                return false;
            residual = sum(rr);
            rho      = sum(rz);
        }
        else
        {
            if (!vector_step(ImplicitDiffusionVectorStage::Direction, 0, next_rho / rho))
                return false;
            rho = next_rho;
        }
    }
    if (!std::isfinite(residual) || residual > tolerance || !vector_step(ImplicitDiffusionVectorStage::Finalize))
        return false;
    // Reductions control the iteration on the host; every spatial operation,
    // Krylov update, and stored tendency is produced by the device kernels.
    std::copy(r.begin(), r.end(), target.tendency);
    report = {iteration, initial, residual};
    return true;
}
} // namespace nimbus::compute
