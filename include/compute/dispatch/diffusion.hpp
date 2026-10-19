#pragma once

/**
 * @file diffusion.hpp
 * @brief Diffusion compute dispatch data and capability interface.
 */

#include <algorithm>
#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

enum class DiffusionKernelOp
{
    Diffusion,
    Implicit
};

enum class DiffusionFieldLocation : int
{
    Scalar = 0,
    UFace = 1,
    VFace = 2,
    WFace = 3
};

struct DiffusionDispatch
{
    // Native Cartesian finite-volume tendency ABI. `field` is never advanced
    // here: the kernel returns d(field)/dt so the host retains the same
    // guarded commit and energy-ledger path as the CPU reference.
    const float* field = nullptr;
    const float* rho = nullptr;
    const float* dz = nullptr;
    float* tendency = nullptr;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    float dx = 0.0f;
    float dy = 0.0f;
    float K_h = 0.0f;
    float K_v = 0.0f;
    DiffusionFieldLocation location = DiffusionFieldLocation::Scalar;
    bool c_grid = false;
    bool periodic_x = false;
    bool periodic_y = false;
    const float* vertical_K = nullptr;
};

struct DiffusionBatchDispatch
{
    // All fields share one grid, live density, vertical metric, coefficients,
    // and topology. The pointer arrays and locations contain `count` entries.
    const float* const* fields = nullptr;
    float* const* tendencies = nullptr;
    const DiffusionFieldLocation* locations = nullptr;
    int count = 0;
    const float* rho = nullptr;
    const float* dz = nullptr;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    float dx = 0.0f;
    float dy = 0.0f;
    float K_h = 0.0f;
    float K_v = 0.0f;
    bool c_grid = false;
    bool periodic_x = false;
    bool periodic_y = false;
    const float* const* vertical_K = nullptr;
};

struct ImplicitDiffusionStencilDispatch
{
    DiffusionDispatch diffusion;
    float             dt          = 0;
    bool              form_rhs    = false;
    float             field_scale = 1, density_scale = 1, thickness_scale = 1;
    float*            diagonal            = nullptr;
    float*            dot                 = nullptr;
    float*            preconditioned_norm = nullptr;
    float*            norm                = nullptr;
};

enum class ImplicitDiffusionVectorStage : int
{
    Initialize,
    Update,
    Direction,
    Restart,
    Finalize
};

struct ImplicitDiffusionVectorDispatch
{
    const float*                 original            = nullptr;
    const float*                 rhs                 = nullptr;
    const float*                 diagonal            = nullptr;
    const float*                 solution            = nullptr;
    const float*                 residual            = nullptr;
    const float*                 direction           = nullptr;
    const float*                 image               = nullptr;
    float*                       next_solution       = nullptr;
    float*                       next_residual       = nullptr;
    float*                       next_direction      = nullptr;
    float*                       preconditioned_norm = nullptr;
    float*                       norm                = nullptr;
    int                          nx = 0, ny = 0, nz = 0;
    ImplicitDiffusionVectorStage stage = ImplicitDiffusionVectorStage::Initialize;
    float                        alpha = 0, beta = 0, dt = 0, field_scale = 1;
};

class DiffusionComputeKernels
{
public:
    virtual ~DiffusionComputeKernels() = default;

    // ── Diffusion dispatch ───────────────────────────────────────────

    virtual bool supports(DiffusionKernelOp /*op*/) const { return false; }

    virtual bool dispatch_implicit_stencil(const ImplicitDiffusionStencilDispatch&)
    {
        return false;
    }
    virtual bool dispatch_implicit_vector(const ImplicitDiffusionVectorDispatch&)
    {
        return false;
    }

    /** Execute the live-density Cartesian finite-volume diffusion tendency. */
    virtual bool dispatch_diffusion(const DiffusionDispatch& dispatch)
    {
        (void)dispatch;
        return false;
    }

    /**
     * Execute independent diffusion targets with shared metrics/density.
     *
     * The default preserves backend compatibility by invoking the qualified
     * single-field operation. Results are staged and published only after
     * every target succeeds, so a failed compatibility dispatch cannot leave
     * a partially updated batch. Backends may override to record the same
     * atomic operation in one submission without changing its equations.
     */
    virtual bool dispatch_diffusion_batch(
        const DiffusionBatchDispatch& dispatch)
    {
        if (dispatch.fields == nullptr || dispatch.tendencies == nullptr ||
            dispatch.locations == nullptr || dispatch.rho == nullptr ||
            dispatch.dz == nullptr || dispatch.count < 1 ||
            dispatch.nx < 1 || dispatch.ny < 1 || dispatch.nz < 1)
        {
            return false;
        }

        const std::size_t nx = static_cast<std::size_t>(dispatch.nx);
        const std::size_t ny = static_cast<std::size_t>(dispatch.ny);
        const std::size_t nz = static_cast<std::size_t>(dispatch.nz);
        constexpr std::size_t maximum =
            std::numeric_limits<std::size_t>::max();
        if (nx > maximum / ny) return false;
        const std::size_t horizontal_cells = nx * ny;
        if (horizontal_cells > maximum / nz) return false;
        const std::size_t cells = horizontal_cells * nz;
        if (cells > maximum / sizeof(float)) return false;

        const std::size_t count = static_cast<std::size_t>(dispatch.count);
        if (cells > maximum / count) return false;
        const std::size_t staged_values = cells * count;
        if (staged_values > maximum / sizeof(float)) return false;

        for (int i = 0; i < dispatch.count; ++i)
        {
            if (dispatch.fields[i] == nullptr ||
                dispatch.tendencies[i] == nullptr)
            {
                return false;
            }
        }

        std::vector<float> staged;
        if (staged_values > staged.max_size()) return false;
        try
        {
            staged.resize(staged_values);
        }
        catch (const std::bad_alloc&)
        {
            return false;
        }
        catch (const std::length_error&)
        {
            return false;
        }

        for (int i = 0; i < dispatch.count; ++i)
        {
            float* const candidate = staged.data() +
                static_cast<std::size_t>(i) * cells;
            std::copy_n(dispatch.tendencies[i], cells, candidate);
            const DiffusionDispatch item{
                dispatch.fields[i], dispatch.rho, dispatch.dz,
                candidate,
                dispatch.nx, dispatch.ny, dispatch.nz,
                dispatch.dx, dispatch.dy, dispatch.K_h, dispatch.K_v,
                dispatch.locations[i], dispatch.c_grid,
                dispatch.periodic_x, dispatch.periodic_y,
                dispatch.vertical_K == nullptr
                    ? nullptr : dispatch.vertical_K[i]};
            if (!dispatch_diffusion(item)) return false;
        }

        for (int i = 0; i < dispatch.count; ++i)
        {
            const float* const candidate = staged.data() +
                static_cast<std::size_t>(i) * cells;
            std::copy_n(candidate, cells, dispatch.tendencies[i]);
        }
        return true;
    }
};
