#pragma once

#include "compute/backend_runtime.hpp"
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace nimbus::compute
{
class NativeConservativeReduction
{
public:
    explicit NativeConservativeReduction(ComputeBackendRuntimeState* runtime) : runtime_(runtime)
    {
        enabled_ = runtime && (runtime->active_kind != ComputeBackendKind::Cpu || runtime->config.backend == "metal" ||
                               runtime->config.backend == "vulkan" || runtime->config.backend == "gpu");
        if (enabled_ && (runtime->fallback_active || !runtime->backend ||
                         !runtime->backend->supports_dynamics(DynamicsKernelOp::ConservativeReduction)))
            throw std::runtime_error("native conservative feedback requires its reduction kernel");
    }
    bool enabled() const noexcept
    {
        return enabled_;
    }
    void clear()
    {
        headers_.clear();
        terms_.clear();
        output_.clear();
    }
    void begin_row()
    {
        headers_.push_back({float(terms_.size()), 0, 0, 0});
    }
    void term(double density, double quantity, double measure)
    {
        if (terms_.size() >= 16777216)
            throw std::overflow_error("native conservative reduction term capacity exceeded");
        std::array<float, 6> term;
        split(density, term[0], term[1]);
        split(quantity, term[2], term[3]);
        split(measure, term[4], term[5]);
        terms_.push_back(term);
    }
    void denominator()
    {
        auto& header = headers_.back();
        header[1]    = float(terms_.size()) - header[0];
        header[2]    = float(terms_.size());
        denominator_ = true;
    }
    void end_row()
    {
        auto& header = headers_.back();
        if (denominator_)
            header[3] = float(terms_.size()) - header[2];
        else
            header[1] = float(terms_.size()) - header[0];
        denominator_ = false;
    }
    void dispatch()
    {
        if (!enabled_ || headers_.empty())
            return;
        if (headers_.size() > 16777216)
            throw std::overflow_error("native conservative reduction row capacity exceeded");
        std::vector<float> packed;
        packed.reserve(headers_.size() * 4 + terms_.size() * 6);
        for (const auto& header : headers_)
            packed.insert(packed.end(), header.begin(), header.end());
        for (const auto& term : terms_)
            packed.insert(packed.end(), term.begin(), term.end());
        output_.resize(headers_.size() * 5);
        ConservativeReductionDispatch d{packed.data(), output_.data(), {int(headers_.size()), int(terms_.size())}};
        if (!runtime_->backend->dynamics_kernels()->dispatch_conservative_reduction(d))
            throw std::runtime_error("native conservative reduction failed before publication");
    }
    double inventory(std::size_t row) const
    {
        return double(output_.at(row * 5)) + output_.at(row * 5 + 1);
    }
    float stored(std::size_t row) const
    {
        return output_.at(row * 5 + 2);
    }

private:
    static void split(double value, float& high, float& low)
    {
        high = float(value);
        low  = float(value - double(high));
        if (!std::isfinite(high) || !std::isfinite(low))
            throw std::overflow_error("native conservative reduction operand is not representable");
    }
    ComputeBackendRuntimeState*       runtime_     = nullptr;
    bool                              enabled_     = false;
    bool                              denominator_ = false;
    std::vector<std::array<float, 4>> headers_;
    std::vector<std::array<float, 6>> terms_;
    std::vector<float>                output_;
};
} // namespace nimbus::compute
