#include "chaos/scheme.hpp"

#include <memory>
#include <string>

class MinimalChaosScheme final : public tmv::chaos::ChaosScheme
{
public:
    std::unique_ptr<tmv::chaos::ChaosScheme> clone() const override
    {
        return std::make_unique<MinimalChaosScheme>(*this);
    }

    std::string name() const override { return "minimal_external"; }

    void initialize(const tmv::chaos::ChaosConfig& config,
                    const tmv::GridMetrics&) override
    {
        initialized_ = config.member_id >= 0;
    }

    bool initialized() const { return initialized_; }

private:
    bool initialized_ = false;
};

int main()
{
    tmv::chaos::ChaosConfig config;
    config.member_id = 7;
    tmv::GridMetrics grid;

    MinimalChaosScheme scheme;
    scheme.initialize(config, grid);
    auto copy = scheme.clone();

    return scheme.initialized() && copy &&
                   copy->name() == "minimal_external"
        ? 0
        : 1;
}
