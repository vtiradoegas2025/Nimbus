#pragma once

#include <memory>

namespace tmv
{
template <class State>
class TransientWorkspace
{
public:
    TransientWorkspace() = default;
    TransientWorkspace(const TransientWorkspace&) noexcept {}
    TransientWorkspace& operator=(const TransientWorkspace& other) noexcept
    {
        if (this != &other) state_.reset();
        return *this;
    }
    TransientWorkspace(TransientWorkspace&&) noexcept = default;
    TransientWorkspace& operator=(TransientWorkspace&&) noexcept = default;

    State& get()
    {
        if (!state_) state_ = std::make_unique<State>();
        return *state_;
    }

private:
    std::unique_ptr<State> state_;
};
}
