#pragma once

#include <array>
#include <tuple>
#include <type_traits>
#include <utility>

namespace tmv::fields
{
template <class Accessor>
struct Entry
{
    const char* name;
    Accessor get;
};
template <class Accessor>
constexpr auto entry(const char* name, Accessor accessor)
{
    return Entry<Accessor>{name, accessor};
}

inline constexpr auto prognostic = std::make_tuple(
    entry("rho", [](auto& state) -> auto& { return state.rho; }),
    entry("p", [](auto& state) -> auto& { return state.p; }),
    entry("u", [](auto& state) -> auto& { return state.u; }),
    entry("w", [](auto& state) -> auto& { return state.w; }),
    entry("v", [](auto& state) -> auto& { return state.v; }),
    entry("tracer", [](auto& state) -> auto& { return state.tracer; }),
    entry("theta", [](auto& state) -> auto& { return state.theta; }),
    entry("qv", [](auto& state) -> auto& { return state.qv; }),
    entry("qc", [](auto& state) -> auto& { return state.qc; }),
    entry("qr", [](auto& state) -> auto& { return state.qr; }),
    entry("qi", [](auto& state) -> auto& { return state.qi; }),
    entry("qs", [](auto& state) -> auto& { return state.qs; }),
    entry("qh", [](auto& state) -> auto& { return state.qh; }),
    entry("qg", [](auto& state) -> auto& { return state.qg; }),
    entry("tke", [](auto& state) -> auto& { return state.tke; }));

inline constexpr auto physics_tendencies = std::make_tuple(
    entry("dtheta_dt_rad", [](auto& state) -> auto& { return state.dtheta_dt_rad; }),
    entry("dtheta_dt_pbl", [](auto& state) -> auto& { return state.dtheta_dt_pbl; }),
    entry("dqv_dt_pbl", [](auto& state) -> auto& { return state.dqv_dt_pbl; }),
    entry("du_dt_pbl", [](auto& state) -> auto& { return state.du_dt_pbl; }),
    entry("dv_dt_pbl", [](auto& state) -> auto& { return state.dv_dt_pbl; }),
    entry("dtke_dt_pbl", [](auto& state) -> auto& { return state.dtke_dt_pbl; }),
    entry("dqc_dt_pbl", [](auto& state) -> auto& { return state.dqc_dt_pbl; }),
    entry("dqi_dt_pbl", [](auto& state) -> auto& { return state.dqi_dt_pbl; }));

inline constexpr auto microphysics_state = std::make_tuple(
    entry("theta", [](auto& state) -> auto& { return state.theta; }),
    entry("qv", [](auto& state) -> auto& { return state.qv; }),
    entry("qc", [](auto& state) -> auto& { return state.qc; }),
    entry("qr", [](auto& state) -> auto& { return state.qr; }),
    entry("qi", [](auto& state) -> auto& { return state.qi; }),
    entry("qs", [](auto& state) -> auto& { return state.qs; }),
    entry("qg", [](auto& state) -> auto& { return state.qg; }),
    entry("qh", [](auto& state) -> auto& { return state.qh; }));

inline constexpr auto diagnostics = std::make_tuple(
    entry("radar_reflectivity", [](auto& state) -> auto& { return state.radar_reflectivity; }),
    entry("vorticity_r", [](auto& state) -> auto& { return state.vorticity_r; }),
    entry("vorticity_theta", [](auto& state) -> auto& { return state.vorticity_theta; }),
    entry("vorticity_z", [](auto& state) -> auto& { return state.vorticity_z; }),
    entry("stretching_term", [](auto& state) -> auto& { return state.stretching_term; }),
    entry("tilting_term", [](auto& state) -> auto& { return state.tilting_term; }),
    entry("baroclinic_term", [](auto& state) -> auto& { return state.baroclinic_term; }),
    entry("angular_momentum", [](auto& state) -> auto& { return state.angular_momentum; }),
    entry("angular_momentum_tendency", [](auto& state) -> auto& { return state.angular_momentum_tendency; }),
    entry("p_prime", [](auto& state) -> auto& { return state.p_prime; }),
    entry("dynamic_pressure", [](auto& state) -> auto& { return state.dynamic_pressure; }),
    entry("buoyancy_pressure", [](auto& state) -> auto& { return state.buoyancy_pressure; }));

inline constexpr auto nested = std::make_tuple(
    entry("rho", [](auto& state) -> auto& { return state.rho; }),
    entry("p", [](auto& state) -> auto& { return state.p; }),
    entry("u", [](auto& state) -> auto& { return state.u; }),
    entry("w", [](auto& state) -> auto& { return state.w; }),
    entry("v", [](auto& state) -> auto& { return state.v; }),
    entry("theta", [](auto& state) -> auto& { return state.theta; }),
    entry("qv", [](auto& state) -> auto& { return state.qv; }),
    entry("qc", [](auto& state) -> auto& { return state.qc; }),
    entry("qr", [](auto& state) -> auto& { return state.qr; }),
    entry("qi", [](auto& state) -> auto& { return state.qi; }),
    entry("qs", [](auto& state) -> auto& { return state.qs; }),
    entry("qh", [](auto& state) -> auto& { return state.qh; }),
    entry("qg", [](auto& state) -> auto& { return state.qg; }),
    entry("tracer", [](auto& state) -> auto& { return state.tracer; }));

template <class Value>
auto pointer(Value& value)
{
    if constexpr (std::is_pointer_v<Value>) return value;
    else return &value;
}

template <class State, class Inventory>
auto pointers(State& state, const Inventory& inventory)
{
    return std::apply([&](const auto&... members) {
        return std::array{pointer(members.get(state))...};
    }, inventory);
}

template <class State, class Inventory, class Function>
void visit(State& state, const Inventory& inventory, Function&& function)
{
    std::apply([&](const auto&... members) {
        (function(members.name, members.get(state)), ...);
    }, inventory);
}
}
