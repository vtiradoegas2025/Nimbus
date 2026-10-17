#include "dynamics/runtime.hpp"

#include "dynamics/dynamics_base.hpp"
#include "dynamics/factory.hpp"
#include "util/log.hpp"

#include <memory>
#include <stdexcept>

void tmv::initialize_dynamics_scheme(const std::string& scheme_name)
{
    std::unique_ptr<DynamicsScheme> selected = create_dynamics_scheme(scheme_name);
    if (!selected)
    {
        throw std::runtime_error(
            "dynamics.scheme '" + scheme_name + "' did not resolve to a "
            "dynamics kernel (unsupported coordinate, staggering, or "
            "axisymmetric combination)");
    }

    tmv::log_info("Initialized dynamics scheme: ", selected->get_scheme_name());
    dynamics_scheme = std::move(selected);
}
