#pragma once

// Compatibility umbrella for existing backend consumers. New code should
// include backend_interface.hpp or backend_runtime.hpp according to whether it
// implements a backend or owns/selects one.
#include "compute/backend_interface.hpp"
#include "compute/backend_runtime.hpp"
#include "compute/dispatch_graph.hpp"

#include <utility>
