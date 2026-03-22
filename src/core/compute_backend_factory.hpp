#pragma once

#include <memory>

class ComputeBackend;

// Internal backend-construction hooks used by runtime orchestration.
std::unique_ptr<ComputeBackend> create_cpu_compute_backend();
std::unique_ptr<ComputeBackend> create_vulkan_compute_backend();
