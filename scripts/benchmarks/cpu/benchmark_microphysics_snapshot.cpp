#include "core/field/field3d.hpp"
#include "util/parallel_copy.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#if defined(__APPLE__) || defined(__linux__)
#include <sys/resource.h>
#endif

namespace
{
using Clock = std::chrono::steady_clock;
constexpr std::size_t field_count = 16;
using Fields = std::array<Field3D, field_count>;

template<class Operation>
double milliseconds(Operation operation)
{
    const auto start = Clock::now();
    operation();
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

void print_values(const char* name, const std::vector<double>& values)
{
    std::cout << ",\"" << name << "\":[";
    for (std::size_t index = 0; index < values.size(); ++index)
        std::cout << (index ? "," : "") << values[index];
    std::cout << ']';
}
}

int main(int argc, char** argv)
{
    try
    {
        if (argc != 3) throw std::invalid_argument("usage: benchmark_microphysics_snapshot WORKERS NX");
        const int workers = std::stoi(argv[1]);
        const int nx = std::stoi(argv[2]);
        if (workers < 1 || workers > 64 || nx < 32 || nx > 640)
            throw std::invalid_argument("use 1–64 workers and 32–640 horizontal cells");
#ifdef _OPENMP
        omp_set_dynamic(0);
        omp_set_num_threads(workers);
#else
        if (workers != 1) throw std::invalid_argument("serial build requires one worker");
#endif
        Fields source, snapshot;
        for (std::size_t field = 0; field < field_count; ++field)
        {
            source[field].resize(nx, nx, 180);
            for (std::size_t index = 0; index < source[field].size(); ++index)
                source[field].data()[index] = static_cast<float>(field + index % 8191) / 8192.0f;
        }
        const auto elements = source[0].size();
        const auto bytes = elements * sizeof(float);
        const double allocation_ms = milliseconds([&]
        {
            for (auto& field : snapshot) field.resize(nx, nx, 180);
        });
        for (auto& field : snapshot) field.release_storage();
        const double cold_ms = milliseconds([&]
        {
            for (std::size_t field = 0; field < field_count; ++field) snapshot[field] = source[field];
        });
        std::array<tmv::ByteCopyRange, field_count> ranges;
        std::array<const float*, field_count> owners;
        for (std::size_t field = 0; field < field_count; ++field)
        {
            ranges[field] = {snapshot[field].data(), source[field].data()};
            owners[field] = snapshot[field].data();
        }
        const auto copy = [&](int variant)
        {
            if (variant == 0)
                for (std::size_t field = 0; field < field_count; ++field) snapshot[field] = source[field];
            else tmv::copy_byte_ranges(ranges, bytes);
            std::atomic_signal_fence(std::memory_order_seq_cst);
        };
        const auto verify = [&]
        {
            for (std::size_t field = 0; field < field_count; ++field)
                if (snapshot[field].data() != owners[field] || snapshot[field].capacity() < elements ||
                    std::memcmp(snapshot[field].data(), source[field].data(), bytes) != 0)
                    throw std::runtime_error("snapshot ownership or bytes changed");
        };
        for (int warmup = 0; warmup < 4; ++warmup) { copy(0); copy(1); }
        std::vector<double> serial, batch, deltas;
        for (int trial = 0; trial < 9; ++trial)
        {
            double elapsed[2]{};
            for (int order = 0; order < 2; ++order)
            {
                const int variant = trial % 2 ? 1 - order : order;
                elapsed[variant] = milliseconds([&]
                {
                    for (int repeat = 0; repeat < 4; ++repeat) copy(variant);
                }) / 4;
                verify();
            }
            serial.push_back(elapsed[0]);
            batch.push_back(elapsed[1]);
            deltas.push_back(elapsed[0] - elapsed[1]);
        }
        const double restore_ms = milliseconds([&]
        {
            for (int repeat = 0; repeat < 10000; ++repeat)
            {
                for (std::size_t field = 0; field < field_count; ++field)
                    std::swap(source[field], snapshot[field]);
                std::atomic_signal_fence(std::memory_order_seq_cst);
            }
        }) / 10000;
        verify();
        const double delta = median(deltas);
        std::vector<double> deviations;
        for (double value : deltas) deviations.push_back(std::abs(value - delta));
        const double noise = 3 * 1.4826 * median(deviations);
        const double percent = 100 * delta / median(serial);
        const std::size_t alignment = PageAlignedAllocator<float>::kAlignmentBytes;
        std::cout << std::setprecision(17)
                  << "{\"shape\":[" << nx << ',' << nx << ",180],\"workers\":" << workers
                  << ",\"fields\":" << field_count << ",\"snapshot_bytes\":" << field_count * bytes
                  << ",\"retained_snapshot_allocation_bytes\":" << field_count * ((bytes + alignment - 1) / alignment) * alignment
                  << ",\"allocation_and_initialization_ms\":" << allocation_ms
                  << ",\"cold_capture_ms\":" << cold_ms << ",\"restore_swap_ms\":" << restore_ms
                  << ",\"serial_median_ms\":" << median(serial) << ",\"batch_median_ms\":" << median(batch)
                  << ",\"paired_delta_ms\":" << delta << ",\"paired_improvement_percent\":" << percent
                  << ",\"three_mad_ms\":" << noise << ",\"speed_noise_passed\":"
                  << (delta > noise && percent >= 5 ? "true" : "false")
                  << ",\"byte_identity\":true,\"warm_owner_identity\":true";
        print_values("serial_trial_ms", serial);
        print_values("batch_trial_ms", batch);
#if defined(__APPLE__) || defined(__linux__)
        rusage usage{};
        if (getrusage(RUSAGE_SELF, &usage) != 0) throw std::runtime_error("cannot read peak memory usage");
#ifdef __APPLE__
        std::cout << ",\"peak_rss_bytes\":" << usage.ru_maxrss;
#else
        std::cout << ",\"peak_rss_bytes\":" << usage.ru_maxrss * 1024;
#endif
#endif
        std::cout << "}\n";
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
