#include "util/parallel_copy.hpp"
#include "core/field/field3d.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
constexpr std::size_t fields = 5;
constexpr std::size_t guard = PageAlignedAllocator<unsigned char>::kAlignmentBytes;
using Clock = std::chrono::steady_clock;

void serial_copy(const std::array<tmv::ByteCopyRange, fields>& ranges, std::size_t bytes)
{
    for (const auto& range : ranges) std::memcpy(range.destination, range.source, bytes);
}

double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}
}

int main(int argc, char** argv)
{
    try
    {
        if (argc < 3) throw std::invalid_argument("usage: benchmark_acoustic_snapshot_copy WORKERS MIB_PER_FIELD [...]");
        std::size_t parsed = 0;
        const int requested_workers = std::stoi(argv[1], &parsed);
        if (argv[1][parsed] != '\0') throw std::invalid_argument("invalid worker budget");
        if (requested_workers < 1 || requested_workers > 256) throw std::invalid_argument("invalid worker budget");
        int workers = 1;
#ifdef _OPENMP
        omp_set_dynamic(0);
        omp_set_num_threads(requested_workers);
        workers = omp_get_max_threads();
#endif
        std::cout << std::setprecision(17);
        for (int argument = 2; argument < argc; ++argument)
        {
            const auto mib = std::stoull(argv[argument], &parsed);
            if (argv[argument][parsed] != '\0') throw std::invalid_argument("invalid field size");
            if (mib < 1 || mib > 256) throw std::invalid_argument("use 1–256 MiB per field");
            const std::size_t bytes = mib * 1024 * 1024;
            int copy_workers = 1;
#ifdef _OPENMP
            if (bytes >= 8 * 1024 * 1024)
            {
                const int planned = static_cast<int>(std::min({std::size_t(workers), std::size_t(32), bytes / (256 * 1024)}));
                #pragma omp parallel num_threads(planned)
                {
                    #pragma omp single
                    copy_workers = omp_get_num_threads();
                }
            }
#endif
            const std::size_t iterations = std::max<std::size_t>(4, (1024ULL * 1024 * 1024) / (fields * bytes));
            using AlignedBytes = std::vector<unsigned char, PageAlignedAllocator<unsigned char>>;
            std::array<AlignedBytes, fields> source, destination;
            std::array<tmv::ByteCopyRange, fields> ranges;
            for (std::size_t field = 0; field < fields; ++field)
            {
                source[field].resize(bytes + 2 * guard, 0xa5);
                destination[field].resize(bytes + 2 * guard, 0x6c);
                for (std::size_t n = 0; n < bytes; ++n)
                    source[field][n + guard] = static_cast<unsigned char>(n * 31 + n / 97 + field * 23);
                ranges[field] = {destination[field].data() + guard, source[field].data() + guard};
            }
            const auto verify = [&]
            {
                for (std::size_t field = 0; field < fields; ++field)
                    if (std::memcmp(destination[field].data() + guard, source[field].data() + guard, bytes) != 0 ||
                        !std::all_of(destination[field].begin(), destination[field].begin() + guard, [](auto v) { return v == 0x6c; }) ||
                        !std::all_of(destination[field].end() - guard, destination[field].end(), [](auto v) { return v == 0x6c; }))
                        throw std::runtime_error("copy payload or guard mismatch");
            };
            for (int warmup = 0; warmup < 4; ++warmup)
            {
                serial_copy(ranges, bytes);
                tmv::copy_byte_ranges(ranges, bytes);
            }
            verify();
            std::vector<double> serial, batch, deltas;
            for (int trial = 0; trial < 9; ++trial)
            {
                double elapsed[2]{};
                for (int order = 0; order < 2; ++order)
                {
                    const int variant = trial % 2 ? 1 - order : order;
                    const auto start = Clock::now();
                    for (std::size_t repeat = 0; repeat < iterations; ++repeat)
                    {
                        if (variant == 0) serial_copy(ranges, bytes);
                        else tmv::copy_byte_ranges(ranges, bytes);
                        std::atomic_signal_fence(std::memory_order_seq_cst);
                    }
                    elapsed[variant] = std::chrono::duration<double, std::milli>(Clock::now() - start).count() / iterations;
                    verify();
                }
                serial.push_back(elapsed[0]); batch.push_back(elapsed[1]);
                deltas.push_back(elapsed[0] - elapsed[1]);
            }
            const double delta = median(deltas);
            std::vector<double> deviations;
            for (double value : deltas) deviations.push_back(std::abs(value - delta));
            const double noise = 3 * 1.4826 * median(deviations);
            const double percent = 100 * delta / median(serial);
            std::cout << "{\"fields\":" << fields << ",\"bytes_per_field\":" << bytes
                      << ",\"alignment_bytes\":" << guard << ",\"worker_budget\":" << workers << ",\"copy_workers\":" << copy_workers << ",\"trials\":9,\"iterations_per_trial\":" << iterations
                      << ",\"serial_median_ms\":" << median(serial) << ",\"batch_median_ms\":" << median(batch)
                      << ",\"paired_delta_ms\":" << delta << ",\"paired_improvement_percent\":" << percent
                      << ",\"three_mad_ms\":" << noise << ",\"speed_noise_passed\":"
                      << (delta > noise && percent >= 5 ? "true" : "false") << ",\"byte_identity\":true,\"serial_trial_ms\":[";
            for (std::size_t i = 0; i < serial.size(); ++i) std::cout << (i ? "," : "") << serial[i];
            std::cout << "],\"batch_trial_ms\":[";
            for (std::size_t i = 0; i < batch.size(); ++i) std::cout << (i ? "," : "") << batch[i];
            std::cout << "]}\n";
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
