#include "core/runtime/simulation_checkpoint.hpp"
#include <chrono>
#include <iostream>
#include <stdexcept>
#if !defined(_WIN32)
#include <sys/resource.h>
#endif

int main(int argc, char** argv)
{
    try
    {
        if (argc < 2 || argc > 3)
            throw std::runtime_error("usage: benchmark_checkpoint_restore checkpoint [roundtrip-output]");
        const std::filesystem::path input(argv[1]);
        if (std::filesystem::file_size(input) > (4ULL << 30))
            throw std::runtime_error("bounded restore benchmark accepts at most 4 GiB");
        SimulationStorageState state;
        tmv::checkpoint::SimulationCheckpointAttachments attachments;
        tmv::checkpoint::SimulationCheckpointIntegrity integrity;
        std::string error;
        const auto start=std::chrono::steady_clock::now();
        if (!tmv::checkpoint::read_simulation_checkpoint(input,state,attachments,integrity,error))
            throw std::runtime_error(error);
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::uint64_t peak_rss=0;
#if !defined(_WIN32)
        struct rusage usage{};
        if (::getrusage(RUSAGE_SELF,&usage)!=0) throw std::runtime_error("cannot inspect RSS");
        peak_rss=static_cast<std::uint64_t>(usage.ru_maxrss);
#if !defined(__APPLE__)
        peak_rss*=1024;
#endif
#endif
        if (argc==3)
        {
            const std::filesystem::path output(argv[2]);
            if (attachments.trajectories.has_value() || std::filesystem::exists(output))
                throw std::runtime_error("roundtrip requires no trajectories and a fresh output path");
            tmv::checkpoint::SimulationCheckpointIntegrity actual;
            if (!tmv::checkpoint::write_simulation_checkpoint(output,state,nullptr,actual,error,1ULL<<30))
                throw std::runtime_error(error);
            if (actual.bytes!=integrity.bytes || actual.fnv1a64!=integrity.fnv1a64 || actual.sha256!=integrity.sha256)
                throw std::runtime_error("checkpoint roundtrip changed saved bytes");
        }
        std::cout.precision(17);
        std::cout << "[CHECKPOINT READ BENCHMARK] {\"bytes\":" << integrity.bytes << ",\"total_s\":" << seconds
                  << ",\"sha256\":\"" << integrity.sha256 << "\",\"fnv1a64\":\"" << integrity.fnv1a64
                  << "\",\"peak_rss_bytes\":" << peak_rss << "}\n";
        return 0;
    }
    catch (const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
