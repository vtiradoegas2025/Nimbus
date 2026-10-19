#include "core/runtime/checkpoint_format.hpp"
#include "core/runtime/simulation_checkpoint.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#if !defined(_WIN32)
#include <sys/resource.h>
#endif

int main(int argc, char** argv)
{
    try
    {
        if (argc < 2 || argc > 3)
            throw std::runtime_error("usage: benchmark_checkpoint_read checkpoint [roundtrip-output]");
        const std::filesystem::path input(argv[1]);
        if (std::filesystem::file_size(input) > (4ULL << 30))
            throw std::runtime_error("bounded reader benchmark accepts at most 4 GiB");
        namespace wire = tmv::checkpoint::wire;
        using Clock = std::chrono::steady_clock;
        const auto elapsed = [](auto start) { return std::chrono::duration<double>(Clock::now()-start).count(); };
        std::string error;
        std::vector<std::uint8_t> bytes;
        auto start = Clock::now();
        if (!wire::read_file(input,bytes,error,"benchmark")) throw std::runtime_error(error);
        const double io_s = elapsed(start);
        start = Clock::now();
        std::uint64_t fnv = 14695981039346656037ULL;
        for (const auto byte : bytes) {fnv ^= byte;fnv *= 1099511628211ULL;}
        const double fnv_s = elapsed(start);
        start = Clock::now();
        nimbus::crypto::Sha256 digest;
        digest.update(bytes.data(),bytes.size());
        const std::string hash = digest.hex_digest();
        const double sha_s = elapsed(start);
        start = Clock::now();
        SimulationStorageState state;
        tmv::checkpoint::SimulationCheckpointAttachments attachments;
        if (!tmv::checkpoint::deserialize_simulation_storage(bytes,state,attachments,error))
            throw std::runtime_error(error);
        const double decode_s = elapsed(start);
        std::uint64_t peak_rss = 0;
#if !defined(_WIN32)
        struct rusage usage{};
        if (::getrusage(RUSAGE_SELF,&usage)!=0) throw std::runtime_error("cannot inspect RSS");
        peak_rss = static_cast<std::uint64_t>(usage.ru_maxrss);
#if !defined(__APPLE__)
        peak_rss *= 1024;
#endif
#endif
        if (argc == 3)
        {
            if (attachments.trajectories.has_value())
                throw std::runtime_error("roundtrip benchmark requires a checkpoint without trajectories");
            const std::filesystem::path output(argv[2]);
            if (std::filesystem::exists(output)) throw std::runtime_error("roundtrip output must not exist");
            tmv::checkpoint::SimulationCheckpointIntegrity result;
            if (!tmv::checkpoint::write_simulation_checkpoint(output,state,nullptr,result,error,1ULL<<30))
                throw std::runtime_error(error);
            if (result.sha256!=hash || result.bytes!=bytes.size())
                throw std::runtime_error("restored state does not re-encode exactly");
        }
        std::ostringstream fnv_text;fnv_text<<std::hex<<std::setfill('0')<<std::setw(16)<<fnv;
        std::cout << std::setprecision(17) << "[CHECKPOINT READ BENCHMARK] {\"bytes\":" << bytes.size()
                  << ",\"io_s\":" << io_s << ",\"fnv_s\":" << fnv_s << ",\"sha_s\":" << sha_s
                  << ",\"validate_decode_s\":" << decode_s << ",\"total_s\":" << io_s+fnv_s+sha_s+decode_s
                  << ",\"sha256\":\"" << hash << "\",\"fnv1a64\":\"" << fnv_text.str()
                  << "\",\"peak_rss_bytes\":" << peak_rss << "}\n";
        return 0;
    }
    catch (const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
