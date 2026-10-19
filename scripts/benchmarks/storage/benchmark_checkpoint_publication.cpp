#include "core/runtime/checkpoint_format.hpp"
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
        if (argc != 4) throw std::runtime_error("usage: benchmark_checkpoint_publication output elements fields");
        const std::filesystem::path path(argv[1]);
        const auto elements=std::stoull(argv[2]);
        const auto fields=std::stoull(argv[3]);
        if (elements==0 || fields==0 || fields>256 || elements>(1ULL<<29)/fields)
            throw std::runtime_error("bounded benchmark requires a nonempty payload of at most 2 GiB");
        std::vector<float> values(elements);
        std::uint32_t bits=0x73821ab4U;
        for (float& value : values)
        {
            bits=bits*1664525U+1013904223U;
            const std::uint32_t finite_bits=0x3e000000U | (bits&0x01ffffffU);
            std::memcpy(&value,&finite_bits,sizeof(value));
        }
        namespace wire=tmv::checkpoint::wire;
        const std::vector<wire::StreamingSection> sections{{1,1,wire::section_required,
            [&](wire::Encoder& out,std::string&) {
                out.unsigned_value<std::uint64_t>(fields);
                for (std::uint64_t field=0; field<fields; ++field)
                {
                    out.unsigned_value<std::uint64_t>(elements);
#ifdef NIMBUS_CHECKPOINT_SCALAR_BASELINE
                    for (float value : values) out.floating(value);
#else
                    out.floating_array(values.data(),values.size());
#endif
                }
                return out.good();
            }}};
        std::string error;
        wire::PublishedFileIntegrity integrity;
        const auto start=std::chrono::steady_clock::now();
        if (!wire::write_streaming_sections_atomic(path,std::array<std::uint8_t,8>{{'N','I','M','B','B','E','N','C'}},1,
                                                   sections,error,"benchmark",&integrity,1ULL<<30))
            throw std::runtime_error(error);
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::uint64_t peak_rss=0;
#if !defined(_WIN32)
        struct rusage usage{};
        if (::getrusage(RUSAGE_SELF,&usage)!=0) throw std::runtime_error("cannot inspect peak RSS");
        peak_rss=static_cast<std::uint64_t>(usage.ru_maxrss);
#if !defined(__APPLE__)
        peak_rss*=1024;
#endif
#endif
        std::cout.precision(17);
        std::cout << "{\"seconds\":" << seconds << ",\"bytes\":" << integrity.bytes
                  << ",\"fnv1a64\":" << integrity.fnv1a64
                  << ",\"sha256\":\"" << integrity.sha256 << "\",\"peak_rss_bytes\":" << peak_rss << "}\n";
        return 0;
    }
    catch (const std::exception& e) { std::cerr << e.what() << '\n';return 1; }
}
