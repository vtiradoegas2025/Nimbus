#include "core/output/zfp_reader.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
constexpr std::array<const char*, 5> fields = {
    "reflectivity_dbz", "radial_velocity", "zdr", "kdp", "rhohv"};
constexpr std::array<const char*, 2> steps = {
    "step_000000", "step_000001"};
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "usage: native_decode_radar_zfp ZFP_RUN OUTPUT_DIRECTORY\n";
        return 2;
    }

    const std::filesystem::path run(argv[1]);
    const std::filesystem::path output(argv[2]);
    if (!std::filesystem::is_directory(run) || std::filesystem::exists(output))
    {
        std::cerr << "input run is missing or output directory already exists\n";
        return 2;
    }
    std::filesystem::create_directories(output);

    ZfpFrameReader reader;
    for (std::size_t step_index = 0; step_index < steps.size(); ++step_index)
    {
        const auto target = output / steps[step_index];
        std::filesystem::create_directories(target);
        for (const char* field : fields)
        {
            const auto path = run / steps[step_index] / (std::string(field) + ".zfp3d");
            Zfp3dHeader header;
            std::vector<float> values;
            std::string error;
            if (!reader.read_frame(field, path.string(), header, values, error))
            {
                std::cerr << field << ' ' << steps[step_index] << ": " << error << '\n';
                return 1;
            }
            if (header.version != 3 || header.mode != 0 ||
                header.dim0 <= 0 || header.dim1 <= 0 || header.dim2 <= 0 ||
                values.size() != header.total_elements() ||
                (step_index == 0 && !header.is_keyframe()) ||
                (step_index == 1 && !header.is_simple_delta()))
            {
                std::cerr << field << ' ' << steps[step_index]
                          << ": unexpected v3 accuracy lineage or shape\n";
                return 1;
            }
            const auto destination = target / (std::string(field) + ".f32");
            std::ofstream stream(destination, std::ios::binary);
            stream.write(reinterpret_cast<const char*>(values.data()),
                         static_cast<std::streamsize>(values.size() * sizeof(float)));
            if (!stream)
            {
                std::cerr << "failed to write decoded field " << destination << '\n';
                return 1;
            }
            std::cout << field << ' ' << steps[step_index] << ' '
                      << header.dim0 << 'x' << header.dim1 << 'x' << header.dim2
                      << " tolerance=" << header.mode_param << " flags="
                      << static_cast<int>(header.flags) << '\n';
        }
    }
    return 0;
}
