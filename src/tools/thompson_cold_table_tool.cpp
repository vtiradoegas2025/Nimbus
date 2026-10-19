#include "microphysics/schemes/thompson/cold_process_tables.hpp"
#include "core/infra/sha256.hpp"

#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace
{

std::string file_sha256(const char* path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) return {};
    nimbus::crypto::Sha256 hash;
    std::array<char, 1 << 16> buffer{};
    while (input)
    {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        hash.update(buffer.data(), static_cast<std::size_t>(input.gcount()));
    }
    return hash.hex_digest();
}

}

int main(int argc, char** argv)
{
    using namespace tmv::thompson;
    if (argc == 2 && std::string(argv[1]) == "--samples")
    {
        const auto rain = generate_rain_freezing_cell({18, 18, 14, 27});
        const auto cloud = generate_cloud_freezing_cell({18, 49, 14, 27});
        const auto ice = generate_ice_snow_cell({36, 27});
        std::cout << std::setprecision(17)
                  << rain.ice_mass << ' ' << rain.graupel_mass << ' '
                  << rain.ice_number << ' ' << rain.rain_number << '\n'
                  << cloud.ice_mass << ' ' << cloud.ice_number << '\n'
                  << ice.snow_mass << ' ' << ice.ice_number << ' '
                  << ice.ice_deposition_fraction << '\n';
        return 0;
    }
    if (argc != 2)
    {
        std::cerr << "usage: thompson_cold_table_tool OUTPUT|--samples\n";
        return 2;
    }
    const auto tables = generate_cold_process_tables();
    std::string error;
    if (!write_cold_process_tables(argv[1], tables, error))
    {
        std::cerr << error << '\n';
        return 1;
    }
    ColdProcessTables verified;
    if (!load_cold_process_tables(argv[1], verified, error)
        || verified.payload_sha256 != tables.payload_sha256
        || verified.payload_sha256
            != ColdProcessTableMetadata::packaged_payload_sha256)
    {
        std::cerr << (error.empty() ? "cold-table verification failed" : error)
                  << '\n';
        return 1;
    }
    const std::string container_sha256 = file_sha256(argv[1]);
    if (container_sha256 != ColdProcessTableMetadata::packaged_sha256)
    {
        std::cerr << "cold-table container hash mismatch\n";
        return 1;
    }
    std::cout << "upstream_source_sha256="
              << ColdProcessTableMetadata::upstream_source_sha256 << '\n'
              << "constants_sha256="
              << ColdProcessTableMetadata::constants_sha256 << '\n'
              << "payload_sha256=" << verified.payload_sha256 << '\n'
              << "container_sha256=" << container_sha256 << '\n'
              << "resident_bytes=" << cold_resident_bytes(verified) << '\n';
    return 0;
}
