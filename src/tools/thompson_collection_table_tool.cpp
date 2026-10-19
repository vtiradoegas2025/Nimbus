#include "microphysics/schemes/thompson/collection_tables.hpp"
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
    if (argc == 2 && std::string(argv[1]) == "--samples")
    {
        const auto low = tmv::thompson::generate_rain_graupel_cell(
            {0, 0, 0, 0, 0});
        const auto middle = tmv::thompson::generate_rain_graupel_cell(
            {18, 18, 0, 18, 18});
        const auto snow = tmv::thompson::generate_rain_snow_cell(
            {18, 4, 18, 18});
        std::cout << std::setprecision(17);
        for (const auto& cell : {low, middle})
            std::cout << cell.graupel_mass_by_rain << ' '
                      << cell.rain_mass_by_rain << ' '
                      << cell.rain_mass_by_graupel << ' '
                      << cell.graupel_number_by_rain << ' '
                      << cell.rain_number_by_graupel << '\n';
        for (double value : snow.values)
            std::cout << value << ' ';
        std::cout << '\n';
        return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--measure")
    {
        const auto measurement = tmv::thompson::measure_generator_cost(100);
        std::cout << std::setprecision(17)
                  << "sampled_cells=" << measurement.sampled_cells << '\n'
                  << "sampled_pair_evaluations="
                  << measurement.sampled_pair_evaluations << '\n'
                  << "elapsed_seconds=" << measurement.elapsed_seconds << '\n'
                  << "projected_full_generation_seconds="
                  << measurement.projected_full_generation_seconds << '\n'
                  << "checksum=" << measurement.checksum << '\n';
        return 0;
    }
    if (argc != 2)
    {
        std::cerr << "usage: thompson_collection_table_tool OUTPUT|--samples|--measure\n";
        return 2;
    }
    const auto tables = tmv::thompson::generate_collection_tables();
    std::string error;
    if (!tmv::thompson::write_collection_tables(argv[1], tables, error))
    {
        std::cerr << error << '\n';
        return 1;
    }
    tmv::thompson::CollectionTables verified;
    if (!tmv::thompson::load_collection_tables(argv[1], verified, error)
        || verified.payload_sha256 != tables.payload_sha256
        || verified.payload_sha256
            != tmv::thompson::CollectionTableMetadata::packaged_payload_sha256)
    {
        std::cerr << (error.empty() ? "collection-table verification failed" : error)
                  << '\n';
        return 1;
    }
    const std::string container_sha256 = file_sha256(argv[1]);
    if (container_sha256
        != tmv::thompson::CollectionTableMetadata::packaged_sha256)
    {
        std::cerr << "collection-table container hash mismatch\n";
        return 1;
    }
    std::cout << "upstream_source_sha256="
              << tmv::thompson::CollectionTableMetadata::upstream_source_sha256
              << '\n'
              << "constants_sha256="
              << tmv::thompson::CollectionTableMetadata::constants_sha256
              << '\n'
              << "payload_sha256=" << verified.payload_sha256 << '\n'
              << "container_sha256=" << container_sha256 << '\n'
              << "resident_bytes="
              << tmv::thompson::resident_bytes(verified) << '\n';
    return 0;
}
