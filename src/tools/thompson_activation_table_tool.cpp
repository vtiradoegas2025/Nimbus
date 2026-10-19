#include "microphysics/schemes/thompson/activation_table.hpp"
#include "core/infra/sha256.hpp"

#include <array>
#include <fstream>
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
    if (argc != 3)
    {
        std::cerr << "usage: thompson_activation_table_tool SOURCE OUTPUT\n";
        return 2;
    }
    tmv::thompson::ActivationTable table;
    std::string error;
    if (!tmv::thompson::import_wrf_activation_table(argv[1], table, error))
    {
        std::cerr << error << '\n';
        return 1;
    }
    if (!tmv::thompson::write_activation_table(argv[2], table, error))
    {
        std::cerr << error << '\n';
        return 1;
    }
    tmv::thompson::ActivationTable verified;
    if (!tmv::thompson::load_activation_table(argv[2], verified, error)
        || verified.payload_sha256 != table.payload_sha256)
    {
        std::cerr << (error.empty() ? "activation-table verification failed" : error)
                  << '\n';
        return 1;
    }
    const std::string container_sha256 = file_sha256(argv[2]);
    if (container_sha256
        != tmv::thompson::ActivationTableMetadata::packaged_sha256)
    {
        std::cerr << "activation-table container hash mismatch\n";
        return 1;
    }
    std::cout << "upstream_sha256="
              << tmv::thompson::ActivationTableMetadata::upstream_sha256
              << '\n'
              << "payload_sha256=" << verified.payload_sha256 << '\n'
              << "container_sha256=" << container_sha256 << '\n'
              << "elements=" << verified.fractions.size() << '\n';
    return 0;
}
