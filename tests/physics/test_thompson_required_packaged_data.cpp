#include "catch2/catch.hpp"
#include "microphysics/schemes/thompson/required_packaged_data.hpp"

#include <filesystem>
#include <fstream>

namespace
{

std::filesystem::path temporary_path(const char* name)
{
    return std::filesystem::temp_directory_path() / name;
}

}

TEST_CASE("Thompson required data reports the exact offline preparation command",
          "[microphysics][thompson][data][missing][corrupt]")
{
    const auto missing = temporary_path("nimbus_thompson_required_missing.bin");
    const auto corrupt = temporary_path("nimbus_thompson_required_corrupt.bin");
    std::filesystem::remove(missing);
    std::filesystem::remove(corrupt);
    {
        std::ofstream file(corrupt, std::ios::binary | std::ios::trunc);
        REQUIRE(file);
        file.put('x');
    }

    REQUIRE_THROWS_WITH(
        tmv::thompson::require_packaged_activation_table(missing),
        "unable to open Thompson activation table; run `make thompson-data` to prepare the pinned tables");
    REQUIRE_THROWS_WITH(
        tmv::thompson::require_packaged_collection_tables(missing),
        "unable to open Thompson collection tables; run `make thompson-data` to prepare the pinned tables");
    REQUIRE_THROWS_WITH(
        tmv::thompson::require_packaged_cold_process_tables(missing),
        "unable to open Thompson cold tables; run `make thompson-data` to prepare the pinned tables");
    REQUIRE_THROWS_WITH(
        tmv::thompson::require_packaged_activation_table(corrupt),
        Catch::Matchers::Contains("run `make thompson-data` to prepare the pinned tables"));
    REQUIRE_THROWS_WITH(
        tmv::thompson::require_packaged_collection_tables(corrupt),
        Catch::Matchers::Contains("run `make thompson-data` to prepare the pinned tables"));
    REQUIRE_THROWS_WITH(
        tmv::thompson::require_packaged_cold_process_tables(corrupt),
        Catch::Matchers::Contains("run `make thompson-data` to prepare the pinned tables"));

    std::filesystem::remove(corrupt);
}
