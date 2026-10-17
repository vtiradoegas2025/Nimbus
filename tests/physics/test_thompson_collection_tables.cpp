#include "catch2/catch.hpp"
#include "microphysics/schemes/thompson/collection_tables.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>

namespace
{

using tmv::thompson::CollectionTableMetadata;
using tmv::thompson::CollectionTables;
using tmv::thompson::RainGraupelIndex;
using tmv::thompson::RainSnowIndex;

CollectionTables empty_tables()
{
    CollectionTables tables;
    tables.rain_graupel.resize(tmv::thompson::checked_rain_graupel_size());
    tables.rain_snow.resize(tmv::thompson::checked_rain_snow_size());
    return tables;
}

std::filesystem::path temporary_path(const char* name)
{
    return std::filesystem::temp_directory_path() / name;
}

void require_close(double actual, double expected)
{
    REQUIRE(actual == Approx(expected).epsilon(2.0e-13));
}

} // namespace

TEST_CASE("Thompson collection metadata pins the official mp=28 source",
          "[microphysics][thompson][tables]")
{
    REQUIRE(std::string(CollectionTableMetadata::upstream_release) == "WRF v4.7.1");
    REQUIRE(std::string(CollectionTableMetadata::upstream_commit) == "f52c197");
    REQUIRE(std::string(CollectionTableMetadata::upstream_source_sha256)
            == "fabf19e2a9073cff886e882b187080bfdf089d3fd40c0fce1d19bc93b1e5e802");
    REQUIRE(std::string(CollectionTableMetadata::constants_sha256)
            == "d7accde31ef645d72673bb963339e6be62689ddf4bb1292c0d164a53449ae77c");
    REQUIRE(CollectionTableMetadata::bins == 100);
    REQUIRE(CollectionTableMetadata::graupel_densities == 1);
    REQUIRE(tmv::thompson::checked_rain_graupel_size() == 1'874'161);
    REQUIRE(tmv::thompson::checked_rain_snow_size() == 455'877);
    REQUIRE(tmv::thompson::checked_payload_bytes() == 118'730'632);
    REQUIRE(tmv::thompson::checked_full_pair_evaluations() == 23'300'380'000ULL);
}

TEST_CASE("Thompson collection generation cost is measurable without full regeneration",
          "[microphysics][thompson][tables][cost]")
{
    const auto measurement = tmv::thompson::measure_generator_cost(4);
    REQUIRE(measurement.sampled_cells == 4);
    REQUIRE(measurement.sampled_pair_evaluations == 40'000);
    REQUIRE(measurement.elapsed_seconds >= 0.0);
    REQUIRE(measurement.projected_full_generation_seconds >= 0.0);
    REQUIRE(std::isfinite(measurement.checksum));
}

TEST_CASE("Thompson rain-graupel cells match the official v4.7.1 integration",
          "[microphysics][thompson][tables][literal]")
{
    const auto low = tmv::thompson::generate_rain_graupel_cell({0, 0, 0, 0, 0});
    require_close(low.graupel_mass_by_rain, 1.9857834208898785e-12);
    require_close(low.rain_mass_by_rain, 3.0583290143955339e-13);
    require_close(low.rain_mass_by_graupel, 1.2258696979777394e-12);
    require_close(low.graupel_number_by_rain, 1.7668054935070198e-6);
    require_close(low.rain_number_by_graupel, 1.7300896761443716e-4);

    const auto middle = tmv::thompson::generate_rain_graupel_cell({18, 18, 0, 18, 18});
    require_close(middle.graupel_mass_by_rain, 1.9857834208898741e-8);
    require_close(middle.rain_mass_by_rain, 3.0583290143955204e-9);
    require_close(middle.rain_mass_by_graupel, 1.2258696979777433e-8);
    require_close(middle.graupel_number_by_rain, 1.7668054935070131e-2);
    require_close(middle.rain_number_by_graupel, 1.7300896761443603);
}

TEST_CASE("Thompson rain-snow cells preserve the official table ordering",
          "[microphysics][thompson][tables][literal]")
{
    const auto cell = tmv::thompson::generate_rain_snow_cell({18, 4, 18, 18});
    const std::array<double, 12> expected = {
        6.2455097549390434e-8, 5.5991049180776331e-7,
        4.3467861395007152e-8, 3.3354903620929700e-8,
        0.0, 0.0, 3.9635917329850351e-9, 2.1667681664856442e-7,
        3.1518912066977345, 1.1442737477658025, 0.0, 4.0462040026691541};
    for (std::size_t index = 0; index < expected.size(); ++index)
        require_close(cell.values[index], expected[index]);
}

TEST_CASE("Thompson collection indices fail closed",
          "[microphysics][thompson][tables][invalid]")
{
    REQUIRE_THROWS_AS(tmv::thompson::flatten(RainGraupelIndex{37, 0, 0, 0, 0}),
                      std::out_of_range);
    REQUIRE_THROWS_AS(tmv::thompson::flatten(RainGraupelIndex{0, 0, 1, 0, 0}),
                      std::out_of_range);
    REQUIRE_THROWS_AS(tmv::thompson::flatten(RainSnowIndex{0, 9, 0, 0}),
                      std::out_of_range);
}

TEST_CASE("Thompson collection loader rejects corruption without publication",
          "[microphysics][thompson][tables][integrity][atomic]")
{
    const auto missing_path = temporary_path("nimbus_thompson_collection_missing.bin");
    const auto valid_path = temporary_path("nimbus_thompson_collection_valid.bin");
    const auto corrupt_path = temporary_path("nimbus_thompson_collection_corrupt.bin");
    std::filesystem::remove(missing_path);
    std::filesystem::remove(valid_path);
    std::filesystem::remove(corrupt_path);

    CollectionTables tables = empty_tables();
    std::string error;
    REQUIRE(tmv::thompson::write_collection_tables(valid_path, tables, error));

    tmv::thompson::CollectionTableStore store;
    REQUIRE_FALSE(store.publish_packaged_file(missing_path, error));
    REQUIRE(error == "unable to open Thompson collection tables");
    REQUIRE_FALSE(store.snapshot());
    REQUIRE(store.publish_file(valid_path, error));
    const auto published = store.snapshot();
    REQUIRE(published);
    REQUIRE_FALSE(store.publish_packaged_file(valid_path, error));
    REQUIRE(error.find("packaged collection-table payload mismatch")
            != std::string::npos);
    REQUIRE(store.snapshot() == published);

    std::filesystem::copy_file(valid_path, corrupt_path);
    {
        std::fstream file(corrupt_path, std::ios::binary | std::ios::in | std::ios::out);
        REQUIRE(file);
        file.seekp(-1, std::ios::end);
        char byte = 1;
        file.write(&byte, 1);
    }
    REQUIRE_FALSE(store.publish_file(corrupt_path, error));
    REQUIRE(error.find("hash mismatch") != std::string::npos);
    REQUIRE(store.snapshot() == published);

    REQUIRE_FALSE(store.publish_file(
        valid_path, error, tmv::thompson::checked_payload_bytes() - 1));
    REQUIRE(error.find("resident-memory budget") != std::string::npos);
    REQUIRE(store.snapshot() == published);

    std::filesystem::remove(valid_path);
    std::filesystem::remove(corrupt_path);
}

TEST_CASE("Thompson packaged collection artifact has the sealed mp=28 identity",
          "[microphysics][thompson][tables][artifact]")
{
    tmv::thompson::CollectionTableStore store;
    std::string error;
    REQUIRE(store.publish_packaged_file(
        CollectionTableMetadata::packaged_path,
        error,
        tmv::thompson::checked_payload_bytes()));
    const auto tables = store.snapshot();
    REQUIRE(tables);
    REQUIRE(tables->payload_sha256
            == CollectionTableMetadata::packaged_payload_sha256);
    REQUIRE(tmv::thompson::resident_bytes(*tables)
            == tmv::thompson::checked_payload_bytes());
}

TEST_CASE("Thompson collection validation rejects nonfinite values",
          "[microphysics][thompson][tables][invalid]")
{
    CollectionTables tables = empty_tables();
    tables.rain_snow[0].values[0] = std::numeric_limits<double>::quiet_NaN();
    std::string error;
    REQUIRE_FALSE(tmv::thompson::validate_collection_tables(tables, error));
    REQUIRE(error.find("invalid values") != std::string::npos);
}
