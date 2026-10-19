#include "catch2/catch.hpp"

#include "core/output/domain_recycling_swath.hpp"
#include "core/output/zfp_reader.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#ifdef HAVE_NETCDF
#include <netcdf.h>
#endif

namespace
{

std::filesystem::path temporary_swath(const std::string& label)
{
    const auto serial = std::chrono::high_resolution_clock::now()
                            .time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path()
        / ("nimbus_swath_" + label + "_" + std::to_string(serial));
    std::filesystem::remove_all(path);
    return path;
}

std::string read_text(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    REQUIRE(input.is_open());
    return {std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>()};
}

std::uint64_t fnv1a64(std::string_view text)
{
    std::uint64_t hash = UINT64_C(14695981039346656037);
    for (const unsigned char byte : text)
    {
        hash ^= byte;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

std::vector<float> read_npy_f32(const std::filesystem::path& path,
                                const std::string& shape,
                                std::size_t values)
{
    std::ifstream input(path, std::ios::binary);
    REQUIRE(input.is_open());
    char magic[6] = {};
    input.read(magic, sizeof(magic));
    REQUIRE(std::string(magic + 1, magic + 6) == "NUMPY");
    char version[2] = {};
    input.read(version, sizeof(version));
    std::uint16_t header_length = 0;
    input.read(reinterpret_cast<char*>(&header_length), sizeof(header_length));
    std::string header(header_length, '\0');
    input.read(header.data(), static_cast<std::streamsize>(header.size()));
    REQUIRE(header.find("'descr': '<f4'") != std::string::npos);
    REQUIRE(header.find(shape) != std::string::npos);
    std::vector<float> result(values);
    input.read(reinterpret_cast<char*>(result.data()),
               static_cast<std::streamsize>(values * sizeof(float)));
    REQUIRE(input.good());
    return result;
}

tmv::domain::WholeCellShiftPlan diagonal_plan()
{
    return tmv::domain::plan_whole_cell_shift_for_physical_domain(
        8, 9, 2, 2, -1, 1, 7, 1, 8);
}

std::vector<float> packed_volume(
    const tmv::domain::WholeCellShiftPlan& plan, float offset = 0.0f)
{
    std::vector<float> packed;
    packed.reserve(plan.trailing_cells);
    for (int i = plan.physical_i_begin; i < plan.physical_i_end; ++i)
        for (int j = plan.physical_j_begin; j < plan.physical_j_end; ++j)
        {
            if (i >= plan.source_i_begin && i < plan.source_i_end
                && j >= plan.source_j_begin && j < plan.source_j_end)
                continue;
            for (int k = 0; k < plan.nz; ++k)
                packed.push_back(offset
                    + static_cast<float>(i * 100 + j * 10 + k));
        }
    return packed;
}

std::vector<float> packed_surface(
    const tmv::domain::WholeCellShiftPlan& plan)
{
    std::vector<float> packed;
    for (int i = plan.physical_i_begin; i < plan.physical_i_end; ++i)
        for (int j = plan.physical_j_begin; j < plan.physical_j_end; ++j)
        {
            if (i >= plan.source_i_begin && i < plan.source_i_end
                && j >= plan.source_j_begin && j < plan.source_j_end)
                continue;
            packed.push_back(static_cast<float>(i * 100 + j * 10));
        }
    return packed;
}

nimbus::output::RetiringSwathPublication base_publication(
    const std::filesystem::path& root,
    const tmv::domain::WholeCellShiftPlan& plan)
{
    nimbus::output::RetiringSwathPublication publication;
    publication.swath_root = root;
    publication.transaction_id = "d5_test_0001";
    publication.publication_generation = "generation-test";
    publication.transaction_sequence = 1;
    publication.simulation_time_s = 20.0;
    publication.dx_m = 100.0;
    publication.dy_m = 200.0;
    publication.dz_m = 50.0;
    publication.domain_motion_mode = "constant_velocity";
    publication.native_staggering = "c_grid";
    publication.pre_shift_frame.earth_origin_at_t0_m = {1000.0, -500.0};
    publication.pre_shift_frame.velocity_ms = {10.0, -5.0};
    publication.plan = plan;
    publication.output.format = OutputFormat::npy_3d;
    publication.output.async_io = true;
    return publication;
}

} // namespace

TEST_CASE("D5 retirement decomposes into exact nonoverlapping rectangles",
          "[core][output][domain_recycling][swath][D5]")
{
    for (const auto shift : std::array<std::array<int, 2>, 8>{{
             {{1, 0}}, {{-1, 0}}, {{0, 1}}, {{0, -1}},
             {{1, 1}}, {{1, -1}}, {{-1, 1}}, {{-1, -1}},
         }})
    {
        const auto plan =
            tmv::domain::plan_whole_cell_shift_for_physical_domain(
                8, 9, 2, shift[0], shift[1], 1, 7, 1, 8);
        const auto tiles =
            nimbus::output::plan_retiring_scalar_swath_tiles(plan);
        REQUIRE(tiles.size() >= 1);
        REQUIRE(tiles.size() <= 4);
        std::set<std::pair<int, int>> represented;
        for (const auto& tile : tiles)
        {
            REQUIRE(tile.nx() > 0);
            REQUIRE(tile.ny() > 0);
            for (int i = tile.i_begin; i < tile.i_end; ++i)
                for (int j = tile.j_begin; j < tile.j_end; ++j)
                    REQUIRE(represented.insert({i, j}).second);
        }
        std::set<std::pair<int, int>> expected;
        for (int i = plan.physical_i_begin; i < plan.physical_i_end; ++i)
            for (int j = plan.physical_j_begin;
                 j < plan.physical_j_end; ++j)
                if (!(i >= plan.source_i_begin && i < plan.source_i_end
                      && j >= plan.source_j_begin
                      && j < plan.source_j_end))
                    expected.insert({i, j});
        REQUIRE(represented == expected);
        REQUIRE(represented.size() * static_cast<std::size_t>(plan.nz)
                == plan.trailing_cells);
    }
}

TEST_CASE("native D5 swath publication is exact durable and Earth located",
          "[core][output][domain_recycling][swath][native][D5]")
{
    const auto root = temporary_swath("native");
    const auto plan = diagonal_plan();
    auto theta = packed_volume(plan);
    auto rain = packed_surface(plan);
    auto publication = base_publication(root, plan);
    publication.output.resolved_fields = {"theta", "accumulated_rainfall"};
    publication.fields = {
        {"theta", &theta, true},
        {"accumulated_rainfall", &rain, false},
    };

    // A crash before the final directory rename can leave only this private
    // staging name. Retrying the same not-yet-published transaction replaces
    // it without exposing its contents.
    const auto stale_staging = root / "transaction_d5_test_0001.tmp";
    std::filesystem::create_directories(stale_staging);
    {
        std::ofstream stale(stale_staging / "partial.bin");
        stale << "incomplete";
    }

    nimbus::output::RetiringSwathPublicationResult result;
    std::string error;
    const bool published = nimbus::output::publish_retiring_swath_precommit(
        publication, result, error);
    INFO(error);
    REQUIRE(published);
    REQUIRE(error.empty());
    REQUIRE(result.tiles_published == 2);
    REQUIRE(result.fields_published == 4);
    REQUIRE(result.values_published ==
            plan.trailing_cells + plan.trailing_cells / plan.nz);
    REQUIRE(result.bytes_published > 0);
    REQUIRE(std::filesystem::is_directory(result.event_directory));
    REQUIRE_FALSE(std::filesystem::exists(
        result.event_directory.string() + ".tmp"));

    const auto west = result.event_directory / "tile_00";
    const auto north = result.event_directory / "tile_01";
    REQUIRE(read_npy_f32(west / "theta.npy", "(2, 7, 2)", 28)
            == std::vector<float>(theta.begin(), theta.begin() + 28));
    REQUIRE(read_npy_f32(north / "theta.npy", "(4, 1, 2)", 8)
            == std::vector<float>(theta.begin() + 28, theta.end()));
    REQUIRE(read_npy_f32(west / "accumulated_rainfall.npy", "(2, 7)", 14)
            == std::vector<float>(rain.begin(), rain.begin() + 14));

    const std::string event = read_text(result.event_directory / "event.json");
    REQUIRE(result.event_manifest_bytes == event.size());
    REQUIRE(result.event_manifest_fnv1a64 == fnv1a64(event));
    REQUIRE(event.find("\"publication_phase\": \"captured_precommit\"")
            != std::string::npos);
    REQUIRE(event.find("\"transaction_sequence\": 1")
            != std::string::npos);
    REQUIRE(event.find(
                "\"previous_committed_chain_fnv1a64\": \"0000000000000000\"")
            != std::string::npos);
    REQUIRE(event.find(
                "\"transaction_chain_algorithm\": \"checkpoint_bound_d5_transaction_provenance_v1\"")
            != std::string::npos);
    const std::string west_theta_bytes = read_text(west / "theta.npy");
    REQUIRE(event.find("\"path\": \"tile_00/theta.npy\"")
            != std::string::npos);
    std::ostringstream west_theta_integrity;
    west_theta_integrity << "\"bytes\": " << west_theta_bytes.size()
                         << ", \"fnv1a64\": \"" << std::hex
                         << std::setfill('0') << std::setw(16)
                         << fnv1a64(west_theta_bytes) << "\"";
    REQUIRE(event.find(west_theta_integrity.str()) != std::string::npos);
    REQUIRE(event.find("\"pre_shift_grid_origin_earth_m\": [1200, -600]")
            != std::string::npos);
    REQUIRE(event.find("\"lower\": [1300, -400]")
            != std::string::npos);
    const std::string tile_manifest = read_text(west / "manifest.json");
    REQUIRE(tile_manifest.find("\"kind\": \"retiring_swath_tile\"")
            != std::string::npos);
    REQUIRE(tile_manifest.find("\"valid_index_range\": {\"lower\": [0, 0, 0], \"upper\": [1, 6, 1]}")
            != std::string::npos);
    REQUIRE(tile_manifest.find("\"scalar_origin_m\": [1350, -300, 25]")
            != std::string::npos);

    // A completed transaction ID is immutable. Restart/retry cannot replace
    // its global history with different bytes.
    theta.front() += 1000.0f;
    nimbus::output::RetiringSwathPublicationResult duplicate_result;
    REQUIRE_FALSE(nimbus::output::publish_retiring_swath_precommit(
        publication, duplicate_result, error));
    REQUIRE(error.find("already exists") != std::string::npos);
    REQUIRE(read_text(result.event_directory / "event.json") == event);
    REQUIRE(read_npy_f32(west / "theta.npy", "(2, 7, 2)", 28).front()
            != theta.front());
    std::filesystem::remove_all(root);
}

TEST_CASE("D5 swath validation failure publishes no event",
          "[core][output][domain_recycling][swath][rollback][D5]")
{
    const auto root = temporary_swath("rollback");
    const auto plan = diagonal_plan();
    auto theta = packed_volume(plan);
    theta[3] = std::numeric_limits<float>::quiet_NaN();
    auto publication = base_publication(root, plan);
    publication.output.resolved_fields = {"theta"};
    publication.fields = {{"theta", &theta, true}};

    nimbus::output::RetiringSwathPublicationResult result;
    std::string error;
    REQUIRE_FALSE(nimbus::output::publish_retiring_swath_precommit(
        publication, result, error));
    REQUIRE(error.find("non-finite") != std::string::npos);
    REQUIRE_FALSE(std::filesystem::exists(
        root / "transaction_d5_test_0001"));
    REQUIRE_FALSE(std::filesystem::exists(
        root / "transaction_d5_test_0001.tmp"));
    std::filesystem::remove_all(root);
}

#ifdef HAVE_NETCDF
TEST_CASE("netCDF D5 swath embeds exact Earth and transaction coordinates",
          "[core][output][domain_recycling][swath][netcdf][D5]")
{
    const auto root = temporary_swath("netcdf");
    const auto plan = diagonal_plan();
    auto theta = packed_volume(plan);
    auto publication = base_publication(root, plan);
    publication.output.format = OutputFormat::netcdf;
    publication.output.resolved_fields = {"theta"};
    publication.fields = {{"theta", &theta, true}};
    nimbus::output::RetiringSwathPublicationResult result;
    std::string error;
    REQUIRE(nimbus::output::publish_retiring_swath_precommit(
        publication, result, error));

    int ncid = -1;
    REQUIRE(nc_open((result.event_directory / "tile_00" / "history.nc")
                        .string().c_str(), NC_NOWRITE, &ncid) == NC_NOERR);
    int x_var = -1;
    REQUIRE(nc_inq_varid(ncid, "x", &x_var) == NC_NOERR);
    std::array<double, 2> x{};
    REQUIRE(nc_get_var_double(ncid, x_var, x.data()) == NC_NOERR);
    REQUIRE(x == std::array<double, 2>{{1350.0, 1450.0}});
    std::size_t length = 0;
    REQUIRE(nc_inq_attlen(ncid, NC_GLOBAL, "nimbus_data_product", &length)
            == NC_NOERR);
    std::string product(length, '\0');
    REQUIRE(nc_get_att_text(ncid, NC_GLOBAL, "nimbus_data_product",
                            product.data()) == NC_NOERR);
    REQUIRE(product == "retiring_swath_tile");
    int ordinal = -1;
    REQUIRE(nc_get_att_int(ncid, NC_GLOBAL,
                           "domain_recycling_tile_ordinal", &ordinal)
            == NC_NOERR);
    REQUIRE(ordinal == 0);
    REQUIRE(nc_close(ncid) == NC_NOERR);
    std::filesystem::remove_all(root);
}
#endif

#ifdef HAVE_ZFP
TEST_CASE("qualified ZFP D5 swaths are independent bounded-error keyframes",
          "[core][output][domain_recycling][swath][zfp][D5]")
{
    static constexpr std::array<const char*, 8> names{{
        "u", "v", "w", "theta", "p", "rho", "qv", "qr"}};
    const auto root = temporary_swath("zfp");
    const auto plan = diagonal_plan();
    std::array<std::vector<float>, 8> values;
    auto publication = base_publication(root, plan);
    publication.output.format = OutputFormat::zfp;
    publication.output.preset = FieldPreset::minimal;
    publication.output.zfp_mode = "accuracy";
    publication.output.zfp_per_field_tolerances = true;
    publication.output.zfp_sparse_threshold = 0.0f;
    publication.output.zfp_float16_prefilter = false;
    publication.output.zfp_keyframe_interval = 7;
    publication.output.keep_last_frames = 3;
    publication.output.minimum_free_space_mib = 4096;
    publication.output.maximum_retained_history_files = 256;
    for (std::size_t index = 0; index < names.size(); ++index)
    {
        values[index] = packed_volume(plan, static_cast<float>(index));
        publication.output.resolved_fields.insert(names[index]);
        const auto budget = qualified_zfp_field_budgets().at(names[index]);
        publication.output.zfp_field_tolerances[names[index]] =
            budget.max_configured_absolute_error;
        publication.fields.push_back({names[index], &values[index], true});
    }

    nimbus::output::RetiringSwathPublicationResult result;
    std::string error;
    const bool published = nimbus::output::publish_retiring_swath_precommit(
        publication, result, error);
    INFO(error);
    REQUIRE(published);
    Zfp3dHeader header;
    std::vector<float> decoded;
    REQUIRE(read_zfp_3d(
        (result.event_directory / "tile_00" / "theta.zfp3d").string(),
        header, decoded, error));
    REQUIRE(header.is_keyframe());
    REQUIRE(header.dim0 == 2);
    REQUIRE(header.dim1 == 7);
    REQUIRE(header.dim2 == 2);
    REQUIRE(decoded.size() == 28);
    for (std::size_t index = 0; index < decoded.size(); ++index)
        REQUIRE(std::abs(decoded[index] - values[3][index]) <= 1.0e-3f);
    const std::string manifest =
        read_text(result.event_directory / "tile_00" / "manifest.json");
    REQUIRE(manifest.find("\"frame_role\": \"keyframe\"")
            != std::string::npos);
    REQUIRE(manifest.find("\"parent_step_indices\": []")
            != std::string::npos);
    std::filesystem::remove_all(root);
}
#endif
