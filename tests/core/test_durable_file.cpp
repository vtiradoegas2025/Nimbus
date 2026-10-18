/**
 * @file test_durable_file.cpp
 * @brief Unit tests for the output durability helpers.
 *
 * These guard the publication contract's crash semantics: a manifest is only
 * allowed to become visible once the payloads it advertises are on stable
 * storage. fsync success itself cannot be observed from userspace, so these
 * verify the contract the callers depend on -- that the helpers succeed on
 * every shape the writer produces, report failure rather than silently
 * passing on paths that do not exist, and leave file contents untouched.
 */
#include "catch2/catch.hpp"
#include "core/output/durable_file.hpp"
#include "core/infra/sha256.hpp"

#include <filesystem>
#include <fstream>
#include <string>

namespace
{

std::filesystem::path unique_dir(const std::string& label)
{
    static int serial = 0;
    const auto path = std::filesystem::temp_directory_path()
                      / ("nimbus_durable_" + label + "_" + std::to_string(++serial));
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
    return path;
}

void write_text(const std::filesystem::path& path, const std::string& body)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(body.data(), static_cast<std::streamsize>(body.size()));
}

} // namespace

TEST_CASE("portable SHA-256 matches standard vectors and streaming boundaries",
          "[core][integrity][sha256]")
{
    CHECK(nimbus::crypto::sha256_hex("") ==
          "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(nimbus::crypto::sha256_hex("abc") ==
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    const std::string message =
        "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    CHECK(nimbus::crypto::sha256_hex(message) ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");

    nimbus::crypto::Sha256 streamed;
    for (char value : message)
        streamed.update(&value, 1);
    CHECK(streamed.hex_digest() == nimbus::crypto::sha256_hex(message));
    // Finalization is observational: callers can inspect a digest and continue
    // feeding the same stream without resetting it.
    streamed.update("x", 1);
    CHECK(streamed.hex_digest() == nimbus::crypto::sha256_hex(message + "x"));
}

TEST_CASE("sync_file_to_disk flushes a written file and preserves its bytes",
          "[output][durability]")
{
    const auto dir  = unique_dir("file");
    const auto path = dir / "w.npy";
    write_text(path, "payload-bytes");

    std::string error;
    REQUIRE(nimbus::output::sync_file_to_disk(path, error));
    REQUIRE(error.empty());

    std::ifstream in(path, std::ios::binary);
    const std::string round_trip((std::istreambuf_iterator<char>(in)),
                                 std::istreambuf_iterator<char>());
    REQUIRE(round_trip == "payload-bytes");
    in.close();
    std::filesystem::remove_all(dir);
}

TEST_CASE("sync_file_to_disk handles an empty file", "[output][durability]")
{
    const auto  dir  = unique_dir("empty");
    const auto  path = dir / "empty.npy";
    write_text(path, "");
    std::string error;
    REQUIRE(nimbus::output::sync_file_to_disk(path, error));
    std::filesystem::remove_all(dir);
}

TEST_CASE("sync_file_to_disk reports a missing file instead of passing silently",
          "[output][durability]")
{
    const auto  dir = unique_dir("missing");
    std::string error;
    REQUIRE_FALSE(nimbus::output::sync_file_to_disk(dir / "absent.npy", error));
    REQUIRE_FALSE(error.empty());
    std::filesystem::remove_all(dir);
}

TEST_CASE("sync_directory_to_disk flushes a real directory", "[output][durability]")
{
    const auto  dir = unique_dir("dir");
    std::string error;
    REQUIRE(nimbus::output::sync_directory_to_disk(dir, error));
    REQUIRE(error.empty());
    std::filesystem::remove_all(dir);
}

TEST_CASE("sync_directory_to_disk rejects a regular file", "[output][durability]")
{
    const auto dir  = unique_dir("notdir");
    const auto path = dir / "w.npy";
    write_text(path, "x");
    std::string error;
    REQUIRE_FALSE(nimbus::output::sync_directory_to_disk(path, error));
    REQUIRE_FALSE(error.empty());
    std::filesystem::remove_all(dir);
}

TEST_CASE("sync_directory_contents_to_disk covers every published product",
          "[output][durability]")
{
    // The step directory shape the writer actually produces: field payloads
    // from several writers plus JSON sidecars.
    const auto dir = unique_dir("contents");
    for (const char* name : {"w.npy", "theta.npy", "qv.zfp3d", "validation_report.json"})
    {
        write_text(dir / name, "body");
    }
    // A subdirectory must not derail the sweep -- it is not a regular file.
    std::filesystem::create_directories(dir / "nested");

    std::string error;
    REQUIRE(nimbus::output::sync_directory_contents_to_disk(dir, error));
    REQUIRE(error.empty());
    std::filesystem::remove_all(dir);
}

TEST_CASE("sync_directory_contents_to_disk accepts an empty directory",
          "[output][durability]")
{
    const auto  dir = unique_dir("noproducts");
    std::string error;
    REQUIRE(nimbus::output::sync_directory_contents_to_disk(dir, error));
    std::filesystem::remove_all(dir);
}

TEST_CASE("sync_directory_contents_to_disk fails closed on a missing directory",
          "[output][durability]")
{
    const auto  dir = unique_dir("gone");
    const auto  absent = dir / "never_created";
    std::string error;
    REQUIRE_FALSE(nimbus::output::sync_directory_contents_to_disk(absent, error));
    REQUIRE_FALSE(error.empty());
    std::filesystem::remove_all(dir);
}

TEST_CASE("atomic replacement publishes a new file without a visibility gap",
          "[output][durability][atomic-replace][windows]")
{
    const auto dir = unique_dir("replace");
    const auto destination = dir / "manifest.json";
    const auto temporary = dir / "manifest.json.tmp";
    write_text(destination, "old-generation");
    write_text(temporary, "new-generation");

    std::string error;
    REQUIRE(nimbus::output::atomic_replace_file(temporary, destination, error));
    REQUIRE(error.empty());
    REQUIRE_FALSE(std::filesystem::exists(temporary));

    std::ifstream in(destination, std::ios::binary);
    const std::string round_trip((std::istreambuf_iterator<char>(in)),
                                 std::istreambuf_iterator<char>());
    REQUIRE(round_trip == "new-generation");
    in.close();
    std::filesystem::remove_all(dir);
}

TEST_CASE("completion publisher runs only after payload durability succeeds",
          "[output][durability][manifest][transaction]")
{
    const auto dir = unique_dir("manifest_order");
    write_text(dir / "cold_pool_boundary.npy", "complete-payload");
    bool publisher_called = false;
    std::string error;
    REQUIRE(nimbus::output::publish_manifest_after_durable_payloads(
        dir,
        [&]
        {
            publisher_called = true;
            return true;
        },
        error));
    REQUIRE(publisher_called);
    REQUIRE(error.empty());
    std::filesystem::remove_all(dir);
}

TEST_CASE("payload durability failure suppresses completion publication",
          "[output][durability][manifest][transaction][fail-closed]")
{
    const auto dir = unique_dir("manifest_failure");
    const auto missing = dir / "missing_step";
    bool publisher_called = false;
    std::string error;
    REQUIRE_FALSE(nimbus::output::publish_manifest_after_durable_payloads(
        missing,
        [&]
        {
            publisher_called = true;
            return true;
        },
        error));
    REQUIRE_FALSE(publisher_called);
    REQUIRE_FALSE(error.empty());
    std::filesystem::remove_all(dir);
}
