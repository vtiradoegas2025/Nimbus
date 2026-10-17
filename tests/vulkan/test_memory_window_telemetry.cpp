#include "catch2/catch.hpp"
#include "compute/vulkan_compute_telemetry.hpp"

#include <limits>
#include <sstream>
#include <string>

namespace
{
constexpr const char* token = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
using tmv::vulkan_compute::Telemetry;

void boundary(Telemetry& telemetry, std::uint64_t index, std::ostringstream& output)
{
    telemetry.memory_window.boundary(index, telemetry.memory_counters(),
                                     telemetry.memory_accounting_valid, output, 1234);
}
}

TEST_CASE("Host readbacks stay with their completed submission including the final operation",
          "[vulkan][telemetry][readback]")
{
    using namespace tmv::vulkan_compute;
    Telemetry total;
    ActivityCounters attributed;
    total.note_host_upload(100);
    total.note_device_download(80);
    total.note_submission(2);
    auto first = activity_delta(total.activity(), attributed);
    attributed = total.activity();
    total.note_host_download(80);
    attribute_host_readback(first, attributed, 80);
    total.note_host_upload(20);
    total.note_device_download(40);
    total.note_submission(1);
    auto second = activity_delta(total.activity(), attributed);
    attributed = total.activity();
    REQUIRE(first.host_download_bytes == 80);
    REQUIRE(second.host_download_bytes == 0);
    REQUIRE(second.host_upload_bytes == 20);
    REQUIRE(second.device_download_bytes == 40);
    total.note_host_download(40);
    attribute_host_readback(second, attributed, 40);
    REQUIRE(second.host_download_bytes == 40);
    REQUIRE(first.host_download_bytes + second.host_download_bytes == total.host_download_bytes);
    REQUIRE(activity_delta(total.activity(), attributed).host_download_bytes == 0);
}

TEST_CASE("Vulkan memory window is default-off and preserves legacy accounting",
          "[vulkan][memory-window]")
{
    Telemetry telemetry;
    std::ostringstream output;
    telemetry.note_allocation(80, false);
    telemetry.note_memory_allocation(128, false);
    boundary(telemetry, 1, output);
    telemetry.note_release(80, false);
    telemetry.note_memory_release(128, false);
    boundary(telemetry, 2, output);
    REQUIRE(output.str().empty());
    REQUIRE(telemetry.staging_allocated_bytes == 0);
    REQUIRE(telemetry.staging_high_water_bytes == 80);
    REQUIRE(telemetry.staging_memory_reserved_high_water_bytes == 128);
    REQUIRE(telemetry.memory_accounting_valid);
}

TEST_CASE("Vulkan memory window measures overlapping extents separately from reserved memory",
          "[vulkan][memory-window]")
{
    Telemetry telemetry;
    std::ostringstream output;
    telemetry.note_allocation(500, false);
    telemetry.note_release(450, false);
    telemetry.note_memory_allocation(512, false);
    REQUIRE(telemetry.memory_window.configure(token, 2, 4));
    boundary(telemetry, 1, output);
    REQUIRE(output.str().empty());
    boundary(telemetry, 2, output);
    telemetry.note_allocation(100, false);
    telemetry.note_allocation(30, true);
    telemetry.note_memory_allocation(64, true);
    telemetry.note_release(100, false);
    boundary(telemetry, 3, output);
    telemetry.note_release(30, true);
    telemetry.note_memory_release(64, true);
    boundary(telemetry, 4, output);
    boundary(telemetry, 5, output);
    telemetry.memory_window.abort(telemetry.memory_counters(), true, output, 1234);
    const auto text = output.str();
    REQUIRE(std::count(text.begin(), text.end(), '\n') == 2);
    REQUIRE(text.find("\"firstBoundary\":2,\"endBoundaryExclusive\":4") != std::string::npos);
    REQUIRE(text.find("\"staging_buffer_extent_bytes\":{\"unit\":\"bytes\",\"begin\":50,\"current\":50,\"highWater\":150}") != std::string::npos);
    REQUIRE(text.find("\"device_buffer_extent_bytes\":{\"unit\":\"bytes\",\"begin\":0,\"current\":0,\"highWater\":30}") != std::string::npos);
    REQUIRE(text.find("\"staging_memory_reserved_bytes\":{\"unit\":\"bytes\",\"begin\":512,\"current\":512,\"highWater\":512}") != std::string::npos);
    REQUIRE(text.find("\"device_memory_reserved_bytes\":{\"unit\":\"bytes\",\"begin\":0,\"current\":0,\"highWater\":64}") != std::string::npos);
    REQUIRE(text.find("\"device_memory_allocation_count\":{\"unit\":\"allocations\",\"begin\":0,\"current\":0,\"highWater\":1}") != std::string::npos);
    REQUIRE(text.find("\"allocationHookEvents\":6") != std::string::npos);
    REQUIRE(text.find("\"event\":\"end\",\"boundary\":4,\"completed\":true,\"valid\":true") != std::string::npos);
    REQUIRE(telemetry.staging_high_water_bytes == 500);
}

TEST_CASE("Vulkan memory window rejects malformed requests without partial configuration",
          "[vulkan][memory-window]")
{
    Telemetry telemetry;
    REQUIRE_FALSE(telemetry.memory_window.configure(nullptr, 1, 2));
    REQUIRE_FALSE(telemetry.memory_window.configure("a", 1, 2));
    REQUIRE_FALSE(telemetry.memory_window.configure(token, 0, 2));
    REQUIRE_FALSE(telemetry.memory_window.configure(token, 2, 2));
    REQUIRE_FALSE(telemetry.memory_window.configure(token, 3, 2));
    std::string uppercase(token);
    uppercase[30] = 'A';
    REQUIRE_FALSE(telemetry.memory_window.configure(uppercase.c_str(), 1, 2));
    REQUIRE(telemetry.memory_window.configure(token, 1, 2));
    REQUIRE_FALSE(telemetry.memory_window.configure(token, 2, 3));
}

TEST_CASE("Vulkan memory window emits decimal counters without altering shared stream formatting",
          "[vulkan][memory-window]")
{
    Telemetry telemetry;
    std::ostringstream output;
    output << std::hex << std::showbase << std::showpos;
    output.width(20);
    const auto flags = output.flags();
    REQUIRE(telemetry.memory_window.configure(token, 1, 2));
    boundary(telemetry, 1, output);
    telemetry.note_allocation(123, true);
    boundary(telemetry, 2, output);
    REQUIRE(output.str().find("\"pid\":1234") != std::string::npos);
    REQUIRE(output.str().find("\"highWater\":123") != std::string::npos);
    REQUIRE(output.str().find("0x") == std::string::npos);
    REQUIRE(output.flags() == flags);
    REQUIRE(output.width() == 20);
}

TEST_CASE("Vulkan memory window rejects extent and reserved counter arithmetic failures",
          "[vulkan][memory-window]")
{
    for (int defect = 0; defect < 6; ++defect)
    {
        Telemetry telemetry;
        std::ostringstream output;
        REQUIRE(telemetry.memory_window.configure(token, 1, 2));
        boundary(telemetry, 1, output);
        switch (defect)
        {
        case 0:
            telemetry.note_release(1, false);
            REQUIRE(telemetry.staging_allocated_bytes == 0);
            break;
        case 1:
            telemetry.note_allocation(std::numeric_limits<std::uint64_t>::max(), true);
            telemetry.note_allocation(1, true);
            REQUIRE(telemetry.device_allocated_bytes == 0);
            break;
        case 2:
            telemetry.note_memory_release(1, false);
            REQUIRE(telemetry.staging_memory_allocations == 0);
            break;
        case 3:
            telemetry.note_memory_allocation(std::numeric_limits<std::uint64_t>::max(), true);
            telemetry.note_memory_allocation(1, true);
            REQUIRE(telemetry.device_memory_reserved_bytes == 0);
            break;
        case 4:
            telemetry.device_memory_allocations = std::numeric_limits<std::uint64_t>::max();
            telemetry.note_memory_allocation(1, true);
            REQUIRE(telemetry.device_memory_allocations == 0);
            break;
        default:
            telemetry.note_memory_allocation(1, false);
            telemetry.note_memory_release(2, false);
            REQUIRE(telemetry.staging_memory_reserved_bytes == 0);
            break;
        }
        boundary(telemetry, 2, output);
        REQUIRE_FALSE(telemetry.memory_accounting_valid);
        REQUIRE(output.str().find("\"event\":\"end\",\"boundary\":2,\"completed\":true,\"valid\":false") != std::string::npos);
    }
}

TEST_CASE("Vulkan memory window does not hide invalid accounting before its first boundary",
          "[vulkan][memory-window]")
{
    Telemetry telemetry;
    std::ostringstream output;
    telemetry.note_release(1, true);
    REQUIRE(telemetry.memory_window.configure(token, 1, 2));
    boundary(telemetry, 1, output);
    boundary(telemetry, 2, output);
    REQUIRE(output.str().find("\"valid\":true") == std::string::npos);
}

TEST_CASE("Vulkan memory window rejects aborted and discontinuous boundary sequences",
          "[vulkan][memory-window]")
{
    for (int defect = 0; defect < 3; ++defect)
    {
        Telemetry telemetry;
        std::ostringstream output;
        REQUIRE(telemetry.memory_window.configure(token, 1, 3));
        boundary(telemetry, 1, output);
        telemetry.note_allocation(10, true);
        if (defect == 0)
        {
            telemetry.memory_window.abort(telemetry.memory_counters(), true, output, 1234);
            REQUIRE(output.str().find("\"completed\":false,\"valid\":false") != std::string::npos);
        }
        else
        {
            boundary(telemetry, defect == 1 ? 4 : 1, output);
            boundary(telemetry, 3, output);
            REQUIRE(output.str().find("\"completed\":true,\"valid\":false") != std::string::npos);
        }
        const auto text = output.str();
        REQUIRE(std::count(text.begin(), text.end(), '\n') == 2);
        REQUIRE(telemetry.device_allocated_bytes == 10);
    }
}
