#include "catch2/catch.hpp"

#include "rendering/capture_format.hpp"
#include "rendering/frame_policy.hpp"
#include "rendering/swapchain_policy.hpp"
#include "rendering/volume_backend.hpp"
#include "core/context.hpp"

#include <array>

TEST_CASE("native volume defaults to unembellished scientific texture",
          "[vulkan][volume][quality]")
{
    const vkcpp::VolumeBackend::Config config;
    REQUIRE(config.texture_mode == "smooth");
}

TEST_CASE("capture format policy accepts only packed RGBA and BGRA", "[vulkan][capture]")
{
    using vkcpp::capture::PixelOrder;
    using vkcpp::capture::pixel_order;

    REQUIRE(pixel_order(VK_FORMAT_R8G8B8A8_UNORM) == PixelOrder::rgba);
    REQUIRE(pixel_order(VK_FORMAT_R8G8B8A8_SRGB) == PixelOrder::rgba);
    REQUIRE(pixel_order(VK_FORMAT_B8G8R8A8_UNORM) == PixelOrder::bgra);
    REQUIRE(pixel_order(VK_FORMAT_B8G8R8A8_SRGB) == PixelOrder::bgra);
    REQUIRE(pixel_order(VK_FORMAT_A2R10G10B10_UNORM_PACK32) == PixelOrder::unsupported);
}

TEST_CASE("capture direct readback requires both surface usage and a supported format",
          "[vulkan][capture]")
{
    using vkcpp::capture::direct_readback_supported;

    REQUIRE(direct_readback_supported(
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VK_FORMAT_B8G8R8A8_SRGB));
    REQUIRE_FALSE(direct_readback_supported(
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        VK_FORMAT_B8G8R8A8_SRGB));
    REQUIRE_FALSE(direct_readback_supported(
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VK_FORMAT_R16G16B16A16_SFLOAT));
}

TEST_CASE("capture pixel conversion honors the selected byte order", "[vulkan][capture]")
{
    std::array<std::uint8_t, 4> packed{11, 22, 33, 44};
    std::array<std::uint8_t, 3> rgb{};

    vkcpp::capture::to_rgb(vkcpp::capture::PixelOrder::rgba, packed.data(), rgb.data());
    REQUIRE(rgb == std::array<std::uint8_t, 3>{11, 22, 33});

    vkcpp::capture::to_rgb(vkcpp::capture::PixelOrder::bgra, packed.data(), rgb.data());
    REQUIRE(rgb == std::array<std::uint8_t, 3>{33, 22, 11});
}

TEST_CASE("swapchain composite alpha is selected only from the advertised mask",
          "[vulkan][swapchain]")
{
    using vkcpp::swapchain::choose_composite_alpha;

    REQUIRE(choose_composite_alpha(VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR |
                                   VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR) ==
            VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR);
    REQUIRE(choose_composite_alpha(VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR) ==
            VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR);
    REQUIRE(choose_composite_alpha(0) == 0);
}

TEST_CASE("queue policy prefers a shared family and supports split presentation",
          "[vulkan][swapchain]")
{
    std::vector<VkQueueFamilyProperties> families(3);
    families[0].queueFlags = VK_QUEUE_GRAPHICS_BIT;
    families[1].queueFlags = 0;
    families[2].queueFlags = VK_QUEUE_GRAPHICS_BIT;

    SECTION("shared graphics and present family wins")
    {
        const auto selected = vkcpp::choose_queue_families(
            families, {VK_FALSE, VK_TRUE, VK_TRUE});
        REQUIRE(selected.graphics_family == 2);
        REQUIRE(selected.present_family == 2);
        REQUIRE(selected.complete(true));
    }

    SECTION("separate families remain valid when no shared family exists")
    {
        const auto selected = vkcpp::choose_queue_families(
            families, {VK_FALSE, VK_TRUE, VK_FALSE});
        REQUIRE(selected.graphics_family == 0);
        REQUIRE(selected.present_family == 1);
        REQUIRE(selected.complete(true));
    }

    SECTION("missing presentation support fails the window requirement")
    {
        const auto selected = vkcpp::choose_queue_families(
            families, {VK_FALSE, VK_FALSE, VK_FALSE});
        REQUIRE(selected.has_graphics());
        REQUIRE_FALSE(selected.complete(true));
    }
}

TEST_CASE("native viewer redraw and wait policy idles unchanged scenes", "[vulkan][frame-policy]")
{
    using vkcpp::frame_policy::should_redraw;
    using vkcpp::frame_policy::wait_timeout_seconds;

    REQUIRE_FALSE(should_redraw(false, false, false, false, false));
    REQUIRE(should_redraw(true, false, false, false, false));
    REQUIRE(should_redraw(false, true, false, false, false));
    REQUIRE(should_redraw(false, false, true, false, false));
    REQUIRE(should_redraw(false, false, false, true, false));
    REQUIRE(should_redraw(false, false, false, false, true));

    REQUIRE(wait_timeout_seconds(true, false, 30.0f) == 0.0);
    REQUIRE(wait_timeout_seconds(false, false, 30.0f) == Approx(0.05));
    REQUIRE(wait_timeout_seconds(false, true, 20.0f) == Approx(0.05));
}

TEST_CASE("native viewer qualification resize fires exactly once at its frame",
          "[vulkan][frame-policy]")
{
    using vkcpp::frame_policy::qualification_resize_due;

    REQUIRE_FALSE(qualification_resize_due(-1, 0, false));
    REQUIRE_FALSE(qualification_resize_due(20, 19, false));
    REQUIRE(qualification_resize_due(20, 20, false));
    REQUIRE_FALSE(qualification_resize_due(20, 20, true));
    REQUIRE_FALSE(qualification_resize_due(20, 21, false));
}

TEST_CASE("native viewer wall duration is an explicit graceful deadline",
          "[vulkan][frame-policy]")
{
    using vkcpp::frame_policy::duration_complete;

    REQUIRE_FALSE(duration_complete(0.0, 5000.0));
    REQUIRE_FALSE(duration_complete(20.0, 19.999));
    REQUIRE(duration_complete(20.0, 20.0));
    REQUIRE(duration_complete(20.0, 20.001));
}

TEST_CASE("native viewer interval capture limit is deterministic",
          "[vulkan][frame-policy]")
{
    using vkcpp::frame_policy::interval_capture_allowed;

    REQUIRE(interval_capture_allowed(0, 500));
    REQUIRE(interval_capture_allowed(2, 0));
    REQUIRE(interval_capture_allowed(2, 1));
    REQUIRE_FALSE(interval_capture_allowed(2, 2));
}

TEST_CASE("validation error tally ignores warnings and counts errors fail-closed",
          "[vulkan][validation]")
{
    vkcpp::ValidationMessageTally tally;
    REQUIRE(tally.error_count() == 0);
    tally.record(VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT);
    REQUIRE(tally.error_count() == 0);
    tally.record(VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT);
    tally.record(VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT);
    REQUIRE(tally.error_count() == 2);
    tally.reset();
    REQUIRE(tally.error_count() == 0);
}

TEST_CASE("present-mode policy defaults or falls back to FIFO", "[vulkan][frame-policy]")
{
    const std::vector<VkPresentModeKHR> supported{
        VK_PRESENT_MODE_FIFO_KHR,
        VK_PRESENT_MODE_MAILBOX_KHR,
    };
    REQUIRE(vkcpp::swapchain::choose_present_mode(supported, VK_PRESENT_MODE_FIFO_KHR) ==
            VK_PRESENT_MODE_FIFO_KHR);
    REQUIRE(vkcpp::swapchain::choose_present_mode(supported, VK_PRESENT_MODE_MAILBOX_KHR) ==
            VK_PRESENT_MODE_MAILBOX_KHR);
    REQUIRE(vkcpp::swapchain::choose_present_mode(supported, VK_PRESENT_MODE_IMMEDIATE_KHR) ==
            VK_PRESENT_MODE_FIFO_KHR);
}
