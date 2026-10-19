#include "catch2/catch.hpp"
#include "compute/vulkan_loader_policy.hpp"

#include <array>
#include <cerrno>
#include <optional>
#include <string>
#include <thread>
#include <vector>

TEST_CASE("Vulkan loader lifetime default is installed before library construction", "[vulkan][loader-policy]")
{
    std::optional<std::string> value;
    std::vector<std::string>   operations;
    std::string                error;
    const auto                 read = [&](const char* name)
    {
        REQUIRE(std::string(name) == tmv_vulkan::loader_policy::environment_name);
        operations.emplace_back("read");
        return value ? value->c_str() : nullptr;
    };
    const auto write = [&](const char* name, const char* requested)
    {
        REQUIRE(std::string(name) == tmv_vulkan::loader_policy::environment_name);
        operations.emplace_back("write");
        value = requested;
        return 0;
    };
    const auto open = [&]
    {
        REQUIRE(value == "1");
        operations.emplace_back("open");
    };
    REQUIRE(tmv_vulkan::loader_policy::open_with_lifetime_policy(read, write, open, error));
    REQUIRE(tmv_vulkan::loader_policy::open_with_lifetime_policy(read, write, open, error));
    REQUIRE(operations == std::vector<std::string>{"read", "write", "open", "read", "open"});
    REQUIRE(error.empty());
}

TEST_CASE("Vulkan loader lifetime respects every explicit override", "[vulkan][loader-policy]")
{
    for (const std::string value : {"0", "1", "", "custom"})
    {
        INFO("explicit value='" << value << "'");
        bool        opened = false;
        unsigned    writes = 0;
        std::string error;
        REQUIRE(tmv_vulkan::loader_policy::open_with_lifetime_policy([&](const char*) { return value.c_str(); },
                                                                     [&](const char*, const char*)
                                                                     {
                                                                         ++writes;
                                                                         return 0;
                                                                     },
                                                                     [&] { opened = true; },
                                                                     error));
        REQUIRE(opened);
        REQUIRE(writes == 0);
        REQUIRE(error.empty());
    }
}

TEST_CASE("Vulkan loader setup failure prevents library construction and permits retry", "[vulkan][loader-policy]")
{
    std::optional<std::string> value;
    std::string                error;
    unsigned                   opens = 0;
    const auto                 read  = [&](const char*) { return value ? value->c_str() : nullptr; };
    const auto                 open  = [&] { ++opens; };
    REQUIRE_FALSE(tmv_vulkan::loader_policy::open_with_lifetime_policy(
        read, [](const char*, const char*) { return ENOMEM; }, open, error));
    REQUIRE(opens == 0);
    REQUIRE_FALSE(value);
    REQUIRE(error.find(tmv_vulkan::loader_policy::environment_name) != std::string::npos);
    error.clear();
    REQUIRE(tmv_vulkan::loader_policy::open_with_lifetime_policy(
        read,
        [&](const char*, const char* requested)
        {
            value = requested;
            return 0;
        },
        open,
        error));
    REQUIRE(opens == 1);
    REQUIRE(value == "1");
    REQUIRE(error.empty());
}

TEST_CASE("Concurrent Vulkan library opens share one lifetime policy setup", "[vulkan][loader-policy]")
{
    constexpr std::size_t          count = 16;
    std::optional<std::string>     value;
    unsigned                       writes = 0;
    unsigned                       opens  = 0;
    std::array<bool, count>        ok{};
    std::array<std::string, count> errors;
    std::vector<std::thread>       threads;
    for (std::size_t index = 0; index < count; ++index)
    {
        threads.emplace_back(
            [&, index]
            {
                ok[index] = tmv_vulkan::loader_policy::open_with_lifetime_policy(
                    [&](const char*) { return value ? value->c_str() : nullptr; },
                    [&](const char*, const char* requested)
                    {
                        ++writes;
                        value = requested;
                        return 0;
                    },
                    [&]
                    {
                        if (value == "1")
                            ++opens;
                    },
                    errors[index]);
            });
    }
    for (auto& thread : threads)
        thread.join();
    REQUIRE(writes == 1);
    REQUIRE(opens == count);
    for (std::size_t index = 0; index < count; ++index)
    {
        REQUIRE(ok[index]);
        REQUIRE(errors[index].empty());
    }
}
