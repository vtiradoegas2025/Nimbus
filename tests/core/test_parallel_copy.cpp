#include "catch2/catch.hpp"
#include "util/parallel_copy.hpp"
#include <vector>

TEST_CASE("Large host transfers preserve unaligned payloads and guard bytes",
          "[core][copy][thread]")
{
#ifdef _OPENMP
    struct RestoreWorkers
    {
        int count = omp_get_max_threads();
        int dynamic = omp_get_dynamic();
        ~RestoreWorkers() { omp_set_num_threads(count); omp_set_dynamic(dynamic); }
    } restore;
    omp_set_dynamic(0);
    omp_set_num_threads(GENERATE(1, 4));
#endif
    const auto bytes = GENERATE(std::size_t(0), std::size_t(511), std::size_t(8 * 1024 * 1024 + 37));
    std::vector<unsigned char> source(bytes + 32, 0xa5), destination(bytes + 32, 0x6c);
    for (std::size_t n = 0; n < bytes; ++n)
        source[n + 5] = static_cast<unsigned char>(n * 31 + n / 97);
    const auto original = source;
    tmv::copy_bytes(destination.data() + 9, source.data() + 5, bytes);
    CHECK(std::equal(source.begin(), source.end(), original.begin()));
    CHECK(std::equal(source.begin() + 5, source.begin() + 5 + bytes, destination.begin() + 9));
    CHECK(std::all_of(destination.begin(), destination.begin() + 9, [](auto v) { return v == 0x6c; }));
    CHECK(std::all_of(destination.begin() + 9 + bytes, destination.end(), [](auto v) { return v == 0x6c; }));
    const auto copied = destination;
    tmv::copy_bytes(destination.data(), destination.data(), destination.size());
    CHECK(std::equal(destination.begin(), destination.end(), copied.begin()));
    CHECK_NOTHROW(tmv::copy_bytes(nullptr, nullptr, 0));
}

TEST_CASE("Host transfer respects an existing worker team", "[core][copy][thread]")
{
    constexpr std::size_t bytes = 8 * 1024 * 1024 + 13;
    constexpr int fields = 4;
    std::vector<unsigned char> source(fields * bytes), destination(fields * bytes);
    for (std::size_t n = 0; n < source.size(); ++n)
        source[n] = static_cast<unsigned char>(n * 17 + n / 37);
    #pragma omp parallel for num_threads(fields)
    for (int field = 0; field < fields; ++field)
        tmv::copy_bytes(destination.data() + field * bytes, source.data() + field * bytes, bytes);
    CHECK(std::equal(destination.begin(), destination.end(), source.begin()));
}

TEST_CASE("Snapshot batches preserve bit patterns, uneven tails and independent guard bytes",
          "[core][copy][batch][thread]")
{
#ifdef _OPENMP
    struct RestoreWorkers
    {
        int count = omp_get_max_threads();
        int dynamic = omp_get_dynamic();
        ~RestoreWorkers() { omp_set_num_threads(count); omp_set_dynamic(dynamic); }
    } restore;
    omp_set_dynamic(0);
    omp_set_num_threads(GENERATE(1, 4));
#endif
    const auto bytes = GENERATE(std::size_t(0), std::size_t(513),
                                std::size_t(8 * 1024 * 1024 - 1),
                                std::size_t(8 * 1024 * 1024),
                                std::size_t(8 * 1024 * 1024 + 37));
    std::array<std::vector<unsigned char>, 5> source, destination, expected;
    std::array<tmv::ByteCopyRange, 5> ranges;
    for (std::size_t field = 0; field < ranges.size(); ++field)
    {
        source[field].resize(bytes + 32, 0xa5);
        destination[field].resize(bytes + 32, 0x6c);
        for (std::size_t n = 0; n < bytes; ++n)
            source[field][n + 5] = static_cast<unsigned char>(n * 31 + n / 97 + field * 23);
        expected[field] = destination[field];
        if (field == 2)
            ranges[field] = {destination[field].data() + 9, destination[field].data() + 9};
        else
        {
            ranges[field] = {destination[field].data() + 9, source[field].data() + 5};
            std::copy_n(source[field].begin() + 5, bytes, expected[field].begin() + 9);
        }
    }
    tmv::copy_byte_ranges(ranges, bytes);
    for (std::size_t field = 0; field < ranges.size(); ++field)
    {
        CAPTURE(field, bytes);
        CHECK(destination[field] == expected[field]);
        bool source_intact = true;
        for (std::size_t n = 0; n < source[field].size(); ++n)
            source_intact &= source[field][n] == (n < 5 || n >= bytes + 5
                ? 0xa5 : static_cast<unsigned char>((n - 5) * 31 + (n - 5) / 97 + field * 23));
        CHECK(source_intact);
    }
    CHECK_NOTHROW(tmv::copy_byte_ranges(std::array<tmv::ByteCopyRange, 0>{}, bytes));
    CHECK_NOTHROW(tmv::copy_byte_ranges(std::array<tmv::ByteCopyRange, 1>{{{nullptr, nullptr}}}, 0));
}

TEST_CASE("Snapshot batches can run inside an existing worker team", "[core][copy][batch][thread]")
{
    constexpr std::size_t bytes = 8 * 1024 * 1024 + 13;
    constexpr int fields = 4;
    std::vector<unsigned char> source(fields * bytes), destination(fields * bytes);
    for (std::size_t n = 0; n < source.size(); ++n)
        source[n] = static_cast<unsigned char>(n * 17 + n / 37);
    #pragma omp parallel for num_threads(2)
    for (int group = 0; group < 2; ++group)
    {
        const auto offset = group * 2 * bytes;
        const std::array<tmv::ByteCopyRange, 2> ranges{{
            {destination.data() + offset, source.data() + offset},
            {destination.data() + offset + bytes, source.data() + offset + bytes},
        }};
        tmv::copy_byte_ranges(ranges, bytes);
    }
    CHECK(destination == source);
}
