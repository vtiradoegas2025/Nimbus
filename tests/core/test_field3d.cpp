/**
 * @file test_field3d.cpp
 * @brief Unit tests for Field3D container.
 */
#include "catch2/catch.hpp"
#include "core/field/field3d.hpp"

#include <cmath>
#include <cstdint>
#include <limits>

TEST_CASE("Field storage allocator preserves page alignment across size boundaries", "[core][field3d][allocator]")
{
    PageAlignedAllocator<float> allocator;
    constexpr std::size_t       page_elements = PageAlignedAllocator<float>::kAlignmentBytes / sizeof(float);
    for (const std::size_t count :
         {std::size_t{0}, std::size_t{1}, page_elements - 1, page_elements, page_elements + 1})
    {
        float* values = allocator.allocate(count);
        REQUIRE(values != nullptr);
        REQUIRE(reinterpret_cast<std::uintptr_t>(values) % PageAlignedAllocator<float>::kAlignmentBytes == 0);
        if (count > 0)
        {
            std::fill_n(values, count, 7.0f);
            REQUIRE(values[0] == 7.0f);
            REQUIRE(values[count - 1] == 7.0f);
        }
        allocator.deallocate(values, count);
    }
    allocator.deallocate(nullptr, 0);
    REQUIRE_THROWS_AS(allocator.allocate(std::numeric_limits<std::size_t>::max()), std::bad_array_new_length);
    PageAlignedAllocator<unsigned char> bytes;
    REQUIRE_THROWS_AS(bytes.allocate(std::numeric_limits<std::size_t>::max()), std::bad_array_new_length);
}

TEST_CASE("Field3D default construction", "[core][field3d]")
{
    Field3D f;
    REQUIRE(f.size_r() == 0);
    REQUIRE(f.size_th() == 0);
    REQUIRE(f.size_z() == 0);
    REQUIRE(f.size() == 0);
    REQUIRE(f.empty());
}

TEST_CASE("Field3D sized construction zero-initializes", "[core][field3d]")
{
    Field3D f(4, 8, 16);
    REQUIRE(f.size_r() == 4);
    REQUIRE(f.size_th() == 8);
    REQUIRE(f.size_z() == 16);
    REQUIRE(f.size() == 4 * 8 * 16);
    REQUIRE_FALSE(f.empty());

    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 8; ++j)
            for (int k = 0; k < 16; ++k)
                REQUIRE(f(i, j, k) == 0.0f);
}

TEST_CASE("Field3D construction with init value", "[core][field3d]")
{
    Field3D f(2, 3, 4, 7.5f);
    REQUIRE(f.size() == 24);
    for (size_t n = 0; n < f.size(); ++n)
        REQUIRE(f.data()[n] == 7.5f);
}

TEST_CASE("Field3D bracket and paren indexing agree", "[core][field3d]")
{
    Field3D f(4, 6, 8);
    f(2, 3, 5) = 42.0f;
    float val = f[2][3][5];
    REQUIRE(val == 42.0f);

    f[1][0][7] = -1.0f;
    REQUIRE(f(1, 0, 7) == -1.0f);
}

TEST_CASE("Field3D const bracket indexing", "[core][field3d]")
{
    Field3D f(2, 2, 2, 3.14f);
    const Field3D& cf = f;
    float val = cf[1][1][1];
    REQUIRE(val == Approx(3.14f));
}

TEST_CASE("Field3D fill", "[core][field3d]")
{
    Field3D f(3, 3, 3);
    f.fill(99.0f);
    for (size_t n = 0; n < f.size(); ++n)
        REQUIRE(f.data()[n] == 99.0f);
}

TEST_CASE("Field3D resize", "[core][field3d]")
{
    Field3D f(2, 2, 2);
    f(0, 0, 0) = 1.0f;
    f.resize(4, 4, 4);
    REQUIRE(f.size_r() == 4);
    REQUIRE(f.size() == 64);
}

TEST_CASE("Field3D resize with init value", "[core][field3d]")
{
    Field3D f(2, 2, 2);
    f.resize(3, 3, 3, -1.0f);
    REQUIRE(f.size() == 27);
    for (size_t n = 0; n < f.size(); ++n)
        REQUIRE(f.data()[n] == -1.0f);
}

TEST_CASE("Field3D copy construction", "[core][field3d]")
{
    Field3D src(3, 4, 5, 2.0f);
    src(1, 2, 3) = 100.0f;

    Field3D dst(src);
    REQUIRE(dst.size_r() == 3);
    REQUIRE(dst(1, 2, 3) == 100.0f);

    // Verify deep copy
    dst(1, 2, 3) = 0.0f;
    REQUIRE(src(1, 2, 3) == 100.0f);
}

TEST_CASE("Field3D move construction", "[core][field3d]")
{
    Field3D src(3, 4, 5, 1.0f);
    size_t orig_size = src.size();

    Field3D dst(std::move(src));
    REQUIRE(dst.size() == orig_size);
    REQUIRE(src.empty());
}

TEST_CASE("Field3D copy assignment", "[core][field3d]")
{
    Field3D a(2, 2, 2, 5.0f);
    Field3D b;
    b = a;
    REQUIRE(b.size_r() == 2);
    REQUIRE(b(1, 1, 1) == 5.0f);
}

TEST_CASE("Field3D move assignment", "[core][field3d]")
{
    Field3D a(4, 4, 4, 3.0f);
    Field3D b;
    b = std::move(a);
    REQUIRE(b.size() == 64);
    REQUIRE(a.empty());
}

TEST_CASE("Field3D release_storage drops shape and supports lazy reuse",
          "[core][field3d][cpu016]")
{
    Field3D field(5, 4, 3, 7.0f);
    REQUIRE(field.size() == 60);
    REQUIRE_FALSE(field.empty());

    field.release_storage();

    REQUIRE(field.empty());
    REQUIRE(field.size() == 0);
    REQUIRE(field.size_r() == 0);
    REQUIRE(field.size_th() == 0);
    REQUIRE(field.size_z() == 0);

    field.resize(2, 3, 4, 1.5f);
    REQUIRE(field.size() == 24);
    REQUIRE(field[1][2][3] == Approx(1.5f));
}

TEST_CASE("Field3D assign from nested vector", "[core][field3d]")
{
    std::vector<std::vector<std::vector<float>>> nested(2,
        std::vector<std::vector<float>>(3,
            std::vector<float>(4, 1.0f)));
    nested[1][2][3] = 77.0f;

    Field3D f;
    f.assign(2, 3, 4, nested);
    REQUIRE(f.size_r() == 2);
    REQUIRE(f.size_th() == 3);
    REQUIRE(f.size_z() == 4);
    REQUIRE(f(1, 2, 3) == 77.0f);
    REQUIRE(f(0, 0, 0) == 1.0f);
}

TEST_CASE("Field3D negative dimensions throw", "[core][field3d]")
{
    REQUIRE_THROWS_AS(Field3D(-1, 2, 3), std::invalid_argument);
    REQUIRE_THROWS_AS(Field3D(2, -1, 3), std::invalid_argument);
    REQUIRE_THROWS_AS(Field3D(2, 3, -1), std::invalid_argument);
}

TEST_CASE("Field3D assign dimension mismatch throws", "[core][field3d]")
{
    std::vector<std::vector<std::vector<float>>> wrong_r(3,
        std::vector<std::vector<float>>(2, std::vector<float>(2, 0.0f)));

    Field3D f;
    REQUIRE_THROWS_AS(f.assign(2, 2, 2, wrong_r), std::invalid_argument);
}

TEST_CASE("Field3D data pointer is contiguous", "[core][field3d]")
{
    Field3D f(4, 8, 16);
    f(2, 3, 5) = 1.0f;

    // Verify row-major layout: index = i*NTH*NZ + j*NZ + k
    size_t expected = 2 * 8 * 16 + 3 * 16 + 5;
    REQUIRE(f.data()[expected] == 1.0f);
}

TEST_CASE("Field3D rejected storage growth preserves shape and values", "[core][field3d][transaction]")
{
    Field3D field(2, 3, 4, 7.0f);
    const int huge = std::numeric_limits<int>::max();
    SECTION("resize") { REQUIRE_THROWS(field.resize(huge, huge, 1)); }
    SECTION("filled resize") { REQUIRE_THROWS(field.resize(huge, huge, 1, 1.0f)); }
    REQUIRE(field.size_r() == 2);
    REQUIRE(field.size_th() == 3);
    REQUIRE(field.size_z() == 4);
    REQUIRE(field.size() == 24);
    REQUIRE(field[1][2][3] == 7.0f);
}
