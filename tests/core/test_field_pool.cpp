/**
 * @file test_field_pool.cpp
 * @brief Unit tests for FieldPool singleton and RAII guard.
 */
#include "catch2/catch.hpp"
#include "core/field_pool.hpp"

TEST_CASE("FieldPool acquire returns zero-filled buffer", "[core][field_pool]")
{
    auto& pool = FieldPool::instance();
    pool.clear();

    Field3D buf = pool.acquire(4, 4, 4);
    REQUIRE(buf.size_r() == 4);
    REQUIRE(buf.size_th() == 4);
    REQUIRE(buf.size_z() == 4);

    for (size_t n = 0; n < buf.size(); ++n)
        REQUIRE(buf.data()[n] == 0.0f);

    pool.release(std::move(buf));
}

TEST_CASE("FieldPool recycles released buffers", "[core][field_pool]")
{
    auto& pool = FieldPool::instance();
    pool.clear();

    Field3D buf1 = pool.acquire(8, 8, 8);
    float* ptr1 = buf1.data();
    pool.release(std::move(buf1));

    REQUIRE(pool.free_count() == 1);

    Field3D buf2 = pool.acquire(8, 8, 8);
    // Should recycle the same allocation
    REQUIRE(buf2.data() == ptr1);
    REQUIRE(pool.free_count() == 0);

    pool.release(std::move(buf2));
}

TEST_CASE("FieldPool recycle zero-fills the buffer", "[core][field_pool]")
{
    auto& pool = FieldPool::instance();
    pool.clear();

    Field3D buf = pool.acquire(4, 4, 4);
    buf.fill(999.0f);
    pool.release(std::move(buf));

    Field3D recycled = pool.acquire(4, 4, 4);
    for (size_t n = 0; n < recycled.size(); ++n)
        REQUIRE(recycled.data()[n] == 0.0f);

    pool.release(std::move(recycled));
}

TEST_CASE("FieldPool does not recycle mismatched dimensions", "[core][field_pool]")
{
    auto& pool = FieldPool::instance();
    pool.clear();

    Field3D buf = pool.acquire(4, 4, 4);
    float* ptr1 = buf.data();
    pool.release(std::move(buf));

    // Request different dimensions — should NOT recycle
    Field3D buf2 = pool.acquire(8, 8, 8);
    REQUIRE(buf2.data() != ptr1);

    // The 4x4x4 buffer should still be in the pool
    REQUIRE(pool.free_count() == 1);

    pool.release(std::move(buf2));
    pool.clear();
}

TEST_CASE("FieldPool acquire_copy duplicates data", "[core][field_pool]")
{
    auto& pool = FieldPool::instance();
    pool.clear();

    Field3D src(4, 4, 4, 7.0f);
    src(2, 2, 2) = 42.0f;

    Field3D copy = pool.acquire_copy(src);
    REQUIRE(copy.size_r() == 4);
    REQUIRE(copy(2, 2, 2) == 42.0f);
    REQUIRE(copy(0, 0, 0) == 7.0f);

    // Verify independent
    copy(2, 2, 2) = 0.0f;
    REQUIRE(src(2, 2, 2) == 42.0f);

    pool.release(std::move(copy));
}

TEST_CASE("FieldPool ScopedField auto-releases", "[core][field_pool]")
{
    auto& pool = FieldPool::instance();
    pool.clear();

    {
        auto guard = pool.scoped_acquire(4, 4, 4);
        guard.field.fill(1.0f);
        REQUIRE(pool.free_count() == 0);
    }
    // Guard destroyed — buffer should be back in pool
    REQUIRE(pool.free_count() == 1);

    pool.clear();
}

TEST_CASE("FieldPool release ignores empty fields", "[core][field_pool]")
{
    auto& pool = FieldPool::instance();
    pool.clear();

    Field3D empty;
    pool.release(std::move(empty));
    REQUIRE(pool.free_count() == 0);
}

TEST_CASE("FieldPool clear frees all pooled memory", "[core][field_pool]")
{
    auto& pool = FieldPool::instance();
    pool.clear();

    pool.release(pool.acquire(4, 4, 4));
    pool.release(pool.acquire(8, 8, 8));
    REQUIRE(pool.free_count() == 2);

    pool.clear();
    REQUIRE(pool.free_count() == 0);
}
