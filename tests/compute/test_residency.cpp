/**
 * @file test_residency.cpp
 * @brief Backend-neutral resident allocation and authority-state tests.
 */

#include "catch2/catch.hpp"
#include "compute/compute_residency.hpp"
#include "compute/resident_field_store.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{

using nimbus::compute::FieldExtent;
using nimbus::compute::FieldResidencyGraph;
using nimbus::compute::ResidentFieldStore;

struct FakeAllocator
{
    ResidentFieldStore::Allocation next = 1;
    int acquired = 0;
    int batch_calls = 0;
    bool fail_next = false;
    bool fail_batch = false;
    bool throw_batch = false;
    std::size_t partial_batch_count = 0;
    std::vector<ResidentFieldStore::Allocation> released;
};

ResidentFieldStore::Allocation acquire(void* user_data, std::size_t)
{
    auto& allocator = *static_cast<FakeAllocator*>(user_data);
    if (allocator.fail_next)
    {
        allocator.fail_next = false;
        return ResidentFieldStore::invalid_allocation;
    }
    ++allocator.acquired;
    return allocator.next++;
}

void release(void* user_data, ResidentFieldStore::Allocation allocation)
{
    static_cast<FakeAllocator*>(user_data)->released.push_back(allocation);
}

bool acquire_batch(
    void* user_data, const std::size_t*, std::size_t count,
    ResidentFieldStore::Allocation* allocations)
{
    auto& allocator = *static_cast<FakeAllocator*>(user_data);
    ++allocator.batch_calls;
    if (allocator.fail_batch || allocator.throw_batch)
    {
        const std::size_t partial =
            allocator.partial_batch_count < count
                ? allocator.partial_batch_count
                : count;
        for (std::size_t i = 0; i < partial; ++i)
        {
            ++allocator.acquired;
            allocations[i] = allocator.next++;
        }
        if (allocator.throw_batch)
            throw std::runtime_error("injected batch failure");
        return false;
    }
    for (std::size_t i = 0; i < count; ++i)
    {
        ++allocator.acquired;
        allocations[i] = allocator.next++;
    }
    return true;
}

} // namespace

static_assert(!std::is_copy_constructible_v<ResidentFieldStore>);
static_assert(!std::is_copy_assignable_v<ResidentFieldStore>);
static_assert(std::is_nothrow_move_constructible_v<ResidentFieldStore>);
static_assert(!std::is_move_assignable_v<ResidentFieldStore>);

TEST_CASE("Single resident allocation growth is failure-atomic",
          "[compute][residency][allocation]")
{
    FakeAllocator allocator;
    ResidentFieldStore store;
    store.set_allocator(&acquire, &release, &allocator);

    float host[16]{};
    const FieldExtent extent{16, sizeof(float)};
    store.begin_step();
    REQUIRE(store.acquire("theta", sizeof(host)) == 1);
    store.commit_host_mirror("theta", host, extent);
    REQUIRE(store.reusable("theta", host, extent, true));

    allocator.fail_next = true;
    REQUIRE(store.acquire("theta", sizeof(host) * 2) ==
            ResidentFieldStore::invalid_allocation);
    REQUIRE(store.find("theta") != nullptr);
    CHECK(store.find("theta")->allocation == 1);
    CHECK(store.find("theta")->capacity_bytes == sizeof(host));
    CHECK(allocator.released.empty());
    CHECK(store.reusable("theta", host, extent, true));

    REQUIRE(store.acquire("theta", sizeof(host) * 2) == 2);
    REQUIRE(allocator.released.size() == 1);
    CHECK(allocator.released.front() == 1);
    CHECK_FALSE(store.reusable("theta", host, extent, true));

    allocator.fail_next = true;
    CHECK(store.acquire("new_field", sizeof(host)) ==
          ResidentFieldStore::invalid_allocation);
    CHECK(store.find("new_field") == nullptr);
}

TEST_CASE("Resident allocation rejects empty names without allocator side effects",
          "[compute][residency][allocation][contract]")
{
    FakeAllocator allocator;
    ResidentFieldStore store;
    REQUIRE(store.set_allocator(&acquire, &release, &allocator));

    CHECK(store.acquire("", 64) == ResidentFieldStore::invalid_allocation);
    CHECK(allocator.acquired == 0);
    CHECK(store.find("") == nullptr);
}

TEST_CASE("Resident allocator bindings cannot change while claims are live",
          "[compute][residency][allocation][ownership]")
{
    FakeAllocator original;
    FakeAllocator replacement;
    ResidentFieldStore store;
    REQUIRE(store.set_allocator(&acquire, &release, &original));
    REQUIRE(store.set_batch_allocator(&acquire_batch));
    REQUIRE(store.acquire("theta", 64) == 1);

    // Repeating an identical binding is not a rebind.
    CHECK(store.set_allocator(&acquire, &release, &original));
    CHECK(store.set_batch_allocator(&acquire_batch));

    CHECK_FALSE(store.set_allocator(&acquire, &release, &replacement));
    CHECK_FALSE(store.set_batch_allocator(nullptr));
    REQUIRE(store.acquire("rho", 64) == 2);
    CHECK(original.acquired == 2);
    CHECK(replacement.acquired == 0);

    store.clear(true);
    REQUIRE(original.released.size() == 2);
    CHECK(store.set_allocator(&acquire, &release, &replacement));
    CHECK(store.set_batch_allocator(nullptr));
    REQUIRE(store.acquire("qv", 64) == 1);
    CHECK(replacement.acquired == 1);
}

TEST_CASE("Invalid resident lineages fail closed and revoke device authority",
          "[compute][residency][authority][contract]")
{
    FakeAllocator allocator;
    ResidentFieldStore store;
    REQUIRE(store.set_allocator(&acquire, &release, &allocator));
    float host[8]{};
    const FieldExtent valid{8, sizeof(float)};

    store.begin_step();
    REQUIRE(store.acquire("theta", sizeof(host)) == 1);

    const auto restore_current = [&] {
        store.commit_host_mirror("theta", host, valid);
        REQUIRE(store.reusable("theta", host, valid, true));
    };

    const std::array<std::pair<const void*, FieldExtent>, 4>
        invalid_lineages{{
            {nullptr, valid},
            {host, FieldExtent{0, sizeof(float)}},
            {host, FieldExtent{8, 0}},
            {host, FieldExtent{
                       std::numeric_limits<std::size_t>::max(), 2}},
        }};
    for (const auto& invalid : invalid_lineages)
    {
        restore_current();
        CHECK_FALSE(store.reusable(
            "theta", invalid.first, invalid.second, true));
        CHECK_FALSE(store.reusable("theta", host, valid, true));

        restore_current();
        store.commit_host_mirror("theta", invalid.first, invalid.second);
        CHECK_FALSE(store.reusable("theta", host, valid, true));

        restore_current();
        store.note_device_write("theta", invalid.first, invalid.second);
        CHECK_FALSE(store.reusable("theta", host, valid, true));
    }

    const auto* state = store.version_state("theta");
    REQUIRE(state != nullptr);
    CHECK(state->host_identity == host);
    CHECK(state->extent == valid);
}

TEST_CASE("Resident store move construction transfers each allocation once",
          "[compute][residency][ownership]")
{
    FakeAllocator allocator;
    ResidentFieldStore source;
    source.set_allocator(&acquire, &release, &allocator);
    REQUIRE(source.acquire("qv", 64) == 1);

    ResidentFieldStore destination(std::move(source));
    CHECK(source.find("qv") == nullptr);
    CHECK(source.acquire("unused", 64) ==
          ResidentFieldStore::invalid_allocation);
    REQUIRE(destination.find("qv") != nullptr);
    CHECK(destination.find("qv")->allocation == 1);

    destination.release("qv");
    REQUIRE(allocator.released.size() == 1);
    CHECK(allocator.released.front() == 1);
}

TEST_CASE("Residency graph keeps a single authoritative field lineage",
          "[compute][residency][authority]")
{
    FieldResidencyGraph graph;
    float host[8]{};
    const FieldExtent extent{8, sizeof(float)};

    graph.note_host_write("rho", host, extent);
    REQUIRE(graph.host_current("rho", host, extent));
    CHECK_FALSE(graph.device_current("rho"));

    REQUIRE(graph.note_device_synced_from_host("rho", host, extent));
    CHECK(graph.device_current("rho"));
    CHECK(graph.device_current("rho", host, extent));

    graph.note_device_write("rho", host, extent);
    CHECK(graph.device_current("rho"));
    CHECK_FALSE(graph.host_current("rho", host, extent));
    REQUIRE(graph.note_host_synced_from_device("rho", host, extent));
    CHECK(graph.host_current("rho", host, extent));

    CHECK_FALSE(graph.device_current("rho", host + 1, extent));
    graph.invalidate_device("rho");
    CHECK_FALSE(graph.device_current("rho"));
}

TEST_CASE("Explicit host writes are observed only once per generation",
          "[compute][residency][generation]")
{
    FakeAllocator allocator;
    ResidentFieldStore store;
    store.set_allocator(&acquire, &release, &allocator);
    float host[8]{};
    const FieldExtent extent{8, sizeof(float)};

    store.begin_step();
    REQUIRE(store.acquire("u", sizeof(host)) == 1);
    store.commit_host_mirror("u", host, extent);
    const auto initial_version = store.version_state("u")->version;

    store.begin_step();
    store.note_host_write("u");
    REQUIRE(store.version_state("u")->version == initial_version + 1);
    CHECK_FALSE(store.reusable("u", host, extent, true));
    CHECK(store.version_state("u")->version == initial_version + 1);
}

TEST_CASE("Compute field residency tracks one authoritative version",
          "[compute][residency][authority]")
{
    FieldResidencyGraph graph;
    float host_a[8]{};
    float host_b[8]{};
    const FieldExtent extent{8, sizeof(float)};

    graph.begin_step();
    graph.note_host_write("theta", host_a, extent);
    REQUIRE(graph.host_current("theta", host_a, extent));
    REQUIRE_FALSE(graph.device_current("theta", host_a, extent));

    REQUIRE(graph.note_device_synced_from_host("theta", host_a, extent));
    REQUIRE(graph.device_current("theta", host_a, extent));

    graph.note_device_write("theta", host_a, extent);
    REQUIRE(graph.device_current("theta", host_a, extent));
    REQUIRE_FALSE(graph.host_current("theta", host_a, extent));
    REQUIRE(graph.note_host_synced_from_device("theta", host_a, extent));
    REQUIRE(graph.host_current("theta", host_a, extent));

    graph.note_host_write("theta", host_a, extent);
    REQUIRE_FALSE(graph.device_current("theta", host_a, extent));
    REQUIRE_FALSE(graph.device_current("theta", host_b, extent));

    const FieldExtent resized{4, sizeof(float)};
    graph.note_host_write("theta", host_a, resized);
    REQUIRE(graph.host_current("theta", host_a, resized));
    REQUIRE_FALSE(graph.device_current("theta", host_a, extent));
}

TEST_CASE("Resident field store owns allocation growth and copy versions",
          "[compute][residency][allocation]")
{
    FakeAllocator allocator;
    allocator.next = 0;
    ResidentFieldStore store;
    store.set_allocator(&acquire, &release, &allocator);
    float host[16]{};
    const FieldExtent extent{16, sizeof(float)};

    store.begin_step();
    REQUIRE(store.acquire("theta", sizeof(host)) == 0);
    store.commit_host_mirror("theta", host, extent);
    REQUIRE(store.reusable("theta", host, extent, true));
    REQUIRE_FALSE(store.reusable("theta", host + 1, extent, true));

    // A larger extent releases and replaces the pinned backend allocation.
    REQUIRE(store.acquire("theta", sizeof(host) * 2) == 1);
    REQUIRE(allocator.acquired == 2);
    REQUIRE(allocator.released.size() == 1);
    REQUIRE_FALSE(store.reusable("theta", host, extent, true));

    store.invalidate("theta");
    REQUIRE_FALSE(store.reusable("theta", host, extent, true));

    store.release("theta");
    REQUIRE(store.find("theta") == nullptr);
    REQUIRE(allocator.released.size() == 2);
    // Releasing an absent name is idempotent.
    store.release("theta");
    REQUIRE(allocator.released.size() == 2);
}

TEST_CASE("Fixed resident field scopes preserve root authority across child use",
          "[compute][residency][scope]")
{
    FakeAllocator allocator;
    allocator.next = 0;
    std::array<ResidentFieldStore, 2> stores;
    for (auto& store : stores)
        store.set_allocator(&acquire, &release, &allocator);
    auto& root = stores[0];
    auto& child = stores[1];
    float parent_theta[16]{};
    float child_theta[8]{};
    const FieldExtent parent_extent{16, sizeof(float)};
    const FieldExtent child_extent{8, sizeof(float)};

    root.begin_step();
    REQUIRE(root.acquire("theta", sizeof(parent_theta)) == 0);
    root.commit_host_mirror("theta", parent_theta, parent_extent);
    REQUIRE(root.reusable("theta", parent_theta, parent_extent, true));

    child.begin_step();
    REQUIRE(child.acquire("theta", sizeof(child_theta)) == 1);
    child.commit_host_mirror("theta", child_theta, child_extent);
    REQUIRE(child.reusable("theta", child_theta, child_extent, true));
    REQUIRE(root.reusable("theta", parent_theta, parent_extent, true));
    REQUIRE(root.find("theta")->allocation !=
            child.find("theta")->allocation);

    const auto* root_before_child_invalidation = root.version_state("theta");
    REQUIRE(root_before_child_invalidation != nullptr);
    const auto root_lineage_before_child_invalidation =
        *root_before_child_invalidation;

    REQUIRE(child.invalidate_all() == 1);
    REQUIRE_FALSE(child.reusable("theta", child_theta, child_extent, true));
    REQUIRE(root.reusable("theta", parent_theta, parent_extent, true));
    const auto* root_after_child_invalidation = root.version_state("theta");
    REQUIRE(root_after_child_invalidation != nullptr);
    REQUIRE(root_after_child_invalidation->version ==
            root_lineage_before_child_invalidation.version);
    REQUIRE(root_after_child_invalidation->host_version ==
            root_lineage_before_child_invalidation.host_version);
    REQUIRE(root_after_child_invalidation->device_version ==
            root_lineage_before_child_invalidation.device_version);
    REQUIRE(root.invalidate_all() == 1);
    REQUIRE_FALSE(root.reusable("theta", parent_theta, parent_extent, true));
}

TEST_CASE("Exact resident invalidation preserves unrelated device authority",
          "[compute][residency][authority]")
{
    FakeAllocator allocator;
    allocator.next = 0;
    ResidentFieldStore store;
    store.set_allocator(&acquire, &release, &allocator);
    float u[8]{};
    float theta[8]{};
    const FieldExtent extent{8, sizeof(float)};

    store.begin_step();
    REQUIRE(store.acquire("u", sizeof(u)) == 0);
    REQUIRE(store.acquire("theta", sizeof(theta)) == 1);
    store.commit_host_mirror("u", u, extent);
    store.commit_host_mirror("theta", theta, extent);
    REQUIRE(store.reusable("u", u, extent, true));
    REQUIRE(store.reusable("theta", theta, extent, true));
    const auto* u_before = store.version_state("u");
    REQUIRE(u_before != nullptr);
    const std::uint64_t u_version_before = u_before->version;

    u[0] = 1.0f;
    store.note_host_write("u");
    const auto* u_after = store.version_state("u");
    REQUIRE(u_after != nullptr);
    REQUIRE(u_after->version == u_version_before + 1);
    REQUIRE(u_after->host_version == u_after->version);
    REQUIRE(u_after->device_version != u_after->version);
    REQUIRE_FALSE(store.reusable("u", u, extent, true));
    REQUIRE(store.reusable("theta", theta, extent, true));

    REQUIRE(store.invalidate_all() == 1);
    REQUIRE_FALSE(store.reusable("theta", theta, extent, true));
}

TEST_CASE("Resident field store batches missing names and preserves reusable entries",
          "[compute][residency][allocation][batch]")
{
    FakeAllocator allocator;
    allocator.next = 0;
    ResidentFieldStore store;
    store.set_allocator(&acquire, &release, &allocator);
    store.set_batch_allocator(&acquire_batch);

    const char* first_names[3] = {"theta", "rho", "qv"};
    const std::size_t first_sizes[3] = {64, 64, 64};
    ResidentFieldStore::Allocation first[3]{};
    REQUIRE(store.acquire_many(first_names, first_sizes, 3, first));
    REQUIRE(first[0] == 0);
    REQUIRE(first[1] == 1);
    REQUIRE(first[2] == 2);
    REQUIRE(allocator.batch_calls == 1);
    REQUIRE(allocator.acquired == 3);

    ResidentFieldStore::Allocation reused[3]{};
    REQUIRE(store.acquire_many(first_names, first_sizes, 3, reused));
    REQUIRE(reused[0] == first[0]);
    REQUIRE(reused[1] == first[1]);
    REQUIRE(reused[2] == first[2]);
    REQUIRE(allocator.batch_calls == 1);

    const char* grown_names[3] = {"theta", "rho", "w"};
    const std::size_t grown_sizes[3] = {128, 64, 64};
    ResidentFieldStore::Allocation grown[3]{};
    REQUIRE(store.acquire_many(grown_names, grown_sizes, 3, grown));
    REQUIRE(grown[0] == 3);
    REQUIRE(grown[1] == first[1]);
    REQUIRE(grown[2] == 4);
    REQUIRE(allocator.batch_calls == 2);
    REQUIRE(allocator.acquired == 5);
    REQUIRE(allocator.released.size() == 1);

    // Growth is transactional: allocator failure retains the smaller
    // published allocation instead of orphaning the field.
    allocator.fail_batch = true;
    const char* failed_growth_names[1] = {"rho"};
    const std::size_t failed_growth_sizes[1] = {256};
    ResidentFieldStore::Allocation failed_growth[1]{};
    REQUIRE_FALSE(store.acquire_many(
        failed_growth_names, failed_growth_sizes, 1, failed_growth));
    REQUIRE(store.find("rho") != nullptr);
    REQUIRE(store.find("rho")->allocation == first[1]);
    REQUIRE(store.find("rho")->capacity_bytes == 64);
    REQUIRE(allocator.released.size() == 1);

    const char* duplicates[2] = {"rho", "rho"};
    const std::size_t duplicate_sizes[2] = {64, 64};
    ResidentFieldStore::Allocation duplicate_out[2]{};
    REQUIRE_FALSE(store.acquire_many(
        duplicates, duplicate_sizes, 2, duplicate_out));

    const char* failed_names[2] = {"p", "u"};
    ResidentFieldStore::Allocation failed[2]{};
    REQUIRE_FALSE(store.acquire_many(
        failed_names, duplicate_sizes, 2, failed));
    REQUIRE(store.find("p") == nullptr);
    REQUIRE(store.find("u") == nullptr);
}

TEST_CASE("Resident batch acquisition rolls back staged names and partial handles",
          "[compute][residency][allocation][batch][atomicity]")
{
    FakeAllocator allocator;
    allocator.next = 10;
    ResidentFieldStore store;
    REQUIRE(store.set_allocator(&acquire, &release, &allocator));
    REQUIRE(store.set_batch_allocator(&acquire_batch));

    const char* names[2] = {"p", "u"};
    const std::size_t sizes[2] = {64, 64};
    ResidentFieldStore::Allocation out[2] = {91, 92};

    allocator.fail_batch = true;
    allocator.partial_batch_count = 1;
    REQUIRE_FALSE(store.acquire_many(names, sizes, 2, out));
    CHECK(out[0] == ResidentFieldStore::invalid_allocation);
    CHECK(out[1] == ResidentFieldStore::invalid_allocation);
    CHECK(store.find("p") == nullptr);
    CHECK(store.find("u") == nullptr);
    REQUIRE(allocator.released.size() == 1);
    CHECK(allocator.released[0] == 10);

    allocator.fail_batch = false;
    allocator.throw_batch = true;
    allocator.partial_batch_count = 1;
    REQUIRE_THROWS_AS(store.acquire_many(names, sizes, 2, out),
                      std::runtime_error);
    CHECK(out[0] == ResidentFieldStore::invalid_allocation);
    CHECK(out[1] == ResidentFieldStore::invalid_allocation);
    CHECK(store.find("p") == nullptr);
    CHECK(store.find("u") == nullptr);
    REQUIRE(allocator.released.size() == 2);
    CHECK(allocator.released[1] == 11);
}
