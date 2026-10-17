#include "core/output/output_writer.hpp"
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <new>
#include <thread>

namespace
{
std::atomic<void*> watched{nullptr};
std::atomic<bool> deleting{false}, release_storage{true};
void deallocate(void* pointer) noexcept
{
    if (pointer && pointer==watched.load(std::memory_order_acquire))
    {
        deleting.store(true,std::memory_order_release);
        while (!release_storage.load(std::memory_order_acquire)) std::this_thread::yield();
    }
    std::free(pointer);
}
bool await(const std::atomic<bool>& condition)
{
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
    while (!condition.load(std::memory_order_acquire) && std::chrono::steady_clock::now()<deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return condition.load(std::memory_order_acquire);
}
}
void* operator new(std::size_t size)
{
    if (void* pointer=std::malloc(size?size:1)) return pointer;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) {return ::operator new(size);}
void operator delete(void* pointer) noexcept {deallocate(pointer);}
void operator delete[](void* pointer) noexcept {deallocate(pointer);}
void operator delete(void* pointer,std::size_t) noexcept {deallocate(pointer);}
void operator delete[](void* pointer,std::size_t) noexcept {deallocate(pointer);}

int main()
{
    bool passed=true;
    for (bool failing : {false,true})
    {
        const auto directory=std::filesystem::temp_directory_path()/
            ("nimbus_lifetime_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(directory);
        OutputConfig config;config.format=OutputFormat::npy_3d;config.async_io=true;
        AsyncOutputWriter writer(config);
        ExportSnapshot snapshot;snapshot.step_dir=directory;
        snapshot.fields.emplace_back("field",std::vector<float>(512,1.0f),failing?9:8,8,8,true);
        watched.store(snapshot.fields[0].data.data(),std::memory_order_release);
        deleting.store(false,std::memory_order_release);
        release_storage.store(false,std::memory_order_release);
        const bool submitted=writer.submit(std::move(snapshot));
        const bool entered=await(deleting);
        std::atomic<bool> started{false};
        auto waiter=std::async(std::launch::async,[&] {started.store(true,std::memory_order_release);return writer.wait_until_ready();});
        const bool started_in_time=await(started);
        const bool blocked=waiter.wait_for(std::chrono::milliseconds(200))==std::future_status::timeout;
        release_storage.store(true,std::memory_order_release);
        const bool result=waiter.get();
        const bool flushed=writer.flush();
        watched.store(nullptr,std::memory_order_release);
        const bool ok=submitted && entered && started_in_time && blocked && result==!failing && flushed==!failing;
        std::cout << "snapshot_failure=" << failing << " deletion_observed=" << entered
                  << " producer_blocked_until_release=" << blocked << " ready_result=" << result
                  << " flush_result=" << flushed << " writer_error=" << writer.has_error() << " passed=" << ok << '\n';
        passed=passed&&ok;
        std::filesystem::remove_all(directory);
    }
    return passed?0:1;
}
