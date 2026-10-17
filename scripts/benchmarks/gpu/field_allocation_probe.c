#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static _Atomic uint64_t calls;
static _Atomic uint64_t requested_bytes;
static void* (*allocate_real)(size_t, size_t);
static size_t selected_bytes;

__attribute__((constructor)) static void initialize_probe(void)
{
    allocate_real = (void* (*)(size_t, size_t))dlsym(RTLD_NEXT, "aligned_alloc");
    const char* value = getenv("NIMBUS_PROBE_FIELD_BYTES");
    selected_bytes = value ? strtoull(value, NULL, 10) : 0;
    if (!allocate_real || !selected_bytes) abort();
}

void* aligned_alloc(size_t alignment, size_t bytes)
{
    if (!allocate_real)
        allocate_real = (void* (*)(size_t, size_t))dlsym(RTLD_NEXT, "aligned_alloc");
    if (!allocate_real) abort();
    void* result = allocate_real(alignment, bytes);
    if (result && alignment == 16384 && bytes == selected_bytes)
    {
        atomic_fetch_add_explicit(&calls, 1, memory_order_relaxed);
        atomic_fetch_add_explicit(&requested_bytes, bytes, memory_order_relaxed);
    }
    return result;
}

__attribute__((destructor)) static void report_probe(void)
{
    fprintf(stderr, "[FIELD ALLOCATION PROBE] bytes=%zu allocations=%llu requested_bytes=%llu\n",
        selected_bytes, (unsigned long long)atomic_load(&calls),
        (unsigned long long)atomic_load(&requested_bytes));
}
