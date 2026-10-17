#include "compute/support/kernel_ownership_catalog.hpp"

#include <cstdio>
#include <exception>

int main()
{
    try
    {
        const std::string catalog = nimbus::parity::serialize_kernel_ownership_catalog_json();
        std::fputs(catalog.c_str(), stdout);
        return 0;
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "kernel ownership catalog failed: %s\n", error.what());
        return 1;
    }
}
