/**
 * @file test_main.cpp
 * @brief Catch2 test runner entry point.
 *
 * Compiled once, linked into every test binary. Defines main() so that
 * individual test files only need TEST_CASE / SECTION macros.
 */
#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
