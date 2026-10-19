#include "catch2/catch.hpp"
#include "util/packaged_runtime.hpp"

TEST_CASE("Packaged runtime paths follow GUI and simulator layouts without guessing source paths",
          "[core][packaged-runtime]")
{
    using tmv::packaged_runtime::frameworks_for_executable;
    REQUIRE(frameworks_for_executable("/tmp/Renamed App.app/Contents/MacOS/nimbus-studio") ==
            "/tmp/Renamed App.app/Contents/Frameworks");
    REQUIRE(frameworks_for_executable("/tmp/Renamed App.app/Contents/Resources/bin/nimbus_sim") ==
            "/tmp/Renamed App.app/Contents/Frameworks");
    REQUIRE(frameworks_for_executable("/project/bin/nimbus_sim").empty());
    REQUIRE(frameworks_for_executable("/tmp/Resources/bin/nimbus_sim").empty());
    REQUIRE(frameworks_for_executable("/tmp/MacOS/nimbus-studio").empty());
}
