#pragma once

#include <string_view>

namespace tmv::runtime
{

inline constexpr std::string_view kFlatMixedMetalCadence1ScopeManifestSha256 =
    "ea8921dabe62cceac5f95a023bdb92f3f5063dc9cd7aea6e8368925712014402";
inline constexpr std::string_view kFlatMixedMetalCadence1ScopeJson = R"json({"schema":"nimbus.qualified-flat-mixed-metal-npy-scope.v1","version":1,"profileCount":1,"backend":"metal","executionMode":"qualified","familyId":"boundary:ed765facac077ef0b331283d","strictCanonicalYamlSha256":"985c698858be9be4be29dc2adf819f31c20788c13a414747c422e21d7ebe51da","outputFormats":["npy_3d"],"preset":"minimal","writeCadenceSeconds":[1],"asyncIo":false,"liveShm":false,"trajectories":false,"allowFallback":false,"memoryAdmissionPolicies":["warn","reject"],"memoryPolicyCanonicalValue":"warn","gridShape":[18,18,18],"gridSpacingMeters":[500,500,250],"topHeightMeters":4500,"dtLargeSeconds":0.1,"acousticSubsteps":6,"adaptiveDt":false,"lateralBoundaries":["rigid_slip","rigid_slip","rigid_slip","rigid_slip"],"bottomBoundary":"rigid_noslip","topBoundary":"open","scalarTransportOwner":"metal_native_conservative_mc","initialConditionChaosOwner":"host","lateralDampingOwner":"host","lateralSpongeEnabled":true,"lateralSpongeWidth":20,"lateralSpongeAlpha":0.05,"wOnlyTopSponge":true,"wOnlyDepth":8,"wOnlyAlpha":0.2,"effectiveWOnlySponge":false,"sourceProfileOwnerSymbol":"is_public_qualified_flat_mixed_metal_cadence1_profile"})json";

}
