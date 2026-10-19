import {
  VOLUME_APPEARANCE_ABLATIONS,
  volumeAppearanceAblation,
} from "../../../../ui/src/lib/vtk/volumeAppearanceQualification.ts";
import {
  VOLUME_CONTEXT_OPACITY_FACTOR,
  volumeOpacityReferenceM,
  volumeOpacitySelectionKey,
  resolveVolumeTransferPolicy,
  volumeOpacityPoints,
} from "../../../../ui/src/lib/plot/volumeTransferPolicy.ts";

let passed = 0;

function check(condition: boolean, message: string): void {
  if (!condition) throw new Error(message);
  passed += 1;
}

for (const field of ["vorticity_magnitude", "buoyancy", "dense_signed", "temperature", "reflectivity_dbz"]) {
  for (const scalarTransform of ["none", "horizontal-level-anomaly"] as const) {
    const selections = new Map<string, string>();
    for (const colormap of ["Inferno", "Viridis", "Cool to Warm", "dBZ"]) {
      for (const opacityScale of [0, 0.25, 1, 2.5]) {
        const transfer = { colormap, opacityScale, scalarTransform };
        const key = volumeOpacitySelectionKey(field, transfer);
        const policy = resolveVolumeTransferPolicy(field, colormap, { min: -10, max: 20, p2: -8, p98: 15 }, scalarTransform);
        const support = JSON.stringify(volumeOpacityPoints(policy, opacityScale).map((point) => [point.scalar, point.opacity > 0]));
        check(!selections.has(key) || selections.get(key) === support, `${field} cached brick selection preserves opacity support`);
        selections.set(key, support);
      }
    }
  }
}

check(VOLUME_CONTEXT_OPACITY_FACTOR === 0.08, "qualified sparse context defaults to eight percent");
check(volumeOpacityReferenceM("sparse-positive") === 100, "qualified sparse detail uses a 100 m opacity reference");
check(volumeAppearanceAblation(null).name === "baseline", "missing qualification selector preserves product defaults");
check(volumeAppearanceAblation("unknown").name === "baseline", "unknown qualification selector fails closed to baseline");
check(
  VOLUME_APPEARANCE_ABLATIONS["context-018"].contextOpacityFactor === 0.18,
  "context regression can be reproduced as one factor",
);
check(
  VOLUME_APPEARANCE_ABLATIONS["opacity-250m"].opacityReferenceM === 250,
  "opacity-reference regression can be reproduced as one factor",
);
check(
  VOLUME_APPEARANCE_ABLATIONS["sample-quarter-voxel"].sampleDistanceFactor === 0.25
    && VOLUME_APPEARANCE_ABLATIONS["sample-one-voxel"].sampleDistanceFactor === 1,
  "ray-step aliasing can be isolated without changing source or material",
);
check(
  VOLUME_APPEARANCE_ABLATIONS["preview-six"].previewDistanceMultiplier === 6
    && VOLUME_APPEARANCE_ABLATIONS["preview-eight"].previewDistanceMultiplier === 8,
  "interaction ray spacing can be isolated without changing settled fidelity",
);
for (const [name, overrides] of Object.entries(VOLUME_APPEARANCE_ABLATIONS)) {
  if (name === "baseline") continue;
  check(Object.keys(overrides).length === 1, `${name} changes exactly one appearance factor`);
}

console.log(`volume appearance qualification: ${passed} checks passed`);
