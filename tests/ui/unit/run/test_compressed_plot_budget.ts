import assert from "node:assert/strict";
import { registerListedStepMetadata, plotFieldWindows, clearFieldCache, type StepInfo, type FieldLoadWindow } from "../../../../ui/src/lib/run/runOutputs.ts";
const run = "/fixture/compressed-budget";
const step: StepInfo = { index: 0, simulationTimeS: 0, fields: ["cloud_condensate"], fieldUnits: { cloud_condensate: "kg/kg" }, grid: { nr: 512, nth: 512, nz: 160 }, publicationState: "complete", publicationGeneration: "generation" };
const native: FieldLoadWindow[] = [{ aggregation: "cell_mean", origin: [0,0,0], extent: [273,273,160], stride: [1,1,1], maxOutputBytes: 273*273*160*4, maxSourceBytes: 273*273*160*4 }];
registerListedStepMetadata(run, [step]);
assert.equal(plotFieldWindows(run, 0, "cloud_condensate", "generation", native), native, "uncompressed read budgets must stay unchanged");
registerListedStepMetadata(run, [{ ...step, fieldCompression: { cloud_condensate: {
  codec: "zfp", mode: "accuracy", modeParameter: 1e-5, modeParameterUnits: "field_units", errorContract: "zfp_absolute_accuracy_before_prefilters", sparseThreshold: 0, float16Prefilter: false, decodedDtype: "float32",
} } }]);
const adjusted = plotFieldWindows(run, 0, "cloud_condensate", "generation", native)!;
assert.ok(adjusted[0].maxSourceBytes! > native[0].maxSourceBytes!);
assert.ok(adjusted[0].maxSourceBytes! < 64*1024*1024);
assert.deepEqual(adjusted[0].stride, [1,1,1]);
assert.equal(adjusted[0].maxOutputBytes, native[0].maxOutputBytes);
const sparse: FieldLoadWindow[] = [{ aggregation: "point_sample", origin: [0,0,0], extent: [512,512,160], stride: [16,16,5], maxOutputBytes: 131072, maxSourceBytes: 4*1024*1024 }];
assert.equal(plotFieldWindows(run, 0, "cloud_condensate", "generation", sparse)![0], sparse[0], "explicit sparse-point caps must not be raised");
const context = plotFieldWindows(run, 0, "cloud_condensate", "generation", [{ ...native[0], extent: [512,512,160], stride: [3,2,2] }])![0];
assert.ok(context.maxSourceBytes! >= 512*512*160*4 + 171*256*80*21);
assert.ok(context.maxSourceBytes! <= 256*1024*1024);
clearFieldCache();
assert.equal(plotFieldWindows(run, 0, "cloud_condensate", "generation", native), native);
console.log("Compressed plot budgets: bounded codec/reduction headroom, unchanged native resolution, strict sparse caps, and metadata cleanup passed.");
