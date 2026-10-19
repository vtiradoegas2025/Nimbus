import {
  compareVtkAppearanceImages,
  evaluateVtkAppearanceComparison,
  qualifyVtkAppearanceImages,
  type RgbaImage,
  type VtkAppearanceImageTolerance,
} from "../../../../ui/src/lib/vtk/vtkAppearanceComparison.ts";

let passed = 0;

function check(condition: boolean, message: string): void {
  if (!condition) throw new Error(message);
  passed += 1;
}

const image = (values: number[]): RgbaImage => ({
  width: 2,
  height: 2,
  data: Uint8Array.from(values),
});
const reference = image([
  0, 0, 0, 255, 255, 255, 255, 255,
  64, 64, 64, 255, 128, 128, 128, 255,
]);
const identical = image([...reference.data]);
const exact = compareVtkAppearanceImages(reference, identical);
check(exact.rgbRmse === 0 && exact.edgeRmse === 0, "identical frames have zero color and edge error");
check(exact.changedPixelFraction === 0 && exact.alphaMismatchFraction === 0, "identical frames change no pixels");

const shifted = image([
  16, 16, 16, 255, 240, 240, 240, 255,
  64, 64, 64, 0, 150, 128, 128, 255,
]);
const metrics = compareVtkAppearanceImages(reference, shifted);
check(metrics.rgbRmse > 0 && metrics.meanAbsoluteError > 0, "color changes produce bounded normalized errors");
check(metrics.changedPixelFraction === 0.75, "changed-pixel occupancy uses the declared channel threshold");
check(metrics.alphaMismatchFraction === 0.25, "alpha mismatches are measured independently");
check(metrics.edgeRmse > 0, "structure changes alter the edge metric");

const strict: VtkAppearanceImageTolerance = {
  rgbRmse: 0,
  meanAbsoluteError: 0,
  p99AbsoluteError: 0,
  changedPixelFraction: 0,
  edgeRmse: 0,
  alphaMismatchFraction: 0,
  changedChannelThreshold: 8 / 255,
};
const evaluated = evaluateVtkAppearanceComparison(metrics, strict);
check(!evaluated.pass && evaluated.violations.length === 6, "explicit tolerances report every violated image contract");
check(evaluateVtkAppearanceComparison(exact, strict).pass, "strict repeatability accepts an exact frame");
check(qualifyVtkAppearanceImages(reference, identical, strict).pass, "qualification applies one coherent tolerance contract");

let dimensionError = "";
try {
  compareVtkAppearanceImages(reference, { width: 1, height: 1, data: Uint8Array.from([0, 0, 0, 255]) });
} catch (error) {
  dimensionError = error instanceof Error ? error.message : String(error);
}
check(dimensionError.includes("dimensions differ"), "comparison refuses implicit image resizing");

console.log(`vtk appearance comparison: ${passed} checks passed`);
