export interface RgbaImage {
  width: number;
  height: number;
  data: Uint8Array | Uint8ClampedArray;
}

export interface VtkAppearanceImageMetrics {
  pixelCount: number;
  rgbRmse: number;
  meanAbsoluteError: number;
  p99AbsoluteError: number;
  changedPixelFraction: number;
  edgeRmse: number;
  alphaMismatchFraction: number;
}

export interface VtkAppearanceImageTolerance {
  rgbRmse: number;
  meanAbsoluteError: number;
  p99AbsoluteError: number;
  changedPixelFraction: number;
  edgeRmse: number;
  alphaMismatchFraction: number;
  changedChannelThreshold: number;
}

export interface VtkAppearanceComparisonResult {
  pass: boolean;
  metrics: VtkAppearanceImageMetrics;
  violations: string[];
}

function assertImage(image: RgbaImage, label: string): void {
  if (!Number.isInteger(image.width) || !Number.isInteger(image.height)
      || image.width <= 0 || image.height <= 0
      || image.data.length !== image.width * image.height * 4) {
    throw new Error(`${label} is not a complete positive RGBA image.`);
  }
}

function luminance(data: Uint8Array | Uint8ClampedArray, offset: number): number {
  return (0.2126 * data[offset] + 0.7152 * data[offset + 1] + 0.0722 * data[offset + 2]) / 255;
}

function edgeDifference(
  reference: RgbaImage,
  candidate: RgbaImage,
): number {
  if (reference.width < 2 || reference.height < 2) return 0;
  let squared = 0;
  let samples = 0;
  for (let y = 0; y < reference.height - 1; y += 1) {
    for (let x = 0; x < reference.width - 1; x += 1) {
      const offset = (y * reference.width + x) * 4;
      const right = offset + 4;
      const down = offset + reference.width * 4;
      const referenceDx = luminance(reference.data, right) - luminance(reference.data, offset);
      const referenceDy = luminance(reference.data, down) - luminance(reference.data, offset);
      const candidateDx = luminance(candidate.data, right) - luminance(candidate.data, offset);
      const candidateDy = luminance(candidate.data, down) - luminance(candidate.data, offset);
      squared += (referenceDx - candidateDx) ** 2 + (referenceDy - candidateDy) ** 2;
      samples += 2;
    }
  }
  return Math.sqrt(squared / samples);
}

export function compareVtkAppearanceImages(
  reference: RgbaImage,
  candidate: RgbaImage,
  changedChannelThreshold = 8 / 255,
): VtkAppearanceImageMetrics {
  assertImage(reference, "Reference");
  assertImage(candidate, "Candidate");
  if (reference.width !== candidate.width || reference.height !== candidate.height) {
    throw new Error(
      `Appearance image dimensions differ: ${reference.width}×${reference.height} versus ${candidate.width}×${candidate.height}.`,
    );
  }
  const absolute: number[] = [];
  let squared = 0;
  let sum = 0;
  let changedPixels = 0;
  let alphaMismatches = 0;
  const pixelCount = reference.width * reference.height;
  for (let pixel = 0; pixel < pixelCount; pixel += 1) {
    const offset = pixel * 4;
    let changed = false;
    for (let channel = 0; channel < 3; channel += 1) {
      const delta = Math.abs(reference.data[offset + channel] - candidate.data[offset + channel]) / 255;
      absolute.push(delta);
      sum += delta;
      squared += delta * delta;
      changed = changed || delta > changedChannelThreshold;
    }
    if (changed) changedPixels += 1;
    if (reference.data[offset + 3] !== candidate.data[offset + 3]) alphaMismatches += 1;
  }
  absolute.sort((a, b) => a - b);
  const p99Index = Math.min(absolute.length - 1, Math.ceil(absolute.length * 0.99) - 1);
  return {
    pixelCount,
    rgbRmse: Math.sqrt(squared / absolute.length),
    meanAbsoluteError: sum / absolute.length,
    p99AbsoluteError: absolute[p99Index] ?? 0,
    changedPixelFraction: changedPixels / pixelCount,
    edgeRmse: edgeDifference(reference, candidate),
    alphaMismatchFraction: alphaMismatches / pixelCount,
  };
}

export function evaluateVtkAppearanceComparison(
  metrics: VtkAppearanceImageMetrics,
  tolerance: VtkAppearanceImageTolerance,
): VtkAppearanceComparisonResult {
  const violations: string[] = [];
  const checks: Array<[keyof VtkAppearanceImageMetrics, number]> = [
    ["rgbRmse", tolerance.rgbRmse],
    ["meanAbsoluteError", tolerance.meanAbsoluteError],
    ["p99AbsoluteError", tolerance.p99AbsoluteError],
    ["changedPixelFraction", tolerance.changedPixelFraction],
    ["edgeRmse", tolerance.edgeRmse],
    ["alphaMismatchFraction", tolerance.alphaMismatchFraction],
  ];
  for (const [name, limit] of checks) {
    const value = metrics[name];
    if (value > limit) violations.push(`${name} ${value} exceeds ${limit}`);
  }
  return { pass: violations.length === 0, metrics, violations };
}

export function qualifyVtkAppearanceImages(
  reference: RgbaImage,
  candidate: RgbaImage,
  tolerance: VtkAppearanceImageTolerance,
): VtkAppearanceComparisonResult {
  return evaluateVtkAppearanceComparison(
    compareVtkAppearanceImages(reference, candidate, tolerance.changedChannelThreshold),
    tolerance,
  );
}
