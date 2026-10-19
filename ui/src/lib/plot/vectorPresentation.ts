import type { NpyN } from "@/lib/platform/npy";

export interface VectorReferenceKey {
  magnitude: number;
  tailX: number;
  headX: number;
  y: number;
}

function niceAtOrBelow(value: number): number {
  if (!(value > 0) || !Number.isFinite(value)) return 0;
  const exponent = Math.floor(Math.log10(value));
  const power = 10 ** exponent;
  const normalized = value / power;
  const mantissa = normalized >= 5 ? 5 : normalized >= 2 ? 2 : 1;
  return mantissa * power;
}

/**
 * Build an in-axis vector key whose arrow has the same physical scaling as the
 * field arrows. Keeping this in data coordinates makes screenshots honest;
 * a fixed-pixel annotation would cease to be a scale key after resize/export.
 */
export function vectorReferenceKey(
  referenceMagnitude: number,
  arrowSpacing: number,
  effectiveStride: number,
  scale: number,
  xValues: readonly number[] | undefined,
  yValues: readonly number[] | undefined,
): VectorReferenceKey | null {
  if (!(referenceMagnitude > 0) || !(arrowSpacing > 0) || !(scale > 0)) return null;
  if (!xValues || xValues.length < 2 || !yValues || yValues.length < 2) return null;
  const xMin = Math.min(xValues[0], xValues.at(-1)!);
  const xMax = Math.max(xValues[0], xValues.at(-1)!);
  const yMin = Math.min(yValues[0], yValues.at(-1)!);
  const yMax = Math.max(yValues[0], yValues.at(-1)!);
  const xSpan = xMax - xMin;
  const ySpan = yMax - yMin;
  if (!(xSpan > 0) || !(ySpan > 0)) return null;

  const fullReferenceLength = Math.max(1, Math.floor(effectiveStride)) * arrowSpacing * scale;
  const targetLength = Math.min(fullReferenceLength, xSpan * 0.16);
  const targetMagnitude = referenceMagnitude * targetLength / fullReferenceLength;
  const magnitude = niceAtOrBelow(targetMagnitude);
  if (!(magnitude > 0)) return null;
  const length = fullReferenceLength * magnitude / referenceMagnitude;
  const tailX = xMin + xSpan * 0.04;
  return {
    magnitude,
    tailX,
    headX: tailX + length,
    y: yMin + ySpan * 0.065,
  };
}

/** Native-to-display disclosure for bounded vector planes. */
export function describeVectorSampling(
  field: NpyN | null | undefined,
  rowAxis: number,
  colAxis: number,
): string | null {
  const geometry = field?.windowGeometry;
  const stride = geometry?.stride;
  if (!stride) return null;
  const horizontal = Math.max(1, Math.floor(stride[colAxis] ?? 1));
  const vertical = Math.max(1, Math.floor(stride[rowAxis] ?? 1));
  if (horizontal === 1 && vertical === 1) return null;
  const reduction = geometry?.aggregation === "cell_mean"
    ? "native cell mean"
    : geometry?.aggregation === "linear_reflectivity_mean"
      ? "native linear-reflectivity mean"
      : `native ${geometry?.aggregation?.replaceAll("_", "-") ?? "block"} reduction`;
  return `${reduction} ${horizontal}×${vertical}`;
}
