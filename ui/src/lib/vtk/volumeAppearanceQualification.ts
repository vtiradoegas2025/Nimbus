import type { FieldWindowAggregation } from "@/lib/dataset/nativeFieldWindow";

export interface VolumeAppearanceQualificationOverrides {
  contextOpacityFactor?: number;
  detailOpacityFactor?: number;
  opacityReferenceM?: number;
  sampleDistanceFactor?: number;
  previewDistanceMultiplier?: number;
  lighting?: "policy" | "flat";
  overviewAggregation?: FieldWindowAggregation;
  nativeDetail?: boolean;
  frameNativeDetail?: boolean;
  progressiveBricks?: boolean;
  compactBricks?: boolean;
}

export const VOLUME_APPEARANCE_ABLATIONS = {
  baseline: {},
  "uncompacted-bricks": { compactBricks: false },
  "context-off": { contextOpacityFactor: 0 },
  "context-018": { contextOpacityFactor: 0.18 },
  "detail-075": { detailOpacityFactor: 0.75 },
  "flat-detail": { lighting: "flat" },
  "opacity-250m": { opacityReferenceM: 250 },
  "sample-quarter-voxel": { sampleDistanceFactor: 0.25 },
  "sample-one-voxel": { sampleDistanceFactor: 1 },
  "preview-six": { previewDistanceMultiplier: 6 },
  "preview-eight": { previewDistanceMultiplier: 8 },
  "no-native-detail": { nativeDetail: false },
  "overview-camera": { frameNativeDetail: false },
} as const satisfies Readonly<Record<string, VolumeAppearanceQualificationOverrides>>;

export type VolumeAppearanceAblation = keyof typeof VOLUME_APPEARANCE_ABLATIONS;

export function volumeAppearanceAblation(
  name: string | null | undefined,
): { name: VolumeAppearanceAblation; overrides: VolumeAppearanceQualificationOverrides } {
  const key = name && Object.hasOwn(VOLUME_APPEARANCE_ABLATIONS, name)
    ? name as VolumeAppearanceAblation
    : "baseline";
  return { name: key, overrides: VOLUME_APPEARANCE_ABLATIONS[key] };
}
