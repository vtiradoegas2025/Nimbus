import type { FieldBrickHierarchy, VolumeBrickPlan } from "../plot/fieldBrickHierarchy";
import { planVolumeTextureLayout } from "./volumeTexturePacking";

export function volumeImageBytes(image: { values: Float32Array; validity?: Uint8Array; shape?: readonly number[] } | null | undefined,
  maximumTextureDimension?: number): number {
  if (!image) return 0;
  if (maximumTextureDimension != null && image.shape) {
    try {
      return planVolumeTextureLayout(image.shape, maximumTextureDimension).textureCells * (4 + Number(image.validity != null));
    } catch {
      return image.values.byteLength + (image.validity?.byteLength ?? 0);
    }
  }
  return image.values.byteLength + (image.validity?.byteLength ?? 0);
}

export function volumeValidityBudgetFactor(hierarchy: FieldBrickHierarchy, coverageUnknown: boolean): number {
  return coverageUnknown || hierarchy.levels.some((level) => level.bricks.some((brick) => brick.scalarRange.nonFiniteCount > 0)) ? 1.25 : 1;
}

export function reserveVolumeValidity(plan: VolumeBrickPlan, factor: number): VolumeBrickPlan {
  if (factor === 1) return plan;
  const items = plan.items.map((item) => ({ ...item, preparedScalarBytes: Math.ceil(item.preparedScalarBytes * factor) }));
  return { ...plan, items, residentBytes: items.reduce((sum, item) => sum + item.preparedScalarBytes, 0) };
}
