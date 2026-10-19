import type { VolumeBrickPlan } from "@/lib/plot/fieldBrickHierarchy";
import { MAX_NATIVE_FIELD_WINDOW_ENVELOPE_OVERHEAD_BYTES } from "@/lib/dataset/nativeFieldWindow";
import type { VisualizationResourceClaim } from "@/lib/core/visualizationResourceGovernor";

export interface NestedVolumeLayerPlan {
  levelId: string;
  plan: VolumeBrickPlan;
}

export interface NestedVolumeGenerationPlan {
  identity: string;
  field: string;
  layers: NestedVolumeLayerPlan[];
  claims: VisualizationResourceClaim[];
}

export function planNestedVolumeGeneration(
  identity: string,
  layers: readonly NestedVolumeLayerPlan[],
): NestedVolumeGenerationPlan {
  if (!identity || layers.length < 2) throw new Error("Nested volume generation requires a parent and child");
  const field = layers[0].plan.field;
  if (layers.some((layer, index) => !layer.levelId
      || layer.plan.field !== field
      || index > 0 && layer.levelId === layers[index - 1].levelId)) {
    throw new Error("Nested volume layers do not share one field and unique hierarchy chain");
  }
  const claims = layers.flatMap((layer) => layer.plan.items.flatMap((item) => {
    const prefix = `nested:${identity}:${layer.levelId}:${item.brickId}`;
    return [
      { allocationId: `${prefix}:input`, stage: "native-read" as const,
        bytes: (item.window.maxOutputBytes ?? 0) + MAX_NATIVE_FIELD_WINDOW_ENVELOPE_OVERHEAD_BYTES,
        accounting: "reserved" as const },
      { allocationId: `${prefix}:prepared`, stage: "prepared-main" as const,
        bytes: item.preparedScalarBytes, accounting: "reserved" as const },
      { allocationId: `${prefix}:gpu`, stage: "gpu" as const,
        bytes: item.preparedScalarBytes, accounting: "reserved" as const },
    ];
  }));
  return { identity, field, layers: [...layers], claims };
}

export class NestedVolumeCommitBarrier<T> {
  private identity = "";
  private required = new Set<string>();
  private prepared = new Map<string, T>();

  reset(plan: NestedVolumeGenerationPlan): void {
    this.identity = plan.identity;
    this.required = new Set(plan.layers.map((layer) => layer.levelId));
    this.prepared.clear();
  }

  publish(identity: string, levelId: string, value: T): T[] | null {
    if (identity !== this.identity || !this.required.has(levelId)) return null;
    this.prepared.set(levelId, value);
    if (this.prepared.size !== this.required.size) return null;
    return [...this.required].map((id) => this.prepared.get(id)!);
  }
}
