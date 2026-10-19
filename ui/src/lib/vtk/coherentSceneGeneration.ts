export interface SceneGenerationLayer {
  layer: string;
  identity: string;
  ready: boolean;
  error: string | null;
}

export function coherentSceneGeneration(
  identity: string,
  layers: readonly SceneGenerationLayer[],
): { ready: boolean; error: string | null } {
  if (!identity || new Set(layers.map(layer => layer.layer)).size !== layers.length) {
    throw new Error("Scene generation requires one identity and unique layer owners");
  }
  const current = layers.filter(layer => layer.identity === identity);
  const error = current.find(layer => layer.error != null)?.error ?? null;
  return { ready: error == null && current.length === layers.length && current.every(layer => layer.ready), error };
}

export class CoherentSceneCommit<T> {
  private requested: string | null = null;
  private committed: { identity: string; value: T } | null = null;

  request(identity: string): void { this.requested = identity; }
  commit(identity: string, layers: readonly SceneGenerationLayer[], value: T): boolean {
    if (identity !== this.requested || !coherentSceneGeneration(identity, layers).ready) return false;
    this.committed = { identity, value };
    return true;
  }
  snapshot(): { identity: string; value: T } | null { return this.committed; }
}
