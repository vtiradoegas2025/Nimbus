export interface JointVolumeFrame {
  owner: object;
  layers: number;
  lead: object;
  mapper: object;
  volume: object;
}

const bindings = new WeakMap<object, () => JointVolumeFrame | null>();
const cleanups = new WeakMap<object, () => void>();
export function hasJointVolumeBinding(mapper: object): boolean { return bindings.has(mapper); }

export function setJointVolumeCleanup(owner: object, cleanup: () => void): void { cleanups.set(owner, cleanup); }
export function releaseJointVolume(owner: object): void {
  cleanups.get(owner)?.();
  cleanups.delete(owner);
}

export function registerJointVolume(mapper: object, resolve: () => JointVolumeFrame | null): () => void {
  if (bindings.has(mapper)) throw new Error("Volume mapper already belongs to a joint renderer");
  bindings.set(mapper, resolve);
  return () => { bindings.delete(mapper); };
}

export function resolveJointVolume(mapper: object): JointVolumeFrame | null {
  return bindings.get(mapper)?.() ?? null;
}
