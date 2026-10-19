import assert from "node:assert/strict";
import { register } from "node:module";

register(`data:text/javascript,${encodeURIComponent(`
export function resolve(specifier, context, nextResolve) {
  return nextResolve(specifier.startsWith('@kitware/vtk.js/') && !specifier.endsWith('.js')
    ? specifier + '.js' : specifier, context);
}
`)}`, import.meta.url);
const originalWindow = Object.getOwnPropertyDescriptor(globalThis, "window");
const originalDocument = Object.getOwnPropertyDescriptor(globalThis, "document");
Object.assign(globalThis, { window: { location: { search: "" } } });
try {
  const { qualifyPackagedPhysicalGpuResidency } = await import("../../../../ui/src/lib/vtk/packagedAppearanceQualification.ts");
  const { visualizationResourceGovernor: governor } = await import("../../../../ui/src/lib/core/visualizationResourceGovernor.ts");
  const field = { shape: [2, 2, 2] as [number, number, number], data: new Float32Array(8) };
  let createdHosts = 0;
  Object.assign(globalThis, { document: { createElement: () => { createdHosts++; throw new Error("must not create a scene"); } } });
  governor.setCapacity({ totalBytes: null, stageBytes: { gpu: 0 } });
  await assert.rejects(qualifyPackagedPhysicalGpuResidency(field, async () => undefined), /gpu needs 64 bytes/);
  assert.equal(createdHosts, 0, "rejected GPU admission must not create DOM or rendering owners");
  assert.equal(governor.stats().residentBytes, 0, "rejection releases prepared input ownership");
  assert.equal(governor.stats().reservedBytes, 0);

  governor.setCapacity({ totalBytes: null });
  await assert.rejects(qualifyPackagedPhysicalGpuResidency(field, async () => undefined), /must not create a scene/);
  assert.equal(createdHosts, 1);
  assert.equal(governor.stats().residentBytes, 0, "construction failure releases input and GPU reservations");
  assert.equal(governor.stats().reservedBytes, 0);

  let removedHost = false;
  Object.assign(globalThis, { document: {
    createElement: () => ({ style: {}, remove: () => { removedHost = true; } }),
    body: { append: () => { throw new Error("host attachment failed"); } },
  } });
  await assert.rejects(qualifyPackagedPhysicalGpuResidency(field, async () => undefined), /host attachment failed/);
  assert.equal(removedHost, true, "partial host construction retires its DOM owner");
  assert.equal(governor.stats().residentBytes, 0);
  assert.equal(governor.stats().reservedBytes, 0);
} finally {
  if (originalWindow) Object.defineProperty(globalThis, "window", originalWindow);
  else Reflect.deleteProperty(globalThis, "window");
  if (originalDocument) Object.defineProperty(globalThis, "document", originalDocument);
  else Reflect.deleteProperty(globalThis, "document");
}
console.log("Packaged GPU admission and partial construction failures retire every reservation: passed");
