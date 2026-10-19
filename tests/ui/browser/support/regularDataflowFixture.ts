import type { GridInfo, StepInfo } from "@nimbus/lib/run/runOutputs";
import type { NpyN } from "@nimbus/lib/platform/npy";
import { encodeNpy } from "@nimbus/lib/platform/pythonExport";
import { encodeFieldWindow } from "./fixtures";

export const REGULAR_RUN = "/__nimbus_browser_fixture__/regular-history";
const requestedNz = Number(new URLSearchParams(window.location.search).get("regularNz"));
const nz = Number.isSafeInteger(requestedNz) && requestedNz > 0 && requestedNz <= 256 ? requestedNz : 2;
const sectionFields = new URLSearchParams(window.location.search).get("namedSectionFields") === "1"
  ? ["cloud_condensate", "p", "theta_prime"] : [];
export const regularGrid: GridInfo = { coordinateSystem: "cartesian", nr: 3, nth: 2, nz,
  drM: 1000, dyM: 2000, dzM: 500, scalarOriginM: [500, 1000, 250],
  physicalBoundsM: { lower: [0, 0, 0], upper: [3000, 4000, nz * 500] } };
const versions = new Map<number, number>();
const offsets = new Map<number, number>();
const missing = new Set<number>();
const units = new Map<number, string>();
const delays = new Map<string, number>();
const samples = new Map<string, number>();
const componentUnits = new Map<string, string>();
const heldFutureReads = new URLSearchParams(window.location.search).get("heldPrefetch") === "1";
const failedField = new URLSearchParams(window.location.search).get("failedRegularField");
const pendingReads = new Map<string, { field: string; step: number; resolve: () => void; reject: (error: Error) => void }>();
const cancelledReadIds: string[] = [];
export const regularCalls: { command: string; step: number; field: string;
  request?: { origin: number[]; extent: number[]; stride: number[] } }[] = [];

export function regularSteps(indices: number[]): StepInfo[] {
  return indices.map(index => ({ index, simulationTimeS: index * 30, publicationState: "complete",
    publicationGeneration: `regular-${index}-${versions.get(index) ?? 0}`,
    fields: ["w", "u", "v", "column_max_w", ...sectionFields], fieldOrigins: {
      w: "native", u: "native", v: "native", column_max_w: "native",
      ...Object.fromEntries(sectionFields.map(field => [field, "native" as const])),
    },
    fieldShapes: { w: [3, 2, nz], u: [3, 2, nz], v: [3, 2, nz], column_max_w: [3, 2],
      ...Object.fromEntries(sectionFields.map(field => [field, [3, 2, nz]])),
    },
    fieldUnits: { w: units.get(index) ?? "m/s", u: componentUnits.get(`${index}:u`) ?? "m/s",
      v: componentUnits.get(`${index}:v`) ?? "m/s", column_max_w: units.get(index) ?? "m/s",
      ...Object.fromEntries(sectionFields.map(field => [field, field === "p" ? "Pa" : field === "theta_prime" ? "K" : "kg/kg"])),
    }, grid: regularGrid }));
}

export function regularSource(step: number, field: string): NpyN {
  const shape = field === "column_max_w" ? [3, 2] : [3, 2, nz];
  const data = new Float32Array(shape.reduce((count, size) => count * size, 1));
  for (let x = 0; x < 3; x++) for (let y = 0; y < 2; y++) for (let z = 0; z < (shape[2] ?? 1); z++) {
    const index = shape.length === 3 ? (x * 2 + y) * nz + z : x * 2 + y;
    data[index] = missing.has(step) ? NaN : field === "u" ? 2 : field === "v" ? -3
      : field === "cloud_condensate" ? (1 + x + y + z + step) / 65536
      : field === "p" ? 100000 - z * 500 + x * 100 + y * 10 + step
      : field === "theta_prime" ? z / 4 + x * 2 - y - 12 + step
      : x * 2 * nz + y * nz + (shape.length === 3 ? z + 1 : nz) + step * 10 + (offsets.get(step) ?? 0);
    if (samples.has(`${step}:${field}:${index}`)) data[index] = samples.get(`${step}:${field}:${index}`)!;
  }
  return { shape, data };
}

export const regularFixture = {
  readOwnership: () => ({ pending: [...pendingReads].map(([requestId, value]) => ({ requestId, field: value.field, step: value.step })), cancelledReadIds: [...cancelledReadIds] }),
  releaseReads: () => {
    const pending = [...pendingReads.values()];
    pendingReads.clear();
    for (const read of pending) read.resolve();
  },
  revise: (index: number, offset: number) => { offsets.set(index, offset); versions.set(index, (versions.get(index) ?? 0) + 1); },
  setMissing: (index: number, value: boolean) => { if (value) missing.add(index); else missing.delete(index); versions.set(index, (versions.get(index) ?? 0) + 1); },
  setUnit: (index: number, unit: string) => { units.set(index, unit); versions.set(index, (versions.get(index) ?? 0) + 1); },
  setDelay: (field: string, milliseconds: number) => { delays.set(field, milliseconds); },
  setSample: (index: number, field: string, cell: number, value: number) => { samples.set(`${index}:${field}:${cell}`, value); versions.set(index, (versions.get(index) ?? 0) + 1); },
  setComponentUnit: (index: number, field: string, unit: string) => { componentUnits.set(`${index}:${field}`, unit); versions.set(index, (versions.get(index) ?? 0) + 1); },
};

function reduction(values: readonly number[], threshold: number | null) {
  const finite = values.filter(Number.isFinite);
  const sum = finite.reduce((total, value) => total + value, 0);
  return { min: finite.length ? Math.min(...finite) : null, max: finite.length ? Math.max(...finite) : null,
    mean: finite.length ? sum / finite.length : null, rms: finite.length ? Math.sqrt(finite.reduce((total, value) => total + value * value, 0) / finite.length) : null,
    finite: finite.length, total: values.length, countAbove: finite.filter(value => threshold != null && value > threshold).length,
    countBelow: finite.filter(value => threshold != null && value < threshold).length,
    countAbsAbove: finite.filter(value => threshold != null && Math.abs(value) > threshold).length };
}

export async function regularInvoke(command: string, payload: unknown): Promise<{ value: unknown } | null> {
  const args = (payload ?? {}) as Record<string, unknown>;
  if (command === "cancel_field_read") {
    const requestId = String(args.requestId);
    const pending = pendingReads.get(requestId);
    if (!pending) return null;
    pendingReads.delete(requestId);
    cancelledReadIds.push(requestId);
    pending.reject(new Error("Analytical fixture field read cancelled"));
    return { value: { requestId, found: true, cancellationRequested: true, phase: "read", interruptibility: "cooperative" } };
  }
  if (args.runDir !== REGULAR_RUN) return null;
  if (command === "list_nested_history_levels") return { value: [] };
  if (!["reduce_field", "reduce_field_axis", "read_field_bytes", "read_field_window"].includes(command)) return null;
  const step = Number(args.stepIndex), field = String(args.fieldName);
  const call: (typeof regularCalls)[number] = { command, step, field };
  regularCalls.push(call);
  if (field === failedField && step === 0 && command === "read_field_window") {
    throw new Error("Analytical fixture primary field read failed");
  }
  if (heldFutureReads && field === "w" && step > 0 && command === "read_field_window") {
    const requestId = String((args.request as { requestId: string }).requestId);
    await new Promise<void>((resolve, reject) => pendingReads.set(requestId, { field, step, resolve, reject }));
  }
  const source = regularSource(step, field);
  const delay = delays.get(field) ?? 0;
  if (delay > 0) await new Promise(resolve => setTimeout(resolve, delay));
  if (command === "read_field_bytes") return { value: encodeNpy(source.data, source.shape).buffer };
  if (command === "read_field_window") {
    const request = args.request as { requestId: string; origin: number[]; extent: number[]; stride: number[];
      aggregation: Parameters<typeof encodeFieldWindow>[5]; maxOutputBytes: number; maxSourceBytes: number };
    call.request = {
      origin: [...request.origin], extent: [...request.extent], stride: [...request.stride],
    };
    return { value: encodeFieldWindow(request.requestId, source, request.origin, request.extent, request.stride,
      request.aggregation, request.maxOutputBytes, request.maxSourceBytes, false, undefined, regularGrid) };
  }
  if (command === "reduce_field") return { value: reduction([...source.data], args.threshold as number | null) };
  const axis = Number(args.axis);
  const values = Array.from({ length: source.shape[axis] }, (_, level) => {
    const slab: number[] = [];
    for (let x = 0; x < 3; x++) for (let y = 0; y < 2; y++) for (let z = 0; z < nz; z++) {
      if ([x, y, z][axis] === level) slab.push(source.data[(x * 2 + y) * nz + z]);
    }
    return reduction(slab, null)[args.reducer as "max" | "min" | "mean" | "rms"];
  });
  return { value: { values, axis, axisLen: values.length } };
}
