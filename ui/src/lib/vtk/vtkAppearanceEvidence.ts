export const VTK_APPEARANCE_EVIDENCE_SCHEMA = "nimbus.studio.vtk-appearance" as const;
export const VTK_APPEARANCE_EVIDENCE_VERSION = 1 as const;

export type VtkAppearancePlot =
  | "volume"
  | "isosurface"
  | "streamlines"
  | "supercell"
  | "trajectories";

export type VtkAppearanceValue =
  | boolean
  | number
  | string
  | null
  | readonly VtkAppearanceValue[]
  | { readonly [key: string]: VtkAppearanceValue };

export interface VtkAppearanceDescriptor {
  plot: VtkAppearancePlot;
  backend: "webgl" | "webgpu";
  datasetId?: string | null;
  frameIndex?: number | null;
  field?: string | null;
  fidelityMode?: "scientific" | "presentation";
  parameters?: Readonly<Record<string, VtkAppearanceValue>>;
}

export interface VtkAppearanceCameraEvidence {
  position: [number, number, number] | null;
  focalPoint: [number, number, number] | null;
  viewUp: [number, number, number] | null;
  clippingRange: [number, number] | null;
  viewAngle: number | null;
  parallelProjection: boolean | null;
  parallelScale: number | null;
}

export interface VtkAppearanceLayerEvidence {
  kind: "actor" | "volume";
  index: number;
  visible: boolean | null;
  opacity: number | null;
  pointCount: number | null;
  cellCount: number | null;
  scalarVisibility: boolean | null;
  sampleDistance: number | null;
  maximumSamplesPerRay: number | null;
  shade: boolean | null;
  ambient: number | null;
  diffuse: number | null;
  specular: number | null;
}

export interface VtkAppearanceEvidence {
  schema: typeof VTK_APPEARANCE_EVIDENCE_SCHEMA;
  schemaVersion: typeof VTK_APPEARANCE_EVIDENCE_VERSION;
  descriptor: VtkAppearanceDescriptor;
  viewport: {
    cssWidth: number;
    cssHeight: number;
    backingWidth: number;
    backingHeight: number;
    pixelRatio: number;
  };
  camera: VtkAppearanceCameraEvidence;
  background: [number, number, number] | null;
  layers: readonly VtkAppearanceLayerEvidence[];
  metadataFingerprint: string;
}

interface CallableRecord {
  [key: string]: unknown;
}

export interface VtkAppearanceRenderWindowLike {
  getRenderer: () => unknown;
  getApiSpecificRenderWindow: () => {
    captureNextImage?: (format: string) => Promise<string> | string | undefined;
    [key: string]: unknown;
  };
  getRenderWindow: () => { render: () => void };
}

function objectRecord(value: unknown): CallableRecord | null {
  return value != null && typeof value === "object" ? value as CallableRecord : null;
}

function call(record: unknown, name: string): unknown {
  const object = objectRecord(record);
  const method = object?.[name];
  if (typeof method !== "function") return null;
  try {
    return method.call(record);
  } catch {
    return null;
  }
}

function finite(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? normalizeNumber(value) : null;
}

function boolean(value: unknown): boolean | null {
  return typeof value === "boolean" ? value : null;
}

function tuple2(value: unknown): [number, number] | null {
  if (!Array.isArray(value) && !ArrayBuffer.isView(value)) return null;
  const list = Array.from(value as ArrayLike<unknown>);
  const x = finite(list[0]);
  const y = finite(list[1]);
  return x == null || y == null ? null : [x, y];
}

function tuple3(value: unknown): [number, number, number] | null {
  if (!Array.isArray(value) && !ArrayBuffer.isView(value)) return null;
  const list = Array.from(value as ArrayLike<unknown>);
  const x = finite(list[0]);
  const y = finite(list[1]);
  const z = finite(list[2]);
  return x == null || y == null || z == null ? null : [x, y, z];
}

function normalizeNumber(value: number): number {
  if (Object.is(value, -0)) return 0;
  return Number(value.toPrecision(12));
}

function dataSetFor(layer: unknown): unknown {
  const mapper = call(layer, "getMapper");
  return call(mapper, "getInputData") ?? call(mapper, "getInputDataSet");
}

function pointCount(data: unknown): number | null {
  const direct = finite(call(data, "getNumberOfPoints"));
  if (direct != null) return direct;
  return finite(call(call(data, "getPoints"), "getNumberOfPoints"));
}

function cellCount(data: unknown): number | null {
  const direct = finite(call(data, "getNumberOfCells"));
  if (direct != null) return direct;
  const counts = ["getPolys", "getLines", "getStrips", "getVerts"]
    .map((name) => finite(call(call(data, name), "getNumberOfCells")))
    .filter((value): value is number => value != null);
  return counts.length > 0 ? counts.reduce((sum, value) => sum + value, 0) : null;
}

function layerEvidence(
  kind: VtkAppearanceLayerEvidence["kind"],
  layer: unknown,
  index: number,
): VtkAppearanceLayerEvidence {
  const mapper = call(layer, "getMapper");
  const property = call(layer, "getProperty");
  const data = dataSetFor(layer);
  return {
    kind,
    index,
    visible: boolean(call(layer, "getVisibility")),
    opacity: finite(call(property, "getOpacity")),
    pointCount: pointCount(data),
    cellCount: cellCount(data),
    scalarVisibility: boolean(call(mapper, "getScalarVisibility")),
    sampleDistance: kind === "volume" ? finite(call(mapper, "getSampleDistance")) : null,
    maximumSamplesPerRay: kind === "volume"
      ? finite(call(mapper, "getMaximumSamplesPerRay"))
      : null,
    shade: boolean(call(property, "getShade")),
    ambient: finite(call(property, "getAmbient")),
    diffuse: finite(call(property, "getDiffuse")),
    specular: finite(call(property, "getSpecular")),
  };
}

function arrayResult(value: unknown): readonly unknown[] {
  return Array.isArray(value) ? value : [];
}

function canvasFor(apiWindow: unknown): HTMLCanvasElement | null {
  const direct = call(apiWindow, "getCanvas");
  if (direct instanceof HTMLCanvasElement) return direct;
  const container = call(apiWindow, "getContainer");
  return container instanceof HTMLElement ? container.querySelector("canvas") : null;
}

function cameraEvidence(renderer: unknown): VtkAppearanceCameraEvidence {
  const camera = call(renderer, "getActiveCamera");
  return {
    position: tuple3(call(camera, "getPosition")),
    focalPoint: tuple3(call(camera, "getFocalPoint")),
    viewUp: tuple3(call(camera, "getViewUp")),
    clippingRange: tuple2(call(camera, "getClippingRange")),
    viewAngle: finite(call(camera, "getViewAngle")),
    parallelProjection: boolean(call(camera, "getParallelProjection")),
    parallelScale: finite(call(camera, "getParallelScale")),
  };
}

function normalizeValue(value: VtkAppearanceValue): VtkAppearanceValue {
  if (typeof value === "number") return Number.isFinite(value) ? normalizeNumber(value) : null;
  if (Array.isArray(value)) return value.map(normalizeValue);
  if (value != null && typeof value === "object") {
    return Object.fromEntries(Object.entries(value)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([key, item]) => [key, normalizeValue(item)]));
  }
  return value;
}

export function canonicalAppearanceJson(value: VtkAppearanceValue): string {
  return JSON.stringify(normalizeValue(value));
}

export function appearanceFingerprint(value: VtkAppearanceValue): string {
  let hash = 0xcbf29ce484222325n;
  for (const byte of new TextEncoder().encode(canonicalAppearanceJson(value))) {
    hash ^= BigInt(byte);
    hash = BigInt.asUintN(64, hash * 0x100000001b3n);
  }
  return `fnv1a64:${hash.toString(16).padStart(16, "0")}`;
}

export function collectVtkAppearanceEvidence(
  renderWindow: VtkAppearanceRenderWindowLike,
  descriptor: VtkAppearanceDescriptor,
): VtkAppearanceEvidence {
  const renderer = renderWindow.getRenderer();
  const apiWindow = renderWindow.getApiSpecificRenderWindow();
  const canvas = canvasFor(apiWindow);
  const rect = canvas?.getBoundingClientRect();
  const backingWidth = canvas?.width ?? 0;
  const backingHeight = canvas?.height ?? 0;
  const cssWidth = normalizeNumber(rect?.width ?? 0);
  const cssHeight = normalizeNumber(rect?.height ?? 0);
  const horizontalRatio = cssWidth > 0 ? backingWidth / cssWidth : 0;
  const verticalRatio = cssHeight > 0 ? backingHeight / cssHeight : 0;
  const pixelRatio = horizontalRatio > 0 && verticalRatio > 0
    ? normalizeNumber(Math.min(horizontalRatio, verticalRatio))
    : 0;
  const actors = arrayResult(call(renderer, "getActors"));
  const volumes = arrayResult(call(renderer, "getVolumes"));
  const unsigned = {
    schema: VTK_APPEARANCE_EVIDENCE_SCHEMA,
    schemaVersion: VTK_APPEARANCE_EVIDENCE_VERSION,
    descriptor: normalizeValue(descriptor as unknown as VtkAppearanceValue) as unknown as VtkAppearanceDescriptor,
    viewport: { cssWidth, cssHeight, backingWidth, backingHeight, pixelRatio },
    camera: cameraEvidence(renderer),
    background: tuple3(call(renderer, "getBackground")),
    layers: [
      ...actors.map((actor, index) => layerEvidence("actor", actor, index)),
      ...volumes.map((volume, index) => layerEvidence("volume", volume, index)),
    ],
  };
  return {
    ...unsigned,
    metadataFingerprint: appearanceFingerprint(unsigned as unknown as VtkAppearanceValue),
  };
}
