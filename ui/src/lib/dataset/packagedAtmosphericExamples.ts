import {
  createLocalDatasetSource,
  type LocalDatasetSource,
} from "./datasetSources";
import { validateFieldShape, type NpyN } from "@/lib/platform/npy";
import type { FieldReduction, FieldReducer, GridInfo, StepInfo } from "@/lib/run/runOutputs";
import type { FieldWindowAggregation } from "./nativeFieldWindow";

/**
 * Simulator-free examples are immutable, bundled analysis sources. They are
 * deliberately identified by a URI rather than a fake filesystem path so a
 * saved project can never imply that the source is a user-owned run directory.
 */
export const PACKAGED_EXAMPLE_URI_PREFIX = "nimbus-example://" as const;

export interface PackagedAtmosphericExampleManifest {
  schema: "nimbus.studio.packaged-atmospheric-example";
  schemaVersion: 1;
  id: string;
  uri: `${typeof PACKAGED_EXAMPLE_URI_PREFIX}${string}`;
  title: string;
  shortTitle: string;
  summary: string;
  dataClass: "analytical_manufactured_atmosphere";
  provenance: {
    creator: string;
    generator: string;
    generatorVersion: string;
    externalData: false;
    license: "MIT";
    sourceNotice: string;
    scientificUseNotice: string;
  };
  grid: Required<Pick<GridInfo,
    "nr" | "nth" | "nz" | "coordinateSystem" | "geometryKind" | "staggering"
    | "drM" | "dyM" | "dzM" | "scalarOriginM" | "physicalBoundsM"
  >>;
  frames: readonly { index: number; simulationTimeS: number; stormCenterXM: number }[];
  fields: Readonly<Record<string, { unit: string; description: string }>>;
  recommendedView: {
    fieldName: string;
    category: "plan-views";
    plotType: "slice";
  };
}

const MANUFACTURED_SUPERCELL: PackagedAtmosphericExampleManifest = Object.freeze({
  schema: "nimbus.studio.packaged-atmospheric-example" as const,
  schemaVersion: 1 as const,
  id: "manufactured-supercell-v1",
  uri: "nimbus-example://manufactured-supercell-v1" as const,
  title: "Manufactured supercell anatomy",
  shortTitle: "Manufactured supercell",
  summary: "A compact, deterministic storm-like atmosphere for learning Nimbus Studio without installing or running the simulator.",
  dataClass: "analytical_manufactured_atmosphere" as const,
  provenance: Object.freeze({
    creator: "Nimbus Studio contributors",
    generator: "packagedAtmosphericExamples.ts analytic field evaluator",
    generatorVersion: "1.0.0",
    externalData: false as const,
    license: "MIT" as const,
    sourceNotice: "Generated from analytic Gaussian, shear, and circulation functions shipped in the Nimbus source tree; no external observations or model output are included.",
    scientificUseNotice: "Designed for interface learning and renderer qualification. It is not a numerical weather simulation and must not be used for scientific validation.",
  }),
  grid: Object.freeze({
    nr: 36,
    nth: 32,
    nz: 20,
    coordinateSystem: "cartesian",
    geometryKind: "physical" as const,
    staggering: "cell_centered",
    drM: 500,
    dyM: 500,
    dzM: 500,
    scalarOriginM: [250, 250, 250] as [number, number, number],
    physicalBoundsM: Object.freeze({
      lower: [0, 0, 0] as [number, number, number],
      upper: [18_000, 16_000, 10_000] as [number, number, number],
    }),
  }),
  frames: Object.freeze(Array.from({ length: 8 }, (_, index) => Object.freeze({
    index,
    simulationTimeS: index * 300,
    stormCenterXM: 7_000 + index * 350,
  }))),
  fields: Object.freeze({
    reflectivity_dbz: Object.freeze({ unit: "dBZ", description: "Manufactured hook-like reflectivity envelope" }),
    w: Object.freeze({ unit: "m s-1", description: "Manufactured updraft and compensating downdraft" }),
    u: Object.freeze({ unit: "m s-1", description: "Background shear plus storm-relative circulation" }),
    v: Object.freeze({ unit: "m s-1", description: "Background flow plus storm-relative circulation" }),
    vorticity_z: Object.freeze({ unit: "s-1", description: "Manufactured low-level vertical-vorticity couplet" }),
    theta_prime: Object.freeze({ unit: "K", description: "Manufactured cold-pool and warm-updraft perturbation" }),
    p_prime: Object.freeze({ unit: "Pa", description: "Manufactured mesolow and perturbation-pressure couplet" }),
    qv: Object.freeze({ unit: "kg kg-1", description: "Idealized water-vapor mixing ratio" }),
    theta: Object.freeze({ unit: "K", description: "Idealized potential-temperature profile plus perturbation" }),
    p: Object.freeze({ unit: "Pa", description: "Hydrostatic reference pressure plus perturbation" }),
  }),
  recommendedView: Object.freeze({
    fieldName: "reflectivity_dbz",
    category: "plan-views" as const,
    plotType: "slice" as const,
  }),
});

export const PACKAGED_ATMOSPHERIC_EXAMPLES: readonly PackagedAtmosphericExampleManifest[] =
  Object.freeze([MANUFACTURED_SUPERCELL]);

export function isPackagedAtmosphericExampleUri(uri: string): boolean {
  return uri.startsWith(PACKAGED_EXAMPLE_URI_PREFIX);
}

export function packagedAtmosphericExample(uriOrId: string): PackagedAtmosphericExampleManifest | null {
  return PACKAGED_ATMOSPHERIC_EXAMPLES.find(({ uri, id }) => uri === uriOrId || id === uriOrId) ?? null;
}

export function createPackagedAtmosphericExampleSource(
  uriOrId: string = MANUFACTURED_SUPERCELL.id,
): LocalDatasetSource {
  const example = packagedAtmosphericExample(uriOrId);
  if (!example) throw new Error(`Unknown packaged atmospheric example '${uriOrId}'`);
  return {
    ...createLocalDatasetSource("nimbus_run", example.uri),
    displayName: `${example.shortTitle} · included example`,
  };
}

function generation(example: PackagedAtmosphericExampleManifest, stepIndex: number): string {
  return `packaged:${example.id}:${example.provenance.generatorVersion}:frame-${stepIndex}`;
}

export function listPackagedAtmosphericExampleSteps(uri: string): StepInfo[] {
  const example = packagedAtmosphericExample(uri);
  if (!example) throw new Error(`Unknown packaged atmospheric example '${uri}'`);
  const fields = Object.keys(example.fields);
  const fieldUnits = Object.fromEntries(Object.entries(example.fields).map(([id, field]) => [id, field.unit]));
  return example.frames.map((frame) => ({
    index: frame.index,
    simulationTimeS: frame.simulationTimeS,
    fields: [...fields],
    grid: { ...example.grid },
    fieldUnits,
    fieldOrigins: Object.fromEntries(fields.map((field) => [field, "native" as const])),
    publicationState: "complete",
    publicationGeneration: generation(example, frame.index),
  }));
}

interface PackagedFieldWindow {
  aggregation?: FieldWindowAggregation;
  origin: readonly number[];
  extent: readonly number[];
  stride: readonly number[];
  maxOutputBytes?: number;
  maxSourceBytes?: number;
}

function validatedSelection(
  sourceShape: readonly number[],
  window?: PackagedFieldWindow,
): { origin: number[]; extent: number[]; stride: number[]; outputShape: number[] } {
  if (!window) {
    return {
      origin: sourceShape.map(() => 0),
      extent: [...sourceShape],
      stride: sourceShape.map(() => 1),
      outputShape: [...sourceShape],
    };
  }
  if (window.origin.length !== 3 || window.extent.length !== 3 || window.stride.length !== 3) {
    throw new Error("Packaged-example field windows require three origin, extent, and stride values");
  }
  const origin = [...window.origin];
  const extent = [...window.extent];
  const stride = [...window.stride];
  const outputShape = sourceShape.map((length, axis) => {
    const start = origin[axis];
    const count = extent[axis];
    const step = stride[axis];
    if (!Number.isSafeInteger(start) || start < 0 || !Number.isSafeInteger(count) || count <= 0
      || !Number.isSafeInteger(step) || step <= 0) {
      throw new Error(`Packaged-example window axis ${axis} is invalid`);
    }
    const exclusiveEnd = start + count;
    if (!Number.isSafeInteger(exclusiveEnd) || exclusiveEnd > length) {
      throw new Error(`Packaged-example window axis ${axis} exceeds source shape ${sourceShape.join("×")}`);
    }
    return Math.floor((count - 1) / step) + 1;
  });
  const { decodedBytes } = validateFieldShape(outputShape);
  if (window.maxOutputBytes != null
    && (!Number.isSafeInteger(window.maxOutputBytes) || window.maxOutputBytes <= 0 || decodedBytes > window.maxOutputBytes)) {
    throw new Error(`Packaged-example window requires ${decodedBytes} bytes; request budget is ${window.maxOutputBytes ?? 0}`);
  }
  const sourceBytes = extent.reduce((product, count) => product * count, 1) * Float32Array.BYTES_PER_ELEMENT;
  if (!Number.isSafeInteger(sourceBytes)
    || (window.maxSourceBytes != null
      && (!Number.isSafeInteger(window.maxSourceBytes) || window.maxSourceBytes <= 0 || sourceBytes > window.maxSourceBytes))) {
    throw new Error(`Packaged-example window requires ${sourceBytes} source bytes; request budget is ${window.maxSourceBytes ?? 0}`);
  }
  return { origin, extent, stride, outputShape };
}

function analyticField(
  fieldName: string,
  xM: number,
  yM: number,
  zM: number,
  stormCenterXM: number,
  frameIndex: number,
): number {
  const cx = stormCenterXM;
  const cy = 8_400;
  const dx = (xM - cx) / 3_000;
  const dy = (yM - cy) / 2_700;
  const dz = (zM - 4_500) / 3_600;
  const core = Math.exp(-(dx * dx + dy * dy + dz * dz));
  const lowLevel = Math.exp(-(((zM - 1_100) / 2_200) ** 2));
  const rotation = Math.exp(-(dx * dx + dy * dy) * 1.45) * lowLevel;
  const rearFlank = Math.exp(-(((xM - (cx - 2_500)) / 2_700) ** 2 + ((yM - (cy - 1_900)) / 2_000) ** 2))
    * Math.exp(-((zM / 2_200) ** 2));
  const forwardFlank = Math.exp(-(((xM - (cx + 2_300)) / 4_300) ** 2 + ((yM - (cy + 1_100)) / 2_800) ** 2))
    * Math.exp(-(((zM - 2_500) / 3_200) ** 2));
  const thetaPrime = 1.4 * core - 7.5 * rearFlank - 2.2 * forwardFlank;
  switch (fieldName) {
    case "reflectivity_dbz": {
      const tower = Math.exp(-(dx * dx * 0.75 + dy * dy + ((zM - 4_000) / 4_600) ** 2));
      const hook = Math.exp(-(((xM - (cx - 1_800)) / 2_700) ** 2 + ((yM - (cy - 2_000)) / 1_250) ** 2)) * lowLevel;
      return Math.max(-10, 8 + 53 * Math.max(tower, 0.92 * forwardFlank, 0.88 * hook));
    }
    case "w": {
      const downdraft = Math.exp(-(((xM - (cx + 2_700)) / 2_100) ** 2 + ((yM - (cy + 600)) / 2_000) ** 2 + ((zM - 3_000) / 3_000) ** 2));
      return 36 * core - 17 * downdraft;
    }
    case "u":
      return 4 + zM * 0.00145 - 17 * dy * rotation;
    case "v":
      return 9 + zM * 0.00055 + 17 * dx * rotation + frameIndex * 0.15;
    case "vorticity_z": {
      const anticyclonic = Math.exp(-(((xM - (cx + 3_300)) / 2_300) ** 2 + ((yM - (cy + 600)) / 2_000) ** 2)) * lowLevel;
      return 0.019 * rotation - 0.008 * anticyclonic;
    }
    case "theta_prime":
      return thetaPrime;
    case "p_prime":
      return -760 * rotation + 310 * rearFlank - 190 * core;
    case "qv":
      return Math.max(0.001, 0.0145 * Math.exp(-zM / 2_800) + 0.0017 * core - 0.0008 * rearFlank);
    case "theta":
      return 300 + zM * 0.004 + thetaPrime;
    case "p":
      return 100_000 * Math.exp(-zM / 8_200) - 760 * rotation + 310 * rearFlank - 190 * core;
    default:
      throw new Error(`Field '${fieldName}' is not included in the packaged example`);
  }
}

export function readPackagedAtmosphericExampleField(
  uri: string,
  stepIndex: number,
  fieldName: string,
  expectedGeneration: string | null,
  window?: PackagedFieldWindow,
): NpyN {
  const example = packagedAtmosphericExample(uri);
  if (!example) throw new Error(`Unknown packaged atmospheric example '${uri}'`);
  const frame = example.frames.find(({ index }) => index === stepIndex);
  if (!frame) throw new Error(`Packaged example has no frame ${stepIndex}`);
  if (!(fieldName in example.fields)) throw new Error(`Field '${fieldName}' is not included in '${example.title}'`);
  const actualGeneration = generation(example, stepIndex);
  if (expectedGeneration !== null && expectedGeneration !== actualGeneration) {
    throw new Error(`Packaged example generation changed: expected '${expectedGeneration}', found '${actualGeneration}'`);
  }
  const sourceShape = [example.grid.nr, example.grid.nth, example.grid.nz];
  const selection = validatedSelection(sourceShape, window);
  const aggregation = window?.aggregation ?? "cell_mean";
  if (!["cell_mean", "linear_reflectivity_mean", "block_max", "block_min", "magnitude_extreme", "point_sample"].includes(aggregation)) {
    throw new Error(`Unsupported packaged-example window aggregation: ${aggregation}`);
  }
  const { elementCount } = validateFieldShape(selection.outputShape);
  const data = new Float32Array(elementCount);
  let target = 0;
  for (let ox = 0; ox < selection.outputShape[0]; ox += 1) {
    const x = selection.origin[0] + ox * selection.stride[0];
    for (let oy = 0; oy < selection.outputShape[1]; oy += 1) {
      const y = selection.origin[1] + oy * selection.stride[1];
      for (let oz = 0; oz < selection.outputShape[2]; oz += 1) {
        const z = selection.origin[2] + oz * selection.stride[2];
        const xEnd = Math.min(selection.origin[0] + selection.extent[0], x + (aggregation === "point_sample" ? 1 : selection.stride[0]));
        const yEnd = Math.min(selection.origin[1] + selection.extent[1], y + (aggregation === "point_sample" ? 1 : selection.stride[1]));
        const zEnd = Math.min(selection.origin[2] + selection.extent[2], z + (aggregation === "point_sample" ? 1 : selection.stride[2]));
        let sum = 0;
        let count = 0;
        let invalid = false;
        for (let sx = x; sx < xEnd; sx += 1) {
          for (let sy = y; sy < yEnd; sy += 1) {
            for (let sz = z; sz < zEnd; sz += 1) {
              const value = Math.fround(analyticField(
                fieldName,
                example.grid.scalarOriginM[0] + sx * example.grid.drM,
                example.grid.scalarOriginM[1] + sy * example.grid.dyM,
                example.grid.scalarOriginM[2] + sz * example.grid.dzM,
                frame.stormCenterXM,
                frame.index,
              ));
              if (!Number.isFinite(value)) invalid = true;
              switch (aggregation) {
                case "cell_mean": sum += value; break;
                case "linear_reflectivity_mean": sum += 10 ** (value / 10); break;
                case "block_max": if (count === 0 || value > sum) sum = value; break;
                case "block_min": if (count === 0 || value < sum) sum = value; break;
                case "magnitude_extreme": if (count === 0 || Math.abs(value) > Math.abs(sum)) sum = value; break;
                case "point_sample": sum = value; break;
              }
              count += 1;
            }
          }
        }
        data[target++] = invalid ? NaN
          : aggregation === "cell_mean" ? sum / count
          : aggregation === "linear_reflectivity_mean" ? 10 * Math.log10(sum / count)
          : sum;
      }
    }
  }
  return {
    shape: selection.outputShape,
    data,
    ...(window ? { windowGeometry: {
      aggregation,
      weighting: "uniform_cell_volume" as const,
      sampleLocation: aggregation === "point_sample" ? "source_cell_center" as const : "block_center" as const,
      centroidsM: null,
      sourceShape,
      origin: selection.origin,
      extent: selection.extent,
      stride: selection.stride,
    } } : {}),
  };
}

export function reducePackagedAtmosphericExampleField(
  uri: string,
  stepIndex: number,
  fieldName: string,
  expectedGeneration: string | null,
  threshold?: number,
): FieldReduction {
  const { data } = readPackagedAtmosphericExampleField(uri, stepIndex, fieldName, expectedGeneration);
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let sumSquares = 0;
  let finite = 0;
  let countAbove = 0;
  let countBelow = 0;
  let countAbsAbove = 0;
  for (const value of data) {
    if (!Number.isFinite(value)) continue;
    min = Math.min(min, value);
    max = Math.max(max, value);
    sum += value;
    sumSquares += value * value;
    finite += 1;
    if (threshold != null) {
      if (value > threshold) countAbove += 1;
      if (value < threshold) countBelow += 1;
      if (Math.abs(value) > threshold) countAbsAbove += 1;
    }
  }
  return {
    min: finite ? min : Number.NaN,
    max: finite ? max : Number.NaN,
    mean: finite ? sum / finite : Number.NaN,
    rms: finite ? Math.sqrt(sumSquares / finite) : Number.NaN,
    finite,
    total: data.length,
    countAbove,
    countBelow,
    countAbsAbove,
  };
}

export function reducePackagedAtmosphericExampleFieldAxis(
  uri: string,
  stepIndex: number,
  fieldName: string,
  axis: 0 | 1 | 2,
  reducer: FieldReducer,
  expectedGeneration: string | null,
): number[] {
  const field = readPackagedAtmosphericExampleField(uri, stepIndex, fieldName, expectedGeneration);
  const [nx, ny, nz] = field.shape;
  const length = field.shape[axis];
  const values = Array.from({ length }, () => reducer === "min" ? Infinity : reducer === "max" ? -Infinity : 0);
  const counts = new Uint32Array(length);
  for (let x = 0; x < nx; x += 1) for (let y = 0; y < ny; y += 1) for (let z = 0; z < nz; z += 1) {
    const value = field.data[(x * ny + y) * nz + z];
    if (!Number.isFinite(value)) continue;
    const coordinate = axis === 0 ? x : axis === 1 ? y : z;
    if (reducer === "min") values[coordinate] = Math.min(values[coordinate], value);
    else if (reducer === "max") values[coordinate] = Math.max(values[coordinate], value);
    else values[coordinate] += reducer === "rms" ? value * value : value;
    counts[coordinate] += 1;
  }
  return values.map((value, index) => {
    if (!counts[index]) return Number.NaN;
    if (reducer === "mean") return value / counts[index];
    if (reducer === "rms") return Math.sqrt(value / counts[index]);
    return value;
  });
}
