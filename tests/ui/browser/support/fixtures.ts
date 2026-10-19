import { mockIPC } from "@tauri-apps/api/mocks";
import type { InvokeArgs } from "@tauri-apps/api/core";
import type { NpyN } from "@nimbus/lib/platform/npy";
import type { GridInfo, StepInfo } from "@nimbus/lib/run/runOutputs";
import type { FieldBrickHierarchy } from "@nimbus/lib/plot/fieldBrickHierarchy";
import { MOCK_SCHEME_CATALOG } from "@nimbus/lib/config/schemeCatalog";
import { publishNativeIoBudgetLimit } from "@nimbus/lib/core/nativeWindowBudget";
import {
  ANALYTIC_FIDELITY_FIELD_NAMES,
  createAnalyticFidelityFixture,
  isAnalyticFidelityFieldName,
} from "@nimbus/lib/vtk/analyticFidelityFixtures";

export const FIELD_SHAPE = [48, 36, 24] as const;
export const COARSE_FIELD_SHAPE = [11, 10, 7] as const;
export const CONTOUR_SHAPE = [96, 72, 48] as const;
const CONTEXT_READ_FAILURE_FIELD = typeof window === "undefined" ? null
  : new URLSearchParams(window.location.search).get("volumeContextReadFailure");
let contextReadFailureEnabled = true;
declare global {
  interface Window {
    __nimbusVolumeBudgetFixture?: {
      allowContextRead: () => void;
      publishIoLimit: (bytes: number) => void;
    };
  }
}
if (typeof window !== "undefined" && CONTEXT_READ_FAILURE_FIELD) {
  window.__nimbusVolumeBudgetFixture = {
    allowContextRead: () => { contextReadFailureEnabled = false; },
    publishIoLimit: publishNativeIoBudgetLimit,
  };
}
export const LARGE_FIELD_SHAPE: readonly [number, number, number] = CONTEXT_READ_FAILURE_FIELD
  ? [320, 320, 180] : [256, 256, 160];
export const BEAST_REFERENCE_RUN_DIR = "/__nimbus_browser_fixture__/beast-xl-reference";
export const FIELD_NAMES = [
  "w",
  "u",
  "v",
  "reflectivity_dbz",
  "buoyancy",
  "theta_prime",
  "cloud_condensate",
  "qr",
  "qg",
  "qh",
  "cloud_liquid_condensate",
  "cloud_frozen_condensate",
  "vorticity_r",
  "vorticity_theta",
  "vorticity_z",
  "okubo_weiss",
  // Exact min=max field used to qualify the 2D uniform-data disclosure.
  "uniform_scalar",
  /* Rank 2, unlike everything above it. The engine exports this as the column
   * maximum of reflectivity_dbz, and the radar floor reads it in preference to
   * reducing the column itself. Without it in the fixture the floor's rank-2
   * path — different vector rank, different response shape — is never executed
   * in a browser at all. */
  "composite_reflectivity",
  ...ANALYTIC_FIDELITY_FIELD_NAMES,
] as const;

const CM1_NATIVE_TO_FIXTURE_FIELD: Readonly<Record<string, string>> = {
  winterp: "w",
  uinterp: "u",
  vinterp: "v",
  dbz: "reflectivity_dbz",
  thpert: "theta_prime",
  prspert: "p_prime",
  th: "theta",
  prs: "p",
  qv: "qv",
  zvort: "vorticity_z",
};

const cm1Dimensions = [
  { name: "time", length: 3, unlimited: true },
  { name: "xh", length: FIELD_SHAPE[0], unlimited: false },
  { name: "yh", length: FIELD_SHAPE[1], unlimited: false },
  { name: "zh", length: FIELD_SHAPE[2], unlimited: false },
];

const cm1CoordinateVariable = (name: string, length: number, units: string) => ({
  name,
  dataType: "f32",
  dimensions: [name],
  shape: [length],
  units,
  longName: null,
});

const cm1VolumeVariable = (name: string, units: string) => ({
  name,
  dataType: "f32",
  dimensions: ["time", "xh", "yh", "zh"],
  shape: [3, ...FIELD_SHAPE],
  units,
  longName: null,
});

const cm1MetadataFixture = {
  schema: "nimbus.studio.netcdf-metadata",
  schemaVersion: 1,
  path: "/__nimbus_browser_fixture__/cm1out.nc",
  containerFormat: "netCDF4/HDF5",
  recordCount: 3,
  dimensions: cm1Dimensions,
  variables: [
    cm1CoordinateVariable("time", 3, "s"),
    cm1CoordinateVariable("xh", FIELD_SHAPE[0], "m"),
    cm1CoordinateVariable("yh", FIELD_SHAPE[1], "m"),
    cm1CoordinateVariable("zh", FIELD_SHAPE[2], "m"),
    cm1VolumeVariable("winterp", "m/s"),
    cm1VolumeVariable("uinterp", "m/s"),
    cm1VolumeVariable("vinterp", "m/s"),
    cm1VolumeVariable("dbz", "dBZ"),
    cm1VolumeVariable("thpert", "K"),
    cm1VolumeVariable("prspert", "Pa"),
    cm1VolumeVariable("th", "K"),
    cm1VolumeVariable("prs", "Pa"),
    cm1VolumeVariable("qv", "kg/kg"),
    cm1VolumeVariable("zvort", "1/s"),
  ],
  sourceClassification: "cm1_likely",
  plottingSupported: true,
  plotUnavailableReason: "",
};

export interface IpcMetrics {
  startedJobs: number;
  completedJobs: number;
  inFlightJobs: number;
  maxInFlightJobs: number;
  inFlightBytes: number;
  maxInFlightBytes: number;
  deliveredBytes: number;
  failures: number;
  failureMessages: string[];
}

export interface FixtureControl {
  publishLiveThrough: (count: number) => void;
  setListDelay: (runDir: string, milliseconds: number) => void;
  /** Slow every field read, so a spec can observe a multi-frame read while it
   *  is still in flight (busy labels, cancel buttons, progress counters). */
  setFieldReadDelay: (milliseconds: number) => void;
  /** Put a file on the virtual disk, so a spec can say "this already exists"
   *  instead of relying on the reader inventing plausible contents. */
  writeFile: (path: string, contents: string) => void;
  /** Put raw bytes on the virtual disk without the config-command extension
   * allowlist, matching the scoped fs plugin used by Open data. */
  writeBinaryFile: (path: string, contents: readonly number[]) => void;
  /** What is on the virtual disk now, or null when nothing is. */
  readFile: (path: string) => string | null;
  /** The same, unmodified — for the export paths that write binary. */
  readFileBytes: (path: string) => Uint8Array | null;
  /** Choose what the next save dialog returns: a path, or null for Cancel.
   *  `undefined` restores the derived default. */
  setSaveDialogPath: (path: string | null | undefined) => void;
  /** Same, for the open dialog. Defaults to Cancel, since a spec that has not
   *  named a file has not chosen one. */
  setOpenDialogPath: (path: string | null | undefined) => void;
  snapshot: () => {
    liveStepCount: number;
    listCalls: Record<string, number>;
    ipc: IpcMetrics;
    savedGif: SavedGifMetrics;
    /** The virtual disk, path → contents. */
    files: Record<string, string>;
  };
}

export interface SavedGifMetrics {
  path: string | null;
  byteLength: number;
  signature: string;
  writes: number;
}

let liveStepCount = 3;
let fieldReadDelayMs = 0;
const listDelays = new Map<string, number>();
const listCalls = new Map<string, number>();
export const savedGifMetrics: SavedGifMetrics = {
  path: null,
  byteLength: 0,
  signature: "",
  writes: 0,
};

/* ---------- A filesystem the text commands can actually see ---------- */

/**
 * Where `plugins_dir` says the gallery lives.
 *
 * The real command resolves `$HOME/.nimbus/plugins` and creates it; the path
 * itself is not the contract, so a fixture-owned one keeps specs from depending
 * on whose machine they run on.
 */
export const FIXTURE_PLUGINS_DIR = "/__nimbus_browser_fixture__/plugins";

/**
 * Ported from `ALLOWED_CONFIG_EXTENSIONS` in src-tauri/src/commands/config.rs.
 *
 * Copied rather than approximated on purpose. A double that accepted anything
 * would let a spec certify a save path the shipped app rejects — the UI would
 * pass its whole test suite and then refuse the same file in production.
 */
const ALLOWED_CONFIG_EXTENSIONS = ["yaml", "yml", "json", "py"] as const;

/**
 * Bytes, not text.
 *
 * Two commands write here — `write_config` with a string and `plugin:fs|write_file`
 * with a buffer — and a disk that only held text could not represent what the
 * second one does. That is not academic: exporting a figure, a GIF or an .npy
 * all go through the binary path, so a text-only disk made every one of them
 * unobservable and their specs could only assert that the UI said it had
 * saved something.
 */
interface VirtualFile {
  bytes: Uint8Array;
  /** Monotonic, so `list_plugins`' newest-first order is deterministic. */
  modifiedMs: number;
}

/*
 * Backed by sessionStorage, so the disk survives a reload.
 *
 * A real filesystem does not forget when the page navigates, and a double that
 * did would make "reopened from disk after a restart" — the whole point of
 * storing paths rather than text in the recovery store — impossible to test.
 * sessionStorage rather than localStorage because the harness clears the
 * latter to give each spec the same starting workspace, and because Playwright
 * gives every test its own context so nothing leaks between them.
 */
const VIRTUAL_DISK_KEY = "__nimbus_fixture_disk__";

function loadVirtualDisk(): Map<string, VirtualFile> {
  try {
    const raw = window.sessionStorage.getItem(VIRTUAL_DISK_KEY);
    if (raw) {
      const stored = JSON.parse(raw) as Record<string, { bytes: number[]; modifiedMs: number }>;
      return new Map(Object.entries(stored).map(([path, file]) => [
        path,
        { bytes: Uint8Array.from(file.bytes), modifiedMs: file.modifiedMs },
      ]));
    }
  } catch {
    // A blocked or corrupt store is an empty disk, which is the initial state
    // anyway; a spec that needed a file will fail on the file, not on this.
  }
  return new Map();
}

function saveVirtualDisk(): void {
  try {
    window.sessionStorage.setItem(
      VIRTUAL_DISK_KEY,
      JSON.stringify(Object.fromEntries(
        [...virtualFiles].map(([path, file]) => [
          path,
          { bytes: [...file.bytes], modifiedMs: file.modifiedMs },
        ]),
      )),
    );
  } catch {
    // Best effort; the in-memory map is still authoritative for this page.
  }
}

const virtualFiles = loadVirtualDisk();
/* A fixed epoch plus a counter rather than Date.now(): two writes in the same
 * millisecond would tie, and the tie-break is by name, so the ordering a spec
 * asserts would depend on how fast the machine ran. Resumed past whatever the
 * restored disk already holds so a write after a reload still sorts newest. */
const FIXTURE_EPOCH_MS = 1_700_000_000_000;
let writeSequence = [...virtualFiles.values()]
  .reduce((highest, file) => Math.max(highest, file.modifiedMs - FIXTURE_EPOCH_MS), 0);

/** Rejection shaped like `AppError`'s wire form — `{ code, message }`, never an
 *  Error. `appInvoke` branches on `code`, so an Error here would be a different
 *  failure than the one production produces. */
function appError(code: string, message: string): { code: string; message: string } {
  return { code, message };
}

/** Ported from `validate_config_path`. Returns nothing; throws like Rust. */
function assertConfigPathAllowed(path: string): void {
  const trimmed = path.trim();
  if (trimmed.length === 0) throw appError("invalid", "invalid argument: empty path");
  const base = trimmed.split(/[/\\]/).pop() ?? "";
  const dot = base.lastIndexOf(".");
  const extension = dot > 0 ? base.slice(dot + 1).toLowerCase() : "";
  if (!ALLOWED_CONFIG_EXTENSIONS.includes(extension as typeof ALLOWED_CONFIG_EXTENSIONS[number])) {
    throw appError(
      "invalid",
      `invalid argument: refusing to access '${trimmed}': only `
      + `${ALLOWED_CONFIG_EXTENSIONS.join(", ")} files are permitted through this command`,
    );
  }
}

/** The text commands' write. Validated the way `write_config` validates. */
function writeVirtualFile(path: string, contents: string): void {
  assertConfigPathAllowed(path);
  writeVirtualBytes(path, new TextEncoder().encode(contents));
}

/**
 * The binary write, from `plugin:fs|write_file`.
 *
 * Deliberately NOT extension-checked: the fs plugin is scoped by capability
 * rather than by suffix, and it is what an export dialog's chosen path goes
 * through — including the .png, .gif and .npy that `write_config` would refuse.
 */
function writeVirtualBytes(path: string, bytes: Uint8Array): void {
  writeSequence += 1;
  virtualFiles.set(path.trim(), { bytes, modifiedMs: FIXTURE_EPOCH_MS + writeSequence });
  saveVirtualDisk();
}

/**
 * What the save dialog returns next, when a spec needs to choose.
 *
 * `undefined` means "derive one", which is what an unconfigured spec gets;
 * `null` is the user pressing Cancel, which is a distinct outcome the UI has
 * to leave the buffer dirty for.
 */
let nextSavePath: string | null | undefined;
let nextOpenPath: string | null | undefined;

/**
 * Where the next binary write lands.
 *
 * `plugin:fs|write_file` carries its path in HTTP-style headers rather than in
 * the payload, and `mockIPC` hands the callback only `(cmd, args)` — the third
 * argument, where those headers live, is dropped before it gets here. So the
 * destination has to come from the save dialog that immediately preceded the
 * write, which is how the flow works anyway: pick a path, then write to it.
 */
let lastChosenSavePath: string | null = null;

export const fixtureControl: FixtureControl = {
  publishLiveThrough: (count) => {
    liveStepCount = Math.max(0, Math.min(steps.length, Math.floor(count)));
  },
  setListDelay: (runDir, milliseconds) => {
    listDelays.set(runDir, Math.max(0, milliseconds));
  },
  setFieldReadDelay: (milliseconds) => {
    fieldReadDelayMs = Math.max(0, milliseconds);
  },
  writeFile: (path, contents) => {
    writeVirtualFile(path, contents);
  },
  writeBinaryFile: (path, contents) => {
    writeVirtualBytes(path, Uint8Array.from(contents));
  },
  readFile: (path) => {
    const file = virtualFiles.get(path.trim());
    return file ? new TextDecoder().decode(file.bytes) : null;
  },
  readFileBytes: (path) => virtualFiles.get(path.trim())?.bytes ?? null,
  setSaveDialogPath: (path) => {
    nextSavePath = path;
  },
  setOpenDialogPath: (path) => {
    nextOpenPath = path;
  },
  snapshot: () => ({
    liveStepCount,
    listCalls: Object.fromEntries(listCalls),
    ipc: { ...ipcMetrics },
    savedGif: { ...savedGifMetrics },
    files: Object.fromEntries(
      [...virtualFiles].map(([path, file]) => [path, new TextDecoder().decode(file.bytes)]),
    ),
  }),
};

export const ipcMetrics: IpcMetrics = {
  startedJobs: 0,
  completedJobs: 0,
  inFlightJobs: 0,
  maxInFlightJobs: 0,
  inFlightBytes: 0,
  maxInFlightBytes: 0,
  deliveredBytes: 0,
  failures: 0,
  failureMessages: [],
};

export const grid: GridInfo = {
  nr: FIELD_SHAPE[0],
  nth: FIELD_SHAPE[1],
  nz: FIELD_SHAPE[2],
  coordinateSystem: "cartesian",
  geometryKind: "physical",
  staggering: "c_grid",
  drM: 125,
  dyM: 175,
  dthetaRad: 175,
  dzM: 90,
  scalarOriginM: [0, 0, 0],
  scalarCoordinatesM: [
    Array.from({ length: FIELD_SHAPE[0] }, (_, index) => index * 125),
    Array.from({ length: FIELD_SHAPE[1] }, (_, index) => index * 175),
    Array.from({ length: FIELD_SHAPE[2] }, (_, index) => index * 90),
  ],
  physicalBoundsM: {
    lower: [0, 0, 0],
    upper: [5_875, 6_125, 2_070],
  },
};

/** A deliberately non-flat physical mesh over the same horizontal lattice.
 * Model level k follows this surface, so consumers cannot honestly interpret
 * k as one separable height. Exact centroids/volumes are emitted by the window
 * fixture below, matching the native schema-v4 terrain contract. */
export const terrainGrid: GridInfo = {
  ...grid,
  terrainFollowing: true,
  horizontalScalarOriginM: [0, 0],
  scalarCoordinatesM: undefined,
  scalarOriginM: undefined,
  physicalBoundsM: {
    lower: [0, 0, 0],
    upper: [5_875, 6_125, 2_430],
  },
};

const stretchedZM = Array.from({ length: FIELD_SHAPE[2] }, (_, index) =>
  Math.round(35 * index + 4.2 * index * index));
export const stretchedGrid: GridInfo = {
  ...grid,
  dzM: undefined,
  scalarCoordinatesM: [
    [...grid.scalarCoordinatesM![0]],
    [...grid.scalarCoordinatesM![1]],
    stretchedZM,
  ],
  physicalBoundsM: {
    lower: [0, 0, stretchedZM[0] - (stretchedZM[1] - stretchedZM[0]) / 2],
    upper: [
      grid.physicalBoundsM!.upper[0],
      grid.physicalBoundsM!.upper[1],
      stretchedZM.at(-1)! + (stretchedZM.at(-1)! - stretchedZM.at(-2)!) / 2,
    ],
  },
};

const LARGE_PLANE_SHAPE = [2_300, 2_300] as const;
export const largePlaneGrid: GridInfo = {
  nr: LARGE_PLANE_SHAPE[0],
  nth: LARGE_PLANE_SHAPE[1],
  nz: 1,
  coordinateSystem: "cartesian",
  geometryKind: "physical",
  staggering: "c_grid",
  drM: 100,
  dyM: 100,
  scalarOriginM: [0, 0, 0],
  physicalBoundsM: {
    lower: [-50, -50, -50],
    upper: [229_950, 229_950, 50],
  },
};

/** Resample `grid` onto a different cell count over the same physical box.
 *  Every shape-dependent field is recomputed: spreading `grid` and patching
 *  only nr/nth/nz would leave `scalarCoordinatesM` sized for FIELD_SHAPE, and
 *  the consumers that index it without a length check would then read
 *  coordinates belonging to a different lattice. */
function resampledGrid(shape: readonly [number, number, number]): GridInfo {
  const spacing = grid.physicalBoundsM!.upper.map(
    (upper, axis) => upper / (shape[axis] - 1),
  ) as [number, number, number];
  return {
    ...grid,
    nr: shape[0],
    nth: shape[1],
    nz: shape[2],
    drM: spacing[0],
    dyM: spacing[1],
    dthetaRad: spacing[1],
    dzM: spacing[2],
    scalarCoordinatesM: spacing.map((step, axis) =>
      Array.from({ length: shape[axis] }, (_, index) => index * step),
    ) as [number[], number[], number[]],
    physicalBoundsM: grid.physicalBoundsM,
  };
}

/** Same synthetic storm sampled on a deliberately coarse physical lattice. */
export const coarseGrid: GridInfo = resampledGrid(COARSE_FIELD_SHAPE);

export const contourGrid: GridInfo = resampledGrid(CONTOUR_SHAPE);
if (typeof window !== "undefined" && new URLSearchParams(window.location.search).get("stretchedBricks") === "1") {
  const last = contourGrid.scalarCoordinatesM![2].at(-1)!;
  const z = Array.from({ length: CONTOUR_SHAPE[2] }, (_, index) => {
    const fraction = index / (CONTOUR_SHAPE[2] - 1);
    return last * (0.2 * fraction + 0.8 * fraction * fraction);
  });
  contourGrid.scalarCoordinatesM = [contourGrid.scalarCoordinatesM![0], contourGrid.scalarCoordinatesM![1], z];
  contourGrid.physicalBoundsM = {
    lower: [contourGrid.physicalBoundsM!.lower[0], contourGrid.physicalBoundsM!.lower[1], -(z[1] - z[0]) / 2],
    upper: [contourGrid.physicalBoundsM!.upper[0], contourGrid.physicalBoundsM!.upper[1], last + (last - z.at(-2)!) / 2],
  };
}
export const terrainContourGrid: GridInfo = {
  ...contourGrid,
  terrainFollowing: true,
  horizontalScalarOriginM: [0, 0],
  scalarCoordinatesM: undefined,
  scalarOriginM: undefined,
  physicalBoundsM: {
    lower: [0, 0, 0],
    upper: [
      contourGrid.physicalBoundsM!.upper[0],
      contourGrid.physicalBoundsM!.upper[1],
      contourGrid.physicalBoundsM!.upper[2] + 360,
    ],
  },
};
export const largeGrid: GridInfo = {
  nr: LARGE_FIELD_SHAPE[0],
  nth: LARGE_FIELD_SHAPE[1],
  nz: LARGE_FIELD_SHAPE[2],
  coordinateSystem: "cartesian",
  geometryKind: "physical",
  staggering: "c_grid",
  drM: 100,
  dyM: 100,
  dthetaRad: 100,
  dzM: 100,
  scalarOriginM: [0, 0, 0],
  physicalBoundsM: {
    lower: [0, 0, 0],
    upper: [(LARGE_FIELD_SHAPE[0] - 1) * 100, (LARGE_FIELD_SHAPE[1] - 1) * 100, (LARGE_FIELD_SHAPE[2] - 1) * 100],
  },
};

export const MOVING_CLOUD_FIXTURE = typeof window !== "undefined" ? new URLSearchParams(window.location.search).get("movingCloud") : null;
function movingCloudGrid(step: number, child = false): GridInfo {
  const terrain = MOVING_CLOUD_FIXTURE?.startsWith("terrain");
  const dx = child ? 62.5 : 125, dy = child ? 87.5 : 175, dz = 90;
  const lower: [number,number,number] = [step * 125 + (child ? 1000 : 0), child ? 1500 : 0, 0];
  const center: [number,number,number] = [lower[0] + dx / 2, lower[1] + dy / 2, dz / 2];
  return { ...grid, drM: dx, dyM: dy, dzM: dz, scalarCoordinatesM: undefined,
    terrainFollowing: terrain, horizontalScalarOriginM: terrain ? [center[0],center[1]] : undefined,
    scalarOriginM: terrain ? undefined : center,
    physicalBoundsM: { lower, upper: [lower[0] + FIELD_SHAPE[0] * dx, lower[1] + FIELD_SHAPE[1] * dy, FIELD_SHAPE[2] * dz + (terrain ? 250 : 0)] } };
}
function movingTerrainHeight(x: number, y: number): number {
  return 100 + 60 * Math.sin(x / 1700) * Math.sin(y / 1500);
}
function movingCloudField(step: number, fieldName: string, child = false): NpyN {
  const g = movingCloudGrid(step, child), origin = g.scalarOriginM ?? [...g.horizontalScalarOriginM!, 45];
  const data = new Float32Array(FIELD_SHAPE[0] * FIELD_SHAPE[1] * FIELD_SHAPE[2]);
  for (let i = 0; i < FIELD_SHAPE[0]; i++) for (let j = 0; j < FIELD_SHAPE[1]; j++) for (let k = 0; k < FIELD_SHAPE[2]; k++) {
    const x = origin[0] + i * g.drM!, y = origin[1] + j * g.dyM!;
    const z = (g.terrainFollowing ? movingTerrainHeight(x,y) : 0) + 45 + k * 90;
    const q = 0.0015 * Math.exp(-0.5 * (((x - 2500 - step * 60) / 450) ** 2 + ((y - 3000 - step * 30) / 450) ** 2 + ((z - 1050 - step * 6) / 250) ** 2));
    data[(i * FIELD_SHAPE[1] + j) * FIELD_SHAPE[2] + k] = fieldName === "u" ? 2 : fieldName === "v" ? 1 : fieldName === "w" ? 0.2 : fieldName === "cloud_condensate" ? q : scalarValue(FIELD_SHAPE, fieldName, step, i,j,k);
  }
  return { shape: [...FIELD_SHAPE], data };
}

const irregularPlayback = typeof window !== "undefined"
  && new URLSearchParams(window.location.search).get("irregularPlayback") === "1";
export const steps: StepInfo[] = Array.from({ length: 20 }, (_, index) => ({
  index,
  simulationTimeS: irregularPlayback ? index === 0 ? 0 : index === 1 ? 2 : 10 + (index - 2) * 30 : index * 30,
  fields: [...FIELD_NAMES],
  grid: MOVING_CLOUD_FIXTURE ? movingCloudGrid(index) : grid,
  motion: MOVING_CLOUD_FIXTURE ? { mode: "constant_velocity", horizontalFrameContract: "constant_translation_earth_wind_grid_transport_v1", windsEarthRelative: true, frameVelocityMs: [125 / 30, 0], earthOriginM: [index * 125, 0] } : typeof window !== "undefined" && new URLSearchParams(window.location.search).get("movingAxes") === "1"
    ? { mode: "constant_velocity", windsEarthRelative: true,
      horizontalFrameContract: index === 1 ? "future_contract" : "constant_translation_earth_wind_grid_transport_v1" }
    : undefined,
  fieldCompression: typeof window !== "undefined" && new URLSearchParams(window.location.search).get("compressedPlotBudget") === "1"
    ? Object.fromEntries(["cloud_condensate", "u", "v", "w", "vorticity_z", "qr"].map(field => [field, {
        codec: "zfp", mode: "accuracy", modeParameter: 1e-5, modeParameterUnits: "field_units",
        errorContract: "zfp_absolute_accuracy_before_prefilters", sparseThreshold: 0, float16Prefilter: false, decodedDtype: "float32",
      }])) : undefined,
  fieldUnits: {
    w: "m s-1",
    u: "m s-1",
    v: "m s-1",
    reflectivity_dbz: "dBZ",
    composite_reflectivity: "dBZ",
    buoyancy: "m s-2",
    theta_prime: "K",
    cloud_condensate: "kg kg-1",
    qr: "kg kg-1",
    qg: "kg kg-1",
    qh: "kg kg-1",
    cloud_liquid_condensate: "kg kg-1",
    cloud_frozen_condensate: "kg kg-1",
    vorticity_r: "s-1",
    vorticity_theta: "s-1",
    vorticity_z: "s-1",
    okubo_weiss: "1/s^2",
    uniform_scalar: "1",
  },
  publicationState: "complete",
  publicationGeneration: MOVING_CLOUD_FIXTURE ? "browser-moving-cloud" : `browser-fixture-${index}`,
}));

const beastReferenceStep = (index: number, simulationTimeS: number): StepInfo => ({
  index,
  simulationTimeS,
  fields: typeof window !== "undefined" && new URLSearchParams(window.location.search).has("beastFields")
    ? ["vorticity_magnitude", "theta", "theta_prime", "p", "p_prime", "qv", "buoyancy", "cloud_condensate", "reflectivity_dbz", "temperature", "u", "v", "w", "dewpoint", "horizontal_wind_speed", "qr", "rho", "total_condensate", "vorticity_z", "relative_humidity", "theta_e"] : ["vorticity_magnitude"],
  grid: {
    nr: 512,
    nth: 512,
    nz: 160,
    coordinateSystem: "cartesian",
    geometryKind: "physical",
    staggering: "c_grid",
    drM: 100,
    dyM: 100,
    dzM: 100,
    scalarOriginM: [-25_550, -25_550, 50],
    physicalBoundsM: {
      lower: [-25_600, -25_600, 0],
      upper: [25_600, 25_600, 16_000],
    },
  },
  fieldUnits: { vorticity_magnitude: "s-1", theta:"K", theta_prime:"K", p:"Pa", p_prime:"Pa", qv:"kg/kg", buoyancy:"m/s^2", cloud_condensate:"kg/kg", reflectivity_dbz:"dBZ", temperature:"K", u:"m/s", v:"m/s", w:"m/s", dewpoint:"K", horizontal_wind_speed:"m/s", qr:"kg/kg", rho:"kg/m³", total_condensate:"kg/kg", vorticity_z:"s-1", relative_humidity:"%", theta_e:"K" },
  fieldOrigins: { vorticity_magnitude: "native", dewpoint: "studio_derived", horizontal_wind_speed: "studio_derived", relative_humidity: "studio_derived", theta_e: "studio_derived" },
  publicationState: "complete",
  publicationGeneration: `beast-xl-reference-step-${index}`,
});

// Step 23 remains first so the accepted historical-reference lane keeps its
// stable index. The additional frames reproduce the broad early support and
// late playback transition reported from the preserved sandbox run.
export const beastReferenceSteps: StepInfo[] = [
  beastReferenceStep(23, 1380.21),
  beastReferenceStep(2, 120),
  beastReferenceStep(5, 300.475),
  beastReferenceStep(24, 1440.18),
  beastReferenceStep(0, 0),
  beastReferenceStep(1, 60),
  beastReferenceStep(3, 180),
];

export const terrainSteps: StepInfo[] = steps.map((step) => ({
  ...step,
  grid: terrainGrid,
  publicationGeneration: `browser-terrain-fixture-${step.index}`,
}));

export const stretchedSteps: StepInfo[] = steps.map((step) => ({
  ...step,
  grid: stretchedGrid,
  publicationGeneration: `browser-stretched-fixture-${step.index}`,
}));

export const largePlaneSteps: StepInfo[] = [{
  index: 0,
  simulationTimeS: 0,
  fields: ["composite_reflectivity"],
  grid: largePlaneGrid,
  fieldUnits: { composite_reflectivity: "dBZ" },
  publicationState: "complete",
  publicationGeneration: "browser-large-plane-fixture-0",
}];

export const coarseSteps: StepInfo[] = steps.map((step) => ({
  ...step,
  grid: coarseGrid,
  publicationGeneration: `browser-coarse-fixture-${step.index}`,
}));

export const largeSteps: StepInfo[] = [{
  index: 0,
  simulationTimeS: 0,
  fields: ["vorticity_z", "w", "u", "v", "cloud_condensate"],
  grid: largeGrid,
  fieldUnits: {
    vorticity_z: "s-1",
    w: "m s-1",
    u: "m s-1",
    v: "m s-1",
    cloud_condensate: "kg kg-1",
  },
  publicationState: "complete",
  publicationGeneration: "browser-large-fixture-0",
  fieldBrickHierarchies: {
    cloud_condensate: {
      path: "cloud_condensate.bricks.json",
      contentSha256: "a".repeat(64),
      schemaVersion: 1,
      nativeBrickShape: [64, 64, 32],
      extremaSemantics: "decoded_value_bounds",
      extremaAbsolutePadding: 1e-8,
      cullingSafe: true,
    },
  },
}];

export const brickedLargeSteps: StepInfo[] = largeSteps.map((step) => ({
  ...step,
  fieldBrickHierarchies: {
    ...step.fieldBrickHierarchies,
    vorticity_z: {
      path: "vorticity_z.bricks.json",
      contentSha256: "b".repeat(64),
      schemaVersion: 1,
      nativeBrickShape: [64, 64, 32],
      extremaSemantics: "decoded_value_bounds",
      extremaAbsolutePadding: 1e-6,
      cullingSafe: true,
    },
  },
}));

const derivedBrickInfo = (field: string, hashDigit: string) => ({
  path: `${field}.bricks.json`,
  contentSha256: hashDigit.repeat(64),
  schemaVersion: 2,
  nativeBrickShape: [64, 64, 32] as [number, number, number],
  extremaSemantics: "stored_values" as const,
  extremaAbsolutePadding: 0,
  cullingSafe: true,
});

export const derivedBrickSteps: StepInfo[] = Array.from({ length: 3 }, (_, index) => ({
  index,
  simulationTimeS: index * 30,
  fields: ["u", "v", "w", "dense_signed", "temperature", ...ANALYTIC_FIDELITY_FIELD_NAMES],
  grid: contourGrid,
  fieldUnits: { u: "m s-1", v: "m s-1", w: "m s-1", dense_signed: "1", temperature: "K" },
  publicationState: "complete",
  publicationGeneration: `browser-derived-brick-fixture-${index}`,
  fieldBrickHierarchies: {
    u: derivedBrickInfo("u", "a"),
    v: derivedBrickInfo("v", "b"),
    w: derivedBrickInfo("w", "c"),
    dense_signed: derivedBrickInfo("dense_signed", "e"),
    temperature: derivedBrickInfo("temperature", "f"),
    ...Object.fromEntries(ANALYTIC_FIDELITY_FIELD_NAMES.map((name) => [name, derivedBrickInfo(name, "d")])),
  },
}));

export const terrainBrickSteps: StepInfo[] = Array.from({ length: 3 }, (_, index) => ({
  index,
  simulationTimeS: index * 30,
  fields: ["u", "v", "w", "temperature"],
  grid: terrainContourGrid,
  fieldUnits: { u: "m s-1", v: "m s-1", w: "m s-1", temperature: "K" },
  publicationState: "complete" as const,
  publicationGeneration: `browser-terrain-brick-fixture-${index}`,
  fieldBrickHierarchies: {
    u: { ...derivedBrickInfo("u", "b"), schemaVersion: 3 },
    v: { ...derivedBrickInfo("v", "c"), schemaVersion: 3 },
    w: { ...derivedBrickInfo("w", "d"), schemaVersion: 3 },
    temperature: { ...derivedBrickInfo("temperature", "e"), schemaVersion: 3 },
  },
}));

function nestedChildGrid(step: number): GridInfo {
  const dx = contourGrid.drM! * 0.5;
  const dy = contourGrid.dyM! * 0.5;
  const dz = contourGrid.dzM!;
  const offsetX = step === 0 ? 1_000 : 2_000;
  const offsetY = step === 0 ? 800 : 1_400;
  const upper: [number, number, number] = [
    Math.min(contourGrid.physicalBoundsM!.upper[0], offsetX + CONTOUR_SHAPE[0] * dx),
    Math.min(contourGrid.physicalBoundsM!.upper[1], offsetY + CONTOUR_SHAPE[1] * dy),
    contourGrid.physicalBoundsM!.upper[2],
  ];
  return {
    ...contourGrid,
    drM: dx,
    dyM: dy,
    dzM: dz,
    scalarOriginM: [offsetX + 0.5 * dx, offsetY + 0.5 * dy, 0.5 * dz],
    scalarCoordinatesM: undefined,
    physicalBoundsM: { lower: [offsetX, offsetY, 0], upper },
  };
}

export const nestedChildBrickSteps: StepInfo[] = derivedBrickSteps.map((step) => ({
  ...step,
  grid: nestedChildGrid(step.index),
}));

export const nestedGrandchildBrickSteps: StepInfo[] = nestedChildBrickSteps.map((step) => {
  const parent = step.grid!;
  const dx = parent.drM! * 0.5;
  const dy = parent.dyM! * 0.5;
  const lower = parent.physicalBoundsM!.lower;
  const offsetX = lower[0] + 300;
  const offsetY = lower[1] + 250;
  return {
    ...step,
    grid: {
      ...parent,
      drM: dx,
      dyM: dy,
      scalarOriginM: [offsetX + 0.5 * dx, offsetY + 0.5 * dy, parent.scalarOriginM![2]],
      physicalBoundsM: {
        lower: [offsetX, offsetY, lower[2]],
        upper: [
          Math.min(parent.physicalBoundsM!.upper[0], offsetX + CONTOUR_SHAPE[0] * dx),
          Math.min(parent.physicalBoundsM!.upper[1], offsetY + CONTOUR_SHAPE[1] * dy),
          parent.physicalBoundsM!.upper[2],
        ],
      },
    },
  };
});

function scalarValue(
  shape: readonly [number, number, number],
  fieldName: string,
  step: number,
  x: number,
  y: number,
  z: number,
): number {
  const [nx, ny, nz] = shape;
  const cx = (nx - 1) * (0.42 + step * 0.004);
  const cy = (ny - 1) * 0.53;
  const cz = (nz - 1) * 0.42;
  const dx = (x - cx) / (nx * 0.22);
  const dy = (y - cy) / (ny * 0.24);
  const dz = (z - cz) / (nz * 0.3);
  const storm = Math.exp(-(dx * dx + dy * dy + dz * dz) * 2.8);
  const positive = Math.exp(-((dx + 0.34) ** 2 + dy * dy + dz * dz) * 3.4);
  const negative = Math.exp(-((dx - 0.34) ** 2 + dy * dy + dz * dz) * 3.4);
  const wave = Math.sin(x * 0.23 + step * 0.17) * Math.cos(y * 0.19) * Math.sin((z + 1) * 0.31);
  // A localized, vertically coherent ~0.02 s^-1 circulation. The unequal
  // coefficients account for the fixture's 175 m y and 125 m x spacing, so
  // u=-Omega*y and v=Omega*x represent the same physical angular rate. Its
  // tapered edge supplies a strain-dominated exterior around a bounded Q>0
  // core instead of an unphysical domain-wide solid-body rotation.
  const rotationEnvelope = Math.exp(
    -(dx * dx + dy * dy) * 2.2 - dz * dz * 1.2,
  );
  if (fieldName === "cloud_liquid_condensate" || fieldName === "cloud_frozen_condensate") {
    const aggregate = scalarValue(shape, "cloud_condensate", step, x, y, z);
    const frozenFraction = Math.max(0, Math.min(1, (z / Math.max(1, nz - 1) - 0.35) / 0.35));
    return fieldName === "cloud_frozen_condensate"
      ? aggregate * frozenFraction
      : aggregate * (1 - frozenFraction);
  }
  switch (fieldName) {
    case "uniform_scalar": return 0;
    case "reflectivity_dbz": return -10 + 70 * storm + 3 * wave;
    case "buoyancy": return 0.14 * positive - 0.18 * negative + 0.01 * wave;
    case "theta_prime": return 2.5 * positive - 4.5 * negative + 0.2 * wave;
    case "cloud_condensate": {
      // Multi-scale synthetic supercell: a vertically coherent tower, anvil,
      // low wall-cloud lobe, and resolved billow modulation. This is still a
      // deterministic scalar fixture, but unlike a single Gaussian sphere it
      // exercises coarse cloud silhouette, normals, lighting, and framing.
      const anvilDx = (x - cx - nx * 0.1) / (nx * 0.34);
      const anvilDy = (y - cy) / (ny * 0.27);
      const anvilDz = (z - nz * 0.7) / (nz * 0.1);
      const anvil = Math.exp(-(anvilDx * anvilDx + anvilDy * anvilDy + anvilDz * anvilDz) * 2.1);
      const wallDx = (x - cx + nx * 0.06) / (nx * 0.14);
      const wallDy = (y - cy + ny * 0.04) / (ny * 0.15);
      const wallDz = (z - nz * 0.18) / (nz * 0.16);
      const wall = Math.exp(-(wallDx * wallDx + wallDy * wallDy + wallDz * wallDz) * 2.5);
      const billows = 0.78 + 0.22 * Math.sin(x * 0.58 + z * 0.41) * Math.cos(y * 0.49 - z * 0.22);
      return Math.max(0, 0.0036 * storm * billows + 0.0018 * anvil + 0.0014 * wall);
    }
    case "qr": {
      const lowLevel = Math.exp(-(((z - nz * 0.2) / (nz * 0.24)) ** 2));
      return Math.max(0, 0.0024 * storm * lowLevel * (0.9 + 0.1 * wave));
    }
    case "qg": {
      const midLevel = Math.exp(-(((z - nz * 0.48) / (nz * 0.25)) ** 2));
      return Math.max(0, 0.0012 * storm * midLevel * (0.92 + 0.08 * wave));
    }
    case "qh": {
      const hailCore = Math.exp(-((dx + 0.14) ** 2 + (dy - 0.04) ** 2 + dz * dz) * 5.2);
      return Math.max(0, 0.0007 * hailCore * (0.94 + 0.06 * wave));
    }
    case "w": return 18 * storm + 2 * wave - 1;
    case "u": return -(y - cy) * 3.5 * rotationEnvelope + 0.2 * wave * storm;
    case "v": return (x - cx) * 2.5 * rotationEnvelope - 0.2 * wave * storm;
    case "dense_signed": return 5 * Math.sin(4 * Math.PI * x / Math.max(1, nx - 1))
      + 2 * Math.sin(2 * Math.PI * y / Math.max(1, ny - 1))
      + Math.cos(2 * Math.PI * z / Math.max(1, nz - 1));
    case "temperature": return 278 + 0.35 * z + 5 * storm + 0.2 * wave;
    case "vorticity_r": return -0.08 * dy * positive + 0.08 * dy * negative;
    case "vorticity_theta": return 0.08 * (dx + 0.34) * positive - 0.08 * (dx - 0.34) * negative;
    case "vorticity_z": {
      // Counter-rotating resolved cores exercise the signed ± contour path.
      // Both lobes move with the synthetic storm and remain smooth at coarse
      // resolution, so a missing sign or an unstable threshold is visible.
      return 0.055 * (positive - negative) + 0.0005 * wave;
    }
    case "okubo_weiss": {
      // Negative rotation-dominated cores embedded in a weak positive
      // strain-dominated environment. The pair follows the same resolved
      // counter-rotating structures as the vorticity fixture.
      return 0.0012 + 0.0004 * storm - 0.006 * (positive + negative);
    }
    default: return storm;
  }
}

function maskGeometryFixture(field: NpyN, step = 0): NpyN {
  if (typeof window === "undefined" || field.shape.length !== 3) return field;
  const mode = new URLSearchParams(window.location.search).get("missingGeometry");
  if (mode === "empty-next") { if (step > 0) field.data.fill(NaN); return field; }
  if (mode !== "1") return field;
  const [nx, ny, nz] = field.shape;
  const lower = Math.floor(ny * 0.45), upper = Math.max(lower + 2, Math.ceil(ny * 0.55));
  for (let x = 0; x < nx; x++) for (let y = lower; y < upper; y++) field.data.fill(NaN, (x * ny + y) * nz, (x * ny + y + 1) * nz);
  return field;
}

function makeSyntheticField(
  shape: readonly [number, number, number],
  step: number,
  fieldName: string,
): NpyN {
  const [nx, ny, nz] = shape;
  const data = new Float32Array(nx * ny * nz);
  let i = 0;
  for (let x = 0; x < nx; x += 1) {
    for (let y = 0; y < ny; y += 1) {
      for (let z = 0; z < nz; z += 1) {
        data[i++] = scalarValue(shape, fieldName, step, x, y, z);
      }
    }
  }
  const windMotion = typeof window !== "undefined" ? new URLSearchParams(window.location.search).get("windMotion") : null;
  if (step === 0 && typeof window !== "undefined"
      && new URLSearchParams(window.location.search).get("uniformAtStart") === fieldName) data.fill(0);
  if ((fieldName === "u" || fieldName === "v") && (windMotion === "missing" || windMotion === "fast")) data.fill(windMotion === "missing" ? NaN : 1000);
  return maskGeometryFixture({ shape: [...shape], data }, step);
}

/**
 * The engine's rank-2 products, built from the rank-3 field they reduce.
 *
 * `composite_reflectivity` is by definition the column maximum of
 * `reflectivity_dbz`, so it is computed here rather than invented — a fixture
 * whose composite disagreed with its own volume would let a floor that reads
 * the wrong array still look right.
 */
function makeColumnMaxField(
  shape: readonly [number, number, number],
  step: number,
  sourceFieldName: string,
): NpyN {
  const [nx, ny, nz] = shape;
  const volume = makeSyntheticField(shape, step, sourceFieldName);
  const data = new Float32Array(nx * ny);
  for (let x = 0; x < nx; x += 1) {
    for (let y = 0; y < ny; y += 1) {
      let max = Number.NEGATIVE_INFINITY;
      for (let z = 0; z < nz; z += 1) {
        const value = volume.data[(x * ny + y) * nz + z]!;
        if (Number.isFinite(value) && value > max) max = value;
      }
      data[x * ny + y] = Number.isFinite(max) ? max : Number.NaN;
    }
  }
  return { shape: [nx, ny], data };
}

const RANK_2_PRODUCTS: Readonly<Record<string, string>> = {
  composite_reflectivity: "reflectivity_dbz",
};

export function makeField(step: number, fieldName = "w"): NpyN {
  if (MOVING_CLOUD_FIXTURE) return movingCloudField(step, fieldName);
  if (isAnalyticFidelityFieldName(fieldName)) {
    const field = createAnalyticFidelityFixture(fieldName, [...FIELD_SHAPE]).field;
    return step === 0 && typeof window !== "undefined" && new URLSearchParams(window.location.search).get("uniformAtStart") === fieldName
      ? { ...field, data: new Float32Array(field.data.length) } : field;
  }
  const reduced = RANK_2_PRODUCTS[fieldName];
  if (reduced) return makeColumnMaxField(FIELD_SHAPE, step, reduced);
  return makeSyntheticField(FIELD_SHAPE, step, fieldName);
}

export function makeCoarseField(step: number, fieldName = "w"): NpyN {
  const reduced = RANK_2_PRODUCTS[fieldName];
  if (reduced) return makeColumnMaxField(COARSE_FIELD_SHAPE, step, reduced);
  return makeSyntheticField(COARSE_FIELD_SHAPE, step, fieldName);
}

function makeLargePlaneField(): NpyN {
  const [nx, ny] = LARGE_PLANE_SHAPE;
  const data = new Float32Array(nx * ny);
  for (let x = 0; x < nx; x += 1) {
    const dx = (x - nx * 0.52) / (nx * 0.18);
    for (let y = 0; y < ny; y += 1) {
      const dy = (y - ny * 0.47) / (ny * 0.2);
      data[x * ny + y] = -10 + 68 * Math.exp(-(dx * dx + dy * dy) * 2.4)
        + 2 * Math.sin(x * 0.017) * Math.cos(y * 0.019);
    }
  }
  return { shape: [nx, ny], data };
}

export function makeContourField(step: number): NpyN {
  return makeSyntheticField(CONTOUR_SHAPE, step, "w");
}

const derivedFieldCache = new Map<string, NpyN>();
function makeDerivedBrickField(step: number, fieldName: string): NpyN {
  const key = `${step}:${fieldName}`;
  let field = derivedFieldCache.get(key);
  if (!field) {
    field = isAnalyticFidelityFieldName(fieldName)
      ? createAnalyticFidelityFixture(fieldName, [...CONTOUR_SHAPE]).field
      : makeSyntheticField(CONTOUR_SHAPE, step, fieldName);
    field = maskGeometryFixture(field, step);
    derivedFieldCache.set(key, field);
  }
  return field;
}

const derivedLodCache = new Map<string, NpyN>();
function makeDerivedBrickLod(step: number, fieldName: string, level: number): NpyN {
  if (!Number.isSafeInteger(level) || level < 1) throw new Error("Invalid derived brick LOD level");
  const key = `${step}:${fieldName}:${level}`;
  const cached = derivedLodCache.get(key);
  if (cached) return cached;
  const source = level === 1
    ? makeDerivedBrickField(step, fieldName)
    : makeDerivedBrickLod(step, fieldName, level - 1);
  const shape = source.shape.map((count) => Math.ceil(count / 2));
  const data = new Float32Array(shape[0]! * shape[1]! * shape[2]!);
  for (let i = 0; i < shape[0]!; i += 1) {
    for (let j = 0; j < shape[1]!; j += 1) {
      for (let k = 0; k < shape[2]!; k += 1) {
        let sum = 0;
        let count = 0;
        let invalid = false;
        for (let di = 0; di < 2 && 2 * i + di < source.shape[0]!; di += 1) {
          for (let dj = 0; dj < 2 && 2 * j + dj < source.shape[1]!; dj += 1) {
            for (let dk = 0; dk < 2 && 2 * k + dk < source.shape[2]!; dk += 1) {
              const value = source.data[
                ((2 * i + di) * source.shape[1]! + 2 * j + dj) * source.shape[2]! + 2 * k + dk
              ]!;
              if (!Number.isFinite(value)) invalid = true;
              else { sum += value; count += 1; }
            }
          }
        }
        data[(i * shape[1]! + j) * shape[2]! + k] = invalid || count === 0
          ? Number.NaN
          : sum / count;
      }
    }
  }
  const lod = { shape, data };
  derivedLodCache.set(key, lod);
  return lod;
}

const terrainBrickFieldCache = new Map<string, NpyN>();
function makeTerrainBrickField(step: number, fieldName = "w"): NpyN {
  const key = `${step}:${fieldName}`;
  const cached = terrainBrickFieldCache.get(key);
  if (cached) return cached;
  const [nx, ny, nz] = CONTOUR_SHAPE;
  const data = new Float32Array(nx * ny * nz);
  for (let i = 0; i < nx; i += 1) {
    for (let j = 0; j < ny; j += 1) {
      for (let k = 0; k < nz; k += 1) {
        const terrainWave = Math.sin(4 * Math.PI * (i + step * 0.2) / (nx - 1));
        data[(i * ny + j) * nz + k] = fieldName === "temperature"
          ? 275 + 0.4 * k + 3 * terrainWave
          : fieldName === "u"
          ? 8 + 0.8 * Math.sin(2 * Math.PI * j / (ny - 1))
          : fieldName === "v"
            ? 2 + 0.6 * terrainWave
            : 10 + 6 * terrainWave
              + 1.5 * Math.sin(2 * Math.PI * j / (ny - 1))
              + 0.75 * Math.cos(2 * Math.PI * k / (nz - 1));
      }
    }
  }
  const field = maskGeometryFixture({ shape: [...CONTOUR_SHAPE], data }, step);
  terrainBrickFieldCache.set(key, field);
  return field;
}

const terrainBrickLodCache = new Map<string, NpyN>();
function makeTerrainBrickLod(step: number, level: number, fieldName = "w"): NpyN {
  const key = `${step}:${fieldName}:${level}`;
  const cached = terrainBrickLodCache.get(key);
  if (cached) return cached;
  const source = level === 1
    ? makeTerrainBrickField(step, fieldName)
    : makeTerrainBrickLod(step, level - 1, fieldName);
  const shape = source.shape.map((count) => Math.ceil(count / 2));
  const data = new Float32Array(shape[0]! * shape[1]! * shape[2]!);
  for (let i = 0; i < shape[0]!; i += 1) {
    for (let j = 0; j < shape[1]!; j += 1) {
      for (let k = 0; k < shape[2]!; k += 1) {
        let sum = 0;
        let count = 0;
        for (let di = 0; di < 2 && 2 * i + di < source.shape[0]!; di += 1) {
          for (let dj = 0; dj < 2 && 2 * j + dj < source.shape[1]!; dj += 1) {
            for (let dk = 0; dk < 2 && 2 * k + dk < source.shape[2]!; dk += 1) {
              sum += source.data[
                ((2 * i + di) * source.shape[1]! + 2 * j + dj) * source.shape[2]!
                  + 2 * k + dk
              ]!;
              count += 1;
            }
          }
        }
        data[(i * shape[1]! + j) * shape[2]! + k] = sum / count;
      }
    }
  }
  const lod = { shape, data };
  terrainBrickLodCache.set(key, lod);
  return lod;
}

export function makeLargeField(fieldName: string): NpyN {
  const elements = LARGE_FIELD_SHAPE[0] * LARGE_FIELD_SHAPE[1] * LARGE_FIELD_SHAPE[2];
  const data = new Float32Array(elements);
  if (fieldName === "vorticity_z") {
    for (let i = 64; i < 192; i += 1) {
      for (let j = 64; j < 192; j += 1) {
        const positiveRadius2 = ((i - 120) / 13) ** 2 + ((j - 128) / 13) ** 2;
        const negativeRadius2 = ((i - 150) / 16) ** 2 + ((j - 128) / 16) ** 2;
        if (positiveRadius2 > 16 && negativeRadius2 > 16) continue;
        for (let k = 0; k < 120; k += 1) {
          const vertical = Math.max(0, 1 - k / 140);
          data[(i * LARGE_FIELD_SHAPE[1] + j) * LARGE_FIELD_SHAPE[2] + k] = vertical * (
            0.08 * Math.exp(-positiveRadius2)
            - 0.05 * Math.exp(-negativeRadius2)
          );
        }
      }
    }
  }
  else if (fieldName === "cloud_condensate") {
    for (let i = 88; i < 168; i += 1) {
      for (let j = 88; j < 168; j += 1) {
        const radial = Math.hypot(i - 128, j - 128) / 40;
        if (radial >= 1) continue;
        for (let k = 0; k < 120; k += 1) {
          const vertical = Math.sin(Math.PI * (k + 1) / 121);
          data[(i * LARGE_FIELD_SHAPE[1] + j) * LARGE_FIELD_SHAPE[2] + k] =
            0.0015 * (1 - radial) * vertical;
        }
      }
    }
  }
  else if (fieldName === "u") data.fill(10);
  else if (fieldName === "v") data.fill(2);
  else if (fieldName === "w") {
    data.fill(1);
    const centerColumn = (
      Math.floor(LARGE_FIELD_SHAPE[0] / 2) * LARGE_FIELD_SHAPE[1]
      + Math.floor(LARGE_FIELD_SHAPE[1] / 2)
    ) * LARGE_FIELD_SHAPE[2];
    data.fill(8, centerColumn, centerColumn + LARGE_FIELD_SHAPE[2]);
  }
  else {
    for (let index = 0; index < elements; index += 1) {
      data[index] = index % LARGE_FIELD_SHAPE[2] - 79.5;
    }
  }
  return { shape: [...LARGE_FIELD_SHAPE], data };
}

const largeBrickHierarchies = new Map<string, FieldBrickHierarchy>();
function makeLargeBrickHierarchy(
  fieldName: "cloud_condensate" | "vorticity_z",
): FieldBrickHierarchy {
  const cached = largeBrickHierarchies.get(fieldName);
  if (cached) return cached;
  const field = makeLargeField(fieldName);
  const padding = fieldName === "vorticity_z" ? 1e-6 : 1e-8;
  const brickShape: [number, number, number] = [64, 64, 32];
  const bricks: FieldBrickHierarchy["levels"][number]["bricks"] = [];
  for (let bi = 0; bi < 4; bi += 1) {
    for (let bj = 0; bj < 4; bj += 1) {
      for (let bk = 0; bk < 5; bk += 1) {
        const origin: [number, number, number] = [bi * 64, bj * 64, bk * 32];
        const extent: [number, number, number] = [64, 64, 32];
        let minimum = Infinity;
        let maximum = -Infinity;
        for (let i = 0; i < extent[0]; i += 1) {
          for (let j = 0; j < extent[1]; j += 1) {
            const row = ((origin[0] + i) * LARGE_FIELD_SHAPE[1] + origin[1] + j)
              * LARGE_FIELD_SHAPE[2] + origin[2];
            for (let k = 0; k < extent[2]; k += 1) {
              const value = field.data[row + k]!;
              minimum = Math.min(minimum, value);
              maximum = Math.max(maximum, value);
            }
          }
        }
        bricks.push({
          id: `l0:${bi}:${bj}:${bk}`,
          index: [bi, bj, bk],
          origin,
          extent,
          scalarRange: {
            minimum: minimum - padding,
            maximum: maximum + padding,
            finiteCount: extent[0] * extent[1] * extent[2],
            nonFiniteCount: 0,
          },
          physicalBoundsM: {
            lower: origin.map((value) => value * 100) as [number, number, number],
            upper: origin.map((value, axis) => (value + extent[axis]) * 100) as [number, number, number],
          },
          contentSha256: (bi * 20 + bj * 5 + bk).toString(16).padStart(64, "0"),
        });
      }
    }
  }
  const hierarchy: FieldBrickHierarchy = {
    schema: "nimbus.field-brick-hierarchy",
    schemaVersion: 1,
    field: fieldName,
    publicationGeneration: "browser-large-fixture-0",
    sourceShape: [...LARGE_FIELD_SHAPE],
    sourceIndexOffset: [0, 0, 0],
    sourceDtype: "float32",
    sampleLocation: "cell_center",
    extremaSemantics: "decoded_value_bounds",
    extremaAbsolutePadding: padding,
    cullingSafe: true,
    levels: [{
      level: 0,
      stride: [1, 1, 1],
      shape: [...LARGE_FIELD_SHAPE],
      brickShape,
      source: { kind: "primary_field" },
      bricks,
    }],
  };
  largeBrickHierarchies.set(fieldName, hierarchy);
  return hierarchy;
}

const derivedBrickHierarchies = new Map<string, FieldBrickHierarchy>();
export function makeDerivedBrickHierarchy(step: number, fieldName: string): FieldBrickHierarchy {
  const cacheKey = `${step}:${fieldName}`;
  const cached = derivedBrickHierarchies.get(cacheKey);
  if (cached) return cached;
  const brickShape: [number, number, number] = [64, 64, 32];
  const makeLevel = (level: number): FieldBrickHierarchy["levels"][number] => {
    const field = level === 0
      ? makeDerivedBrickField(step, fieldName)
      : makeDerivedBrickLod(step, fieldName, level);
    const shape = field.shape as [number, number, number];
    const stride = 2 ** level;
    const counts = shape.map((extent, axis) =>
      Math.ceil(extent / brickShape[axis])) as [number, number, number];
    const bricks: FieldBrickHierarchy["levels"][number]["bricks"] = [];
    for (let bi = 0; bi < counts[0]; bi += 1) {
      for (let bj = 0; bj < counts[1]; bj += 1) {
        for (let bk = 0; bk < counts[2]; bk += 1) {
        const origin: [number, number, number] = [bi * 64, bj * 64, bk * 32];
        const extent = origin.map((value, axis) =>
          Math.min(brickShape[axis], shape[axis] - value)) as [number, number, number];
        let minimum = Infinity;
        let maximum = -Infinity;
        let finiteCount = 0;
        for (let i = 0; i < extent[0]; i += 1) {
          for (let j = 0; j < extent[1]; j += 1) {
            const row = ((origin[0] + i) * shape[1] + origin[1] + j)
              * shape[2] + origin[2];
            for (let k = 0; k < extent[2]; k += 1) {
              const value = field.data[row + k]!;
              if (!Number.isFinite(value)) continue;
              finiteCount++;
              minimum = Math.min(minimum, value);
              maximum = Math.max(maximum, value);
            }
          }
        }
        bricks.push({
          id: `l${level}:${bi}:${bj}:${bk}`,
          index: [bi, bj, bk],
          origin,
          extent,
          scalarRange: {
            minimum: finiteCount ? minimum : null,
            maximum: finiteCount ? maximum : null,
            finiteCount,
            nonFiniteCount: extent[0] * extent[1] * extent[2] - finiteCount,
          },
          physicalBoundsM: {
            lower: origin.map((value, axis) =>
              value * stride * (axis === 0 ? contourGrid.drM! : axis === 1 ? contourGrid.dyM! : contourGrid.dzM!)) as [number, number, number],
            upper: origin.map((value, axis) =>
              Math.min(CONTOUR_SHAPE[axis], (value + extent[axis]) * stride)
                * (axis === 0 ? contourGrid.drM! : axis === 1 ? contourGrid.dyM! : contourGrid.dzM!)) as [number, number, number],
          },
          contentSha256: (level * 64 + bi * 4 + bj * 2 + bk + (fieldName.charCodeAt(0) - 96) * 16)
            .toString(16).padStart(64, "0"),
        });
        }
      }
    }
    return {
      level,
      stride: [stride, stride, stride],
      shape,
      brickShape,
      source: level === 0
        ? { kind: "primary_field" }
        : {
            kind: "npy_lod",
            path: `${fieldName}.lod${level}.npy`,
            contentSha256: String(fieldName.charCodeAt(0) + level).padStart(64, "0"),
            aggregation: "recursive_cell_mean_2x2x2",
          },
      bricks,
    };
  };
  const hierarchy: FieldBrickHierarchy = {
    schema: "nimbus.field-brick-hierarchy",
    schemaVersion: 2,
    field: fieldName,
    publicationGeneration: `browser-derived-brick-fixture-${step}`,
    sourceShape: [...CONTOUR_SHAPE],
    sourceIndexOffset: [0, 0, 0],
    sourceDtype: "float32",
    sampleLocation: "cell_center",
    extremaSemantics: "stored_values",
    extremaAbsolutePadding: 0,
    cullingSafe: true,
    levels: [makeLevel(0), makeLevel(1)],
  };
  derivedBrickHierarchies.set(cacheKey, hierarchy);
  return hierarchy;
}

function makeTerrainBrickHierarchy(step: number, fieldName = "w"): FieldBrickHierarchy {
  const hierarchy = structuredClone(makeDerivedBrickHierarchy(step, fieldName));
  hierarchy.schemaVersion = 3;
  hierarchy.publicationGeneration = `browser-terrain-brick-fixture-${step}`;
  const geometryHash = (component: number, level: number) =>
    String(200 + component * 10 + level).padStart(64, "0");
  const levelOne = hierarchy.levels[1].source;
  if (levelOne.kind !== "npy_lod") throw new Error("Terrain fixture LOD source is invalid");
  levelOne.geometry = {
    kind: "physical_volume_centroids",
    xM: { path: "physical_x_center_m.lod1.npy", contentSha256: geometryHash(0, 1) },
    yM: { path: "physical_y_center_m.lod1.npy", contentSha256: geometryHash(1, 1) },
    zM: { path: "physical_z_center_m.lod1.npy", contentSha256: geometryHash(2, 1) },
    volumeM3: { path: "physical_volume_m3.lod1.npy", contentSha256: geometryHash(3, 1) },
  };
  for (const level of hierarchy.levels) {
    const field = level.level === 0
      ? makeTerrainBrickField(step, fieldName)
      : makeTerrainBrickLod(step, level.level, fieldName);
    const shape = field.shape as [number, number, number];
    for (const brick of level.bricks) {
      let minimum = Infinity;
      let maximum = -Infinity;
      let finiteCount = 0;
      for (let i = 0; i < brick.extent[0]; i += 1) {
        for (let j = 0; j < brick.extent[1]; j += 1) {
          const row = ((brick.origin[0] + i) * shape[1] + brick.origin[1] + j)
            * shape[2] + brick.origin[2];
          for (let k = 0; k < brick.extent[2]; k += 1) {
            const value = field.data[row + k]!;
            if (!Number.isFinite(value)) continue;
            finiteCount++;
            minimum = Math.min(minimum, value);
            maximum = Math.max(maximum, value);
          }
        }
      }
      brick.scalarRange.minimum = finiteCount ? minimum : null;
      brick.scalarRange.maximum = finiteCount ? maximum : null;
      brick.scalarRange.finiteCount = finiteCount;
      brick.scalarRange.nonFiniteCount = brick.extent.reduce((a, b) => a*b, 1) - finiteCount;
    }
  }
  return hierarchy;
}

function makeTrajectories(): NpyN {
  const parcels = 12;
  const samples = 30;
  const variables = 9;
  const data = new Float32Array(parcels * samples * variables);
  const budgetGap = new URLSearchParams(window.location.search).get("trajectoryBudgetGap") === "1";
  for (let p = 0; p < parcels; p += 1) {
    for (let s = 0; s < samples; s += 1) {
      const base = (p * samples + s) * variables;
      const angle = p * 0.5 + s * 0.08;
      data[base] = s * 10;
      data[base + 1] = 2_800 + Math.cos(angle) * (500 + p * 20);
      data[base + 2] = 3_000 + Math.sin(angle) * (500 + p * 20);
      data[base + 3] = 100 + s * 55 + p * 8;
      data[base + 4] = 4 + Math.sin(angle) * 2;
      data[base + 5] = 0.01 + p * 0.001;
      data[base + 6] = 0.001;
      data[base + 7] = budgetGap && s === 15 ? NaN : 0.002;
      data[base + 8] = -0.0005;
    }
  }
  return { shape: [parcels, samples, variables], data };
}

function encodeNpy(array: NpyN): ArrayBuffer {
  const shape = array.shape.length === 1
    ? `${array.shape[0]},`
    : array.shape.join(", ");
  const dictionary = `{'descr': '<f4', 'fortran_order': False, 'shape': (${shape}), }`;
  const preambleBytes = 10;
  const headerBytes = Math.ceil((preambleBytes + dictionary.length + 1) / 64) * 64 - preambleBytes;
  const header = `${dictionary}${" ".repeat(headerBytes - dictionary.length - 1)}\n`;
  const result = new Uint8Array(preambleBytes + headerBytes + array.data.byteLength);
  result.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0], 0);
  result[8] = headerBytes & 0xff;
  result[9] = (headerBytes >>> 8) & 0xff;
  result.set(new TextEncoder().encode(header), preambleBytes);
  result.set(new Uint8Array(array.data.buffer, array.data.byteOffset, array.data.byteLength), preambleBytes + headerBytes);
  return result.buffer;
}


/*
 * Bounded field-window ("NMBLOD1") responses.
 *
 * The radar floor deliberately does NOT go through `read_field_bytes`: it asks
 * for a strided hyperslab so its resolution is independent of whatever voxel
 * cap the 3D volume is under. That is a different command with a different wire
 * format, so mocking `read_field_bytes` alone leaves the floor unexercised in
 * the browser harness — it simply never loads.
 *
 * This encodes the real envelope (magic, u32 header length, JSON header,
 * float32 body) and honours origin/extent/stride, so the specs exercise the
 * protocol rather than a convenient stand-in.
 */
const FIELD_WINDOW_MAGIC = "NMBLOD1\u0000";

/*
 * The backend's two byte budgets, mirrored from
 * `src-tauri/src/commands/outputs.rs`.
 *
 * These are enforced here because omitting them is not a harmless
 * simplification — it is what let a broken caller ship. The radar floor planned
 * its column-maximum window against the OUTPUT budget alone, which every spec
 * accepted because this mock accepted anything; the real backend charges
 * separately for the bytes a cell-mean aggregation must READ, and refused the
 * request outright on any grid deeper than 64 MB. The layer was dead in the
 * built app and green in CI.
 *
 * A mock more permissive than the thing it stands in for does not merely fail
 * to catch bugs — it certifies them.
 */
const MOCK_MAX_OUTPUT_BYTES = 64 * 1024 * 1024;
const MOCK_DEFAULT_OUTPUT_BYTES = 32 * 1024 * 1024;
const MOCK_MAX_SOURCE_BYTES = 256 * 1024 * 1024;
const MOCK_DEFAULT_SOURCE_BYTES = 64 * 1024 * 1024;

export type MockFieldWindowAggregation =
  | "cell_mean"
  | "linear_reflectivity_mean"
  | "block_max"
  | "block_min"
  | "magnitude_extreme"
  | "point_sample";

function mockFieldWindowAggregation(value: unknown): MockFieldWindowAggregation {
  const aggregation = String(value ?? "cell_mean");
  if (![
    "cell_mean",
    "linear_reflectivity_mean",
    "block_max",
    "block_min",
    "magnitude_extreme",
    "point_sample",
  ].includes(aggregation)) {
    throw new Error(`unsupported field-window aggregation '${aggregation}'`);
  }
  return aggregation as MockFieldWindowAggregation;
}

/** Rejects what `validate_field_window_request` rejects, with its wording. */
function assertFieldWindowBudgets(
  outputShape: readonly number[],
  extent: readonly number[],
  aggregation: MockFieldWindowAggregation,
  maxOutputBytes: number | undefined,
  maxSourceBytes: number | undefined,
  outputBytesPerCell = Float32Array.BYTES_PER_ELEMENT,
): void {
  const outputElements = outputShape.reduce((product, count) => product * count, 1);
  const decodedBytes = outputElements * outputBytesPerCell;
  const outputBudget = maxOutputBytes ?? MOCK_DEFAULT_OUTPUT_BYTES;
  if (outputBudget === 0 || outputBudget > MOCK_MAX_OUTPUT_BYTES) {
    throw new Error(`maxOutputBytes must be within 1..=${MOCK_MAX_OUTPUT_BYTES}`);
  }
  if (decodedBytes > outputBudget) {
    throw new Error(`field window needs ${decodedBytes} bytes; request budget is ${outputBudget}`);
  }
  const sourceElements = aggregation === "point_sample"
    ? outputShape.slice(0, -1).reduce((product, count) => product * count, 1)
      * extent[extent.length - 1]
    : extent.reduce((product, count) => product * count, 1);
  const sourceBytes = sourceElements * Float32Array.BYTES_PER_ELEMENT;
  const sourceBudget = maxSourceBytes ?? MOCK_DEFAULT_SOURCE_BYTES;
  if (sourceBudget === 0 || sourceBudget > MOCK_MAX_SOURCE_BYTES) {
    throw new Error(`maxSourceBytes must be within 1..=${MOCK_MAX_SOURCE_BYTES}`);
  }
  if (sourceBytes > sourceBudget) {
    throw new Error(
      `${aggregation} field window must inspect ${sourceBytes} source bytes; request budget is ${sourceBudget}`,
    );
  }
}

export function encodeFieldWindow(
  requestId: string,
  field: NpyN,
  origin: readonly number[],
  extent: readonly number[],
  stride: readonly number[],
  aggregation: MockFieldWindowAggregation,
  maxOutputBytes?: number,
  maxSourceBytes?: number,
  terrain = false,
  headerOverride?: {
    sourceShape: readonly number[];
    origin: readonly number[];
    extent: readonly number[];
    stride: readonly number[];
    sourceFormat: "npy_lod";
  },
  geometryGrid: GridInfo = grid,
): ArrayBuffer {
  const sourceShape = field.shape;
  const rank = sourceShape.length;
  // Every request vector must match the field's own rank. The real backend
  // rejects a three-vector against a rank-2 product, so a mock that quietly
  // padded to rank 3 would hide the floor's 2-D composite path breaking.
  for (const [name, vector] of [["origin", origin], ["extent", extent], ["stride", stride]] as const) {
    if (vector.length !== rank) {
      throw new Error(`Field-window ${name} has rank ${vector.length}; source is rank ${rank}`);
    }
  }

  const axes = sourceShape.map((count, index) => {
    const axisOrigin = origin[index] ?? 0;
    const axisExtent = extent[index] ?? count;
    if (axisOrigin + axisExtent > count) {
      throw new Error(`Field-window request exceeds source axis ${index}`);
    }
    return {
      count,
      origin: axisOrigin,
      extent: axisExtent,
      stride: Math.max(1, stride[index] ?? 1),
    };
  });
  const outputShape = axes.map((a) => Math.max(0, Math.ceil(a.extent / a.stride)));
  const outputElements = outputShape.reduce((product, count) => product * count, 1);
  assertFieldWindowBudgets(
    outputShape,
    extent,
    aggregation,
    maxOutputBytes,
    maxSourceBytes,
    terrain ? 36 : 4,
  );

  // Row-major (C order) walk of the output lattice, mapped back to the source.
  // Rank-generic so a rank-2 product and a rank-3 volume share one path.
  const sourceStrides = sourceShape.map((_, axis) =>
    sourceShape.slice(axis + 1).reduce((product, count) => product * count, 1));
  const data = new Float32Array(outputElements);
  const cursor = new Array<number>(rank).fill(0);
  const sourceAt = (indices: readonly number[]): number => {
    const linear = indices.reduce(
      (total, index, axis) => total + index * sourceStrides[axis]!,
      0,
    );
    return field.data[linear] ?? Number.NaN;
  };
  const directFullCopy = aggregation === "cell_mean"
    && axes.every((axis) => axis.origin === 0 && axis.extent === axis.count && axis.stride === 1);
  if (directFullCopy) data.set(field.data);
  else for (let out = 0; out < outputElements; out += 1) {
    const blockStart = axes.map((axis, index) =>
      axis.origin + cursor[index]! * axis.stride);
    if (aggregation === "point_sample") {
      data[out] = sourceAt(blockStart);
    } else {
      const blockShape = axes.map((axis, index) => Math.min(
        axis.stride,
        axis.origin + axis.extent - blockStart[index]!,
      ));
      const blockElements = blockShape.reduce((product, count) => product * count, 1);
      const blockCursor = new Array<number>(rank).fill(0);
      const values: number[] = [];
      for (let sample = 0; sample < blockElements; sample += 1) {
        values.push(sourceAt(blockStart.map((start, axis) => start + blockCursor[axis]!)));
        for (let axis = rank - 1; axis >= 0; axis -= 1) {
          blockCursor[axis]! += 1;
          if (blockCursor[axis]! < blockShape[axis]!) break;
          blockCursor[axis] = 0;
        }
      }
      if (values.some((value) => !Number.isFinite(value))) {
        data[out] = Number.NaN;
      } else if (aggregation === "block_max") {
        data[out] = Math.max(...values);
      } else if (aggregation === "block_min") {
        data[out] = Math.min(...values);
      } else if (aggregation === "magnitude_extreme") {
        data[out] = values.reduce((selected, value) =>
          Math.abs(value) > Math.abs(selected) ? value : selected);
      } else if (aggregation === "linear_reflectivity_mean") {
        const linearMean = values.reduce((sum, value) => sum + 10 ** (value / 10), 0)
          / values.length;
        data[out] = 10 * Math.log10(linearMean);
      } else {
        data[out] = values.reduce((sum, value) => sum + value, 0) / values.length;
      }
    }
    for (let axis = rank - 1; axis >= 0; axis -= 1) {
      cursor[axis]! += 1;
      if (cursor[axis]! < outputShape[axis]!) break;
      cursor[axis] = 0;
    }
  }

  /*
   * Schema v4 carries the geometry/represented-volume and working-set contract, and the reader
   * validates the whole of it: `separable_uniform` requires all three geometry
   * descriptors to be explicitly null, and the byte accounting has to be a real
   * number rather than a placeholder. This mock serves only uniform separable
   * grids — it has no terrain-following path — so it states the uniform branch
   * exactly as `field_window_payload` in src-tauri/src/commands/outputs.rs does.
   *
   * `sourceWorkingBytes` is the same quantity the npy path reports from
   * `validate_window`: reducers inspect the full requested extent, while a
   * point sample reads one bounded fastest-axis row for every selected prefix.
   * Deriving it rather than hardcoding one is the point — a constant would
   * satisfy the reader's range check while quietly decoupling the mock from
   * the request it answered.
   */
  const inspectedElements = aggregation === "point_sample"
    ? outputShape.slice(0, -1).reduce((product, count) => product * count, 1)
      * axes[axes.length - 1]!.extent
    : axes.reduce((product, a) => product * a.extent, 1);
  const geometry = terrain ? new Float64Array(outputElements * 4) : null;
  if (geometry) {
    const geometryCursor = new Array<number>(rank).fill(0);
    const dx = geometryGrid.drM ?? 1;
    const dy = geometryGrid.dyM ?? dx;
    const dz = geometryGrid.dzM ?? 1;
    const terrainHeight = (x: number, y: number): number => {
      const xn = x / Math.max(1, sourceShape[0]! - 1);
      const yn = y / Math.max(1, sourceShape[1]! - 1);
      return 180 * Math.sin(Math.PI * xn) * Math.sin(Math.PI * yn)
        + 90 * xn;
    };
    for (let out = 0; out < outputElements; out += 1) {
      const blockStart = axes.map((axis, index) =>
        axis.origin + geometryCursor[index]! * axis.stride);
      const blockShape = axes.map((axis, index) => aggregation === "point_sample" ? 1 : Math.min(
        axis.stride,
        axis.origin + axis.extent - blockStart[index]!,
      ));
      let xSum = 0;
      let ySum = 0;
      let zSum = 0;
      let cells = 0;
      for (let xi = 0; xi < blockShape[0]!; xi += 1) {
        for (let yi = 0; yi < blockShape[1]!; yi += 1) {
          for (let zi = 0; zi < blockShape[2]!; zi += 1) {
            const sourceX = blockStart[0]! + xi;
            const sourceY = blockStart[1]! + yi;
            const sourceZ = blockStart[2]! + zi;
            const px = (geometryGrid.horizontalScalarOriginM?.[0] ?? 0) + sourceX * dx;
            const py = (geometryGrid.horizontalScalarOriginM?.[1] ?? 0) + sourceY * dy;
            xSum += px; ySum += py;
            zSum += (MOVING_CLOUD_FIXTURE?.startsWith("terrain") ? movingTerrainHeight(px,py) : terrainHeight(sourceX, sourceY)) + (sourceZ + 0.5) * dz;
            cells += 1;
          }
        }
      }
      geometry[out] = xSum / cells;
      geometry[outputElements + out] = ySum / cells;
      geometry[2 * outputElements + out] = zSum / cells;
      geometry[3 * outputElements + out] = cells * dx * dy * dz;
      for (let axis = rank - 1; axis >= 0; axis -= 1) {
        geometryCursor[axis]! += 1;
        if (geometryCursor[axis]! < outputShape[axis]!) break;
        geometryCursor[axis] = 0;
      }
    }
  }
  const header = {
    schema: "nimbus.studio.field-window",
    schemaVersion: 4,
    requestId,
    aggregation: headerOverride ? "cell_mean" : aggregation,
    weighting: terrain ? "physical_cell_volume" : "uniform_cell_volume",
    missingDataPolicy: "propagate_non_finite",
    sampleLocation: !headerOverride && aggregation === "point_sample" ? "source_cell_center"
      : terrain ? "physical_volume_centroid" : "block_center",
    geometryKind: terrain ? "terrain_following" : "separable_uniform",
    geometryDtype: terrain ? "<f8" : null,
    geometryLayout: terrain ? "component_major_x_y_z_volume" : null,
    geometryComponents: terrain ? ["x_m", "y_m", "z_m", "volume_m3"] : null,
    sourceShape: [...(headerOverride?.sourceShape ?? sourceShape)],
    origin: [...(headerOverride?.origin ?? axes.map((a) => a.origin))],
    extent: [...(headerOverride?.extent ?? axes.map((a) => a.extent))],
    stride: [...(headerOverride?.stride ?? axes.map((a) => a.stride))],
    outputShape,
    elementCount: data.length,
    decodedBytes: data.byteLength,
    geometryDecodedBytes: geometry?.byteLength ?? 0,
    sourceWorkingBytes: headerOverride
      ? data.byteLength
      : inspectedElements * Float32Array.BYTES_PER_ELEMENT,
    sourceWorkingAccounting: "inspected_payload_plus_explicit_working_set",
    dtype: "<f4",
    order: "C",
    sourceFormat: headerOverride?.sourceFormat ?? "npy",
    cancellationMode: "cooperative",
    nonInterruptiblePhase: null,
  };
  const headerBytes = new TextEncoder().encode(JSON.stringify(header));
  const magicBytes = new TextEncoder().encode(FIELD_WINDOW_MAGIC);
  const buffer = new ArrayBuffer(
    magicBytes.byteLength + 4 + headerBytes.byteLength + data.byteLength
      + (geometry?.byteLength ?? 0),
  );
  const bytes = new Uint8Array(buffer);
  bytes.set(magicBytes, 0);
  new DataView(buffer).setUint32(magicBytes.byteLength, headerBytes.byteLength, true);
  bytes.set(headerBytes, magicBytes.byteLength + 4);
  bytes.set(new Uint8Array(data.buffer), magicBytes.byteLength + 4 + headerBytes.byteLength);
  if (geometry) {
    bytes.set(
      new Uint8Array(geometry.buffer),
      magicBytes.byteLength + 4 + headerBytes.byteLength + data.byteLength,
    );
  }
  return buffer;
}

/** CM1/netCDF companion envelope over the same deterministic fixture field. */
function encodeNetcdfFieldWindow(
  requestId: string,
  field: NpyN,
  origin: readonly number[],
  extent: readonly number[],
  stride: readonly number[],
  aggregation: MockFieldWindowAggregation,
  maxOutputBytes?: number,
  maxSourceBytes?: number,
): ArrayBuffer {
  const native = new Uint8Array(encodeFieldWindow(
    requestId,
    field,
    origin,
    extent,
    stride,
    aggregation,
    maxOutputBytes,
    maxSourceBytes,
  ));
  const nativeHeaderBytes = new DataView(native.buffer).getUint32(8, true);
  const nativeHeader = JSON.parse(
    new TextDecoder().decode(native.subarray(12, 12 + nativeHeaderBytes)),
  ) as Record<string, unknown>;
  const body = native.subarray(12 + nativeHeaderBytes);
  const header = {
    schema: "nimbus.studio.netcdf-window",
    schemaVersion: 2,
    requestId,
    aggregation,
    weighting: "uniform_cell_volume",
    missingDataPolicy: "propagate_non_finite",
    sampleLocation: aggregation === "point_sample" ? "source_cell_center" : "block_center",
    sourceShape: nativeHeader.sourceShape,
    origin: nativeHeader.origin,
    extent: nativeHeader.extent,
    stride: nativeHeader.stride,
    outputShape: nativeHeader.outputShape,
    elementCount: nativeHeader.elementCount,
    decodedBytes: nativeHeader.decodedBytes,
    sourceReadBytes: body.byteLength,
    sourceReadAccounting: "converted_hyperslab_output",
    nativeInternalAllocationBounded: false,
    dtype: "<f4",
    order: "C",
    sourceFormat: "netcdf4",
    cancellationMode: "boundary_only",
    nonInterruptiblePhase: "nc_get_vars_float",
  };
  const json = new TextEncoder().encode(JSON.stringify(header));
  const encoded = new Uint8Array(12 + json.byteLength + body.byteLength);
  encoded.set([0x4e, 0x4d, 0x42, 0x4e, 0x43, 0x57, 0x31, 0], 0);
  new DataView(encoded.buffer).setUint32(8, json.byteLength, true);
  encoded.set(json, 12);
  encoded.set(body, 12 + json.byteLength);
  return encoded.buffer;
}

async function deliver(buffer: ArrayBuffer): Promise<ArrayBuffer> {
  const bytes = buffer.byteLength;
  ipcMetrics.startedJobs += 1;
  ipcMetrics.inFlightJobs += 1;
  ipcMetrics.inFlightBytes += bytes;
  ipcMetrics.maxInFlightJobs = Math.max(ipcMetrics.maxInFlightJobs, ipcMetrics.inFlightJobs);
  ipcMetrics.maxInFlightBytes = Math.max(ipcMetrics.maxInFlightBytes, ipcMetrics.inFlightBytes);
  await new Promise((resolve) => setTimeout(resolve, 4));
  ipcMetrics.inFlightJobs -= 1;
  ipcMetrics.inFlightBytes -= bytes;
  ipcMetrics.completedJobs += 1;
  ipcMetrics.deliveredBytes += bytes;
  return buffer;
}

/*
 * The real `preflight_config` command echoes a *canonicalized* context, not the
 * request verbatim: `src-tauri/src/commands/config_preflight.rs` runs every
 * enum-ish field through normalize()/normalize_backend()/normalize_sounding()/
 * normalize_hodograph()/normalize_trigger() before returning it, and
 * assertReportMatchesRequest() in lib/config/configPreflight.ts compares against
 * the same canonical form. A verbatim echo therefore fails for every value the
 * UI exposes under an alias -- the CM1 sounding draft sends the schema default
 * hodograph "custom", whose canonical form is "wk_param".
 *
 * These mirror config_preflight.rs deliberately rather than importing the
 * frontend's copies: the mock stands in for the backend, so a divergence
 * between the two languages must still be able to fail this lane.
 */
const normalizeToken = (value: unknown): string =>
  String(value ?? "").trim().toLowerCase().replaceAll("-", "_");

const ALIASES: Readonly<Record<string, Readonly<Record<string, string>>>> = {
  backend: { vulkan: "gpu", vk: "gpu" },
  soundingSource: {
    cape: "parametric_cape",
    parametriccape: "parametric_cape",
    sharpy: "file",
    raob: "file",
    sounding_file: "file",
    targets: "parametric_targets",
    diagnostics: "parametric_targets",
    from_table: "parametric_targets",
    parametrictargets: "parametric_targets",
  },
  hodographSource: { wk: "wk_param", wkparam: "wk_param", custom: "wk_param", static: "zero", calm: "zero" },
  triggerSource: {
    bubble: "warm_bubble",
    warmbubble: "warm_bubble",
    off: "none",
    disabled: "none",
    vortex: "vortex_seed",
    vortexseed: "vortex_seed",
    rankine: "vortex_seed",
  },
};

/** Canonicalize one field the way the Rust command does. Fields with no alias
 *  table are token-normalized; non-strings (booleans, numbers, tolerance maps)
 *  pass through untouched. */
function canonicalField(key: string, value: unknown): unknown {
  if (typeof value !== "string") return value;
  const token = normalizeToken(value);
  return ALIASES[key]?.[token] ?? token;
}

/** Fields the Rust command copies verbatim -- free text and paths, where
 *  lowercasing would corrupt the value. */
const VERBATIM_CONTEXT_KEYS = new Set([
  "executablePath",
  "configPath",
  "expectedCatalogFingerprint",
  "availableStateFields",
  "diagnostics",
]);

function normalizedPreflightContext(context: Record<string, unknown>): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(context)) {
    if (VERBATIM_CONTEXT_KEYS.has(key)) {
      result[key] = value;
    } else if (value !== null && typeof value === "object" && !Array.isArray(value)) {
      result[key] = normalizedPreflightContext(value as Record<string, unknown>);
    } else {
      result[key] = canonicalField(key, value);
    }
  }
  return result;
}

/**
 * Byte-for-byte mirror of `native_field_read_capabilities`
 * (`src-tauri/src/commands/outputs.rs`). The panel's whole purpose is reporting
 * read boundaries accurately, so a fixture that invented friendlier values would
 * certify a panel that misreports them. Cancellation is deliberately NOT uniform
 * here: NPY and classic netCDF check between bounded rows, while ZFP v3 brick
 * decodes and netCDF4's `nc_get_vars` have no abort hook. If the backend record
 * changes, this must change with it.
 */
const NATIVE_READ_CAPABILITIES = {
  schema: "nimbus.studio.native-field-read-capabilities",
  schemaVersion: 5,
  maxWindowBytes: 64 * 1024 * 1024,
  maxSourceBytes: 256 * 1024 * 1024,
  aggregations: [
    "cell_mean",
    "linear_reflectivity_mean",
    "block_max",
    "block_min",
    "magnitude_extreme",
    "point_sample",
  ],
  weighting: "uniform_or_physical_cell_volume",
  terrainFollowingWindowRead: true,
  terrainFollowingReductions: true,
  geometryLimitation:
    "terrain-following native windows use exact float64 interface-volume weights "
    + "and return physical-volume centroids; scalar/axis reductions stream bounded "
    + "interface slabs with the same weights. Plan views retain honest model levels; "
    + "vertical, volume, vector, Python, comparison, and radar-floor consumers use the "
    + "returned geometry for conservative physical-coordinate remapping or draping",
  maxActiveRequests: 64,
  npy: {
    windowRead: true,
    cancellation: "cooperative",
    limitation: null,
  },
  zfp3d: {
    windowRead: true,
    cancellation: "boundary_only",
    limitation:
      "v3 reads only intersecting 64x64x32 bricks; each libzfp brick decode is "
      + "non-interruptible, while legacy v1/v2 artifacts retain full-volume reconstruction",
  },
  netcdfClassic: {
    windowRead: true,
    cancellation: "cooperative",
    limitation: "bounded classic-netCDF source rows stop cooperatively between row reads",
  },
  netcdf4: {
    windowRead: true,
    cancellation: "boundary_only",
    limitation:
      "nc_get_vars float/double calls read bounded hyperslabs without an abort hook; "
      + "cancellation is checked at call boundaries and a cancelled result is discarded",
  },
} as const;

export interface TauriFixtureOptions {
  /**
   * Answer `native_field_read_capabilities` instead of rejecting it. Off by
   * default: the probe-failure state is itself under test
   * (`resource-budget-capabilities.spec.ts`), and it is reached precisely by
   * leaving this command unhandled.
   */
  nativeCapabilities?: boolean;
  intercept?: (command: string, payload: unknown) => Promise<{ value: unknown } | null>;
}

export function installTauriFixture(options: TauriFixtureOptions = {}): void {
  const trajectoryBuffer = encodeNpy(makeTrajectories());
  const realRunBridge = new URLSearchParams(window.location.search)
    .get("realRunBridge") === "beast-xl";
  mockIPC(async (command: string, payload?: InvokeArgs) => {
    try {
      const intercepted = await options.intercept?.(command, payload);
      if (intercepted) return intercepted.value;
      if (command === "read_field_bytes") {
        const args = (payload ?? {}) as Record<string, unknown>;
        const step = Number(args.stepIndex ?? 0);
        const fieldName = String(args.fieldName ?? "w");
        const runDir = String(args.runDir ?? "");
        // Lets a spec observe an in-flight multi-frame read: without a delay
        // every loop over the fixture's steps completes within one frame and
        // busy/cancel affordances are unobservable.
        if (fieldReadDelayMs > 0) {
          await new Promise((resolve) => setTimeout(resolve, fieldReadDelayMs));
        }
        const field = MOVING_CLOUD_FIXTURE && runDir.endsWith("/moving-child")
          ? movingCloudField(step, fieldName, true)
          : runDir.endsWith("/plane-large")
          ? makeLargePlaneField()
          : runDir.endsWith("/terrain-bricks")
          ? makeTerrainBrickField(step, fieldName)
          : runDir.endsWith("/bricked-derived") || runDir.endsWith("/nested-child") || runDir.endsWith("/nested-grandchild")
          ? makeDerivedBrickField(step, fieldName)
          : runDir.endsWith("/large")
          ? makeLargeField(fieldName)
          : runDir.endsWith("/coarse")
            ? makeCoarseField(step, fieldName)
            : makeField(step, fieldName);
        const ensembleOffset = runDir.endsWith("/ensemble-1") ? 4 : runDir.endsWith("/ensemble-2") ? -4 : 0;
        return deliver(encodeNpy(ensembleOffset === 0 ? field : {
          ...field,
          data: Float32Array.from(field.data, (value) => Number.isFinite(value) ? value + ensembleOffset : value),
        }));
      }
      if (command === "read_field_brick_hierarchy") {
        const args = (payload ?? {}) as Record<string, unknown>;
        const runDir = String(args.runDir ?? "");
        const fieldName = String(args.fieldName ?? "");
        if (runDir.endsWith("/large")
            && (fieldName === "cloud_condensate" || fieldName === "vorticity_z")) {
          return structuredClone(makeLargeBrickHierarchy(fieldName));
        }
        if ((runDir.endsWith("/bricked-derived") || runDir.endsWith("/nested-child") || runDir.endsWith("/nested-grandchild"))
            && derivedBrickSteps[0].fields.includes(fieldName)) {
          return structuredClone(makeDerivedBrickHierarchy(Number(args.stepIndex ?? 0), fieldName));
        }
        if (runDir.endsWith("/terrain-bricks") && ["u", "v", "w", "temperature"].includes(fieldName)) {
          return structuredClone(makeTerrainBrickHierarchy(Number(args.stepIndex ?? 0), fieldName));
        }
        throw new Error(`No brick hierarchy fixture for ${runDir}:${fieldName}`);
      }
      if (command === "process_memory_info") {
        return {
          schema: "nimbus.studio.process-memory",
          schemaVersion: 3,
          currentRssBytes: 384 * 1024 * 1024,
          peakRssBytes: 512 * 1024 * 1024,
          processTreeRssBytes: 640 * 1024 * 1024,
          processTreePeakRssBytes: 768 * 1024 * 1024,
          processTreeProcessCount: 3,
          processTreeCoverage: "host_plus_descendants",
          measurement: "browser-fixture",
        };
      }
      if (command === "read_field_brick_window") {
        const args = (payload ?? {}) as Record<string, unknown>;
        const request = (args.request ?? {}) as Record<string, unknown>;
        const step = Number(args.stepIndex ?? 0);
        const fieldName = String(args.fieldName ?? "w");
        const runDir = String(args.runDir ?? "");
        const level = Number(args.level ?? 0);
        const terrainLod = runDir.endsWith("/terrain-bricks")
          && ["u", "v", "w", "temperature"].includes(fieldName);
        const derivedLod = (runDir.endsWith("/bricked-derived") || runDir.endsWith("/nested-child") || runDir.endsWith("/nested-grandchild"))
          && derivedBrickSteps[0].fields.includes(fieldName);
        if ((!terrainLod && !derivedLod)
            || !Number.isSafeInteger(level) || level < 1) {
          throw new Error(`No durable brick LOD fixture for ${runDir}:${fieldName}:l${level}`);
        }
        if (fieldReadDelayMs > 0) {
          await new Promise((resolve) => setTimeout(resolve, fieldReadDelayMs));
        }
        const original = terrainLod
          ? makeTerrainBrickField(step, fieldName)
          : makeDerivedBrickField(step, fieldName);
        const lod = terrainLod
          ? makeTerrainBrickLod(step, level, fieldName)
          : makeDerivedBrickLod(step, fieldName, level);
        const sourceStride = 2 ** level;
        const origin = (request.origin as number[]) ?? original.shape.map(() => 0);
        const extent = (request.extent as number[]) ?? [...original.shape];
        const stride = (request.stride as number[]) ?? original.shape.map(() => sourceStride);
        if (stride.some((value) => value !== sourceStride)
            || origin.some((value) => value % sourceStride !== 0)) {
          throw new Error("Durable brick LOD fixture request is not level-aligned");
        }
        const lodOrigin = origin.map((value) => value / sourceStride);
        const lodExtent = extent.map((value) => Math.ceil(value / sourceStride));
        return deliver(encodeFieldWindow(
          String(request.requestId ?? ""),
          lod,
          lodOrigin,
          lodExtent,
          [1, 1, 1],
          "point_sample",
          request.maxOutputBytes as number | undefined,
          request.maxSourceBytes as number | undefined,
          terrainLod,
          {
            sourceShape: original.shape,
            origin,
            extent,
            stride,
            sourceFormat: "npy_lod",
          },
          terrainLod ? {
            ...terrainContourGrid,
            nr: lod.shape[0],
            nth: lod.shape[1],
            nz: lod.shape[2],
            drM: terrainContourGrid.drM! * sourceStride,
            dyM: terrainContourGrid.dyM! * sourceStride,
            dzM: terrainContourGrid.dzM! * sourceStride,
          } : grid,
        ));
      }
      if (command === "read_field_window") {
        const args = (payload ?? {}) as Record<string, unknown>;
        const request = (args.request ?? {}) as Record<string, unknown>;
        if (contextReadFailureEnabled && CONTEXT_READ_FAILURE_FIELD === args.fieldName
            && (request.stride as number[]).some(stride => stride > 1)) {
          throw new Error("Analytical fixture: required reduced context read exceeds budget");
        }
        if (MOVING_CLOUD_FIXTURE && request.aggregation === "point_sample") {
          const log = window as unknown as { movingWindReads?: { runDir: string; step: number; field: string; stride: number[] }[] };
          (log.movingWindReads ??= []).push({ runDir: String(args.runDir), step: Number(args.stepIndex), field: String(args.fieldName), stride: request.stride as number[] });
        }
        if (new URLSearchParams(window.location.search).get("compressedPlotBudget") === "1") {
          const log = (window as unknown as { compressedWindowRequests?: Record<string, unknown>[] });
          (log.compressedWindowRequests ??= []).push({ field: args.fieldName, ...request });
          if (request.aggregation !== "point_sample") {
            const sourceBytes = (request.extent as number[]).reduce((bytes, count) => bytes * count, 4);
            if (Number(request.maxSourceBytes) < sourceBytes + 512 * 1024) throw new Error("Compressed plot request omitted decoder scratch");
          }
        }
        if (request.aggregation === "point_sample" && (args.fieldName === "u" || args.fieldName === "v")
            && new URLSearchParams(window.location.search).get("windMotion") === "read-error") throw new Error("Wind point sampling unavailable for fixture");
        const step = Number(args.stepIndex ?? 0);
        const fieldName = String(args.fieldName ?? "w");
        const runDir = String(args.runDir ?? "");
        if (fieldReadDelayMs > 0) {
          await new Promise((resolve) => setTimeout(resolve, fieldReadDelayMs));
        }
        if (realRunBridge && runDir === BEAST_REFERENCE_RUN_DIR) {
          const response = await fetch("/__nimbus_real_run/field-window", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({ stepIndex: step, fieldName, request }),
          });
          if (!response.ok) throw new Error(await response.text());
          return deliver(await response.arrayBuffer());
        }
        const field = MOVING_CLOUD_FIXTURE && runDir.endsWith("/moving-child")
          ? movingCloudField(step, fieldName, true)
          : runDir.endsWith("/plane-large")
          ? makeLargePlaneField()
          : runDir.endsWith("/terrain-bricks")
          ? makeTerrainBrickField(step, fieldName)
          : runDir.endsWith("/bricked-derived") || runDir.endsWith("/nested-child") || runDir.endsWith("/nested-grandchild")
          ? makeDerivedBrickField(step, fieldName)
          : runDir.endsWith("/large")
          ? makeLargeField(fieldName)
          : runDir.endsWith("/coarse")
            ? makeCoarseField(step, fieldName)
            : makeField(step, fieldName);
        const ensembleOffset = runDir.endsWith("/ensemble-1") ? 4
          : runDir.endsWith("/ensemble-2") ? -4 : 0;
        const displayedField = ensembleOffset === 0 ? field : {
          ...field,
          data: Float32Array.from(field.data, (value) =>
            Number.isFinite(value) ? value + ensembleOffset : value),
        };
        return deliver(encodeFieldWindow(
          String(request.requestId ?? ""),
          displayedField,
          (request.origin as number[]) ?? field.shape.map(() => 0),
          (request.extent as number[]) ?? [...field.shape],
          (request.stride as number[]) ?? field.shape.map(() => 1),
          mockFieldWindowAggregation(request.aggregation),
          request.maxOutputBytes as number | undefined,
          request.maxSourceBytes as number | undefined,
          MOVING_CLOUD_FIXTURE?.startsWith("terrain") || runDir.includes("/terrain"),
          undefined,
          MOVING_CLOUD_FIXTURE ? movingCloudGrid(step, runDir.endsWith("/moving-child")) : runDir.endsWith("/terrain-bricks") ? terrainContourGrid : grid,
        ));
      }
      if (command === "inspect_netcdf_metadata") {
        const args = (payload ?? {}) as Record<string, unknown>;
        return { ...cm1MetadataFixture, path: String(args.path ?? cm1MetadataFixture.path) };
      }
      if (command === "read_netcdf_values") {
        const args = (payload ?? {}) as Record<string, unknown>;
        const variable = String(args.variable ?? "");
        if (variable === "time") return [0, 30, 60];
        if (variable === "xh") return Array.from({ length: FIELD_SHAPE[0] }, (_, index) => index * 125);
        if (variable === "yh") return Array.from({ length: FIELD_SHAPE[1] }, (_, index) => index * 175);
        if (variable === "zh") return Array.from({ length: FIELD_SHAPE[2] }, (_, index) => index * 90);
        throw new Error(`Unknown CM1 coordinate fixture ${variable}`);
      }
      if (command === "read_netcdf_field") {
        const args = (payload ?? {}) as Record<string, unknown>;
        const nativeName = String(args.variable ?? "winterp");
        const fieldName = CM1_NATIVE_TO_FIXTURE_FIELD[nativeName];
        if (!fieldName) throw new Error(`Unknown CM1 field fixture ${nativeName}`);
        return deliver(encodeNpy(makeField(Number(args.recordIndex ?? 0), fieldName)));
      }
      if (command === "read_netcdf_field_window") {
        const args = (payload ?? {}) as Record<string, unknown>;
        const request = (args.request ?? {}) as Record<string, unknown>;
        const nativeName = String(args.variable ?? "winterp");
        const fieldName = CM1_NATIVE_TO_FIXTURE_FIELD[nativeName];
        if (!fieldName) throw new Error(`Unknown CM1 field fixture ${nativeName}`);
        const field = makeField(Number(args.recordIndex ?? 0), fieldName);
        return deliver(encodeNetcdfFieldWindow(
          String(request.requestId ?? ""),
          field,
          (request.origin as number[]) ?? field.shape.map(() => 0),
          (request.extent as number[]) ?? [...field.shape],
          (request.stride as number[]) ?? field.shape.map(() => 1),
          mockFieldWindowAggregation(request.aggregation),
          request.maxOutputBytes as number | undefined,
          request.maxSourceBytes as number | undefined,
        ));
      }
      if (command === "read_trajectories") {
        return deliver(trajectoryBuffer.slice(0));
      }
      if (command === "run_has_trajectories") return true;
      /*
       * No storm track in the fixtures. Tracking is gated on
       * `storm_tracking.enabled` engine-side, so an untracked run is the common
       * case — and returning null here is what proves the UI treats absence as
       * ordinary rather than painting an error or a fabricated bearing.
       */
      if (command === "read_storm_track") return null;
      if (command === "save_cm1_sounding_projection") {
        const args = (payload ?? {}) as Record<string, unknown>;
        if (args.reviewed !== true) throw new Error("fixture requires explicit sounding review");
        return {
          schema: "nimbus.studio.saved-cm1-sounding-projection",
          schemaVersion: 1,
          canonicalPath: "/__nimbus_browser_fixture__/cm1-projected-sounding.txt",
          byteLength: String(args.contents ?? "").length,
          levelCount: 2,
        };
      }
      if (command === "inspect_cm1_sounding") {
        const args = (payload ?? {}) as Record<string, unknown>;
        return {
          schema: "nimbus.studio.cm1-sounding-report",
          schemaVersion: 1,
          path: String(args.path ?? "/fixture/input_sounding"),
          surface: { pressureHpa: 1_000, potentialTemperatureK: 300, mixingRatioGkg: 14 },
          levels: [
            { sourceLine: 2, heightM: 0, potentialTemperatureK: 300, mixingRatioGkg: 14, uMs: 5, vMs: 0 },
            { sourceLine: 3, heightM: 18_000, potentialTemperatureK: 410, mixingRatioGkg: 0, uMs: 20, vMs: 5 },
          ],
          topHeightM: 18_000,
          warnings: [],
        };
      }
      if (command === "scheme_catalog") return MOCK_SCHEME_CATALOG;
      if (command === "discover_run_library") {
        return [{
          id: "cm1-guided-run",
          caseName: "cm1-guided-smoke",
          outputDir: "/__nimbus_browser_fixture__/live",
          health: "ready",
          issue: null,
          startedAtMs: 1_700_000_000_000,
          modifiedAtMs: 1_700_000_030_000,
          frameCount: 3,
          latestStep: 2,
          simulationTimeS: 60,
          fieldCount: 13,
          outputFormat: "NPY",
          scanLimited: false,
          scanLimit: 512,
          provenance: {
            health: "verified",
            issue: null,
            runtimeHealth: "verified",
            runtimeIssue: null,
            runId: "cm1-guided-run",
            executablePath: "/__nimbus_browser_fixture__/bin/nimbus_sim",
            configurationDigest: "fnv1a64:f2b2b37c105d34a9",
            catalogFingerprint: "fnv1a64:cm1fixture",
            finalStatus: "completed",
            exitCode: 0,
            completeFrameCount: 3,
          },
        }];
      }
      if (command === "preflight_config") {
        const args = (payload ?? {}) as Record<string, unknown>;
        const request = args.request as {
          schema: string;
          requestVersion: number;
          selections: Array<{ moduleId: string; schemeId: string }>;
          [key: string]: unknown;
        };
        const { schema: _schema, requestVersion: _requestVersion, source: _source, selections, ...rawContext } = request;
        const context = normalizedPreflightContext(rawContext);
        return {
          schema: "nimbus.config-preflight-report.v10",
          reportVersion: 10,
          compatible: true,
          nativeAdmission: {
            state: "not_checked", descriptorProjection: "not_checked",
            sourceSha256: null, sourceFingerprint: null, sourcePathFingerprint: null,
            validatedBinarySha256: null, validationExitCode: null,
          },
          executable: {
            canonicalPath: "/__nimbus_browser_fixture__/bin/nimbus_sim",
            byteLength: 1_024,
            modifiedUnixMs: 1_700_000_000_000,
            catalogFingerprint: "fnv1a64:cm1fixture",
            binaryFingerprint: `sha256:${"0".repeat(64)}`,
          },
          context,
          resolvedSelections: selections.map((selection) => ({
            moduleId: selection.moduleId,
            requestedSchemeId: selection.schemeId,
            canonicalSchemeId: selection.schemeId,
            maturity: "validated",
            constructible: true,
            runtimeSelectable: true,
          })),
          hardErrors: [],
          warnings: [],
        };
      }
      if (command === "validate_cm1_sounding_project") {
        const args = (payload ?? {}) as Record<string, unknown>;
        if (!String(args.configYaml ?? "").includes("initial_conditions:")) {
          throw new Error("fixture requires the exact generated project YAML");
        }
        if (args.expectedCatalogFingerprint !== "fnv1a64:cm1fixture") {
          throw new Error("fixture requires catalog-bound initialization validation");
        }
        return {
          schema: "nimbus.studio.cm1-sounding-project-validation",
          schemaVersion: 1,
          executablePath: "/__nimbus_browser_fixture__/bin/nimbus_sim",
          summary: "[VALIDATE] PASS: fixture file/sharpy initialization",
          warnings: [],
        };
      }
      /*
       * Text file write. `writePySnippet` and the config editor both route
       * through this, and the bytes are KEPT: a save the reader cannot see
       * afterwards would let a spec assert "Saved" and prove nothing.
       */
      if (command === "write_config") {
        const args = (payload ?? {}) as Record<string, unknown>;
        writeVirtualFile(String(args.path ?? ""), String(args.contents ?? ""));
        return null;
      }
      /*
       * Text-file read. `readPySnippet` routes through this, so a Python buffer
       * restored by path alone can actually be re-read rather than staying
       * permanently unloaded.
       *
       * A path nothing has written is `not_found`, exactly as the Rust command
       * reports it. The previous fixture answered every path with invented
       * contents, which meant no spec could ever exercise the missing-file
       * branch and the UI's handling of it was untested by construction.
       * Specs put files on the disk with `fixtureControl.writeFile`.
       */
      if (command === "read_config") {
        const args = (payload ?? {}) as Record<string, unknown>;
        const path = String(args.path ?? "");
        assertConfigPathAllowed(path);
        const file = virtualFiles.get(path.trim());
        if (!file) throw appError("not_found", `path not found: ${path}`);
        return new TextDecoder().decode(file.bytes);
      }
      /** Ported from `plugins_dir`: the directory always exists by the time it
       *  is named, because the real command creates it. */
      if (command === "plugins_dir") return FIXTURE_PLUGINS_DIR;
      /*
       * Ported from `list_plugins`: `.py` files directly in the plugins
       * directory, newest first, ties broken by name. Top-level only — a path
       * in a subdirectory is skipped, as the real reader skips it.
       */
      if (command === "list_plugins") {
        const prefix = `${FIXTURE_PLUGINS_DIR}/`;
        return [...virtualFiles]
          .filter(([path]) => path.startsWith(prefix)
            && !path.slice(prefix.length).includes("/")
            && path.toLowerCase().endsWith(".py"))
          .map(([path, file]) => ({
            name: path.slice(prefix.length),
            path,
            sizeBytes: file.bytes.byteLength,
            modifiedMs: file.modifiedMs,
          }))
          .sort((a, b) => b.modifiedMs - a.modifiedMs || a.name.localeCompare(b.name));
      }
      /*
       * The save dialog stands in for the user choosing a name.
       *
       * A spec can dictate the answer (including Cancel) through
       * `fixtureControl.setSaveDialogPath`; otherwise the choice is derived
       * from what was asked for, so the returned path has the extension the
       * caller filtered on. `.gif` keeps its historical fixed answer because
       * the recorder specs assert on that exact path.
       */
      if (command === "plugin:dialog|save") {
        const options = ((payload ?? {}) as Record<string, unknown>).options as
          { defaultPath?: string } | undefined;
        const requested = options?.defaultPath ?? "";
        const chosen = nextSavePath !== undefined
          ? nextSavePath
          : requested.toLowerCase().endsWith(".gif")
            ? "/__nimbus_browser_fixture__/recording.gif"
            : requested.includes("/")
              ? requested
              : `${FIXTURE_PLUGINS_DIR}/${requested}`;
        if (chosen !== null) lastChosenSavePath = chosen;
        if (chosen !== null && chosen.toLowerCase().endsWith(".gif")) {
          savedGifMetrics.path = chosen;
        }
        return chosen;
      }
      /** The open dialog. Cancel by default: a spec that has not named a file
       *  has not chosen one, and inventing a path would make the cancel branch
       *  unreachable. */
      if (command === "plugin:dialog|open") {
        return nextOpenPath ?? null;
      }
      /** Scoped fs reads used by Python's Open data flow. These deliberately
       * bypass the config extension allowlist, as the production fs capability
       * permits the selected nc/npy/npz file rather than treating it as config. */
      if (command === "plugin:fs|stat") {
        const path = String(((payload ?? {}) as Record<string, unknown>).path ?? "");
        const file = virtualFiles.get(path.trim());
        if (!file) throw appError("not_found", `path not found: ${path}`);
        return {
          isFile: true,
          isDirectory: false,
          isSymlink: false,
          size: file.bytes.byteLength,
          mtime: file.modifiedMs,
          atime: null,
          birthtime: null,
          readonly: false,
          fileAttributes: null,
          dev: 0,
          ino: 0,
          mode: 0,
          nlink: 1,
          uid: 0,
          gid: 0,
          rdev: 0,
          blksize: 0,
          blocks: 0,
        };
      }
      if (command === "plugin:fs|read_file") {
        const path = String(((payload ?? {}) as Record<string, unknown>).path ?? "");
        const file = virtualFiles.get(path.trim());
        if (!file) throw appError("not_found", `path not found: ${path}`);
        return [...file.bytes];
      }
      /*
       * The binary write. Lands on the virtual disk as well as in the metrics:
       * recording only a byte count made every export — figure, GIF, array —
       * assertable on its size and never on its contents, so a spec could
       * confirm that something was written but not that it was the right thing.
       */
      if (command === "plugin:fs|write_file") {
        const raw = payload instanceof Uint8Array
          ? payload
          : Array.isArray(payload)
            ? Uint8Array.from(payload as number[])
            : new Uint8Array();
        savedGifMetrics.byteLength = raw.byteLength;
        savedGifMetrics.signature = new TextDecoder().decode(raw.slice(0, 6));
        savedGifMetrics.writes += 1;
        // See `lastChosenSavePath`: the real path is in headers this mock never
        // receives, so the dialog's answer is the destination.
        if (lastChosenSavePath) writeVirtualBytes(lastChosenSavePath, raw);
        return null;
      }
      if (command === "list_step_dirs") {
        const args = (payload ?? {}) as Record<string, unknown>;
        const runDir = String(args.runDir ?? "/__nimbus_browser_fixture__");
        if (realRunBridge && runDir === BEAST_REFERENCE_RUN_DIR) {
          return beastReferenceSteps.filter((step) => step.index === 23 || step.index === 24).map(step => ({
            ...step, fields: step.fields.filter(field => step.fieldOrigins?.[field] !== "studio_derived"),
          }));
        }
        listCalls.set(runDir, (listCalls.get(runDir) ?? 0) + 1);
        const delay = listDelays.get(runDir) ?? 0;
        if (delay > 0) await new Promise((resolve) => setTimeout(resolve, delay));
        if (runDir.endsWith("/legacy")) {
          return steps.slice(0, 2).map((step) => ({
            ...step,
            publicationState: "legacy",
            publicationGeneration: null,
          }));
        }
        if (runDir.endsWith("/fast")) return steps.slice(0, 4);
        if (runDir.endsWith("/slow")) return steps.slice(0, 1);
        if (runDir.endsWith("/terrain")) return terrainSteps;
        if (runDir.endsWith("/stretched")) return stretchedSteps;
        if (runDir.endsWith("/plane-large")) return largePlaneSteps;
        if (runDir.endsWith("/live")) {
          const visible = steps.slice(0, liveStepCount);
          return [
            ...visible,
            {
              ...(steps[liveStepCount] ?? steps[steps.length - 1]),
              index: liveStepCount,
              simulationTimeS: liveStepCount * 30,
              publicationState: "incomplete",
            },
          ];
        }
        return steps;
      }
      // The default browser fixture models a root-only run. Nested-history
      // discovery is still a real command and must answer with an empty graph,
      // not inject an IPC failure into every plot's source disclosure.
      if (command === "list_nested_history_levels") {
        if (MOVING_CLOUD_FIXTURE?.endsWith("nested")) return [{ levelId: "child_0000", parentLevelId: "root", depth: 1,
          runDir: "/__nimbus_browser_fixture__/moving-child", steps: steps.slice(1,3).map(step => ({ ...step, grid: movingCloudGrid(step.index,true) })) }];
        const nestedDepth = Number(new URLSearchParams(window.location.search).get("nestedVolumes") ?? 0);
        const args = (payload ?? {}) as Record<string, unknown>;
        const source = String(args.runDir ?? "");
        if (nestedDepth > 0 && source.endsWith("/bricked-derived")) {
          return [{
            levelId: "child_0000",
            parentLevelId: "root",
            depth: 1,
            runDir: "/__nimbus_browser_fixture__/nested-child",
            steps: nestedChildBrickSteps,
          }, ...(nestedDepth > 1 ? [{
            levelId: "child_0000/child_0000",
            parentLevelId: "child_0000",
            depth: 2,
            runDir: "/__nimbus_browser_fixture__/nested-grandchild",
            steps: nestedGrandchildBrickSteps,
          }] : [])];
        }
        return [];
      }
      if (command === "native_field_read_capabilities" && options.nativeCapabilities) {
        return NATIVE_READ_CAPABILITIES;
      }
      if (command === "cancel_field_read") {
        const args = (payload ?? {}) as Record<string, unknown>;
        return {
          requestId: String(args.requestId ?? ""),
          found: false,
          cancellationRequested: false,
          phase: null,
          interruptibility: null,
        };
      }
      throw new Error(`Unhandled integration IPC command: ${command}`);
    } catch (error) {
      ipcMetrics.failures += 1;
      if (ipcMetrics.failureMessages.length < 32) {
        ipcMetrics.failureMessages.push(
          `${command}: ${error instanceof Error ? error.message : String(error)}`,
        );
      }
      throw error;
    }
  });
}
