import { PlaybackProvider, type PlaybackContextValue } from "@nimbus/components/plots/playbackContext";
import { PanePresentationProvider } from "@nimbus/components/plots/panePresentationContext";
import { collectVtkAppearanceEvidence } from "@nimbus/lib/vtk/vtkAppearanceEvidence";
import type { PanePresentationSnapshot } from "@nimbus/lib/plot/panePresentationBarrier";
import { RegularDataflowHarness } from "./regular-dataflow";
import { VolumeOwnerLifecycleHarness, volumeOwnerInvoke } from "./volume-owner-lifecycle";
import { PublishedVolumeOverviewHarness } from "./published-overview";
import { PlaybackScreenHarness } from "./playback-screen";
import { PlotResolutionHarness } from "./plot-resolution";
import { ResearchHarness } from "./research";
import { researchInvoke } from "../support/researchFixture";
import { regularInvoke } from "../support/regularDataflowFixture";
import { publishedOverviewInvoke } from "../support/publishedOverviewFixture";
import { recorderFixture, recorderInvoke } from "../support/recorderFixture";
import type vtkPolyData from "@kitware/vtk.js/Common/DataModel/PolyData";
import type vtkActor from "@kitware/vtk.js/Rendering/Core/Actor";
import type vtkRenderer from "@kitware/vtk.js/Rendering/Core/Renderer";
import { getVolumeValidity } from "@nimbus/lib/vtk/volumeValidity";
import { snapshotVtkOverlays } from "@nimbus/lib/vtk/vtkOverlayCapture";
import { flatVolumePreview } from "@nimbus/lib/vtk/volumeShaderBudget";
import { createRoot } from "react-dom/client";
import { qualifyVolumeShadowSampling } from "../support/volumeShadowProbe";
import { qualifyJointVolumeRendering } from "../support/jointVolumeProbe";
import { qualifyVolumeValidity } from "../support/volumeValidityProbe";
import { qualifyVolumeArrayReuse } from "../support/volumeArrayReuseProbe";
import { qualifyVolumeScalarPrecision } from "../support/volumeScalarPrecisionProbe";
import { qualifyVolumeTexturePacking } from "../support/volumeTexturePackingProbe";
import { qualifyVolumeAngularSampling } from "../support/volumeAngularProbe";
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { ComponentType } from "react";
import type {
  FrameDisplayReport,
  PlotComponentProps,
} from "@nimbus/components/plots/common";
import { PaneProvider } from "@nimbus/components/plots/paneContext";
import type { DisplayedFrameIdentity } from "@nimbus/lib/run/frameIdentity";
import type { LocalDatasetSource } from "@nimbus/lib/dataset/datasetSources";
import { clearFieldCache, fieldCacheStats } from "@nimbus/lib/core/fieldCache";
import { fieldLoadSchedulerStats } from "@nimbus/lib/core/loadScheduler";
import { inflightPrefetchCount } from "@nimbus/lib/run/framePrefetch";
import { registerExpressionDerivedFields, registerListedStepMetadata } from "@nimbus/lib/run/runOutputs";
import { augmentStepWithExpressionDerivedFields, createExpressionDerivedField } from "@nimbus/lib/derived/expressionDerivedFields";
import { captureFrameDataUri } from "@nimbus/lib/run/frameCapture";
import {
  lookupVtkAppearanceCapture,
  lookupVtkRenderWindow,
  registerVtkCapture,
  unregisterVtkCapture,
  type VtkAppearanceCapture,
} from "@nimbus/lib/vtk/vtkCaptureRegistry";
import {
  getVisualizationResourceSnapshot,
  visualizationResourceGovernor,
  type VisualizationResourceSnapshot,
} from "@nimbus/lib/core/visualizationResourceGovernor";
import { preparedDatasetRegistrySnapshot } from "@nimbus/lib/vtk/vtkPreparedDatasetRegistry";
import { vtkExtractionServicePool } from "@nimbus/lib/vtk/vtkExtractionServicePool";
import { workerOwnedContourProgressSnapshot } from "@nimbus/lib/vtk/contourWorkerOwnedPreparation";
import {
  compareVtkAppearanceImages,
  type VtkAppearanceImageMetrics,
} from "@nimbus/lib/vtk/vtkAppearanceComparison";
import {
  volumeAppearanceAblation,
  type VolumeAppearanceQualificationOverrides,
} from "@nimbus/lib/vtk/volumeAppearanceQualification";
import {
  installBrowserInstrumentation,
  snapshotBrowserResources,
  type BrowserResourceSnapshot,
} from "../support/instrumentation";
import {
  coarseSteps,
  brickedLargeSteps,
  beastReferenceSteps,
  BEAST_REFERENCE_RUN_DIR,
  derivedBrickSteps,
  fixtureControl,
  installTauriFixture,
  ipcMetrics,
  LARGE_FIELD_SHAPE,
  largeSteps,
  makeField,
  steps,
  terrainBrickSteps,
} from "../support/fixtures";
import {
  qualifyProductionGeometryWorkers,
  type ProductionGeometryQualification,
} from "../support/productionGeometry";
import "@nimbus/styles/theme.css";
import "@nimbus/styles/base.css";
import "../support/styles.css";

installBrowserInstrumentation();
// The production plot harness qualifies the bounded native-reader path. The
// resource-budget harness separately leaves this probe unhandled to exercise
// compatibility with older hosts.
installTauriFixture({ nativeCapabilities: true,
  intercept: new URLSearchParams(window.location.search).get("mode") === "research" ? researchInvoke : new URLSearchParams(window.location.search).get("mode") === "regular-dataflow" ? regularInvoke
    : new URLSearchParams(window.location.search).get("mode") === "published-overview" ? publishedOverviewInvoke
    : new URLSearchParams(window.location.search).get("mode") === "recorder" ? recorderInvoke
    : new URLSearchParams(window.location.search).get("mode") === "volume-owner" ? volumeOwnerInvoke : undefined });
const qualificationRam = Number(new URLSearchParams(window.location.search).get("volumeRamGiB"));
if (Number.isFinite(qualificationRam) && qualificationRam >= 2 && qualificationRam <= 128) {
  visualizationResourceGovernor.setHardwareFacts({ source: "browser", systemRamBytes: qualificationRam * 1024 ** 3, logicalCores: 8 });
}
const BRICKED_LARGE_HARNESS = new URLSearchParams(window.location.search).get("bricks") === "1";
const TERRAIN_BRICK_HARNESS = new URLSearchParams(window.location.search)
  .get("terrainBricks") === "1";
const MONOLITHIC_VECTOR_HARNESS = new URLSearchParams(window.location.search)
  .get("monolithicVectors") === "1";
const MONOLITHIC_VOLUME_HARNESS = new URLSearchParams(window.location.search)
  .get("volumeBricks") === "off";
const EXPRESSION_VOLUME_HARNESS = new URLSearchParams(window.location.search).get("definitionProbe") === "1";
const EXPRESSION_FLOOR_HARNESS = new URLSearchParams(window.location.search).get("floorDefinitionProbe") === "1";
let expressionProbeMultiplier = 2;
let expressionProbeSteps = steps;
function installExpressionProbe(multiplier: number): void {
  expressionProbeMultiplier = multiplier;
  const definition = createExpressionDerivedField({
    id: EXPRESSION_FLOOR_HARNESS ? "theta_prime" : "expression_probe",
    label: "Manufactured expression probe", unit: EXPRESSION_FLOOR_HARNESS ? "K" : "m/s",
    dependencies: ["w"], parameters: [{ id: "gain", value: multiplier }], formula: "w * gain" });
  registerExpressionDerivedFields("/__nimbus_browser_fixture__", [definition]);
  expressionProbeSteps = steps.map(step => augmentStepWithExpressionDerivedFields(EXPRESSION_FLOOR_HARNESS
    ? { ...step, fields: step.fields.filter(field => field !== "theta_prime") } : step, [definition]));
  if (EXPRESSION_FLOOR_HARNESS) registerListedStepMetadata("/__nimbus_browser_fixture__", expressionProbeSteps);
}
if (EXPRESSION_VOLUME_HARNESS || EXPRESSION_FLOOR_HARNESS) installExpressionProbe(expressionProbeMultiplier);
const MONOLITHIC_DERIVED_STEPS = derivedBrickSteps.map((step) => ({
  ...step,
  fieldBrickHierarchies: {},
}));
registerListedStepMetadata("/__nimbus_browser_fixture__", EXPRESSION_FLOOR_HARNESS ? expressionProbeSteps : steps);
registerListedStepMetadata("/__nimbus_browser_fixture__/coarse", coarseSteps);
registerListedStepMetadata(
  "/__nimbus_browser_fixture__/large",
  BRICKED_LARGE_HARNESS ? brickedLargeSteps : largeSteps,
);
registerListedStepMetadata(BEAST_REFERENCE_RUN_DIR, beastReferenceSteps);
registerListedStepMetadata("/__nimbus_browser_fixture__/bricked-derived", derivedBrickSteps);
registerListedStepMetadata("/__nimbus_browser_fixture__/terrain-bricks", terrainBrickSteps);

const [
  { VolumeRender3D },
  { Isosurface3D },
  { Streamlines3D },
  { SupercellRender3D },
  { Trajectories3D },
  { PlotStudio },
  { Cm1CompatibilityReport },
  { MovieRecorderDialog },
] = await Promise.all([
  import("@nimbus/components/plots/VolumeRender3D"),
  import("@nimbus/components/plots/Isosurface3D"),
  import("@nimbus/components/plots/Streamlines3D"),
  import("@nimbus/components/plots/SupercellRender3D"),
  import("@nimbus/components/plots/Trajectories3D"),
  import("@nimbus/components/studio/PlotStudio"),
  import("@nimbus/components/cm1/Cm1CompatibilityReport"),
  import("@nimbus/components/studio/MovieRecorderDialog"),
]);

export const PLOT_TYPES = [
  "volume",
  "isosurface",
  "streamlines",
  "supercell",
  "trajectories",
] as const;
export type HarnessPlotType = typeof PLOT_TYPES[number];

interface ScalarImage {
  shape: number[];
  spacing: number[];
  origin: number[];
  values: Float32Array;
}

function mountedScalarImages(): ScalarImage[] {
  const renderer = lookupVtkRenderWindow(document.querySelector(".volume-render__canvas"))?.getRenderer() as {
    getVolumes(): Array<{
      getVisibility(): boolean;
      getMapper(): { getInputData(): {
        getDimensions(): number[]; getSpacing(): number[]; getOrigin(): number[];
        getPointData(): { getScalars(): { getData(): Float32Array } };
      } };
    }>;
  } | undefined;
  if (!renderer) throw new Error("No scalar scene");
  return renderer.getVolumes().filter((volume) => volume.getVisibility()).map((volume) => {
    const image = volume.getMapper().getInputData();
    return { shape: image.getDimensions(), spacing: image.getSpacing(), origin: image.getOrigin(),
      values: image.getPointData().getScalars().getData() };
  });
}

function inspectBoundedVolumeScalarImages(): Array<Omit<ScalarImage, "values"> & { values: number[] }> {
  const images = mountedScalarImages();
  if (images.reduce((bytes, image) => bytes + image.values.byteLength, 0) > 2 * 1024 * 1024) {
    throw new Error("Scalar value inspection requires bounded analytic images");
  }
  return images.map(image => ({ ...image, values: Array.from(image.values) }));
}

export interface HarnessSnapshot extends BrowserResourceSnapshot {
  plot: HarnessPlotType;
  step: number;
  mounted: boolean;
  coarse: boolean;
  large: boolean;
  paneCount: number;
  canvasCount: number;
  canvasWidth: number | null;
  canvasHeight: number | null;
  canvasMutations: number;
  fixtureArrayBufferBytes: number;
  captureFailures: number;
  visualization: VisualizationResourceSnapshot;
  preparedRegistry: ReturnType<typeof preparedDatasetRegistrySnapshot>;
  extractionPool: ReturnType<typeof vtkExtractionServicePool.snapshot>;
  contourProgress: ReturnType<typeof workerOwnedContourProgressSnapshot>;
  ipc: typeof ipcMetrics;
  fieldCache: ReturnType<typeof fieldCacheStats>;
  fieldScheduler: ReturnType<typeof fieldLoadSchedulerStats>;
  fieldPrefetches: number;
  backgroundLoad: {
    workers: number;
    iterations: number;
    checksum: number;
  };
}

const referenceIds = new WeakMap<object, number>();
let nextReferenceId = 0;

export interface NimbusVtkHarness {
  qualifyVolumeShadows: typeof qualifyVolumeShadowSampling;
  qualifyJointVolumes: typeof qualifyJointVolumeRendering;
  qualifyVolumeValidity: typeof qualifyVolumeValidity;
  qualifyVolumeArrayReuse: typeof qualifyVolumeArrayReuse;
  qualifyVolumeScalarPrecision: typeof qualifyVolumeScalarPrecision;
  qualifyVolumeTexturePacking: typeof qualifyVolumeTexturePacking;
  qualifyVolumeAngularSampling: typeof qualifyVolumeAngularSampling;
  plotTypes: readonly HarnessPlotType[];
  setPlot: (plot: HarnessPlotType) => void;
  setStep: (step: number) => void;
  setMounted: (mounted: boolean) => void;
  setCoarse: (coarse: boolean) => void;
  setPlayback: (playing: boolean, recording?: boolean, durationMs?: number) => void;
  setSize: (width: number, height: number) => void;
  setBrickedVolume: (enabled: boolean) => void;
  captureScalarReference: () => void;
  compareScalarReference: () => { checked: number; maximumError: number };
  inspectSurfaceGap: (lowerY: number, upperY: number) => { triangles: number; crossings: number; nonFinite: number };
  inspectVolumeSupport: () => { index: number; missing: number; finite: boolean }[];
  inspectVolumeLighting: () => { shade: boolean; flatPreview: boolean }[];
  inspectVolumeInputs: () => { volumes: number; images: number; bytes: number; minimum: number; zeros: number; fingerprints: number[] };
  inspectVolumeScalarImages: () => Array<Omit<ScalarImage, "values"> & { values: number[] }>;
  resizeStorm: () => void;
  loseContext: () => boolean;
  restoreContext: () => boolean;
  capture: () => boolean;
  captureAppearance: () => Promise<VtkAppearanceCapture | null>;
  inspectSurfaceNormals: () => Array<{ points: number; normals: number; strips: number }>;
  inspectReferences: () => Array<{ id: number; points: number; bounds: number[] }>;
  inspectLighting: () => Array<{ direction: number[]; color: number[]; intensity: number; type: string }>;
  captureSavedFrame: () => Promise<string>;
  capturePortableViewpoints: (count: number, cancelAfter?: number) => Promise<string>;
  inspectLastSavedFrame: () => VtkAppearanceCapture["evidence"] | null;
  inspectExportOverlays: () => ReturnType<typeof snapshotVtkOverlays>;
  setCamera: (position: [number, number, number], focalPoint: [number, number, number], parallel?: boolean) => Promise<void>;
  compareAppearance: (
    referenceDataUri: string,
    candidateDataUri: string,
    changedChannelThreshold?: number,
  ) => Promise<VtkAppearanceImageMetrics>;
  snapshot: () => HarnessSnapshot;
  /**
   * Drop the decoded-field cache and report the bytes released.
   *
   * The cache is a deliberate LRU with a 512 MB cap, so a soak that cycles 20
   * steps across every plot type grows the heap by design. A retention check
   * that leaves it in place measures how warm the cache got, which is why the
   * soak clears it before reading the heap.
   */
  clearFieldCache: () => number;
  qualifyProductionGeometry: () => Promise<ProductionGeometryQualification>;
  setFieldReadDelay: (milliseconds: number) => void;
  setExpressionMultiplier: (multiplier: number) => void;
  startBackgroundLoad: (workers?: number) => void;
  stopBackgroundLoad: () => void;
}

declare global {
  interface Window {
    __nimbusVtkHarness?: NimbusVtkHarness;
    __nimbusStudioHarness?: {
      setRun: (runDir: string, status?: "running" | "completed") => void;
      publishLiveThrough: (count: number) => void;
      setListDelay: (runDir: string, milliseconds: number) => void;
      setFieldReadDelay: (milliseconds: number) => void;
      clearFieldCache: () => number;
      captureActivePlot: () => Promise<string | null>;
      snapshot: typeof fixtureControl.snapshot;
      resourceSnapshot: () => ReturnType<typeof snapshotBrowserResources> & { visualization: VisualizationResourceSnapshot };
      capturePaneAppearances: () => Promise<Array<VtkAppearanceCapture | null>>;
      inspectPrimaryVolumeScalarImages: typeof inspectBoundedVolumeScalarImages;
      presentationSnapshot: () => PanePresentationSnapshot | null;
      readExportFile: (path: string) => string | null;
    };
    __nimbusRecorderHarness?: {
      snapshot: () => ReturnType<typeof fixtureControl.snapshot> & {
        capturedSteps: number[];
        capturePending: boolean;
        savedPng: number[];
        nativeEncoding: ReturnType<typeof recorderFixture.snapshot>;
      };
      releaseCapture: (changeIdentity?: boolean) => void;
      releasePreparation: () => void;
      releaseProbe: () => void;
    };
  }
}

let canvasMutations = 0;
let captureFailures = 0;
let backgroundLoadWorkers: Worker[] = [];
let backgroundLoadIterations = 0;
let backgroundLoadChecksum = 0;

function stopBackgroundLoad(): void {
  for (const worker of backgroundLoadWorkers) worker.terminate();
  backgroundLoadWorkers = [];
}

function startBackgroundLoad(count = 2): void {
  stopBackgroundLoad();
  backgroundLoadIterations = 0;
  backgroundLoadChecksum = 0;
  const boundedCount = Math.max(1, Math.min(4, Math.floor(count)));
  backgroundLoadWorkers = Array.from({ length: boundedCount }, () => {
    const worker = new Worker(
      new URL("../support/backgroundLoad.worker.ts", import.meta.url),
      { type: "module", name: "nimbus-background-load" },
    );
    worker.onmessage = (event: MessageEvent<{ iterations: number; checksum: number }>) => {
      backgroundLoadIterations += 4;
      backgroundLoadChecksum += event.data.checksum;
    };
    worker.postMessage({ kind: "start" });
    return worker;
  });
}
const rootElement = document.getElementById("root");
if (!rootElement) throw new Error("Harness root missing");
new MutationObserver((records) => {
  canvasMutations += records.filter((record) =>
    record.type === "attributes" && record.target instanceof HTMLCanvasElement
  ).length;
}).observe(rootElement, {
  subtree: true,
  attributes: true,
  attributeFilter: ["width", "height"],
});

function canvas(): HTMLCanvasElement | null {
  return document.querySelector(".harness-scene canvas");
}

async function rgbaImage(dataUri: string) {
  const image = new Image();
  image.src = dataUri;
  await new Promise<void>((resolve, reject) => {
    image.onload = () => resolve();
    image.onerror = () => reject(new Error("appearance PNG decode failed"));
  });
  const target = document.createElement("canvas");
  target.width = image.naturalWidth;
  target.height = image.naturalHeight;
  const context = target.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("appearance comparison canvas is unavailable");
  context.drawImage(image, 0, 0);
  return {
    width: target.width,
    height: target.height,
    data: context.getImageData(0, 0, target.width, target.height).data,
  };
}

let lastLoseExtension: WEBGL_lose_context | null = null;

function contextExtension(): WEBGL_lose_context | null {
  const target = canvas();
  if (!target) return null;
  const gl = target.getContext("webgl2") ?? target.getContext("webgl");
  const extension = gl?.getExtension("WEBGL_lose_context") ?? null;
  if (extension) lastLoseExtension = extension;
  return extension;
}

function snapshot(
  plot: HarnessPlotType,
  step: number,
  mounted: boolean,
  coarse: boolean,
  large: boolean,
  paneCount: number,
  beastReference = false,
): HarnessSnapshot {
  const target = canvas();
  return {
    ...snapshotBrowserResources(),
    plot,
    step,
    mounted,
    coarse,
    large,
    paneCount,
    canvasCount: document.querySelectorAll(".harness-scene canvas").length,
    canvasWidth: target?.width ?? null,
    canvasHeight: target?.height ?? null,
    canvasMutations,
    fixtureArrayBufferBytes: beastReference
      ? 512 * 512 * 160 * Float32Array.BYTES_PER_ELEMENT
      : large
      ? LARGE_FIELD_SHAPE.reduce((product, count) => product * count, 1) * Float32Array.BYTES_PER_ELEMENT
      : makeField(step).data.byteLength,
    captureFailures,
    visualization: getVisualizationResourceSnapshot(),
    preparedRegistry: preparedDatasetRegistrySnapshot(),
    extractionPool: vtkExtractionServicePool.snapshot(),
    contourProgress: workerOwnedContourProgressSnapshot(),
    ipc: { ...ipcMetrics },
    fieldCache: fieldCacheStats(),
    fieldScheduler: fieldLoadSchedulerStats(),
    fieldPrefetches: inflightPrefetchCount(),
    backgroundLoad: {
      workers: backgroundLoadWorkers.length,
      iterations: backgroundLoadIterations,
      checksum: backgroundLoadChecksum,
    },
  };
}

function StandardPlot({ component: Plot, step, coarse, large, bricked, derivedBricks, beastReference, volumeAppearanceQualification, onFrameDisplayChange }: {
  component: ComponentType<PlotComponentProps>;
  step: number;
  coarse: boolean;
  large: boolean;
  bricked: boolean;
  derivedBricks: boolean;
  beastReference?: boolean;
  volumeAppearanceQualification?: VolumeAppearanceQualificationOverrides;
  onFrameDisplayChange: (report: FrameDisplayReport) => void;
}) {
  return (
    <Plot
      runDir={derivedBricks
        ? "/__nimbus_browser_fixture__/bricked-derived"
        : TERRAIN_BRICK_HARNESS
        ? "/__nimbus_browser_fixture__/terrain-bricks"
        : beastReference
        ? BEAST_REFERENCE_RUN_DIR
        : large
        ? "/__nimbus_browser_fixture__/large"
        : coarse ? "/__nimbus_browser_fixture__/coarse" : "/__nimbus_browser_fixture__"}
      steps={derivedBricks
        ? MONOLITHIC_VECTOR_HARNESS && Plot === Streamlines3D
          ? MONOLITHIC_DERIVED_STEPS
          : derivedBrickSteps
        : TERRAIN_BRICK_HARNESS
        ? terrainBrickSteps
        : beastReference
        ? beastReferenceSteps
        : large
          ? bricked ? brickedLargeSteps : largeSteps
          : coarse ? coarseSteps : EXPRESSION_VOLUME_HARNESS || EXPRESSION_FLOOR_HARNESS ? expressionProbeSteps : steps}
      stepIdx={step}
      renderFidelityMode={coarse || new URLSearchParams(window.location.search).get("fidelity") === "presentation" ? "presentation" : "scientific"}
      onFrameDisplayChange={onFrameDisplayChange}
      volumeAppearanceQualification={volumeAppearanceQualification}
    />
  );
}

function Scene({ plot, step, coarse, large, bricked, derivedBricks, beastReference, volumeAppearanceQualification, onFrameDisplayChange }: {
  plot: HarnessPlotType;
  step: number;
  coarse: boolean;
  large: boolean;
  bricked: boolean;
  derivedBricks: boolean;
  beastReference?: boolean;
  volumeAppearanceQualification?: VolumeAppearanceQualificationOverrides;
  onFrameDisplayChange: (plot: HarnessPlotType, report: FrameDisplayReport) => void;
}) {
  const reportFrame = useCallback(
    (report: FrameDisplayReport) => onFrameDisplayChange(plot, report),
    [onFrameDisplayChange, plot],
  );
  switch (plot) {
    case "volume": return <StandardPlot component={VolumeRender3D} step={step} coarse={coarse} large={large} bricked={bricked} derivedBricks={derivedBricks} beastReference={beastReference} volumeAppearanceQualification={volumeAppearanceQualification} onFrameDisplayChange={reportFrame} />;
    case "isosurface": return <StandardPlot component={Isosurface3D} step={step} coarse={coarse} large={large} bricked={bricked} derivedBricks={derivedBricks} beastReference={beastReference} onFrameDisplayChange={reportFrame} />;
    case "streamlines": return <StandardPlot component={Streamlines3D} step={step} coarse={coarse} large={large} bricked={bricked} derivedBricks={derivedBricks} beastReference={beastReference} onFrameDisplayChange={reportFrame} />;
    case "supercell": return <StandardPlot component={SupercellRender3D} step={step} coarse={coarse} large={large} bricked={bricked} derivedBricks={derivedBricks} beastReference={beastReference} onFrameDisplayChange={reportFrame} />;
    case "trajectories": return <StandardPlot component={Trajectories3D} step={step} coarse={coarse} large={large} bricked={bricked} derivedBricks={derivedBricks} beastReference={beastReference} onFrameDisplayChange={reportFrame} />;
  }
}

function Harness() {
  const savedFrameEvidence = useRef<VtkAppearanceCapture["evidence"] | null>(null);
  const scalarReference = useRef<ScalarImage | null>(null);
  const harnessParams = new URLSearchParams(window.location.search);
  const large = harnessParams.get("large") === "1";
  const bricked = harnessParams.get("bricks") === "1";
  const derivedBricks = harnessParams.get("derivedBricks") === "1";
  const forceBrickLod = harnessParams.get("brickLod") === "1";
  const beastReference = harnessParams.get("realRunBridge") === "beast-xl";
  const appearanceAblation = volumeAppearanceAblation(harnessParams.get("volumeAblation"));
  const requestedPaneCount = Number.parseInt(harnessParams.get("panes") ?? "1", 10);
  const paneCount = Math.max(1, Math.min(4, requestedPaneCount || 1));
  const presentationPaneIds = useMemo(() => paneCount > 1
    ? Array.from({ length: paneCount }, (_, index) => `qualification-${index}`) : [], [paneCount]);
  const startsMounted = harnessParams.get("mounted") !== "0";
  const [plot, setPlot] = useState<HarnessPlotType>("volume");
  const [step, setStepState] = useState(0);
  const [mounted, setMounted] = useState(startsMounted);
  const [monolithicVolume, setMonolithicVolume] = useState(MONOLITHIC_VOLUME_HARNESS);
  const [coarse, setCoarse] = useState(false);
  const [playback, setPlayback] = useState<PlaybackContextValue>({ playing: false, recording: false });
  const [size, setSizeState] = useState({ width: 960, height: 620 });
  const [, updateExpressionVersion] = useState(0);
  const [frameDisplays, setFrameDisplays] = useState<Record<number, {
    plot: HarnessPlotType;
    report: FrameDisplayReport;
  }>>({});
  const reportFrameDisplay = useCallback(
    (paneIndex: number, sourcePlot: HarnessPlotType, report: FrameDisplayReport) => {
      setFrameDisplays((current) => {
        const prior = current[paneIndex];
        if (prior?.plot === sourcePlot
            && prior.report.requestedStepIndex === report.requestedStepIndex
            && prior.report.requestedPublicationGeneration === report.requestedPublicationGeneration
            && prior.report.renderedStepIndex === report.renderedStepIndex
            && prior.report.renderedPublicationGeneration === report.renderedPublicationGeneration
            && prior.report.loading === report.loading
            && prior.report.interpolatedDwellMs === report.interpolatedDwellMs) return current;
        return { ...current, [paneIndex]: { plot: sourcePlot, report } };
      });
    },
    [],
  );
  const paneFrameReporters = useMemo(() => Array.from({ length: paneCount }, (_, paneIndex) =>
    (sourcePlot: HarnessPlotType, report: FrameDisplayReport) => reportFrameDisplay(paneIndex, sourcePlot, report)
  ), [paneCount, reportFrameDisplay]);
  const displayedFrames = Object.values(frameDisplays).filter((display) => display.plot === plot);
  const allPanesDisplayed = displayedFrames.length === paneCount
    && displayedFrames.every((display) => !display.report.loading);
  const renderedStep = allPanesDisplayed
    && displayedFrames.every((display) => display.report.renderedStepIndex === displayedFrames[0]?.report.renderedStepIndex)
    ? displayedFrames[0]?.report.renderedStepIndex
    : null;

  useEffect(() => {
    setFrameDisplays({});
  }, [coarse, large, plot]);

  useEffect(() => {
    window.__nimbusVtkHarness = {
      plotTypes: PLOT_TYPES,
      setPlot,
      setBrickedVolume: (enabled) => setMonolithicVolume(!enabled),
      qualifyVolumeShadows: qualifyVolumeShadowSampling,
      qualifyJointVolumes: qualifyJointVolumeRendering,
      qualifyVolumeValidity,
      qualifyVolumeArrayReuse,
      qualifyVolumeScalarPrecision,
      qualifyVolumeTexturePacking,
      qualifyVolumeAngularSampling,
      inspectVolumeLighting: () => {
        const renderer = lookupVtkRenderWindow(document.querySelector(".volume-render__canvas"))!.getRenderer() as ReturnType<typeof vtkRenderer.newInstance>;
        return renderer.getVolumes().filter(volume => volume.getVisibility()).map(volume => ({
          shade: volume.getProperty().getShade(), flatPreview: flatVolumePreview(volume.getMapper()!),
        }));
      },
      inspectVolumeSupport: () => {
        const renderer = lookupVtkRenderWindow(document.querySelector(".volume-render__canvas"))!.getRenderer() as ReturnType<typeof vtkRenderer.newInstance>;
        return renderer.getVolumes().map((volume, index) => {
          if (!volume.getVisibility()) return null;
          const input = volume.getMapper()!.getInputData();
          const mask = getVolumeValidity(input);
          return { index, missing: mask?.reduce((sum, value) => sum + Number(value === 0), 0) ?? 0,
            finite: (input.getPointData().getScalars().getData() as Float32Array).every(Number.isFinite) };
        }).filter((value): value is { index: number; missing: number; finite: boolean } => value != null);
      },
      inspectSurfaceGap: (lowerY, upperY) => {
        const renderer = lookupVtkRenderWindow(document.querySelector(".volume-render__canvas"))!.getRenderer() as ReturnType<typeof vtkRenderer.newInstance>;
        let triangles = 0, crossings = 0, nonFinite = 0;
        for (const actor of renderer.getActors().slice(0, 5) as ReturnType<typeof vtkActor.newInstance>[]) {
          if (!actor.getVisibility()) continue;
          const input = actor.getMapper()!.getInputData();
          const points = input.getPoints().getData() as Float32Array;
          const polys = input.getPolys().getData() as Uint32Array;
          if (points.length > 3_000_000) throw new Error("Surface inspection requires a bounded fixture");
          for (const value of points) if (!Number.isFinite(value)) nonFinite++;
          for (let cell = 0; cell < polys.length;) {
            const count = polys[cell++];
            let min = Infinity, max = -Infinity;
            for (let vertex = 0; vertex < count; vertex++) {
              const y = points[polys[cell++] * 3 + 1];
              min = Math.min(min, y); max = Math.max(max, y);
            }
            triangles++;
            if (max > lowerY && min < upperY) crossings++;
          }
        }
        return { triangles, crossings, nonFinite };
      },
      inspectVolumeInputs: () => {
        const volumes = mountedScalarImages();
        const images = [...new Map(volumes.map((image) => [image.values, image])).values()];
        const bytes = images.reduce((sum, image) => sum + image.values.byteLength, 0);
        if (bytes > 32 * 1024 * 1024) throw new Error("Volume input inspection requires a bounded fixture");
        let minimum = Infinity, zeros = 0;
        for (const image of images) for (const value of image.values) {
          if (!Number.isFinite(value)) throw new Error("Non-finite volume fixture");
          minimum = Math.min(minimum, value);
          if (value === 0) zeros++;
        }
        const fingerprints = images.map(image => {
          let hash = 2166136261;
          const bytes = new Uint8Array(image.values.buffer, image.values.byteOffset, image.values.byteLength);
          for (const byte of bytes) hash = Math.imul(hash ^ byte, 16777619);
          return hash >>> 0;
        });
        return { volumes: volumes.length, images: images.length, bytes, minimum, zeros, fingerprints };
      },
      inspectVolumeScalarImages: inspectBoundedVolumeScalarImages,
      captureScalarReference: () => {
        const images = mountedScalarImages();
        if (images.length !== 1 || images[0].values.byteLength > 16 * 1024 * 1024) {
          throw new Error("Scalar reference requires one bounded fixture image");
        }
        scalarReference.current = { ...images[0], values: images[0].values.slice() };
      },
      compareScalarReference: () => {
        const reference = scalarReference.current;
        if (!reference) throw new Error("No scalar reference");
        let checked = 0;
        let maximumError = 0;
        for (const image of mountedScalarImages()) {
          const offset = image.origin.map((origin, axis) => {
            const index = (origin - reference.origin[axis]) / reference.spacing[axis];
            if (Math.abs(index - Math.round(index)) > 1e-6
                || Math.abs(image.spacing[axis] / reference.spacing[axis] - 1) > 1e-7) {
              throw new Error("Scalar image is not on the reference lattice");
            }
            return Math.round(index);
          });
          const [nx, ny, nz] = image.shape;
          for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
            const position = [x + offset[0], y + offset[1], z + offset[2]];
            if (position.some((value, axis) => value < 0 || value >= reference.shape[axis])) throw new Error("Scalar image exceeds reference bounds");
            const actual = image.values[x + nx * (y + ny * z)];
            const expected = reference.values[position[0] + reference.shape[0] * (position[1] + reference.shape[1] * position[2])];
            if (!Number.isFinite(actual) || !Number.isFinite(expected)) throw new Error("Non-finite scalar fixture");
            maximumError = Math.max(maximumError, Math.abs(actual - expected));
            checked++;
          }
        }
        return { checked, maximumError };
      },
      setStep: (next) => setStepState(Math.max(0, Math.min(
        (derivedBricks
          ? derivedBrickSteps
          : TERRAIN_BRICK_HARNESS
          ? terrainBrickSteps
          : beastReference
          ? beastReferenceSteps
          : large ? bricked ? brickedLargeSteps : largeSteps : steps).length - 1,
        next,
      ))),
      setMounted,
      setCoarse,
      setPlayback: (playing, recording = false, durationMs = 500) => setPlayback({ playing, recording, interpolationDurationMs: durationMs }),
      setSize: (width, height) => setSizeState({ width, height }),
      resizeStorm: () => {
        const host = document.querySelector<HTMLElement>(".harness-scene");
        if (!host) return;
        for (let i = 0; i < 80; i += 1) {
          host.style.width = `${640 + i * 3}px`;
          host.style.height = `${420 + (i % 17)}px`;
        }
        host.style.width = "0px";
        host.style.height = "0px";
        host.getBoundingClientRect();
        host.style.width = `${size.width}px`;
        host.style.height = `${size.height}px`;
      },
      loseContext: () => {
        const extension = contextExtension();
        if (!extension) return false;
        extension.loseContext();
        return true;
      },
      restoreContext: () => {
        const extension = contextExtension() ?? lastLoseExtension;
        if (!extension) return false;
        extension.restoreContext();
        return true;
      },
      capture: () => {
        try {
          const encoded = canvas()?.toDataURL("image/png") ?? "";
          if (!encoded.startsWith("data:image/png")) throw new Error("empty canvas capture");
          return true;
        } catch {
          captureFailures += 1;
          return false;
        }
      },
      captureSavedFrame: () => {
        const target = document.querySelector<HTMLElement>(".volume-render__canvas")!;
        const pending = captureFrameDataUri(target, { width: 960, height: 620 });
        const renderWindow = lookupVtkRenderWindow(target);
        savedFrameEvidence.current = renderWindow ? collectVtkAppearanceEvidence(renderWindow, { plot, backend: "webgl" }) : null;
        return pending;
      },
      capturePortableViewpoints: async (count, cancelAfter) => {
        const { captureViewpoints } = await import("@nimbus/lib/vtk/captureViewpoints");
        const element = document.querySelector<HTMLElement>(".volume-render__canvas")!;
        const canvas = element.querySelector("canvas")!;
        const abort = new AbortController();
        const bytes = await captureViewpoints(element, { width: canvas.width, height: canvas.height }, count, "Portable storm fixture", { frameIndex: step }, { signal: abort.signal }, done => {
          if (cancelAfter && done >= cancelAfter) abort.abort();
        });
        return new TextDecoder().decode(bytes);
      },
      inspectLastSavedFrame: () => savedFrameEvidence.current,
      inspectExportOverlays: () => snapshotVtkOverlays(document.querySelector<HTMLElement>(".volume-render__canvas")!),
      inspectSurfaceNormals: () => {
        const renderer = lookupVtkRenderWindow(document.querySelector(".volume-render__canvas"))?.getRenderer() as ReturnType<typeof vtkRenderer.newInstance> | undefined;
        return renderer?.getActors().filter(actor => actor.getVisibility()).map(actor => {
          const data = (actor as ReturnType<typeof vtkActor.newInstance>).getMapper()?.getInputData() as ReturnType<typeof vtkPolyData.newInstance> | undefined;
          return { points: data?.getNumberOfPoints() ?? 0, normals: data?.getPointData().getNormals()?.getNumberOfTuples() ?? 0, strips: data?.getStrips().getNumberOfCells() ?? 0 };
        }) ?? [];
      },
      inspectReferences: () => {
        const renderer = lookupVtkRenderWindow(document.querySelector(".volume-render__canvas"))?.getRenderer() as ReturnType<typeof vtkRenderer.newInstance> | undefined;
        return (renderer?.getActors() as ReturnType<typeof vtkActor.newInstance>[] ?? []).filter(actor => {
          const data = actor.getMapper()?.getInputData() as ReturnType<typeof vtkPolyData.newInstance> | undefined;
          const count = data?.getNumberOfPoints() ?? 0;
          return actor.getVisibility() && !actor.getProperty().getLighting() && count > 0 && count <= 24;
        }).map(actor => {
          if (!referenceIds.has(actor)) referenceIds.set(actor, ++nextReferenceId);
          return { id: referenceIds.get(actor)!, points: actor.getMapper()!.getInputData().getNumberOfPoints(), bounds: Array.from(actor.getBounds()) };
        });
      },
      inspectLighting: () => {
        const renderer = lookupVtkRenderWindow(document.querySelector(".volume-render__canvas"))?.getRenderer() as ReturnType<typeof vtkRenderer.newInstance> | undefined;
        return renderer?.getLights().map(light => ({
          direction: Array.from(light.getDirection()), color: light.getColor(),
          intensity: light.getIntensity(), type: light.getLightType(),
        })) ?? [];
      },
      captureAppearance: async () => {
        const target = document.querySelector<HTMLElement>(".volume-render__canvas");
        const captureAppearance = lookupVtkAppearanceCapture(target);
        return captureAppearance ? captureAppearance() : null;
      },
      setCamera: async (position, focalPoint, parallel = true) => {
        const window = lookupVtkRenderWindow(document.querySelector(".volume-render__canvas"));
        if (!window) throw new Error("No mounted camera");
        const renderer = window.getRenderer() as {
          getActiveCamera(): {
            setPosition(...position: number[]): void;
            setFocalPoint(...point: number[]): void;
            setViewUp(...up: number[]): void;
            setParallelProjection(value: boolean): void;
            setParallelScale(value: number): void;
          };
          resetCameraClippingRange(): void;
        };
        const camera = renderer.getActiveCamera();
        camera.setPosition(...position);
        camera.setFocalPoint(...focalPoint);
        camera.setViewUp(0, 0, 1);
        camera.setParallelProjection(parallel);
        camera.setParallelScale(5000);
        renderer.resetCameraClippingRange();
        window.getRenderWindow().render();
        await new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve())));
      },
      compareAppearance: async (referenceDataUri, candidateDataUri, changedChannelThreshold) =>
        compareVtkAppearanceImages(
          await rgbaImage(referenceDataUri),
          await rgbaImage(candidateDataUri),
          changedChannelThreshold,
        ),
      snapshot: () => snapshot(plot, step, mounted, coarse, large, paneCount, beastReference),
      clearFieldCache: () => {
        const before = fieldCacheStats().bytes;
        clearFieldCache();
        return before - fieldCacheStats().bytes;
      },
      qualifyProductionGeometry: qualifyProductionGeometryWorkers,
      setFieldReadDelay: fixtureControl.setFieldReadDelay,
      setExpressionMultiplier: multiplier => {
        installExpressionProbe(multiplier);
        updateExpressionVersion(version => version + 1);
      },
      startBackgroundLoad,
      stopBackgroundLoad,
    };
    return () => {
      delete window.__nimbusVtkHarness;
    };
  }, [beastReference, bricked, coarse, derivedBricks, large, mounted, paneCount, plot, size.height, size.width, step]);

  useEffect(() => () => stopBackgroundLoad(), []);

  return (
    <main className="harness-shell" data-plot={plot} data-step={step} data-mounted={mounted} data-coarse={coarse} data-large={large} data-panes={paneCount} data-volume-ablation={appearanceAblation.name}>
      <header>
        <h1>Nimbus VTK lifecycle harness</h1>
        <output data-testid="identity">{plot} · field {step}</output>
      </header>
      <section
        className="harness-scene"
        style={{
          width: size.width,
          height: size.height,
          display: "grid",
          gridTemplateColumns: `repeat(${paneCount > 1 ? 2 : 1}, minmax(0, 1fr))`,
          gridTemplateRows: `repeat(${paneCount > 2 ? 2 : 1}, minmax(0, 1fr))`,
        }}
        data-testid="scene"
        data-frame-plot={allPanesDisplayed ? plot : ""}
        data-frame-loading={!allPanesDisplayed}
        data-rendered-step={renderedStep ?? ""}
      >
        <PanePresentationProvider identity={JSON.stringify([plot, step, coarse, large, bricked, derivedBricks])}
          paneIds={plot === "trajectories" ? [] : presentationPaneIds}>
        {mounted && Array.from({ length: paneCount }, (_, paneIndex) => (
          <div className="harness-pane" data-pane-index={paneIndex} key={paneIndex}>
            <PaneProvider value={{
              bus: null,
              paneId: `qualification-${paneIndex}`,
              cameraLocked: false,
              cameraAuthority: paneIndex === 0,
              voxelBudgetScale: forceBrickLod ? 0.01 : 1 / paneCount,
              ownsSharedOverlays: paneIndex === 0,
            }}>
              <PlaybackProvider value={playback}><Scene
                key={`${plot}:${coarse}:${large}:${appearanceAblation.name}:${paneIndex}`}
                plot={plot}
                step={step}
                coarse={coarse}
                large={large}
                bricked={bricked}
                derivedBricks={derivedBricks}
                beastReference={beastReference}
                volumeAppearanceQualification={monolithicVolume
                  ? { ...appearanceAblation.overrides, progressiveBricks: false }
                  : appearanceAblation.overrides}
                onFrameDisplayChange={paneFrameReporters[paneIndex]}
              /></PlaybackProvider>
            </PaneProvider>
          </div>
        ))}
        </PanePresentationProvider>
      </section>
    </main>
  );
}

function StudioHarness() {
  const presentation = useRef<PanePresentationSnapshot | null>(null);
  const recordPresentation = useCallback((snapshot: PanePresentationSnapshot) => { presentation.current = snapshot; }, []);
  const params = new URLSearchParams(window.location.search);
  const startsEmpty = params.get("empty") === "1";
  const startsCm1 = params.get("cm1") === "1";
  const startsComparison = params.get("compare") === "1";
  const startsEnsemble = params.get("ensemble") === "1";
  const startsTerrain = params.get("terrain") === "1";
  const cm1Dataset = useMemo<LocalDatasetSource>(() => ({
    schema: "nimbus.studio.dataset-source",
    schemaVersion: 1,
    kind: "cm1_netcdf",
    sourceModel: "cm1",
    nativeFormat: "netcdf",
    locator: { kind: "local_path", path: "/__nimbus_browser_fixture__/cm1out.nc" },
    displayName: "CM1 browser supercell",
    readOnly: true,
  }), []);
  const comparisonDataset = useMemo<LocalDatasetSource>(() => ({
    schema: "nimbus.studio.dataset-source",
    schemaVersion: 1,
    kind: "nimbus_run",
    sourceModel: "nimbus",
    nativeFormat: "nimbus_history",
    locator: { kind: "local_path", path: "/__nimbus_browser_fixture__/live" },
    displayName: "Nimbus browser control",
    readOnly: true,
  }), []);
  const ensembleDatasets = useMemo<LocalDatasetSource[]>(() => [1, 2].map((member) => ({
    schema: "nimbus.studio.dataset-source",
    schemaVersion: 1,
    kind: "nimbus_run",
    sourceModel: "nimbus",
    nativeFormat: "nimbus_history",
    locator: { kind: "local_path", path: `/__nimbus_browser_fixture__/ensemble-${member}` },
    displayName: `Nimbus ensemble ${String(member).padStart(2, "0")}`,
    readOnly: true,
  })), []);
  const [run, setRun] = useState<{
    runDir: string | null;
    status: "running" | "completed";
  }>({
    runDir: startsEmpty || startsCm1
      ? null
      : startsTerrain
        ? "/__nimbus_browser_fixture__/terrain"
        : "/__nimbus_browser_fixture__/live",
    status: "running" as "running" | "completed",
  });
  useEffect(() => {
    window.__nimbusStudioHarness = {
      setRun: (runDir, status = "completed") => setRun({ runDir, status }),
      publishLiveThrough: fixtureControl.publishLiveThrough,
      setListDelay: fixtureControl.setListDelay,
      setFieldReadDelay: fixtureControl.setFieldReadDelay,
      clearFieldCache: () => {
        const before = fieldCacheStats().bytes;
        clearFieldCache();
        return before - fieldCacheStats().bytes;
      },
      captureActivePlot: async () => {
        const element = document.querySelector<HTMLElement>(
          ".plot-studio__pane--companion .volume-render__canvas, .plot-canvas",
        );
        if (!element) return null;
        const bounds = element.getBoundingClientRect();
        return captureFrameDataUri(element, {
          width: Math.max(1, Math.round(bounds.width)),
          height: Math.max(1, Math.round(bounds.height)),
        });
      },
      snapshot: fixtureControl.snapshot,
      resourceSnapshot: () => ({ ...snapshotBrowserResources(), visualization: getVisualizationResourceSnapshot() }),
      capturePaneAppearances: async () => {
        const results: Array<VtkAppearanceCapture | null> = [];
        const panes = [...document.querySelectorAll(".plot-studio__pane")];
        const canvases = panes.length > 0
          ? panes.map(pane => pane.querySelector<HTMLElement>(".volume-render__canvas"))
          : [...document.querySelectorAll<HTMLElement>(".plot-studio .volume-render__canvas")];
        for (const canvas of canvases) {
          const capture = lookupVtkAppearanceCapture(canvas);
          results.push(capture ? await capture() : null);
        }
        return results;
      },
      inspectPrimaryVolumeScalarImages: inspectBoundedVolumeScalarImages,
      presentationSnapshot: () => presentation.current,
      readExportFile: path => { const bytes = fixtureControl.readFileBytes(path); return bytes ? new TextDecoder().decode(bytes) : null; },
    };
    return () => {
      delete window.__nimbusStudioHarness;
    };
  }, []);
  return (
    <PlotStudio
      onPanePresentationChange={recordPresentation}
      initialDataset={startsCm1 ? cm1Dataset : null}
      initialComparisonDataset={startsComparison ? comparisonDataset : null}
      initialEnsembleDatasets={startsEnsemble ? ensembleDatasets : undefined}
      initialRunDir={run.runDir}
      initialRunStatus={run.status}
      restoreActiveProject={!startsEmpty}
    />
  );
}

function CompatibilityHarness() {
  const dataset: LocalDatasetSource = {
    schema: "nimbus.studio.dataset-source",
    schemaVersion: 1,
    kind: "cm1_netcdf",
    sourceModel: "cm1",
    nativeFormat: "netcdf",
    locator: { kind: "local_path", path: "/fixture/cm1out.nc" },
    displayName: "CM1 supercell control",
    readOnly: true,
    caseInventory: {
      schema: "nimbus.studio.cm1-case-inventory",
      schemaVersion: 1,
      directory: "/fixture",
      historyFiles: ["/fixture/cm1out.nc"],
      namelistInput: "/fixture/namelist.input",
      inputSounding: "/fixture/input_sounding",
      restartFiles: ["/fixture/cm1rst_000001.dat"],
      statisticsFiles: ["/fixture/cm1out_stats.dat"],
      otherFileCount: 0,
      nestedDirectoryCount: 0,
    },
    namelistReport: {
      schema: "nimbus.studio.cm1-namelist-report",
      schemaVersion: 1,
      path: "/fixture/namelist.input",
      groups: [
        {
          name: "param0",
          assignments: [
            { key: "nx", rawValue: "96" },
            { key: "ny", rawValue: "96" },
            { key: "nz", rawValue: "64" },
            { key: "dx", rawValue: "250.0" },
            { key: "ptype", rawValue: "3" },
            { key: "isnd", rawValue: "7" },
            { key: "mystery_control", rawValue: ".true." },
          ],
        },
      ],
      assignmentCount: 7,
      values: {
        "param0.nx": "96",
        "param0.ny": "96",
        "param0.nz": "64",
        "param0.dx": "250.0",
        "param0.ptype": "3",
        "param0.isnd": "7",
        "param0.mystery_control": ".true.",
      },
      warnings: [],
    },
    soundingReport: {
      schema: "nimbus.studio.cm1-sounding-report",
      schemaVersion: 1,
      path: "/fixture/input_sounding",
      surface: { pressureHpa: 1_000, potentialTemperatureK: 300, mixingRatioGkg: 14 },
      levels: [
        { sourceLine: 2, heightM: 0, potentialTemperatureK: 300, mixingRatioGkg: 14, uMs: 5, vMs: 0 },
        { sourceLine: 3, heightM: 18_000, potentialTemperatureK: 410, mixingRatioGkg: 0, uMs: 20, vMs: 5 },
      ],
      topHeightM: 18_000,
      warnings: [],
    },
  };
  return (
    <main className="plot-studio plot-studio--metadata">
      <header className="plot-studio__header">
        <div className="plot-studio__title-row">
          <h2>CM1 supercell control</h2>
          <span className="plot-studio__source-badge">CM1 / netCDF</span>
        </div>
        <Cm1CompatibilityReport dataset={dataset} />
      </header>
    </main>
  );
}

function RecorderHarness() {
  const [stepIdx, setStepIdx] = useState(() => Number(new URLSearchParams(window.location.search).get("recorderStep") ?? 0));
  const [recorderOpen, setRecorderOpen] = useState(
    () => new URLSearchParams(window.location.search).get("recorderClosed") !== "1",
  );
  const failSecondIdentity = new URLSearchParams(window.location.search)
    .get("recorderFailure") === "stale";
  const recorderTriggerRef = useRef<HTMLButtonElement>(null);
  const captureRootRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const displayedRef = useRef<DisplayedFrameIdentity | null>(null);
  const capturedStepsRef = useRef<number[]>([]);
  const captureReleaseRef = useRef<(() => void) | null>(null);
  const holdCapture = new URLSearchParams(window.location.search).get("recorderCapture") === "held";
  const backingScale = new URLSearchParams(window.location.search).get("recorderRetina") === "1" ? 2 : 1;
  const frameIdentities = useMemo(
    () => steps.slice(0, 3).map((step) => ({
      stepIndex: step.index,
      publicationGeneration: step.publicationGeneration,
    })),
    [],
  );

  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Recorder fixture 2d context unavailable");
    const hue = 205 + stepIdx * 42;
    ctx.fillStyle = `hsl(${hue} 55% 16%)`;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = `hsl(${hue} 88% 68%)`;
    ctx.fillRect(24 + stepIdx * 42, 36, 140, 112);
    ctx.fillStyle = "#f7f9fc";
    ctx.font = "600 32px system-ui";
    ctx.fillText(`Nimbus frame ${stepIdx}`, 28, 205);
    const reportedIdx = failSecondIdentity && stepIdx === 1 ? 0 : stepIdx;
    displayedRef.current = {
      renderedStepIndex: frameIdentities[reportedIdx]?.stepIndex ?? null,
      renderedPublicationGeneration:
        frameIdentities[reportedIdx]?.publicationGeneration ?? null,
      loading: false,
    };
  }, [failSecondIdentity, frameIdentities, stepIdx]);

  useEffect(() => {
    const root = captureRootRef.current;
    const canvas = canvasRef.current;
    if (!root || !canvas) return;
    registerVtkCapture(root, async () => {
      capturedStepsRef.current.push(stepIdx);
      if (holdCapture) await new Promise<void>(resolve => { captureReleaseRef.current = resolve; });
      return canvas.toDataURL("image/png");
    });
    return () => unregisterVtkCapture(root);
  }, [holdCapture, stepIdx]);

  useEffect(() => {
    window.__nimbusRecorderHarness = {
      snapshot: () => ({
        ...fixtureControl.snapshot(),
        capturedSteps: [...capturedStepsRef.current],
        capturePending: captureReleaseRef.current !== null,
        savedPng: Array.from(fixtureControl.readFileBytes("/__nimbus_browser_fixture__/plugins/nimbus-recorder-fixture.png") ?? []),
        nativeEncoding: recorderFixture.snapshot(),
      }),
      releasePreparation: recorderFixture.releasePreparation,
      releaseProbe: recorderFixture.releaseProbe,
      releaseCapture: (changeIdentity = false) => {
        if (changeIdentity && displayedRef.current) displayedRef.current = { ...displayedRef.current, loading: true };
        captureReleaseRef.current?.();
        captureReleaseRef.current = null;
      },
    };
    return () => {
      delete window.__nimbusRecorderHarness;
    };
  }, []);

  return (
    <main className="recorder-fixture">
      <button
        ref={recorderTriggerRef}
        type="button"
        onClick={() => setRecorderOpen(true)}
      >
        Open recorder
      </button>
      <div ref={captureRootRef} className="recorder-fixture__capture">
        <canvas
          ref={canvasRef}
          className="recorder-fixture__canvas"
          width={400 * backingScale}
          height={240 * backingScale}
        />
      </div>
      {recorderOpen && (
        <MovieRecorderDialog
          totalSteps={frameIdentities.length}
          currentStepIdx={stepIdx}
          setStepIdx={setStepIdx}
          plotCanvasSelector=".recorder-fixture__canvas"
          maxFrameWaitMs={failSecondIdentity ? 500 : 4_000}
          frameIdentities={frameIdentities}
          getDisplayedFrameIdentity={() => displayedRef.current}
          defaultFilename="nimbus-recorder-fixture"
          onClose={() => setRecorderOpen(false)}
          returnFocusRef={recorderTriggerRef}
        />
      )}
    </main>
  );
}

const mode = new URLSearchParams(window.location.search).get("mode");
createRoot(rootElement).render(
  mode === "volume-owner"
    ? <VolumeOwnerLifecycleHarness />
    : mode === "research"
    ? <ResearchHarness />
    : mode === "plot-resolution"
    ? <PlotResolutionHarness />
    : mode === "playback-screen"
    ? <PlaybackScreenHarness />
    : mode === "published-overview"
    ? <PublishedVolumeOverviewHarness />
    : mode === "regular-dataflow"
    ? <RegularDataflowHarness />
    : mode === "studio"
    ? <StudioHarness />
    : mode === "compatibility"
      ? <CompatibilityHarness />
    : mode === "recorder"
      ? <RecorderHarness />
      : <Harness />,
);
