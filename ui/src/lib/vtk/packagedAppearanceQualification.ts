import "@kitware/vtk.js/Rendering/Profiles/Geometry";
import "@kitware/vtk.js/Rendering/Profiles/Volume";
import vtkGenericRenderWindow from "@kitware/vtk.js/Rendering/Misc/GenericRenderWindow";
import vtkActor from "@kitware/vtk.js/Rendering/Core/Actor";
import vtkMapper from "@kitware/vtk.js/Rendering/Core/Mapper";
import vtkVolume from "@kitware/vtk.js/Rendering/Core/Volume";
import vtkVolumeMapper from "@kitware/vtk.js/Rendering/Core/VolumeMapper";
import vtkColorTransferFunction from "@kitware/vtk.js/Rendering/Core/ColorTransferFunction";
import vtkPiecewiseFunction from "@kitware/vtk.js/Common/DataModel/PiecewiseFunction";
import vtkImageData from "@kitware/vtk.js/Common/DataModel/ImageData";
import vtkPolyData from "@kitware/vtk.js/Common/DataModel/PolyData";
import vtkTubeFilter from "@kitware/vtk.js/Filters/General/TubeFilter";
import vtkSphereSource from "@kitware/vtk.js/Filters/Sources/SphereSource";
import { setImageScalars, setPreparedImageScalars } from "./vtkImageData";
import { cartesianGridToVtkScalars } from "@/lib/science/cartesianInterp";
import { captureVtkPng } from "./vtkCaptureRegistry";
import { collectVtkAppearanceEvidence, type VtkAppearancePlot } from "./vtkAppearanceEvidence";
import { disposeVtkScene, type VtkDisposable } from "./vtkLifecycle";
import { vtkRenderWindowOptions } from "./vtkBackend";
import { configureCoherentVolumeSampling } from "./coherentVolumeSampling";
import type { Npy3D } from "@/lib/platform/npy";
import {
  visualizationResourceGovernor,
} from "@/lib/core/visualizationResourceGovernor";

export interface PackagedAppearanceSceneEvidence {
  plot: VtkAppearancePlot;
  pngSha256: string;
  repeatPngSha256: string;
  metadataFingerprint: string;
  repeatMetadataFingerprint: string;
  meanLuma: number;
  lumaDeviation: number;
  viewport: {
    cssWidth: number;
    cssHeight: number;
    backingWidth: number;
    backingHeight: number;
    pixelRatio: number;
  };
  visibleLayers: number;
}

export interface PackagedAppearanceEvidence {
  schema: "nimbus.studio.packaged-vtk-appearance";
  schemaVersion: 1;
  scenes: PackagedAppearanceSceneEvidence[];
}

export interface PackagedPhysicalGpuEvidence {
  measurementProtocol: "isolated-prepared-pair-post-scope-v2";
  renderingPath: "nimbus-coherent-volume";
  cycles: number;
  scalarTextureBytesPerCycle: number;
  preparedCpuBufferBytes: number;
  preparedCpuBuffers: number;
  retiredContexts: number;
  retiredOwnedObjects: number[];
  zeroedCanvasBackingStores: number;
  postScopeRetirementMs: number[];
  maximumLogicalGpuResidentBytes: number;
  viewport: { width: number; height: number };
  rss?: {
    currency: "process-tree-rss";
    baselineBytes: number;
    residentBytes: [number, number];
    retiredBytes: [number, number];
    residentDeltaBytes: [number, number];
    restartGrowthBytes: number;
    limitation: string;
  };
  actualRenderedTexturesPerCycle: number;
  contextLost: boolean;
  glErrors: number[];
  maximum3DTextureSize: number;
  renderPngSha256: string[];
}

interface BuiltScene {
  resources: VtkDisposable[];
  detach: () => void;
}

function volumeData() {
  const shape: [number, number, number] = [32, 32, 32];
  const data = new Float32Array(shape[0] * shape[1] * shape[2]);
  let target = 0;
  for (let x = 0; x < shape[0]; x += 1) {
    const dx = (x - 15.5) / 8;
    for (let y = 0; y < shape[1]; y += 1) {
      const dy = (y - 15.5) / 8;
      for (let z = 0; z < shape[2]; z += 1) {
        const dz = (z - 14) / 9;
        data[target++] = Math.exp(-(dx * dx + dy * dy + dz * dz) * 1.8);
      }
    }
  }
  return {
    shape,
    spacing: [0.08, 0.08, 0.08] as [number, number, number],
    origin: [-1.24, -1.24, -1.12] as [number, number, number],
    data,
  };
}

function addVolume(renderer: ReturnType<ReturnType<typeof vtkGenericRenderWindow.newInstance>["getRenderer"]>) {
  const image = vtkImageData.newInstance();
  setImageScalars(image, volumeData());
  const mapper = vtkVolumeMapper.newInstance();
  configureCoherentVolumeSampling(mapper);
  mapper.setInputData(image);
  mapper.setSampleDistance(0.04);
  mapper.setMaximumSamplesPerRay(1000);
  mapper.setAutoAdjustSampleDistances(false);
  const color = vtkColorTransferFunction.newInstance();
  color.addRGBPoint(0, 0.04, 0.02, 0.12);
  color.addRGBPoint(0.35, 0.48, 0.08, 0.32);
  color.addRGBPoint(0.7, 0.95, 0.46, 0.1);
  color.addRGBPoint(1, 1, 0.94, 0.55);
  const opacity = vtkPiecewiseFunction.newInstance();
  opacity.addPoint(0, 0);
  opacity.addPoint(0.1, 0);
  opacity.addPoint(0.2, 0.4);
  opacity.addPoint(0.4, 0.8);
  opacity.addPoint(1, 1);
  const volume = vtkVolume.newInstance();
  volume.setMapper(mapper);
  const property = volume.getProperty();
  property.setRGBTransferFunction(0, color);
  property.setScalarOpacity(0, opacity);
  property.setInterpolationTypeToLinear();
  property.setShade(true);
  property.setAmbient(0.4);
  property.setDiffuse(0.6);
  property.setSpecular(0);
  renderer.addVolume(volume);
  return {
    resources: [volume, mapper, color, opacity, image] as VtkDisposable[],
    detach: () => renderer.removeVolume(volume),
  };
}

function addSphere(
  renderer: ReturnType<ReturnType<typeof vtkGenericRenderWindow.newInstance>["getRenderer"]>,
  center: [number, number, number] = [0, 0, 0],
) {
  const source = vtkSphereSource.newInstance({
    center,
    radius: 0.62,
    thetaResolution: 32,
    phiResolution: 24,
  });
  const mapper = vtkMapper.newInstance();
  mapper.setInputConnection(source.getOutputPort());
  const actor = vtkActor.newInstance();
  actor.setMapper(mapper);
  actor.getProperty().setColor(0.35, 0.72, 1);
  actor.getProperty().setAmbient(0.25);
  actor.getProperty().setDiffuse(0.75);
  renderer.addActor(actor);
  return {
    resources: [actor, mapper, source] as VtkDisposable[],
    detach: () => renderer.removeActor(actor),
  };
}

function addPath(
  renderer: ReturnType<ReturnType<typeof vtkGenericRenderWindow.newInstance>["getRenderer"]>,
  trajectory: boolean,
) {
  const count = 72;
  const points = new Float32Array(count * 3);
  const lines = new Uint32Array(count + 1);
  lines[0] = count;
  for (let index = 0; index < count; index += 1) {
    const t = index / (count - 1);
    const angle = t * Math.PI * (trajectory ? 5 : 3);
    points[index * 3] = trajectory ? Math.cos(angle) * (0.25 + t * 0.7) : -1.1 + t * 2.2;
    points[index * 3 + 1] = trajectory ? Math.sin(angle) * (0.25 + t * 0.7) : Math.sin(angle) * 0.48;
    points[index * 3 + 2] = -0.8 + t * 1.6;
    lines[index + 1] = index;
  }
  const data = vtkPolyData.newInstance();
  data.getPoints().setData(points, 3);
  data.getLines().setData(lines);
  const tube = vtkTubeFilter.newInstance({ radius: 0.055, numberOfSides: 10, capping: true });
  tube.setInputData(data);
  const mapper = vtkMapper.newInstance();
  mapper.setInputConnection(tube.getOutputPort());
  const actor = vtkActor.newInstance();
  actor.setMapper(mapper);
  actor.getProperty().setColor(trajectory ? 1 : 0.42, trajectory ? 0.7 : 0.85, trajectory ? 0.2 : 1);
  renderer.addActor(actor);
  return {
    resources: [actor, mapper, tube, data] as VtkDisposable[],
    detach: () => renderer.removeActor(actor),
  };
}

function buildScene(
  plot: VtkAppearancePlot,
  renderer: ReturnType<ReturnType<typeof vtkGenericRenderWindow.newInstance>["getRenderer"]>,
): BuiltScene {
  const parts = plot === "volume"
    ? [addVolume(renderer)]
    : plot === "isosurface"
      ? [addSphere(renderer)]
      : plot === "streamlines"
        ? [addPath(renderer, false)]
        : plot === "supercell"
          ? [addVolume(renderer), addSphere(renderer, [0.25, 0.1, 0.45])]
          : [addPath(renderer, true)];
  return {
    resources: parts.flatMap((part) => part.resources),
    detach: () => parts.forEach((part) => part.detach()),
  };
}

function pngBytes(dataUri: string): Uint8Array {
  const encoded = dataUri.slice(dataUri.indexOf(",") + 1);
  const binary = atob(encoded);
  return Uint8Array.from(binary, (character) => character.charCodeAt(0));
}

async function sha256(dataUri: string): Promise<string> {
  const source = pngBytes(dataUri);
  const bytes = new ArrayBuffer(source.byteLength);
  new Uint8Array(bytes).set(source);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function pixelSignal(dataUri: string): Promise<{ meanLuma: number; lumaDeviation: number }> {
  const image = new Image();
  image.src = dataUri;
  await image.decode();
  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  try {
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("Packaged appearance 2D evidence context unavailable");
    context.drawImage(image, 0, 0);
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    let sum = 0;
    let squares = 0;
    for (let index = 0; index < pixels.length; index += 4) {
      const value = (pixels[index] * 0.2126 + pixels[index + 1] * 0.7152 + pixels[index + 2] * 0.0722) / 255;
      sum += value;
      squares += value * value;
    }
    const count = pixels.length / 4;
    const meanLuma = sum / count;
    return { meanLuma, lumaDeviation: Math.sqrt(Math.max(0, squares / count - meanLuma * meanLuma)) };
  } finally {
    canvas.width = 0;
    canvas.height = 0;
    image.src = "";
  }
}

async function captureScene(plot: VtkAppearancePlot): Promise<PackagedAppearanceSceneEvidence> {
  const host = document.createElement("div");
  host.style.cssText = "position:fixed;inset:0 auto auto 0;width:640px;height:420px;background:#0e1013";
  document.body.append(host);
  const renderWindow = vtkGenericRenderWindow.newInstance(vtkRenderWindowOptions(
    [0.055, 0.063, 0.075], "webgl",
  ));
  renderWindow.setContainer(host);
  renderWindow.resize();
  const renderer = renderWindow.getRenderer();
  const built = buildScene(plot, renderer);
  renderer.resetCamera();
  renderer.resetCameraClippingRange();
  renderWindow.getRenderWindow().render();
  await new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve())));
  try {
    const first = await captureVtkPng(renderWindow);
    const firstEvidence = collectVtkAppearanceEvidence(renderWindow, {
      plot,
      backend: "webgl",
      datasetId: "packaged-analytic",
      frameIndex: plot === "trajectories" ? null : 0,
      field: plot,
      fidelityMode: "scientific",
      parameters: { qualification: "packaged-wkwebview", ...(plot === "trajectories" ? { temporalScope: "whole-recorded-history" } : {}) },
    });
    const second = await captureVtkPng(renderWindow);
    const secondEvidence = collectVtkAppearanceEvidence(renderWindow, {
      plot,
      backend: "webgl",
      datasetId: "packaged-analytic",
      frameIndex: plot === "trajectories" ? null : 0,
      field: plot,
      fidelityMode: "scientific",
      parameters: { qualification: "packaged-wkwebview", ...(plot === "trajectories" ? { temporalScope: "whole-recorded-history" } : {}) },
    });
    const [pngSha256, repeatPngSha256, signal] = await Promise.all([
      sha256(first),
      sha256(second),
      pixelSignal(first),
    ]);
    return {
      plot,
      pngSha256,
      repeatPngSha256,
      metadataFingerprint: firstEvidence.metadataFingerprint,
      repeatMetadataFingerprint: secondEvidence.metadataFingerprint,
      ...signal,
      viewport: firstEvidence.viewport,
      visibleLayers: firstEvidence.layers.filter((layer) => layer.visible !== false).length,
    };
  } finally {
    disposeVtkScene({
      clearScene: () => undefined,
      cleanupSteps: [built.detach],
      resources: built.resources,
      renderWindow,
    });
    host.remove();
  }
}

export async function capturePackagedAppearanceEvidence(): Promise<PackagedAppearanceEvidence> {
  const plots: VtkAppearancePlot[] = ["volume", "isosurface", "streamlines", "supercell", "trajectories"];
  const scenes = [];
  for (const plot of plots) scenes.push(await captureScene(plot));
  return {
    schema: "nimbus.studio.packaged-vtk-appearance",
    schemaVersion: 1,
    scenes,
  };
}

export async function qualifyPackagedPhysicalGpuResidency(
  field: Npy3D,
  sampleMemory: (label: string) => Promise<void>,
): Promise<PackagedPhysicalGpuEvidence> {
  const reservation = visualizationResourceGovernor.reserve({
    owner: "packaged:physical-gpu-input",
    generation: "immutable-pair",
    priority: "visible",
    claims: [{ allocationId: "packaged:physical-gpu-input:pair", stage: "prepared-main",
      bytes: field.data.byteLength * 2, accounting: "resident" }],
  });
  if (!reservation.ok) throw new Error(reservation.reason);
  try {
    return await qualifyPreparedGpuResidency(field, sampleMemory);
  } finally {
    visualizationResourceGovernor.release(reservation.reservation.id);
  }
}

async function qualifyPreparedGpuCycle(
  prepared: readonly Parameters<typeof setPreparedImageScalars>[1][],
  textureBytes: number,
  cycle: number,
  sampleMemory: (label: string) => Promise<void>,
) {
  const reservation = visualizationResourceGovernor.reserve({
    owner: "packaged:physical-gpu",
    generation: `cycle:${cycle}`,
    priority: "visible",
    claims: [{ allocationId: `packaged:physical-gpu:${cycle}`, stage: "gpu",
      bytes: textureBytes, accounting: "resident" }],
  });
  if (!reservation.ok) throw new Error(reservation.reason);
  let host: HTMLDivElement | null = null;
  let renderWindow: ReturnType<typeof vtkGenericRenderWindow.newInstance> | null = null;
  let renderer: ReturnType<ReturnType<typeof vtkGenericRenderWindow.newInstance>["getRenderer"]> | null = null;
  let canvas: HTMLCanvasElement | null = null;
  let retired: Promise<void> | null = null;
  let confirmRetired: ((event: Event) => void) | undefined;
  let contextLossMessage: string | null = null;
  let renderingStage = "creation";
  const resources: VtkDisposable[] = [];
  const volumes: Array<ReturnType<typeof vtkVolume.newInstance>> = [];
  let ownedObjects: Array<{ isDeleted?: () => boolean }> = [];
  const retirement = { deletedObjects: 0, canvasZeroed: false };
  try {
    host = document.createElement("div");
    host.style.cssText = "position:fixed;inset:0 auto auto 0;width:640px;height:420px;background:#0e1013";
    document.body.append(host);
    renderWindow = vtkGenericRenderWindow.newInstance(vtkRenderWindowOptions(
      [0.055, 0.063, 0.075], "webgl",
    ));
    renderWindow.setContainer(host);
    canvas = renderWindow.getApiSpecificRenderWindow().getCanvas();
    if (!canvas) throw new Error("Packaged physical GPU qualification has no drawing canvas");
    retired = new Promise<void>((resolve) => {
      confirmRetired = event => {
        contextLossMessage = (event as WebGLContextEvent).statusMessage || "driver supplied no status message";
        resolve();
      };
    });
    canvas.addEventListener("webglcontextlost", confirmRetired!, { once: true });
    renderWindow.resize();
    renderer = renderWindow.getRenderer();
    for (const input of prepared) {
      const image = vtkImageData.newInstance();
      resources.push(image);
      setPreparedImageScalars(image, input);
      const mapper = vtkVolumeMapper.newInstance();
      resources.push(mapper);
      configureCoherentVolumeSampling(mapper);
      mapper.setInputData(image);
      mapper.setSampleDistance(100);
      mapper.setMaximumSamplesPerRay(1000);
      mapper.setAutoAdjustSampleDistances(false);
      const color = vtkColorTransferFunction.newInstance();
      resources.push(color);
      color.addRGBPoint(-0.2, 0.03, 0.02, 0.1);
      color.addRGBPoint(0.8, 1, 0.8, 0.2);
      const opacity = vtkPiecewiseFunction.newInstance();
      resources.push(opacity);
      opacity.addPoint(-0.2, 0);
      opacity.addPoint(0.2, 0.05);
      opacity.addPoint(0.8, 0.5);
      const volume = vtkVolume.newInstance();
      resources.push(volume);
      volume.setMapper(mapper);
      volume.getProperty().setRGBTransferFunction(0, color);
      volume.getProperty().setScalarOpacity(0, opacity);
      volumes.push(volume);
      renderer.addVolume(volume);
    }
    ownedObjects = [renderWindow, renderWindow.getRenderWindow(), renderer,
      renderWindow.getInteractor(), renderWindow.getApiSpecificRenderWindow(), ...resources];
    renderer.resetCamera();
    renderingStage = "initial draw";
    renderWindow.getRenderWindow().render();
    await new Promise<void>((resolve) => requestAnimationFrame(() =>
      requestAnimationFrame(() => resolve())));
    renderingStage = "frame capture";
    const captured = await captureVtkPng(renderWindow);
    const renderPngSha256 = await sha256(captured);
    const api = renderWindow.getApiSpecificRenderWindow() as unknown as {
      getContext?: () => WebGL2RenderingContext | WebGLRenderingContext | null;
    };
    const context = api.getContext?.() ?? null;
    if (!context) throw new Error("Packaged physical GPU qualification has no WebGL context");
    const evidence = {
      retirement,
      renderPngSha256,
      contextLost: context.isContextLost(),
      glError: context.getError(),
      maximum3DTextureSize: Number(context.getParameter((context as WebGL2RenderingContext).MAX_3D_TEXTURE_SIZE) ?? 0),
      maximumLogicalGpuResidentBytes: visualizationResourceGovernor.stats().byStage.gpu.residentBytes,
    };
    renderingStage = "resident-memory sample";
    await sampleMemory(`gpu-cycle-${cycle}-resident`);
    return evidence;
  } catch (error) {
    throw new Error(`Physical GPU cycle ${cycle}, ${renderingStage}: ${error instanceof Error ? error.message : String(error)}; context loss: ${contextLossMessage ?? "no event delivered"}`, { cause: error });
  } finally {
    let timeout: ReturnType<typeof setTimeout> | undefined;
    try {
      const cleanupErrors = renderWindow ? disposeVtkScene({
        clearScene: () => undefined,
        cleanupSteps: volumes.map(volume => () => renderer?.removeVolume(volume)),
        resources: [...resources].reverse(),
        renderWindow,
      }) : [];
      if (cleanupErrors.length > 0) throw new Error(`GPU-cycle cleanup failed: ${cleanupErrors.map(String).join("; ")}`);
      retirement.deletedObjects = ownedObjects.filter(object => object.isDeleted?.() === true).length;
      retirement.canvasZeroed = canvas?.width === 0 && canvas.height === 0;
      if (ownedObjects.length > 0 && (retirement.deletedObjects !== ownedObjects.length || !retirement.canvasZeroed)) {
        throw new Error("GPU-cycle owned objects or canvas backing stores survived disposal");
      }
      if (retired) await Promise.race([retired, new Promise<never>((_, reject) => {
        timeout = setTimeout(() => reject(new Error("GPU-cycle context retirement did not complete")), 5_000);
      })]);
    } finally {
      if (timeout != null) clearTimeout(timeout);
      if (confirmRetired) canvas?.removeEventListener("webglcontextlost", confirmRetired);
      host?.remove();
      visualizationResourceGovernor.release(reservation.reservation.id);
    }
  }
}

async function qualifyPreparedGpuResidency(
  field: Npy3D,
  sampleMemory: (label: string) => Promise<void>,
): Promise<PackagedPhysicalGpuEvidence> {
  const textureBytes = field.data.byteLength * 2;
  const prepared = [0, 1].map(() => ({
    shape: field.shape,
    spacing: [100, 100, 100] as [number, number, number],
    origin: [0, 0, 0] as [number, number, number],
    values: cartesianGridToVtkScalars({
      shape: field.shape, spacing: [100, 100, 100], origin: [0, 0, 0], data: field.data,
    }),
  }));
  let maximumLogicalGpuResidentBytes = 0;
  let retiredContexts = 0;
  let contextLost = false;
  let maximum3DTextureSize = 0;
  const glErrors: number[] = [];
  const renderPngSha256: string[] = [];
  const postScopeRetirementMs: number[] = [];
  const retiredOwnedObjects: number[] = [];
  let zeroedCanvasBackingStores = 0;
  await sampleMemory("gpu-baseline");
  for (let cycle = 1; cycle <= 2; cycle += 1) {
    const evidence = await qualifyPreparedGpuCycle(prepared, textureBytes, cycle, sampleMemory);
    retiredContexts += 1;
    retiredOwnedObjects.push(evidence.retirement.deletedObjects);
    if (evidence.retirement.canvasZeroed) zeroedCanvasBackingStores += 1;
    maximumLogicalGpuResidentBytes = Math.max(maximumLogicalGpuResidentBytes, evidence.maximumLogicalGpuResidentBytes);
    maximum3DTextureSize = Math.max(maximum3DTextureSize, evidence.maximum3DTextureSize);
    contextLost ||= evidence.contextLost;
    glErrors.push(evidence.glError);
    renderPngSha256.push(evidence.renderPngSha256);
    const retiredAt = performance.now();
    await new Promise<void>((resolve) => requestAnimationFrame(() =>
      requestAnimationFrame(() => resolve())));
    postScopeRetirementMs.push(performance.now() - retiredAt);
    await sampleMemory(`gpu-cycle-${cycle}-retired`);
  }
  return {
    measurementProtocol: "isolated-prepared-pair-post-scope-v2",
    renderingPath: "nimbus-coherent-volume",
    cycles: 2,
    scalarTextureBytesPerCycle: textureBytes,
    preparedCpuBufferBytes: prepared.reduce((bytes, image) => bytes + image.values.byteLength, 0),
    preparedCpuBuffers: prepared.length,
    retiredContexts,
    retiredOwnedObjects,
    zeroedCanvasBackingStores,
    postScopeRetirementMs,
    maximumLogicalGpuResidentBytes,
    viewport: { width: 640, height: 420 },
    actualRenderedTexturesPerCycle: 2,
    contextLost,
    glErrors,
    maximum3DTextureSize,
    renderPngSha256,
  };
}
