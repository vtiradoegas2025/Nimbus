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
import { setImageScalars } from "./vtkImageData";
import { captureVtkPng } from "./vtkCaptureRegistry";
import { collectVtkAppearanceEvidence, type VtkAppearancePlot } from "./vtkAppearanceEvidence";
import { disposeVtkScene, type VtkDisposable } from "./vtkLifecycle";
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
  cycles: number;
  scalarTextureBytesPerCycle: number;
  maximumLogicalGpuResidentBytes: number;
  viewport: { width: number; height: number };
  firstResidentDeltaBytes?: number;
  secondResidentDeltaBytes?: number;
  maximumPhysicalResidentDeltaBytes?: number;
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
}

async function captureScene(plot: VtkAppearancePlot): Promise<PackagedAppearanceSceneEvidence> {
  const host = document.createElement("div");
  host.style.cssText = "position:fixed;inset:0 auto auto 0;width:640px;height:420px;background:#0e1013";
  document.body.append(host);
  const renderWindow = vtkGenericRenderWindow.newInstance({
    background: [0.055, 0.063, 0.075],
    listenWindowResize: false,
  });
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
  const textureBytes = field.data.byteLength * 2;
  let maximumLogicalGpuResidentBytes = 0;
  let contextLost = false;
  let maximum3DTextureSize = 0;
  const glErrors: number[] = [];
  const renderPngSha256: string[] = [];
  await sampleMemory("gpu-baseline");
  for (let cycle = 1; cycle <= 2; cycle += 1) {
    const host = document.createElement("div");
    host.style.cssText = "position:fixed;inset:0 auto auto 0;width:640px;height:420px;background:#0e1013";
    document.body.append(host);
    const renderWindow = vtkGenericRenderWindow.newInstance({
      background: [0.055, 0.063, 0.075],
      listenWindowResize: false,
    });
    renderWindow.setContainer(host);
    renderWindow.resize();
    const renderer = renderWindow.getRenderer();
    const resources: VtkDisposable[] = [];
    const volumes: Array<ReturnType<typeof vtkVolume.newInstance>> = [];
    const reservation = visualizationResourceGovernor.reserve({
      owner: "packaged:physical-gpu",
      generation: `cycle:${cycle}`,
      priority: "visible",
      claims: [{
        allocationId: `packaged:physical-gpu:${cycle}`,
        stage: "gpu",
        bytes: textureBytes,
        accounting: "resident",
      }],
    });
    if (!reservation.ok) throw new Error(reservation.reason);
    try {
      for (let copy = 0; copy < 2; copy += 1) {
        const image = vtkImageData.newInstance();
        setImageScalars(image, {
          shape: field.shape,
          spacing: [100, 100, 100],
          origin: [0, 0, 0],
          data: field.data,
        });
        const mapper = vtkVolumeMapper.newInstance();
        mapper.setInputData(image);
        mapper.setSampleDistance(100);
        mapper.setMaximumSamplesPerRay(1000);
        mapper.setAutoAdjustSampleDistances(false);
        const color = vtkColorTransferFunction.newInstance();
        color.addRGBPoint(-0.2, 0.03, 0.02, 0.1);
        color.addRGBPoint(0.8, 1, 0.8, 0.2);
        const opacity = vtkPiecewiseFunction.newInstance();
        opacity.addPoint(-0.2, 0);
        opacity.addPoint(0.2, 0.05);
        opacity.addPoint(0.8, 0.5);
        const volume = vtkVolume.newInstance();
        volume.setMapper(mapper);
        volume.getProperty().setRGBTransferFunction(0, color);
        volume.getProperty().setScalarOpacity(0, opacity);
        renderer.addVolume(volume);
        volumes.push(volume);
        resources.push(volume, mapper, color, opacity, image);
      }
      renderer.resetCamera();
      renderWindow.getRenderWindow().render();
      await new Promise<void>((resolve) => requestAnimationFrame(() =>
        requestAnimationFrame(() => resolve())));
      const captured = await captureVtkPng(renderWindow);
      renderPngSha256.push(await sha256(captured));
      const api = renderWindow.getApiSpecificRenderWindow() as unknown as {
        getContext?: () => WebGL2RenderingContext | WebGLRenderingContext | null;
      };
      const context = api.getContext?.() ?? null;
      if (!context) throw new Error("Packaged physical GPU qualification has no WebGL context");
      contextLost = contextLost || context.isContextLost();
      glErrors.push(context.getError());
      maximum3DTextureSize = Math.max(
        maximum3DTextureSize,
        Number(context.getParameter((context as WebGL2RenderingContext).MAX_3D_TEXTURE_SIZE) ?? 0),
      );
      maximumLogicalGpuResidentBytes = Math.max(
        maximumLogicalGpuResidentBytes,
        visualizationResourceGovernor.stats().byStage.gpu.residentBytes,
      );
      await sampleMemory(`gpu-cycle-${cycle}-resident`);
    } finally {
      for (const volume of volumes) renderer.removeVolume(volume);
      disposeVtkScene({
        clearScene: () => undefined,
        resources,
        renderWindow,
      });
      visualizationResourceGovernor.release(reservation.reservation.id);
      host.remove();
    }
    await new Promise<void>((resolve) => requestAnimationFrame(() =>
      requestAnimationFrame(() => resolve())));
    await sampleMemory(`gpu-cycle-${cycle}-retired`);
  }
  return {
    cycles: 2,
    scalarTextureBytesPerCycle: textureBytes,
    maximumLogicalGpuResidentBytes,
    viewport: { width: 640, height: 420 },
    actualRenderedTexturesPerCycle: 2,
    contextLost,
    glErrors,
    maximum3DTextureSize,
    renderPngSha256,
  };
}
