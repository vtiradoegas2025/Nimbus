import vtkGenericRenderWindow from "@kitware/vtk.js/Rendering/Misc/GenericRenderWindow";
import vtkImageData from "@kitware/vtk.js/Common/DataModel/ImageData";
import vtkVolumeMapper from "@kitware/vtk.js/Rendering/Core/VolumeMapper";
import vtkVolume from "@kitware/vtk.js/Rendering/Core/Volume";
import vtkColorTransferFunction from "@kitware/vtk.js/Rendering/Core/ColorTransferFunction";
import vtkPiecewiseFunction from "@kitware/vtk.js/Common/DataModel/PiecewiseFunction";
import { configureCoherentVolumeSampling } from "@nimbus/lib/vtk/coherentVolumeSampling";
import { setPreparedImageScalars } from "@nimbus/lib/vtk/vtkImageData";
import { setCameraAdjustedVolumeRayDistances } from "@nimbus/lib/vtk/volumeOpticalDepth";
import { captureVtkPng } from "@nimbus/lib/vtk/vtkCaptureRegistry";
import { snapshotBrowserResources } from "./instrumentation";

interface VolumeReuseStage {
  name: string;
  sameScalarArray: boolean;
  sameScalarMtime: boolean;
  shape: number[];
  spacing: number[];
  origin: number[];
  scalarValues: number[];
  uploads: number;
  pixels: number[][];
  png: string;
  glError: number;
}

export async function qualifyVolumeArrayReuse() {
  const container = document.createElement("div");
  Object.assign(container.style, { position: "fixed", left: "0", top: "0", width: "128px", height: "128px" });
  document.body.append(container);
  const window = vtkGenericRenderWindow.newInstance({ background: [0, 0, 0] });
  const image = vtkImageData.newInstance();
  const mapper = vtkVolumeMapper.newInstance();
  const actor = vtkVolume.newInstance();
  const color = vtkColorTransferFunction.newInstance();
  const opacity = vtkPiecewiseFunction.newInstance();
  const values = new Float32Array(8 * 16 * 16);
  const validity = Uint8Array.from({ length: values.length }, (_, index) => index % 8 < 4 ? 0 : 255);
  const prepared = { shape: [8, 16, 16] as [number, number, number],
    spacing: [2, 1, 1] as [number, number, number], origin: [1, 0.5, 0.5] as [number, number, number], values, validity };
  const stages: VolumeReuseStage[] = [];
  try {
    window.setContainer(container);
    window.resize();
    const renderer = window.getRenderer();
    const camera = renderer.getActiveCamera();
    camera.setPosition(8, 8, 40);
    camera.setFocalPoint(8, 8, 8);
    camera.setViewUp(0, 1, 0);
    camera.setParallelProjection(true);
    camera.setParallelScale(8);
    configureCoherentVolumeSampling(mapper);
    mapper.setInputData(image);
    mapper.setAutoAdjustSampleDistances(false);
    mapper.setMaximumSamplesPerRay(2000);
    actor.setMapper(mapper);
    renderer.addVolume(actor);
    color.addRGBPoint(0, 1, 0, 0);
    color.addRGBPoint(1, 0, 0, 1);
    opacity.addPoint(0, 0.1);
    opacity.addPoint(1, 0.1);
    const property = actor.getProperty();
    property.setRGBTransferFunction(0, color);
    property.setScalarOpacity(0, opacity);
    property.setShade(false);
    property.setInterpolationTypeToLinear();
    setCameraAdjustedVolumeRayDistances(property, camera,
      { setFullSampleDistanceFor: (target, distance) => target.setSampleDistance(distance) },
      1, 0.25, { xy: 1, z: 1 }, mapper);
    const capture = async (name: string, install: () => void) => {
      const previousArray = image.getPointData().getScalars();
      const previousTime = previousArray?.getMTime();
      const before = snapshotBrowserResources();
      install();
      renderer.resetCameraClippingRange();
      window.getRenderWindow().render();
      const png = await captureVtkPng(window);
      const bitmap = new Image();
      bitmap.src = png;
      await bitmap.decode();
      const target = document.createElement("canvas");
      target.width = bitmap.width;
      target.height = bitmap.height;
      const context = target.getContext("2d")!;
      context.drawImage(bitmap, 0, 0);
      const pixels = [0.15, 0.35, 0.65, 0.85].map(fraction => Array.from(context.getImageData(Math.floor(bitmap.width * fraction), Math.floor(bitmap.height / 2), 1, 1).data));
      const active = image.getPointData().getScalars();
      stages.push({ name, sameScalarArray: active === previousArray, sameScalarMtime: active?.getMTime() === previousTime,
        shape: [...image.getDimensions()], spacing: [...image.getSpacing()], origin: [...image.getOrigin()],
        scalarValues: Array.from({ length: Math.min(8, active.getData().length) }, (_, index) => active.getData()[index]),
        uploads: snapshotBrowserResources().texture3DUploads - before.texture3DUploads,
        pixels, png, glError: window.getApiSpecificRenderWindow().getContext().getError() });
    };
    await capture("initial", () => setPreparedImageScalars(image, prepared));
    await capture("same-preparation", () => setPreparedImageScalars(image, prepared));
    await capture("same-span-wrapper", () => setPreparedImageScalars(image, { ...prepared, values: new Float32Array(values.buffer) }));
    await capture("origin-and-spacing", () => {
      setPreparedImageScalars(image, { ...prepared, spacing: [2, 1, 2], origin: [11, 0.5, 1] });
      camera.setPosition(18, 8, 80);
      camera.setFocalPoint(18, 8, 16);
    });
    const reshaped = { ...prepared, shape: [16, 8, 16] as [number, number, number],
      spacing: [1, 2, 1] as [number, number, number], origin: [0.5, 1, 0.5] as [number, number, number] };
    await capture("same-buffer-reshaped", () => {
      setPreparedImageScalars(image, reshaped);
      camera.setPosition(8, 8, 40);
      camera.setFocalPoint(8, 8, 8);
    });
    const changed = { ...reshaped, values: new Float32Array(values.length).fill(1) };
    await capture("changed-scalars", () => setPreparedImageScalars(image, changed));
    const valid = { ...changed, validity: new Uint8Array(validity.length).fill(255) };
    await capture("changed-mask", () => setPreparedImageScalars(image, valid));
    await capture("mask-removed", () => setPreparedImageScalars(image, { ...valid, validity: undefined }));
    await capture("warm-repeat", () => setPreparedImageScalars(image, { ...valid, validity: undefined }));
  } finally {
    window.getRenderWindow().getInteractor()?.unbindEvents();
    window.setContainer(null as unknown as HTMLElement);
    actor.delete(); mapper.delete(); image.delete(); color.delete(); opacity.delete();
    window.getApiSpecificRenderWindow().getContext()?.getExtension("WEBGL_lose_context")?.loseContext();
    window.delete(); container.remove();
  }
  return stages;
}
