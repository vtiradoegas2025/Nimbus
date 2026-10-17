import vtkGenericRenderWindow from "@kitware/vtk.js/Rendering/Misc/GenericRenderWindow";
import vtkImageData from "@kitware/vtk.js/Common/DataModel/ImageData";
import vtkVolumeMapper from "@kitware/vtk.js/Rendering/Core/VolumeMapper";
import vtkVolume from "@kitware/vtk.js/Rendering/Core/Volume";
import vtkColorTransferFunction from "@kitware/vtk.js/Rendering/Core/ColorTransferFunction";
import vtkPiecewiseFunction from "@kitware/vtk.js/Common/DataModel/PiecewiseFunction";
import { configureCoherentVolumeSampling } from "@nimbus/lib/vtk/coherentVolumeSampling";
import { prepareVtkRequest } from "@nimbus/lib/vtk/vtkPreparation";
import { setPreparedImageScalars } from "@nimbus/lib/vtk/vtkImageData";
import { setCameraAdjustedVolumeRayDistances } from "@nimbus/lib/vtk/volumeOpticalDepth";
import { setFlatVolumePreview } from "@nimbus/lib/vtk/volumeShaderBudget";
import { captureVtkPng } from "@nimbus/lib/vtk/vtkCaptureRegistry";
import { snapshotBrowserResources } from "./instrumentation";

export async function qualifyVolumeValidity() {
  const container = document.createElement("div");
  Object.assign(container.style, { position: "fixed", left: "0", top: "0", width: "128px", height: "128px" });
  document.body.append(container);
  const window = vtkGenericRenderWindow.newInstance({ background: [0, 0, 0] });
  const image = vtkImageData.newInstance();
  const mapper = vtkVolumeMapper.newInstance();
  const actor = vtkVolume.newInstance();
  const color = vtkColorTransferFunction.newInstance();
  const opacity = vtkPiecewiseFunction.newInstance();
  const rows = [];
  try {
    window.setContainer(container); window.resize();
    const renderer = window.getRenderer();
    const camera = renderer.getActiveCamera();
    camera.setPosition(8, 8, 40); camera.setFocalPoint(8, 8, 8); camera.setViewUp(0, 1, 0);
    camera.setParallelProjection(true); camera.setParallelScale(8);
    configureCoherentVolumeSampling(mapper);
    mapper.setInputData(image); mapper.setAutoAdjustSampleDistances(false); mapper.setMaximumSamplesPerRay(2000);
    actor.setMapper(mapper); renderer.addVolume(actor);
    color.addRGBPoint(-10, 1, 0, 0); color.addRGBPoint(310, 1, 0, 0);
    opacity.addPoint(-10, 0.1); opacity.addPoint(310, 0.1);
    const property = actor.getProperty();
    property.setRGBTransferFunction(0, color); property.setScalarOpacity(0, opacity);
    property.setShade(false); property.setInterpolationTypeToLinear();
    setCameraAdjustedVolumeRayDistances(property, camera,
      { setFullSampleDistanceFor: (target, distance) => target.setSampleDistance(distance) },
      1, 0.25, { xy: 1, z: 1 }, mapper);
    for (const mode of ["negative-gap", "positive-gap", "all-missing", "valid-zero", "restored-gap"] as const) {
      const values = new Float32Array(16 ** 3);
      for (let x = 0; x < 16; x++) for (let y = 0; y < 16; y++) for (let z = 0; z < 16; z++) {
        values[(x * 16 + y) * 16 + z] = mode === "all-missing" || (mode !== "valid-zero" && x < 8)
          ? Number.NaN : mode === "positive-gap" ? 300 : mode === "valid-zero" ? 0 : -4;
      }
      const prepared = prepareVtkRequest({ kind: "scalar", field: { shape: [16, 16, 16], data: values },
        grid: { drM: 1, dyM: 1, dzM: 1, scalarOriginM: [0.5, 0.5, 0.5] }, refineForDisplay: false });
      setPreparedImageScalars(image, prepared);
      renderer.resetCameraClippingRange();
      const captures = [];
      const pixels = [];
      for (const preview of [false, true, false]) {
        setFlatVolumePreview(mapper, preview);
        window.getRenderWindow().render();
        const png = await captureVtkPng(window);
        captures.push(png);
        const bitmap = new Image(); bitmap.src = png; await bitmap.decode();
        const target = document.createElement("canvas"); target.width = bitmap.width; target.height = bitmap.height;
        const context = target.getContext("2d")!; context.drawImage(bitmap, 0, 0);
        pixels.push([0.25, 0.75].map((x) => [...context.getImageData(Math.floor(bitmap.width * x), Math.floor(bitmap.height / 2), 1, 1).data]));
      }
      const before = snapshotBrowserResources();
      window.getRenderWindow().render();
      const after = snapshotBrowserResources();
      rows.push({ mode, pixels, captures, repeatedUploads: after.texture3DUploads - before.texture3DUploads,
        validityBytes: prepared.validity?.byteLength ?? 0, gpuBytes: after.gpuLogicalBytes,
        glError: window.getApiSpecificRenderWindow().getContext().getError() });
    }
  } finally {
    window.getRenderWindow().getInteractor()?.unbindEvents();
    window.setContainer(null as unknown as HTMLElement);
    actor.delete(); mapper.delete(); image.delete(); color.delete(); opacity.delete();
    window.getApiSpecificRenderWindow().getContext()?.getExtension("WEBGL_lose_context")?.loseContext();
    window.delete(); container.remove();
  }
  return rows;
}
