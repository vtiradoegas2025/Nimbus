import vtkGenericRenderWindow from "@kitware/vtk.js/Rendering/Misc/GenericRenderWindow";
import vtkImageData from "@kitware/vtk.js/Common/DataModel/ImageData";
import vtkDataArray from "@kitware/vtk.js/Common/Core/DataArray";
import vtkVolumeMapper from "@kitware/vtk.js/Rendering/Core/VolumeMapper";
import vtkVolume from "@kitware/vtk.js/Rendering/Core/Volume";
import vtkColorTransferFunction from "@kitware/vtk.js/Rendering/Core/ColorTransferFunction";
import vtkPiecewiseFunction from "@kitware/vtk.js/Common/DataModel/PiecewiseFunction";
import { configureCoherentVolumeSampling } from "@nimbus/lib/vtk/coherentVolumeSampling";
import { bindJointVolumes, type JointVolumeSource } from "@nimbus/lib/vtk/jointVolumeRendering";
import { setCameraAdjustedVolumeRayDistances } from "@nimbus/lib/vtk/volumeOpticalDepth";
import { captureVtkPng } from "@nimbus/lib/vtk/vtkCaptureRegistry";
import { setVolumeValidity } from "@nimbus/lib/vtk/volumeValidity";
import { snapshotBrowserResources } from "./instrumentation";

export async function qualifyVolumeAngularSampling() {
  const rows = [];
  for (const shape of [[16, 16, 16], [64, 32, 16], [32, 16, 64]]) {
    const spacing = shape.map(count => 16 / count) as [number, number, number];
    for (const field of ["flat", "affine", "masked-affine"] as const) {
      const container = document.createElement("div");
      Object.assign(container.style, { position: "fixed", width: "128px", height: "128px", left: "0", top: "0" });
      document.body.append(container);
      const window = vtkGenericRenderWindow.newInstance({ background: [0, 0, 0] });
      const owned: Array<{ delete(): void }> = [];
      const sources: JointVolumeSource[] = [];
      const opacities: Array<ReturnType<typeof vtkPiecewiseFunction.newInstance>> = [];
      let binding: ReturnType<typeof bindJointVolumes> | null = null;
      try {
        window.setContainer(container); window.resize();
        const renderer = window.getRenderer(), camera = renderer.getActiveCamera();
        camera.setParallelProjection(true); camera.setParallelScale(12); camera.setViewUp(0, 0, 1);
        for (let component = 0; component < 2; component++) {
          const image = vtkImageData.newInstance();
          image.setDimensions(shape[0], shape[1], shape[2]); image.setSpacing(spacing); image.setOrigin(spacing.map(value => value / 2) as [number, number, number]);
          const values = Float32Array.from({ length: shape[0] * shape[1] * shape[2] }, (_, index) =>
            field === "flat" ? 0.5 : 0.4 + 0.2 * (Math.floor(index / (shape[0] * shape[1])) + 0.5) / shape[2]);
          const scalars = vtkDataArray.newInstance({ values, numberOfComponents: 1 });
          image.getPointData().setScalars(scalars);
          if (field.startsWith("masked")) setVolumeValidity(image, Uint8Array.from(values, (_, index) =>
            Math.abs((Math.floor(index / shape[0]) % shape[1] + 0.5) * spacing[1] - 8) < 1 ? 0 : 255));
          const mapper = vtkVolumeMapper.newInstance();
          configureCoherentVolumeSampling(mapper);
          mapper.setInputData(image); mapper.setAutoAdjustSampleDistances(false); mapper.setMaximumSamplesPerRay(3000);
          const actor = vtkVolume.newInstance(); actor.setMapper(mapper);
          const color = vtkColorTransferFunction.newInstance(), opacity = vtkPiecewiseFunction.newInstance();
          color.addRGBPoint(0, 1, 1, 1); color.addRGBPoint(1, 1, 1, 1);
          opacity.addPoint(0, field === "flat" ? 0.03 : 0.02); opacity.addPoint(1, field === "flat" ? 0.03 : 0.04);
          const property = actor.getProperty();
          property.setRGBTransferFunction(0, color); property.setScalarOpacity(0, opacity);
          property.setShade(false); property.setSpecular(0); property.setInterpolationTypeToLinear();
          setCameraAdjustedVolumeRayDistances(property, camera,
            { setFullSampleDistanceFor: (target, distance) => target.setSampleDistance(distance) },
            component === 0 ? 1 : 2, Math.min(...spacing) * 0.5, { xy: 1, z: 1 }, mapper);
          sources.push({ actor, role: "independent" });
          opacities.push(opacity);
          owned.push(actor, mapper, image, scalars, color, opacity);
        }
        renderer.addVolume(sources[0].actor);
        for (const [azimuth, elevation] of [[0, 0], [37, 1], [94, -1], [23, 17]]) {
          const a = azimuth * Math.PI / 180, e = elevation * Math.PI / 180;
          const direction = [Math.cos(a) * Math.cos(e), Math.sin(a) * Math.cos(e), Math.sin(e)];
          camera.setPosition(8 + direction[0] * 40, 8 + direction[1] * 40, 8 + direction[2] * 40); camera.setFocalPoint(8, 8, 8);
          renderer.resetCameraClippingRange();
          const captures = [];
          for (const mode of ["single", "joint-one", "joint-two"]) {
            binding?.dispose(); binding = null;
            sources[1].actor.setVisibility(mode.startsWith("joint"));
            opacities[1].removeAllPoints();
            opacities[1].addPoint(0, mode === "joint-one" ? 0 : field === "flat" ? 0.03 : 0.02);
            opacities[1].addPoint(1, mode === "joint-one" ? 0 : field === "flat" ? 0.03 : 0.04);
            if (mode.startsWith("joint")) renderer.addVolume(sources[1].actor); else renderer.removeVolume(sources[1].actor);
            if (mode.startsWith("joint")) binding = bindJointVolumes(sources);
            window.getRenderWindow().render();
            const png = await captureVtkPng(window);
            const bitmap = new Image(); bitmap.src = png; await bitmap.decode();
            const canvas = document.createElement("canvas"); canvas.width = bitmap.width; canvas.height = bitmap.height;
            const context = canvas.getContext("2d")!; context.drawImage(bitmap, 0, 0);
            const rgba = context.getImageData(0, 0, canvas.width, canvas.height).data;
            const profile = [];
            for (let y = 40; y < 88; y++) profile.push(rgba[(y * canvas.width + 64) * 4] / 255);
            const pixels = [48, 64, 80].map(x => rgba[(64 * canvas.width + x) * 4] / 255);
            captures.push({ mode, png, profile, pixels });
          }
          const single = captures[0], one = captures[1], two = captures[2];
          const oneRmse = Math.sqrt(single.profile.reduce((sum, value, index) => sum + (value - one.profile[index]) ** 2, 0) / single.profile.length);
          const mixedExpected = single.profile.map(value => 1 - (1 - value) ** 1.5);
          const mixtureRmse = Math.sqrt(mixedExpected.reduce((sum, value, index) => sum + (value - two.profile[index]) ** 2, 0) / mixedExpected.length);
          const before = snapshotBrowserResources(); window.getRenderWindow().render(); const after = snapshotBrowserResources();
          rows.push({ shape, spacing, field, azimuth, elevation, oneRmse, mixtureRmse,
            warmUploads: after.texture3DUploads - before.texture3DUploads,
            glError: window.getApiSpecificRenderWindow().getContext().getError(), captures });
        }
      } finally {
        binding?.dispose();
        window.getRenderWindow().getInteractor()?.unbindEvents(); window.setContainer(null as unknown as HTMLElement);
        owned.forEach(value => value.delete());
        window.getApiSpecificRenderWindow().getContext()?.getExtension("WEBGL_lose_context")?.loseContext();
        window.delete(); container.remove();
      }
    }
  }
  return rows;
}
