import { type AffineWind } from "@nimbus/lib/plot/affineWind";
import vtkGenericRenderWindow from "@kitware/vtk.js/Rendering/Misc/GenericRenderWindow";
import vtkImageData from "@kitware/vtk.js/Common/DataModel/ImageData";
import vtkDataArray from "@kitware/vtk.js/Common/Core/DataArray";
import vtkVolumeMapper from "@kitware/vtk.js/Rendering/Core/VolumeMapper";
import vtkVolume from "@kitware/vtk.js/Rendering/Core/Volume";
import vtkColorTransferFunction from "@kitware/vtk.js/Rendering/Core/ColorTransferFunction";
import vtkPiecewiseFunction from "@kitware/vtk.js/Common/DataModel/PiecewiseFunction";
import vtkLight from "@kitware/vtk.js/Rendering/Core/Light";
import { configureCoherentVolumeSampling } from "@nimbus/lib/vtk/coherentVolumeSampling";
import { bindJointVolumes, type JointVolumeSource } from "@nimbus/lib/vtk/jointVolumeRendering";
import { setCameraAdjustedVolumeRayDistances } from "@nimbus/lib/vtk/volumeOpticalDepth";
import { captureVtkPng } from "@nimbus/lib/vtk/vtkCaptureRegistry";
import { setVolumeValidity } from "@nimbus/lib/vtk/volumeValidity";
import { disposeVtkScene } from "@nimbus/lib/vtk/vtkLifecycle";

export async function qualifyJointVolumeRendering() {
  const rows = [];
  for (const mode of ["overlap", "partial", "disjoint", "diagnostic", "native-gradient", "context", "context-empty", "shadow", "shadow-gradient", "four-fields", "large-context", "missing-layer", "all-missing", "temporal-start", "temporal-middle", "temporal-end", "temporal-translation", "temporal-missing", "temporal-fixed-gap", "temporal-start-missing", "temporal-end-missing", "temporal-lit", "temporal-vertical", "temporal-shear", "temporal-rotation", "temporal-affine-fixed-gap", "temporal-affine-normal"] as const) {
    const temporal = mode.startsWith("temporal");
    const shadow = mode === "shadow" || mode === "shadow-gradient" || mode === "temporal-lit" || mode === "temporal-affine-normal";
    const captures: string[] = [];
    for (const reverse of [false, true]) {
      const started = performance.now();
      const progress = (phase: string) => console.info("NIMBUS_JOINT_PROGRESS " + JSON.stringify({ mode, reverse, phase, elapsedMs: performance.now() - started }));
      progress("start");
      const container = document.createElement("div");
      Object.assign(container.style, { position: "fixed", left: "0", top: "0", width: "128px", height: "128px" });
      document.body.append(container);
      const window = vtkGenericRenderWindow.newInstance({ background: [0, 0, 0] });
      const owned: Array<{ delete(): void }> = [];
      let binding: ReturnType<typeof bindJointVolumes> | null = null;
      try {
        window.setContainer(container); window.resize();
        const renderer = window.getRenderer();
        const camera = renderer.getActiveCamera();
        const worldOffset = mode === "large-context" ? 1e9 : 0;
        camera.setPosition(worldOffset + 1, worldOffset + 1, worldOffset + 8);
        camera.setFocalPoint(worldOffset + 1, worldOffset + 1, worldOffset + 1); camera.setViewUp(0, 1, 0);
        if (mode === "temporal-translation") { camera.setPosition(2, 1, 8); camera.setFocalPoint(2, 1, 1); }
        if (mode === "temporal-vertical") { camera.setPosition(8, 1, 2.5); camera.setFocalPoint(1, 1, 2.5); camera.setViewUp(0, 0, 1); }
        if (mode === "temporal-shear") { camera.setPosition(0.25, 1, 8); camera.setFocalPoint(0.25, 1, 1); }
        if (mode === "temporal-rotation") { camera.setPosition(2.04, 1, 8); camera.setFocalPoint(2.04, 1, 1); }
        camera.setParallelProjection(true); camera.setParallelScale(2);
        if (shadow) {
          renderer.setAutomaticLightCreation(false); renderer.setLightFollowCamera(false);
          renderer.removeAllLights();
          const light = vtkLight.newInstance(); light.setLightTypeToSceneLight(); light.setPositional(false);
          light.setPosition(1, 1, 8); light.setFocalPoint(1, 1, 0); light.setColor(1, 1, 1); light.setIntensity(1);
          renderer.addLight(light); owned.push(light);
        }
        const actors: JointVolumeSource[] = [];
        for (let i = 0; i < (mode === "four-fields" ? 4 : 2); i++) {
          const spacing = mode === "large-context" && i === 0 ? 1e9 : 0.25 / 2 ** i;
          const side = mode === "large-context" && i === 0 ? 2 : Math.round(2 / spacing);
          const offset = mode === "large-context" && i === 1 ? worldOffset : 0;
          const lowerZ = offset + (i === 0 ? 0 : mode === "partial" ? 1 : mode === "disjoint" ? 3 : mode === "temporal-vertical" ? 2 : 0);
          const lowerX = offset + (mode === "native-gradient" && i === 1 ? 0.25 : mode === "temporal-translation" && i === 1 ? 2 : 0);
          const image = vtkImageData.newInstance();
          image.setDimensions(side, side, side); image.setSpacing([spacing, spacing, spacing]);
          image.setOrigin([lowerX + spacing / 2, offset + spacing / 2, lowerZ + spacing / 2]);
          const values = new Float32Array(side ** 3).fill(temporal ? i === 0 ? 0.4 : 0.8 : 1);
          if (mode === "native-gradient" || mode === "shadow-gradient" || mode === "temporal-affine-normal") {
            for (let index = 0; index < values.length; index++) values[index] = (lowerX + (index % side + 0.5) * spacing) / 2;
          }
          const data = vtkDataArray.newInstance({ numberOfComponents: 1, values });
          image.getPointData().setScalars(data);
          if (mode === "all-missing" || mode === "missing-layer" && i === 0 || mode === "temporal-missing" && i === 1) setVolumeValidity(image, new Uint8Array(values.length));
          if (mode === "temporal-start-missing" && i === 1 || mode === "temporal-end-missing" && i === 0) setVolumeValidity(image, new Uint8Array(values.length));
          if (mode === "temporal-fixed-gap" || mode === "temporal-affine-fixed-gap") {
            const validity = Uint8Array.from(values, (_, index) => Math.abs((index % side + 0.5) * spacing - 1) < 0.2 ? 0 : 255);
            setVolumeValidity(image, validity);
          }
          const mapper = vtkVolumeMapper.newInstance();
          configureCoherentVolumeSampling(mapper);
          mapper.setInputData(image); mapper.setAutoAdjustSampleDistances(false); mapper.setMaximumSamplesPerRay(3000);
          const volume = vtkVolume.newInstance(); volume.setMapper(mapper);
          const ctf = vtkColorTransferFunction.newInstance();
          const color: [number, number, number] = temporal ? [1, 1, 1] : mode === "four-fields"
            ? [[1,0,0], [0,1,0], [0,0,1], [1,1,1]][i] as [number, number, number]
            : i === 0 || mode === "diagnostic" ? [1, 0, 0] : [0, 1, 0];
          ctf.addRGBPoint(0, ...color); ctf.addRGBPoint(1, ...color);
          const pwf = vtkPiecewiseFunction.newInstance();
          const opacity = mode === "four-fields" ? 1 - 0.8 ** (i + 1) : i === 0 ? 0.2 : mode === "context-empty" ? 0 : 0.36;
          if (temporal) { pwf.addPoint(0, 0); pwf.addPoint(2, 1); }
          else if (mode === "native-gradient") { pwf.addPoint(0, 0); pwf.addPoint(1.5, 0.75); }
          else { pwf.addPoint(0, opacity); pwf.addPoint(1, opacity); }
          const property = volume.getProperty(); property.setRGBTransferFunction(0, ctf); property.setScalarOpacity(0, pwf);
          property.setShade(false); property.setSpecular(0); property.setVolumetricScatteringBlending(0);
          if (shadow) {
            property.setShade(true); property.setAmbient(0); property.setDiffuse(1);
            property.setVolumetricScatteringBlending(1); property.setAnisotropy(0); property.setGlobalIlluminationReach(1);
          }
          if (mode === "temporal-affine-normal") { property.setAmbient(0.2); property.setVolumetricScatteringBlending(0); }
          property.setInterpolationTypeToLinear();
          setCameraAdjustedVolumeRayDistances(property, camera,
            { setFullSampleDistanceFor: (target, distance) => target.setSampleDistance(distance) },
            temporal ? 1 : mode === "large-context" && i === 0 ? 1e9 : mode === "four-fields" ? i + 1 : i === 0 || mode === "native-gradient" ? 1 : 2,
            mode === "large-context" && i === 0 ? 5e8 : 0.0625, { xy: 1, z: 1 }, mapper);
          actors.push({ actor: volume, role: temporal ? i === 0 ? "history" : "cloud" : mode === "diagnostic" ? i === 0 ? "cloud" : "condensation"
            : mode.startsWith("context") || mode === "large-context" ? i === 0 ? "context" : "cloud" : "independent" });
          owned.push(volume, mapper, image, ctf, pwf, property);
        }
        const order = reverse ? [...actors].reverse() : actors;
        for (const source of order) renderer.addVolume(source.actor);
        binding = bindJointVolumes(order);
        if (temporal) binding.setTemporal([{ current: actors[1].actor, previous: actors[0].actor }], mode.startsWith("temporal-start") ? 0 : mode.startsWith("temporal-end") ? 1 : 0.5, [mode === "temporal-translation" ? 2 : mode === "temporal-fixed-gap" ? 1 : 0, 0, 0]);
        if (["temporal-vertical", "temporal-shear", "temporal-rotation", "temporal-affine-fixed-gap", "temporal-affine-normal"].includes(mode)) {
          const model: AffineWind = { centerM: [1, 1, 1], velocityMs: [0, 0, mode === "temporal-vertical" ? 2 : 0],
            gradientPerS: mode === "temporal-rotation" ? [0, -Math.PI / 4, 0, Math.PI / 4, 0, 0, 0, 0, 0]
              : [0, 0, mode === "temporal-shear" || mode === "temporal-affine-fixed-gap" || mode === "temporal-affine-normal" ? 1 : 0, 0, 0, 0, 0, 0, 0] };
          binding.setTemporal([{ current: actors[1].actor, previous: actors[0].actor }], 0.5, [0,0,0], { model, intervalS: 1, scale: { xy: 1, z: 1 } });
        }
        renderer.resetCameraClippingRange();
        camera.setClippingRange(0.1, 20);
        progress("render");
        window.getRenderWindow().render();
        progress("capture");
        captures.push(await captureVtkPng(window));
        progress("captured");
        if (binding.describe().mode !== "joint-optical-depth") throw new Error("Joint fixture did not use the joint renderer");
      } finally {
        const errors = disposeVtkScene({ clearScene: () => {}, cleanupSteps: [() => binding?.dispose(), () => container.remove()],
          resources: owned, renderWindow: window });
        progress(errors.length ? "cleanup-failed" : "disposed");
        if (errors.length) throw new Error(`Joint fixture cleanup failed for ${mode}: ${String(errors[0])}`);
      }
    }
    const pixels = [];
    let width = 0;
    for (const png of captures) {
      const image = new Image(); image.src = png; await image.decode();
      width = image.width;
      const canvas = document.createElement("canvas"); canvas.width = image.width; canvas.height = image.height;
      const context = canvas.getContext("2d")!; context.drawImage(image, 0, 0);
      pixels.push([...context.getImageData(Math.floor(image.width / 2), Math.floor(image.height / 2), 1, 1).data].slice(0, 3).map((v) => v / 255));
    }
    const gradientColor = 0.5 * (1 - (1 - (1 + 2 / width) / 4) ** 4);
    const scattered = 0.125 * (1 - 0.8 ** 8);
    const expected = temporal ? [0, 0, 0].map(() => mode === "temporal-affine-normal" ? 0.2 * (1 - (1 - (1 + 2 / width) / 4) ** 2) : mode === "temporal-lit" ? 0.25 : mode === "temporal-shear" ? 1 - 0.7 ** (4 * (0.25 + 2 / width)) : mode === "temporal-missing" || mode === "temporal-fixed-gap" || mode === "temporal-affine-fixed-gap" ? 0 : mode.startsWith("temporal-start") ? 0.36 : mode.startsWith("temporal-end") ? 0.64 : 0.51) : mode === "four-fields" ? [0, 0, 0].map(() => 0.5 * (1 - 0.8 ** 8))
      : shadow ? [scattered, scattered, 0] : mode === "context" || mode === "large-context" || mode === "missing-layer" ? [0, 0.36, 0] : mode === "context-empty" || mode === "all-missing" ? [0, 0, 0]
      : mode === "native-gradient" ? [gradientColor, gradientColor, 0] : mode === "overlap" ? [0.2952, 0.2952, 0]
      : mode === "partial" ? [0.2464, 0.344, 0]
      : mode === "disjoint" ? [0.2304, 0.36, 0] : [0.36, 0, 0];
    rows.push({ mode, pixels, expected, captures });
  }
  return rows;
}
