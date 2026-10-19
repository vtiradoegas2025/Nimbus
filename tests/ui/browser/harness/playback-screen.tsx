import { useCallback, useLayoutEffect, useRef, useState } from "react";
import vtkGenericRenderWindow from "@kitware/vtk.js/Rendering/Misc/GenericRenderWindow";
import vtkImageData from "@kitware/vtk.js/Common/DataModel/ImageData";
import vtkVolumeMapper from "@kitware/vtk.js/Rendering/Core/VolumeMapper";
import vtkVolume from "@kitware/vtk.js/Rendering/Core/Volume";
import vtkColorTransferFunction from "@kitware/vtk.js/Rendering/Core/ColorTransferFunction";
import vtkPiecewiseFunction from "@kitware/vtk.js/Common/DataModel/PiecewiseFunction";
import { PlaybackProvider } from "@nimbus/components/plots/playbackContext";
import { configureCoherentVolumeSampling } from "@nimbus/lib/vtk/coherentVolumeSampling";
import { setPreparedImageScalars } from "@nimbus/lib/vtk/vtkImageData";
import { attachVtkResize } from "@nimbus/lib/vtk/vtkInteraction";
import { VOLUME_RESOLUTION_POLICY } from "@nimbus/lib/vtk/vtkResolution";
import { attachVtkPlaybackScreenResolution, type VtkPlaybackScreenResolution } from "@nimbus/lib/vtk/vtkPlaybackScreenResolution";
import { useVtkPlaybackScreenResolution } from "@nimbus/lib/vtk/useVtkPlaybackScreenResolution";
import { installVtkContextRenderGuard } from "@nimbus/lib/vtk/vtkContextRenderGuard";
import { captureVtkPng } from "@nimbus/lib/vtk/vtkCaptureRegistry";
import { snapshotBrowserResources } from "../support/instrumentation";
import type { GpuRenderTimer, GpuRenderTimingStatus } from "@nimbus/lib/vtk/vtkGpuRenderTimer";

declare global {
  interface Window {
    __nimbusPlaybackScreenHarness?: {
      setPlayback: (playing: boolean, recording?: boolean) => void;
      completeGpuSamples: (milliseconds: number, count: number) => Promise<void>;
      snapshot: () => {
        timingSource: string; distance: number; source: number[]; values: number[]; validity: number[];
        sameArray: boolean; sameMtime: boolean; dimensions: number[]; spacing: number[]; origin: number[];
        sampleDistance: number; opacityDistance: number; cameraPosition: number[]; cameraFocus: number[];
        controller: ReturnType<VtkPlaybackScreenResolution["describe"]>;
        texture3DUploads: number; glError: number;
      };
      capture: () => Promise<{ png: string; heldDistance: number; restoredDistance: number }>;
      setTimingUnavailable: () => void;
    };
  }
}

function Scene({ setPlayback }: { setPlayback: (playing: boolean, recording?: boolean) => void }) {
  const container = useRef<HTMLDivElement>(null);
  const sceneWindow = useRef<ReturnType<typeof vtkGenericRenderWindow.newInstance> | null>(null);
  const [generation, setGeneration] = useState(0);
  useLayoutEffect(() => {
    installVtkContextRenderGuard();
    const window = vtkGenericRenderWindow.newInstance({ background: [0, 0, 0] });
    window.setContainer(container.current!);
    sceneWindow.current = window;
    const image = vtkImageData.newInstance(), mapper = vtkVolumeMapper.newInstance(), actor = vtkVolume.newInstance();
    const color = vtkColorTransferFunction.newInstance(), opacity = vtkPiecewiseFunction.newInstance();
    const values = Float32Array.from({ length: 4096 }, (_, index) => 0.01 * (index % 16) + 0.02 * (Math.floor(index / 16) % 16));
    const validity = Uint8Array.from({ length: 4096 }, (_, index) => index % 16 < 3 ? 0 : 255);
    const source = [...values];
    setPreparedImageScalars(image, { shape: [16, 16, 16], origin: [0.5, 0.5, 0.5], spacing: [1, 1, 1], values, validity });
    const scalars = image.getPointData().getScalars(), mtime = scalars.getMTime();
    configureCoherentVolumeSampling(mapper);
    mapper.setInputData(image); mapper.setAutoAdjustSampleDistances(false); mapper.setSampleDistance(0.5); mapper.setMaximumSamplesPerRay(256);
    actor.setMapper(mapper); window.getRenderer().addVolume(actor);
    color.addRGBPoint(0, 0.2, 0.4, 0.8); color.addRGBPoint(0.5, 0.8, 0.4, 0.2);
    opacity.addPoint(0, 0); opacity.addPoint(0.5, 0.2);
    actor.getProperty().setRGBTransferFunction(0, color); actor.getProperty().setScalarOpacity(0, opacity);
    actor.getProperty().setScalarOpacityUnitDistance(0, 1); actor.getProperty().setInterpolationTypeToLinear(); actor.getProperty().setShade(false);
    const camera = window.getRenderer().getActiveCamera();
    camera.setPosition(8, 8, 40); camera.setFocalPoint(8, 8, 8); camera.setViewUp(0, 1, 0);
    camera.setParallelProjection(true); camera.setParallelScale(9); window.getRenderer().resetCameraClippingRange();
    let callback: ((milliseconds: number) => void) | null = null;
    let status: GpuRenderTimingStatus = "available";
    const timer: GpuRenderTimer = { begin: () => status === "available" && callback == null,
      end: next => { callback = next; }, cancel: () => { callback = null; }, dispose: () => { callback = null; }, status: () => status };
    const controller = attachVtkPlaybackScreenResolution(window, { getMappers: () => [mapper], timer });
    const detachResize = attachVtkResize(container.current!, window, { policy: VOLUME_RESOLUTION_POLICY });
    window.getRenderWindow().render();
    const browser = globalThis.window;
    browser.__nimbusPlaybackScreenHarness = {
      setPlayback,
      completeGpuSamples: async (milliseconds, count) => {
        for (let index = 0; index < count; index++) {
          window.getRenderWindow().render();
          await new Promise<void>(resolve => requestAnimationFrame(() => { const completed = callback; callback = null; completed?.(milliseconds); resolve(); }));
        }
      },
      snapshot: () => ({ timingSource: "manufactured completed-GPU-query durations; actual VTK source and pixels",
        distance: mapper.getImageSampleDistance(), source, values: [...image.getPointData().getScalars().getData()], validity: [...validity],
        sameArray: image.getPointData().getScalars() === scalars, sameMtime: scalars.getMTime() === mtime,
        dimensions: image.getDimensions(), spacing: image.getSpacing(), origin: image.getOrigin(),
        sampleDistance: mapper.getSampleDistance(), opacityDistance: actor.getProperty().getScalarOpacityUnitDistance(0),
        cameraPosition: [...camera.getPosition()], cameraFocus: [...camera.getFocalPoint()], controller: controller.describe(),
        texture3DUploads: snapshotBrowserResources().texture3DUploads, glError: window.getApiSpecificRenderWindow().getContext().getError() }),
      capture: async () => {
        const capture = captureVtkPng(window), heldDistance = mapper.getImageSampleDistance();
        const png = await capture;
        return { png, heldDistance, restoredDistance: mapper.getImageSampleDistance() };
      },
      setTimingUnavailable: () => { status = "unavailable"; window.getRenderWindow().render(); },
    };
    setGeneration(value => value + 1);
    return () => {
      delete browser.__nimbusPlaybackScreenHarness;
      controller.dispose(); detachResize(); sceneWindow.current = null;
      window.getRenderer().removeVolume(actor); actor.delete(); mapper.delete(); image.delete(); color.delete(); opacity.delete(); window.delete();
    };
  }, [setPlayback]);
  useVtkPlaybackScreenResolution(useCallback(() => sceneWindow.current, []), generation);
  return <div ref={container} style={{ width: 256, height: 256 }} />;
}

export function PlaybackScreenHarness() {
  const [state, setState] = useState({ playing: false, recording: false });
  const setPlayback = useCallback((playing: boolean, recording = false) => setState({ playing, recording }), []);
  return <PlaybackProvider value={{ ...state, interpolationDurationMs: 1000 / 60 }}><Scene setPlayback={setPlayback} /></PlaybackProvider>;
}
