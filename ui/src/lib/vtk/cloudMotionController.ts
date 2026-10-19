import { releaseVtkPointDataArrays } from "./vtkArrayOwnership";
import { setVolumeValidity } from "./volumeValidity";
import vtkImageData from "@kitware/vtk.js/Common/DataModel/ImageData.js";
import vtkVolume from "@kitware/vtk.js/Rendering/Core/Volume.js";
import vtkVolumeMapper from "@kitware/vtk.js/Rendering/Core/VolumeMapper.js";
import { setPreparedImageScalars, setPreparedImageDisplayGeometry } from "./vtkImageData";
import { configureCoherentVolumeSampling } from "./coherentVolumeSampling";
import { bindJointVolumes } from "./jointVolumeRendering";
import { volumeImageBytes } from "./volumeValidityBudget";
import { visualizationBufferClaim, visualizationResourceGovernor } from "@/lib/core/visualizationResourceGovernor";
import { planCloudMotion, type CloudMotionFrame, type CloudMotionSource } from "@/lib/plot/cloudMotion";

type OpticalMapper = { getViewSpecificProperties(): Record<string, unknown>; setViewSpecificProperties(value: Record<string, unknown>): void };
type Actor = ReturnType<typeof vtkVolume.newInstance>;
let sequence = 0;

export function createCloudMotionController(
  renderer: { addVolume(actor: Actor): void; removeVolume(actor: Actor): void },
  render: () => void,
  getContext: () => WebGL2RenderingContext | null,
  setResolutionScale: (scale: number) => void,
) {
  const owner = `cloud-motion:${++sequence}`;
  const history = [0, 1].map(() => {
    const image = vtkImageData.newInstance();
    const mapper = vtkVolumeMapper.newInstance();
    configureCoherentVolumeSampling(mapper);
    mapper.setInputData(image);
    mapper.setAutoAdjustSampleDistances(false);
    const actor = vtkVolume.newInstance(); actor.setMapper(mapper); actor.setVisibility(false);
    actor.getProperty().setSpecular(0);
    renderer.addVolume(actor);
    return { actor, mapper, image, initialProperty: actor.getProperty() };
  });
  let saved: CloudMotionFrame | null = null;
  let savedReservation: number | null = null;
  let transitionReservation: number | null = null;
  let animation = 0, disposed = false, serial = 0;
  let fence: WebGLSync | null = null;
  let fenceContext: WebGL2RenderingContext | null = null;
  let submittedAt = 0;
  let binding: ReturnType<typeof bindJointVolumes> | null = null;
  let finish: (() => void) | null = null;
  let reducedResolution = false;
  let status: { fromSources: readonly CloudMotionSource[]; toSources: readonly CloudMotionSource[]; active: boolean; fromStep: number | null; toStep: number | null; fraction: number; timeS: number | null; toTimeS: number | null; alignment: "wind" | "cloud-centroid" | null; deformationPerS: readonly number[] | null; windWeighting: string | null; displacementM: [number, number, number] | null; reason: string | null } =
    { fromSources: [], toSources: [], active: false, fromStep: null, toStep: null, fraction: 1, timeS: null, toTimeS: null, alignment: null, deformationPerS: null, windWeighting: null, displacementM: null, reason: null };
  const release = (id: number | null) => { if (id != null) visualizationResourceGovernor.release(id); };
  const claims = (...frames: CloudMotionFrame[]) => [...new Map(frames.flatMap(frame =>
    [frame.cloud, frame.context].flatMap(image => image
      ? [visualizationBufferClaim(image.values.buffer, "prepared-main"), ...(image.validity ? [visualizationBufferClaim(image.validity.buffer, "prepared-main")] : [])] : []))
    .map(claim => [claim.allocationId, claim])).values()];
  const remember = (frame: CloudMotionFrame) => {
    const held = visualizationResourceGovernor.reserve({ owner, generation: `saved:${++serial}`, priority: "visible", claims: claims(frame) });
    release(savedReservation); savedReservation = held.ok ? held.reservation.id : null;
    saved = held.ok ? frame : null;
    return held.ok;
  };
  const clearHistory = () => {
    binding?.setTemporal([], 1, [0, 0, 0]);
    if (reducedResolution) setResolutionScale(1);
    reducedResolution = false;
    for (const old of history) {
      old.actor.setVisibility(false);
      releaseVtkPointDataArrays(old.image.getPointData());
      setVolumeValidity(old.image);
    }
    release(transitionReservation); transitionReservation = null;
  };
  const stop = (complete: boolean) => {
    cancelAnimationFrame(animation); animation = 0;
    if (fence) fenceContext?.deleteSync(fence);
    fence = null; fenceContext = null;
    const wasActive = status.active;
    status = { ...status, active: false, fraction: 1, timeS: status.toTimeS };
    clearHistory();
    const completed = finish; finish = null;
    if (wasActive && complete) { render(); completed?.(); }
  };
  return {
    sources: history.map(({ actor }) => ({ actor, role: "history" as const, shadowReachM: 3000 })),
    attach(value: ReturnType<typeof bindJointVolumes>) { binding = value; },
    describe: () => ({ ...status }),
    stop(complete = true) { stop(complete); },
    reset() {
      const redraw = status.active || reducedResolution;
      stop(false); release(savedReservation); savedReservation = null; saved = null;
      if (redraw && !disposed) render();
    },
    commit(input: {
      frame: CloudMotionFrame; enabled: boolean; durationMs: number; scale: { xy: number; z: number };
      current: [Actor, Actor]; onStart(): void; onFinish(elapsedMs: number): void; onProgress(fraction: number, timeS: number): void;
    }): boolean {
      if (disposed) return false;
      try {
        if (input.enabled && saved?.key === input.frame.key && saved.domainKey === input.frame.domainKey) {
          if (saved.cloud.values === input.frame.cloud.values && saved.context?.values === input.frame.context?.values) return status.active;
          stop(false); remember(input.frame); return false;
        }
        stop(false);
        const previous = saved;
        if (!input.enabled) {
          status = { ...status, fromSources: [], toSources: input.frame.sources ?? [], fromStep: null, toStep: input.frame.step, timeS: input.frame.timeS, toTimeS: input.frame.timeS, alignment: null, deformationPerS: null, windWeighting: null, displacementM: null };
          release(savedReservation); savedReservation = null; saved = null; return false;
        }
        const plan = input.frame.wind?.ok === false ? input.frame.wind
          : previous ? planCloudMotion(previous, input.frame) : { ok: false as const, reason: "Waiting for the next saved cloud frame" };
        status = { fromSources: previous?.sources ?? [], toSources: input.frame.sources ?? [], active: false, fromStep: previous?.step ?? null, toStep: input.frame.step, fraction: 1, timeS: input.frame.timeS, toTimeS: input.frame.timeS, alignment: plan.ok ? plan.method : null, deformationPerS: plan.ok && plan.wind ? plan.wind.gradientPerS : null,
          windWeighting: input.frame.wind?.ok ? input.frame.wind.summary.weighting : null, displacementM: plan.ok ? plan.displacement : null, reason: plan.ok ? null : plan.reason };
        if (!previous || !plan.ok || !binding) { remember(input.frame); return false; }
        if (!Number.isFinite(input.durationMs) || input.durationMs <= 0) { status.reason = "Invalid interpolation duration"; remember(input.frame); return false; }
        const gl = getContext();
        if (!gl || typeof gl.fenceSync !== "function") { status.reason = "GPU completion tracking is unavailable"; remember(input.frame); return false; }
        const previousBytes = volumeImageBytes(previous.cloud) + volumeImageBytes(previous.context);
        const admitted = visualizationResourceGovernor.reserve({ owner, generation: `transition:${++serial}`, priority: "visible", claims: [
          ...claims(previous, input.frame),
          { allocationId: `${owner}:history-gpu:${serial}`, stage: "gpu", bytes: previousBytes, accounting: "resident" },
        ] });
        if (!admitted.ok) { status.reason = admitted.reason; remember(input.frame); return false; }
        transitionReservation = admitted.reservation.id;
        reducedResolution = true; setResolutionScale(0.5);
        const pairs: { current: Actor; previous: Actor }[] = [];
        for (let i = 0; i < 2; i++) {
          const image = i === 0 ? previous.context : previous.cloud;
          if (!image) continue;
          const old = history[i], current = input.current[i], mapper = current.getMapper()!;
          setPreparedImageScalars(old.image, image);
          setPreparedImageDisplayGeometry(old.image, image, input.scale);
          old.actor.setProperty(current.getProperty());
          (old.mapper as unknown as OpticalMapper).setViewSpecificProperties((mapper as unknown as OpticalMapper).getViewSpecificProperties());
          old.mapper.setSampleDistance(mapper.getSampleDistance());
          old.mapper.setMaximumSamplesPerRay(mapper.getMaximumSamplesPerRay());
          old.mapper.setImageSampleDistance(mapper.getImageSampleDistance());
          old.mapper.setVolumeShadowSamplingDistFactor(mapper.getVolumeShadowSamplingDistFactor());
          old.actor.setVisibility(true);
          pairs.push({ current, previous: old.actor });
        }
        remember(input.frame);
        let duration = Math.max(50, Math.min(5000, input.durationMs));
        const wind = plan.wind ? { model: plan.wind, intervalS: input.frame.timeS - previous.timeS, scale: input.scale } : undefined;
        const displacement = plan.displacement.map((value, axis) => value * (axis === 2 ? input.scale.z : input.scale.xy)) as [number, number, number];
        status = { ...status, active: true, fraction: 0, timeS: previous.timeS, reason: null };
        let start = 0; let nextDraw = 0;
        finish = () => input.onFinish(start > 0 ? Math.min(duration, performance.now() - start) : 0);
        input.onStart(); input.onProgress(0, previous.timeS);
        const trackDraw = () => {
          fence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
          if (!fence) throw new Error("GPU completion fence could not be created");
          fenceContext = gl; submittedAt = performance.now(); gl.flush();
        };
        const recover = (error: unknown) => {
          stop(false);
          status.reason = `Cloud interpolation unavailable: ${error instanceof Error ? error.message : String(error)}`;
          render(); input.onFinish(0);
        };
        const draw = (now: number) => {
          try {
          if (disposed || !status.active) return;
          if (fence) {
            const completion = gl.clientWaitSync(fence, 0, 0);
            if (completion === gl.TIMEOUT_EXPIRED) {
              if (now - submittedAt > 5000) throw new Error("GPU preview exceeded its completion budget");
              animation = requestAnimationFrame(draw); return;
            }
            if (completion === gl.WAIT_FAILED) throw new Error("GPU completion check failed");
            gl.deleteSync(fence); fence = null;
            if (start === 0) {
              start = now;
              duration = Math.min(5000, Math.max(duration, (now - submittedAt) * 4));
            }
          }
          const fraction = Math.min(1, Math.max(0, (now - start) / duration));
          if (fraction >= 1) {
            status = { ...status, timeS: input.frame.timeS };
            input.onProgress(1, input.frame.timeS);
            stop(true);
            return;
          }
          if (now >= nextDraw) {
            const drawStart = performance.now();
            status = { ...status, fraction, timeS: previous.timeS + fraction * (input.frame.timeS - previous.timeS) };
            input.onProgress(fraction, status.timeS!);
            binding!.setTemporal(pairs, fraction, displacement, wind);
            render();
            trackDraw();
            const drawEnd = performance.now();
            nextDraw = drawEnd + Math.max(16, 50 - (drawEnd - drawStart), (drawEnd - drawStart) * 0.5);
          }
          animation = requestAnimationFrame(draw);
          } catch (error) { recover(error); }
        };
        binding.setTemporal(pairs, 0, displacement, wind);
        render();
        trackDraw();
        animation = requestAnimationFrame(draw);
        return true;
      } catch (error) {
        stop(false);
        status = { ...status, toStep: input.frame.step, timeS: input.frame.timeS, toTimeS: input.frame.timeS,
          reason: `Cloud interpolation unavailable: ${error instanceof Error ? error.message : String(error)}` };
        remember(input.frame);
        return false;
      }
    },
    dispose() {
      disposed = true; stop(false); release(savedReservation); savedReservation = null; saved = null;
      for (const old of history) { renderer.removeVolume(old.actor); old.actor.delete(); old.mapper.delete(); old.image.delete(); old.initialProperty.delete(); }
    },
  };
}
