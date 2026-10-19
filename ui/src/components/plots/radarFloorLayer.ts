import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { fieldDefinitionIdentity, type StepInfo } from "@/lib/run/runOutputs";
import { domainBoundsM } from "@/lib/plot/domainAxes";
import { floorPlaneGeometry } from "@/lib/plot/radarFloor";
import { RadarFloorCommitCache } from "@/lib/plot/radarFloorCommitCache";
import {
  resolveScalarPlanePolicy,
  type ScalarPlanePolicy,
} from "@/lib/plot/scalarPlanePolicy";
import { createRadarFloorTexture, type RadarFloorTexture } from "@/lib/vtk/radarFloorTexture";
import { useRadarFloor, type RadarFloorInput, type RadarFloorState } from "@/lib/vtk/useRadarFloor";
import { useLatchedRange, type SceneScale } from "./common";
import { useVtkGpuResourceOwnership } from "@/lib/vtk/useVtkGpuResourceOwnership";
import {
  RadarFloorEncodingService,
  type RetainedRadarFloorEncoding,
} from "@/lib/vtk/radarFloorEncodingClient";
import { radarFloorPresentationOwnership } from "@/lib/vtk/radarFloorOwnership";

/* Textured ground-plane reflectivity sheet, mirroring useDomainBox lifecycle. Resolution independent of volume quality. */

/** Animation frames the layer waits for a render window before reporting failure (~2 s at 60 Hz). */
const ATTACH_FRAME_BUDGET = 120;
let floorBufferSequence = 0;
const floorBufferIds = new WeakMap<object, number>();

function floorBufferId(buffer: ArrayBufferLike): number {
  let id = floorBufferIds.get(buffer);
  if (id == null) {
    id = ++floorBufferSequence;
    floorBufferIds.set(buffer, id);
  }
  return id;
}

export interface RadarFloorLayerInput extends Omit<RadarFloorInput, "grid"> {
  grid: StepInfo["grid"] | null | undefined;
  scale: SceneScale;
  /** Display-only actor opacity. The encoded NWS colour values are unchanged. */
  opacity?: number;
  lifecycleGeneration?: number;
  /** Stage texture/visibility changes for a parent scene-level commit. */
  deferDataRender?: boolean;
}

export interface RadarFloorLayerState extends RadarFloorState {
  /** Colour mapping actually drawn, with its range latched across frames. */
  policy: ScalarPlanePolicy | null;
  /** Non-zero-alpha cells in the encoded sheet. Null until first encode; zero means no echo at surface. */
  paintedCells: number | null;
  /** True only when the current request has committed visible, nonempty pixels. */
  visible: boolean;
  commitData: () => void;
  presentationIdentity: string;
  committedPresentationIdentity: string | null;
}

// Concrete type required by strictFunctionTypes — vtkProp is not assignable to unknown in addActor.
type RadarFloorActor = RadarFloorTexture["actor"];

export interface RadarFloorLayerRenderWindow {
  getRenderer: () => {
    addActor: (actor: RadarFloorActor) => void;
    removeActor: (actor: RadarFloorActor) => void;
  };
  getRenderWindow: () => { render: () => void };
}

export function useRadarFloorLayer(
  getRenderWindow: () => RadarFloorLayerRenderWindow | null,
  input: RadarFloorLayerInput,
): RadarFloorLayerState {
  const {
    grid,
    scale,
    opacity = 1,
    lifecycleGeneration = 0,
    deferDataRender = false,
    ...request
  } = input;

  // The grid the engine published, in the shape the hyperslab planner wants.
  const dims = grid?.nr && grid?.nz
    ? {
      nr: grid.nr,
      nth: grid.nth ?? grid.nr,
      nz: grid.nz,
      terrainFollowing: grid.terrainFollowing,
      drM: grid.drM,
      dyM: grid.dyM,
    }
    : null;
  const floor = useRadarFloor({ ...request, grid: dims });

  // Latch colour domain so a per-step percentile policy doesn't pulse through a movie.
  const framePolicy = useMemo(
    () => {
      if (!request.enabled || !request.fieldName || floor.loading || floor.error != null
          || floor.values == null || floor.loadedRequestIdentity !== floor.requestedRequestIdentity || !floor.stats) return null;
      const resolved = resolveScalarPlanePolicy(request.fieldName, floor.stats);
      // Override inferred reduction with the user's actual choice.
      return { ...resolved, reduction: request.reduction };
    },
    [request.enabled, request.fieldName, request.reduction, floor.loading, floor.error, floor.values,
      floor.loadedRequestIdentity, floor.requestedRequestIdentity, floor.stats],
  );
  const latchedRange = useLatchedRange(
    JSON.stringify([request.runDir, request.fieldName, request.reduction,
      fieldDefinitionIdentity(request.runDir, request.stepIndex, request.fieldName, request.publicationGeneration),
      fieldDefinitionIdentity(request.runDir, request.stepIndex, floor.source?.fieldName ?? null, request.publicationGeneration)]),
    framePolicy ? { min: framePolicy.min, max: framePolicy.max } : null,
  );
  const policy = useMemo(
    () => (framePolicy && latchedRange
      ? { ...framePolicy, min: latchedRange.min, max: latchedRange.max }
      : framePolicy),
    [framePolicy, latchedRange],
  );

  const encodingGeneration = floor.values && floor.shape && policy
    ? JSON.stringify([
      floorBufferId(floor.values.buffer),
      floor.values.byteOffset,
      floor.values.byteLength,
      floor.shape,
      policy,
    ])
    : "none";
  const [encoded, setEncoded] = useState<{
    generation: string;
    result: RetainedRadarFloorEncoding;
  } | null>(null);
  const [encodingLoading, setEncodingLoading] = useState(false);
  const [encodingError, setEncodingError] = useState<{ generation: string; error: string } | null>(null);
  const encodingServiceRef = useRef<RadarFloorEncodingService>();
  if (!encodingServiceRef.current) encodingServiceRef.current = new RadarFloorEncodingService();
  const currentEncodingRef = useRef<RetainedRadarFloorEncoding | null>(null);
  const pendingEncodingRef = useRef<RetainedRadarFloorEncoding | null>(null);
  const encodedForGeneration = encoded?.generation === encodingGeneration ? encoded.result : null;
  const rgba = encodedForGeneration?.rgba ?? null;
  const paintedCells = encodedForGeneration?.paintedCells ?? null;
  const expectedRgbaBytes = floor.shape ? floor.shape[0] * floor.shape[1] * 4 : 0;
  const terrainGeometryBytes = floor.terrainMesh
    ? floor.terrainMesh.points.byteLength + floor.terrainMesh.polys.byteLength
      + floor.terrainMesh.tcoords.byteLength
    : 0;
  const gpuUploadBytes = expectedRgbaBytes + terrainGeometryBytes;
  const encodingCandidate = request.enabled && !floor.loading && floor.error == null
    && floor.values != null && floor.shape != null && policy != null && expectedRgbaBytes > 0;
  const gpuUpload = useVtkGpuResourceOwnership({
    enabled: encodingCandidate && (paintedCells == null || paintedCells > 0),
    owner: "radar-floor",
    generation: JSON.stringify([
      request.runDir, request.stepIndex, request.publicationGeneration,
      request.fieldName, request.reduction, encodingGeneration, lifecycleGeneration,
    ]),
    bytes: gpuUploadBytes,
  });

  useEffect(() => {
    if (!encodingCandidate || !floor.values || !floor.shape || !policy) {
      pendingEncodingRef.current?.release();
      pendingEncodingRef.current = null;
      setEncoded(null);
      setEncodingLoading(false);
      return;
    }
    if (gpuUpload.error || !gpuUpload.admitted) return;
    const controller = new AbortController();
    setEncodingLoading(true);
    setEncodingError(null);
    encodingServiceRef.current!.encode(
      floor.values,
      floor.shape,
      policy,
      controller.signal,
    ).then((result) => {
      if (controller.signal.aborted) {
        result.release();
        return;
      }
      if (pendingEncodingRef.current
          && pendingEncodingRef.current !== currentEncodingRef.current) {
        pendingEncodingRef.current.release();
      }
      pendingEncodingRef.current = result;
      setEncoded({ generation: encodingGeneration, result });
      setEncodingLoading(false);
    }).catch((reason: unknown) => {
      if (controller.signal.aborted) return;
      setEncodingLoading(false);
      setEncodingError({ generation: encodingGeneration, error: reason instanceof Error ? reason.message : String(reason) });
    });
    return () => controller.abort();
  }, [
    encodingCandidate,
    encodingGeneration,
    floor.shape,
    floor.values,
    gpuUpload.admitted,
    gpuUpload.error,
    policy,
  ]);

  const textureRef = useRef<RadarFloorTexture | null>(null);
  const attachedLifecycleRef = useRef<number | null>(null);
  const scaleXy = scale.xy;
  const scaleZ = scale.z;
  const getRwRef = useRef(getRenderWindow);
  getRwRef.current = getRenderWindow;
  const placement = useMemo(() => floor.terrainMesh ? null : floorPlaneGeometry(
    domainBoundsM(grid), { xy: scaleXy, z: scaleZ },
  ), [floor.terrainMesh, grid, scaleXy, scaleZ]);

  const disposeLayer = (rw: RadarFloorLayerRenderWindow | null) => {
    const texture = textureRef.current;
    textureRef.current = null;
    attachedLifecycleRef.current = null;
    if (!texture) return;
    try {
      rw?.getRenderer().removeActor(texture.actor);
    } catch {
      // A torn-down renderer is not a reason to skip releasing the texture.
    }
    texture.dispose();
  };

  // attachGeneration: bumped on successful attach so the data effect can observe that textureRef is populated.
  const [attachGeneration, setAttachGeneration] = useState(0);
  const [attachment, setAttachment] = useState<{ generation: number | null; error: string | null }>({ generation: null, error: null });

  useEffect(() => {
    let cancelled = false;
    let frame = 0;
    let attemptsLeft = ATTACH_FRAME_BUDGET;

    const attach = () => {
      if (cancelled) return;
      const rw = getRwRef.current();
      if (!rw) {
        if (attemptsLeft-- > 0) {
          frame = requestAnimationFrame(attach);
        } else {
          // Bounded so a scene that never yields a render window cannot spin a
          // rAF chain forever. Giving up is reported, never silent.
          setAttachment({ generation: lifecycleGeneration, error:
            "Floor unavailable: the 3D scene did not provide a render window, so the"
            + " sheet was never attached. Toggle the layer to retry." });
        }
        return;
      }
      disposeLayer(rw);
      try {
        const texture = createRadarFloorTexture();
        try { rw.getRenderer().addActor(texture.actor); }
        catch (reason) { texture.dispose(); throw reason; }
        textureRef.current = texture;
        attachedLifecycleRef.current = lifecycleGeneration;
        rw.getRenderWindow().render();
        setAttachment({ generation: lifecycleGeneration, error: null });
        setAttachGeneration((generation) => generation + 1);
      } catch (reason) {
        disposeLayer(rw);
        setAttachment({ generation: lifecycleGeneration,
          error: `Floor attachment failed: ${reason instanceof Error ? reason.message : String(reason)}` });
      }
    };

    frame = requestAnimationFrame(attach);
    return () => {
      cancelled = true;
      cancelAnimationFrame(frame);
      disposeLayer(getRwRef.current());
    };
  }, [lifecycleGeneration]);

  const geometryIdentity = JSON.stringify(floor.terrainMesh
    ? ["terrain", ...[floor.terrainMesh.points, floor.terrainMesh.polys, floor.terrainMesh.tcoords]
      .map(view => [floorBufferId(view.buffer), view.byteOffset, view.byteLength]), scaleXy, scaleZ, lifecycleGeneration]
    : ["plane", placement, scaleXy, scaleZ, lifecycleGeneration]);
  const presentationIdentity = JSON.stringify([request.enabled, floor.requestedRequestIdentity,
    encodingGeneration, geometryIdentity]);
  const [committedPresentationIdentity, setCommittedPresentationIdentity] = useState<string | null>(null);
  const appliedRef = useRef(new RadarFloorCommitCache());
  const error = request.enabled ? floor.error
    ?? (encodingError?.generation === encodingGeneration ? encodingError.error : null)
    ?? (attachment.generation === lifecycleGeneration ? attachment.error : null)
    ?? gpuUpload.error : null;
  const ownership = radarFloorPresentationOwnership({
    enabled: request.enabled,
    requestedRequestIdentity: floor.requestedRequestIdentity,
    loadedRequestIdentity: floor.loadedRequestIdentity,
    requestLoading: floor.loading,
    encodingLoading,
    encodingRequired: floor.shape != null,
    encodingReady: rgba != null,
    gpuRequired: floor.shape != null && (paintedCells == null || paintedCells > 0),
    gpuAdmitted: gpuUpload.admitted,
    attachedLifecycleGeneration: attachedLifecycleRef.current,
    lifecycleGeneration,
    error,
  });

  const commitData = useCallback(() => {
    const texture = textureRef.current;
    const rw = getRwRef.current();
    if (!texture || !rw) return;
    if (ownership.action === "retain") return;
    if (attachedLifecycleRef.current !== lifecycleGeneration) return;

    if (ownership.action === "hide" || paintedCells === 0 || !rgba || !floor.shape || (!placement && !floor.terrainMesh)) {
      texture.actor.setVisibility(false);
      currentEncodingRef.current?.release();
      if (pendingEncodingRef.current !== currentEncodingRef.current) {
        pendingEncodingRef.current?.release();
      }
      currentEncodingRef.current = null;
      pendingEncodingRef.current = null;
      appliedRef.current.reset();
      setCommittedPresentationIdentity(presentationIdentity);
      if (!deferDataRender) rw.getRenderWindow().render();
      return;
    }
    if (gpuUpload.error || !gpuUpload.admitted) {
      texture.actor.setVisibility(false);
      currentEncodingRef.current?.release();
      if (pendingEncodingRef.current !== currentEncodingRef.current) {
        pendingEncodingRef.current?.release();
      }
      currentEncodingRef.current = null;
      pendingEncodingRef.current = null;
      appliedRef.current.reset();
      if (!deferDataRender) rw.getRenderWindow().render();
      return;
    }

    const shape = floor.shape.join(":");
    appliedRef.current.apply({ texture, rgba, shape, geometry: geometryIdentity }, () => {
      if (floor.terrainMesh) {
      texture.setTerrainMesh(floor.terrainMesh, { xy: scaleXy, z: scaleZ });
      } else if (placement) {
      texture.setPlaneGeometry(placement.origin, placement.point1, placement.point2);
      }
    }, () => texture.setFloorData(rgba, floor.shape!));
    const committedEncoding = encodedForGeneration!;
    const previousEncoding = currentEncodingRef.current;
    currentEncodingRef.current = committedEncoding;
    if (pendingEncodingRef.current === committedEncoding) pendingEncodingRef.current = null;
    if (previousEncoding && previousEncoding !== committedEncoding) previousEncoding.release();
    texture.actor.setVisibility(true);
    setCommittedPresentationIdentity(presentationIdentity);
    if (!deferDataRender) rw.getRenderWindow().render();
  }, [
    rgba,
    encodedForGeneration,
    floor.shape,
    floor.terrainMesh,
    placement,
    scaleXy,
    scaleZ,
    lifecycleGeneration,
    deferDataRender,
    gpuUpload.admitted,
    gpuUpload.error,
    ownership.action,
    geometryIdentity, presentationIdentity,
    paintedCells,
  ]);

  useEffect(() => {
    if (!deferDataRender) commitData();
  }, [commitData, deferDataRender]);

  useEffect(() => {
    const texture = textureRef.current;
    const rw = getRwRef.current();
    if (!texture || !rw) return;
    texture.actor.getProperty().setOpacity(Math.max(0.1, Math.min(1, opacity)));
    if (!deferDataRender) rw.getRenderWindow().render();
  }, [opacity, attachGeneration, deferDataRender]);

  useEffect(() => () => {
    encodingServiceRef.current?.shutdown();
    pendingEncodingRef.current?.release();
    if (currentEncodingRef.current !== pendingEncodingRef.current) {
      currentEncodingRef.current?.release();
    }
    pendingEncodingRef.current = null;
    currentEncodingRef.current = null;
  }, []);

  const loading = ownership.loading;
  const visible = committedPresentationIdentity === presentationIdentity
    && ownership.action === "commit" && !loading && error == null
    && floor.shape != null && rgba != null && policy != null
    && paintedCells != null && paintedCells > 0 && gpuUpload.admitted;
  return { ...floor, loading, error, policy, paintedCells, visible, commitData,
    presentationIdentity, committedPresentationIdentity };
}
