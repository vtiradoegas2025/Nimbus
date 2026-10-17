import { useEffect, useRef, useState, type RefObject } from "react";
import vtkPolyData from "@kitware/vtk.js/Common/DataModel/PolyData";
import vtkMapper from "@kitware/vtk.js/Rendering/Core/Mapper";
import vtkActor from "@kitware/vtk.js/Rendering/Core/Actor";
import type { StepInfo } from "@/lib/run/runOutputs";
import { runVtkCleanupSteps } from "@/lib/vtk/vtkLifecycle";
import {
  buildDomainAxisTicks,
  chooseHorizontalCorners,
  chooseVerticalEdge,
  domainAxisBounds,
  VERTICAL_EDGE_CORNERS,
  type AxisCorners,
  type DomainAxisTick,
} from "@/lib/plot/domainAxes";
import { gridKeyOf, type SceneScale } from "./common";
import type { SceneRgb } from "@/lib/plot/sceneBackground";

/* Domain rulers retain source units while projecting through the scene scale. */

interface AxisCamera {
  onModified: (cb: () => void) => { unsubscribe: () => void };
  getPosition: () => ArrayLike<number>;
}

interface AxisRenderer {
  getActiveCamera: () => AxisCamera;
  /** vtk.js types this as number[]; at runtime it is a Float64Array triple. */
  worldToNormalizedDisplay: (
    x: number,
    y: number,
    z: number,
    aspect: number,
  ) => ArrayLike<number>;
  addActor: (a: AxisTicksActor) => void;
  removeActor: (a: AxisTicksActor) => void;
}

export interface AxisRenderWindow {
  getRenderer: () => AxisRenderer;
  getRenderWindow: () => { render: () => void };
}

export interface ProjectedAxisLabel {
  key: string;
  /** Position within the overlay host, in percent. */
  leftPct: number;
  topPct: number;
  text: string;
  kind: "tick" | "axis";
}

/** Keep labels that stray slightly outside the viewport, drop the rest. */
const VIEWPORT_MARGIN = 0.08;

function makeTicksActor(color: SceneRgb = [0.62, 0.67, 0.75]) {
  const polyData = vtkPolyData.newInstance();
  const mapper = vtkMapper.newInstance();
  mapper.setInputData(polyData);
  const actor = vtkActor.newInstance();
  actor.setMapper(mapper);
  const prop = actor.getProperty();
  prop.setColor(...color);
  prop.setLineWidth(1);
  prop.setLighting(false); // scenery, not data
  actor.setPickable(false);
  return { actor, polyData };
}

type AxisTicksActor = ReturnType<typeof makeTicksActor>["actor"];

/** Rewrite a tick actor's geometry in place, in display coordinates. */
function writeTicks(
  polyData: ReturnType<typeof vtkPolyData.newInstance>,
  ticks: DomainAxisTick[],
): void {
  const points = new Float32Array(ticks.length * 6);
  const lines = new Uint32Array(ticks.length * 3);
  ticks.forEach((tick, i) => {
    points.set(tick.from, i * 6);
    points.set(tick.to, i * 6 + 3);
    lines[i * 3] = 2;
    lines[i * 3 + 1] = i * 2;
    lines[i * 3 + 2] = i * 2 + 1;
  });
  polyData.getPoints().setData(points, 3);
  polyData.getLines().setData(lines);
  polyData.modified();
}

/**
 * Maintain the domain axis rulers for a 3D scene.
 *
 * Returns the projected label positions for `Plot3DAxisLabels`; the tick
 * geometry is added to and removed from the scene internally.
 */
export function useDomainAxes(
  getRenderWindow: () => AxisRenderWindow | null,
  containerRef: RefObject<HTMLElement | null>,
  grid: StepInfo["grid"] | null | undefined,
  scale: SceneScale,
  lifecycleGeneration = 0,
  color: SceneRgb = [0.62, 0.67, 0.75],
): ProjectedAxisLabel[] {
  const [labels, setLabels] = useState<ProjectedAxisLabel[]>([]);
  const getRwRef = useRef(getRenderWindow);
  getRwRef.current = getRenderWindow;
  const gridKey = JSON.stringify([gridKeyOf(grid), grid?.geometryKind === "index" ? grid.validIndexRange : null]);
  const colorKey = color.join(":");
  const ticksRef = useRef<{ actor: AxisTicksActor; polyData: ReturnType<typeof vtkPolyData.newInstance>; owner: AxisRenderer } | null>(null);
  const disposeTicks = () => {
    const current = ticksRef.current;
    ticksRef.current = null;
    if (!current) return;
    const mapper = current.actor.getMapper();
    runVtkCleanupSteps([
      () => current.owner.removeActor(current.actor),
      () => current.actor.delete(),
      () => mapper?.delete?.(),
      () => current.polyData.delete(),
    ]);
  };
  useEffect(() => () => disposeTicks(), [lifecycleGeneration]);

  useEffect(() => {
    const bounds = domainAxisBounds(grid);
    if (!bounds) {
      disposeTicks();
      setLabels([]);
      return;
    }

    let frame = 0;
    let disposed = false;
    let scene = ticksRef.current;
    let cornerKey = "";
    let labelKey = "";
    let projectionAttempts = 0;

    const dispose = () => {
      if (ticksRef.current === scene) disposeTicks();
      scene = null;
    };

    const update = () => {
      frame = 0;
      if (disposed) return;
      const rw = getRwRef.current();
      const host = containerRef.current;
      if (!rw || !host) return;
      const width = host.clientWidth;
      const height = host.clientHeight;
      if (!(width > 0) || !(height > 0)) return;

      const renderer = rw.getRenderer();
      const camera = renderer.getActiveCamera();
      const aspect = width / height;
      const project = (p: readonly number[]) =>
        renderer.worldToNormalizedDisplay(p[0], p[1], p[2], aspect);

      // Camera lives in display space; the corner rules compare against bounds
      // in source coordinates, so undo the scene scale before comparing.
      const position = camera.getPosition();
      const horizontal = chooseHorizontalCorners(
        bounds,
        position[0] / (scale.xy || 1),
        position[1] / (scale.xy || 1),
      );
      const screenXs = VERTICAL_EDGE_CORNERS.map(([atUpperX, atUpperY]) => {
        const projected = project([
          (atUpperX ? bounds.upper[0] : bounds.lower[0]) * scale.xy,
          (atUpperY ? bounds.upper[1] : bounds.lower[1]) * scale.xy,
          bounds.lower[2] * scale.z,
        ]);
        return Number.isFinite(projected[0]) ? projected[0] : null;
      });
      const corners: AxisCorners = {
        ...horizontal,
        zCorner: chooseVerticalEdge(screenXs),
      };

      const axes = buildDomainAxisTicks(grid, scale, corners);
      if (!axes || axes.ticks.length === 0) {
        dispose();
        if (labelKey !== "") {
          labelKey = "";
          setLabels([]);
        }
        return;
      }

      // Geometry only changes when chosen edges change (rare: orbit crossing a face). Skip otherwise.
      const nextCornerKey =
        `${corners.xAtUpperY}|${corners.yAtUpperX}|${corners.zCorner.join(",")}`;
      if (scene && scene.owner !== renderer) dispose();
      if (!scene) {
        scene = { ...makeTicksActor(color), owner: renderer };
        ticksRef.current = scene;
        renderer.addActor(scene.actor);
        cornerKey = "";
      }
      scene.actor.getProperty().setColor(...color);
      if (nextCornerKey !== cornerKey) {
        cornerKey = nextCornerKey;
        writeTicks(scene.polyData, axes.ticks);
        rw.getRenderWindow().render();
      }

      const next: ProjectedAxisLabel[] = [];
      const anchors = [
        ...axes.ticks.map((tick) => ({
          key: `${tick.axis}:${tick.valueM}`,
          at: tick.labelAt,
          text: tick.label,
          kind: "tick" as const,
        })),
        ...axes.axisLabels.map((axis) => ({
          key: `axis:${axis.axis}`,
          at: axis.at,
          text: axis.label,
          kind: "axis" as const,
        })),
      ];
      // Axis names share screen space with tick numbers; nudge them outward in screen (not world) space.
      const centre = project([
        0.5 * (bounds.lower[0] + bounds.upper[0]) * scale.xy,
        0.5 * (bounds.lower[1] + bounds.upper[1]) * scale.xy,
        0.5 * (bounds.lower[2] + bounds.upper[2]) * scale.z,
      ]);
      const nudgeName = (nx: number, ny: number): [number, number] => {
        const dx = nx - centre[0];
        const dy = ny - centre[1];
        const length = Math.hypot(dx, dy);
        if (!Number.isFinite(length) || length < 1e-6) return [nx, ny];
        const NUDGE = 0.05;
        return [nx + (dx / length) * NUDGE, ny + (dy / length) * NUDGE];
      };

      for (const anchor of anchors) {
        const projected = project(anchor.at);
        const [nx, ny] = anchor.kind === "axis"
          ? nudgeName(projected[0], projected[1])
          : [projected[0], projected[1]];
        if (!Number.isFinite(nx) || !Number.isFinite(ny)) continue;
        // vtk.js depth projection is backend-dependent here: WebKit can report
        // a value outside 0..1 for the same visible anchor Chromium accepts.
        // Screen-space clipping below is the portable visibility contract.
        if (
          nx < -VIEWPORT_MARGIN || nx > 1 + VIEWPORT_MARGIN
          || ny < -VIEWPORT_MARGIN || ny > 1 + VIEWPORT_MARGIN
        ) continue;
        next.push({
          key: anchor.key,
          leftPct: nx * 100,
          // Normalized display is y-up; CSS is y-down.
          topPct: (1 - ny) * 100,
          text: anchor.text,
          kind: anchor.kind,
        });
      }

      // Camera existence does not mean its projection matrix has made it to
      // the API-specific view yet. In WebKit the first projection pass can be
      // entirely non-finite even after the camera has mounted. Retry until the
      // first valid ruler is projected; later camera changes schedule their
      // own ordinary updates through onModified.
      if (next.length === 0 && projectionAttempts < MAX_BIND_ATTEMPTS) {
        projectionAttempts += 1;
        frame = requestAnimationFrame(update);
      } else if (next.length > 0) {
        projectionAttempts = MAX_BIND_ATTEMPTS;
      }

      // Skip setLabels when projected positions haven't changed (avoids re-rendering ~20 spans per orbit event).
      const nextLabelKey = next
        .map((l) => `${l.key}@${l.leftPct.toFixed(2)},${l.topPct.toFixed(2)}`)
        .join("|");
      if (nextLabelKey === labelKey) return;
      labelKey = nextLabelKey;
      setLabels(next);
    };

    const schedule = () => {
      if (disposed || frame) return;
      frame = requestAnimationFrame(update);
    };

    // The render window mounts after this effect. WebKit can take several
    // frames to create its WebGL-backed camera, so a single next-frame probe
    // permanently lost every ruler on that browser. Retry for a bounded
    // startup window; a context restoration changes lifecycleGeneration and
    // starts a fresh bind window.
    let subscription: { unsubscribe: () => void } | undefined;
    let bindFrame = 0;
    let bindAttempts = 0;
    const MAX_BIND_ATTEMPTS = 300;
    const bind = () => {
      bindFrame = 0;
      if (disposed || subscription) return;
      const camera = getRwRef.current()?.getRenderer().getActiveCamera();
      if (!camera) {
        bindAttempts += 1;
        if (bindAttempts < MAX_BIND_ATTEMPTS) {
          bindFrame = requestAnimationFrame(bind);
        }
        return;
      }
      subscription = camera.onModified(schedule);
      update();
    };
    bindFrame = requestAnimationFrame(bind);

    const observer = typeof ResizeObserver !== "undefined"
      ? new ResizeObserver(schedule)
      : null;
    if (containerRef.current) observer?.observe(containerRef.current);

    return () => {
      disposed = true;
      if (bindFrame) cancelAnimationFrame(bindFrame);
      if (frame) cancelAnimationFrame(frame);
      subscription?.unsubscribe();
      observer?.disconnect();
    };
    // gridKey stands in for `grid`, which is a fresh object per loaded frame.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gridKey, scale.xy, scale.z, lifecycleGeneration, colorKey]);

  return labels;
}
