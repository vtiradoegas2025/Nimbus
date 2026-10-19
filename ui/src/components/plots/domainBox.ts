import { useEffect, useRef } from "react";
import vtkPolyData from "@kitware/vtk.js/Common/DataModel/PolyData";
import vtkMapper from "@kitware/vtk.js/Rendering/Core/Mapper";
import vtkActor from "@kitware/vtk.js/Rendering/Core/Actor";
import type { StepInfo } from "@/lib/run/runOutputs";
import { runVtkCleanupSteps } from "@/lib/vtk/vtkLifecycle";
import { domainBoundsM } from "@/lib/plot/domainAxes";
import { gridKeyOf, type SceneScale } from "./common";
import type { SceneRgb } from "@/lib/plot/sceneBackground";

/**
 * Wireframe box at the simulation's physical bounds. Keeps camera framing stable as storms advect.
 * (vtk.js CubeAxesActor renders nothing; axis labels live in useDomainAxes.)
 */
export function makeDomainBoxActor(
  grid: StepInfo["grid"] | null | undefined,
  color: SceneRgb = [0.45, 0.5, 0.58],
) {
  const bounds = domainBoundsM(grid);
  if (!bounds) return null;
  if (bounds.lower.some((lo, axis) => !Number.isFinite(lo)
      || !Number.isFinite(bounds.upper[axis]) || bounds.upper[axis]! <= lo)) return null;
  const data = vtkPolyData.newInstance();
  const points = new Float32Array(24);
  for (let corner = 0; corner < 8; corner++) {
    for (let axis = 0; axis < 3; axis++) points[3 * corner + axis] = (corner & (1 << axis)) ? bounds.upper[axis]! : bounds.lower[axis]!;
  }
  data.getPoints().setData(points, 3);
  const edges: number[] = [];
  for (let corner = 0; corner < 8; corner++) {
    for (let axis = 0; axis < 3; axis++) if (!(corner & (1 << axis))) edges.push(2, corner, corner | (1 << axis));
  }
  data.getLines().setData(new Uint32Array(edges));
  const mapper = vtkMapper.newInstance();
  mapper.setInputData(data);
  const actor = vtkActor.newInstance();
  actor.setMapper(mapper);
  const prop = actor.getProperty();
  prop.setColor(...color);
  prop.setOpacity(0.4);
  prop.setLighting(false); // scenery, not data: a constant-tint reference frame
  actor.setPickable(false);
  return actor;
}

type DomainBoxActor = NonNullable<ReturnType<typeof makeDomainBoxActor>>;

interface DomainBoxRenderWindow {
  getRenderer: () => {
    addActor: (a: DomainBoxActor) => void;
    removeActor: (a: DomainBoxActor) => void;
  };
  getRenderWindow: () => { render: () => void };
}

/** Add/maintain a domain bounding-box actor. getRenderWindow read fresh each fire (not an effect dep). */
export function useDomainBox(
  getRenderWindow: () => DomainBoxRenderWindow | null,
  grid: StepInfo["grid"] | null | undefined,
  scale: SceneScale,
  lifecycleGeneration = 0,
  color: SceneRgb = [0.45, 0.5, 0.58],
): void {
  const boxRef = useRef<{ actor: DomainBoxActor; owner: DomainBoxRenderWindow } | null>(null);
  const getRwRef = useRef(getRenderWindow);
  getRwRef.current = getRenderWindow;
  const gridKey = gridKeyOf(grid);
  const colorKey = color.join(":");

  const disposeBox = () => {
    const current = boxRef.current;
    boxRef.current = null;
    if (!current) return;
    const mapper = current.actor.getMapper();
    const data = mapper?.getInputData();
    runVtkCleanupSteps([
      () => current.owner.getRenderer().removeActor(current.actor),
      () => current.actor.delete(),
      () => mapper?.delete?.(),
      () => data?.delete?.(),
    ]);
  };

  useEffect(() => () => disposeBox(), [lifecycleGeneration]);

  useEffect(() => {
    const update = () => {
      const rw = getRwRef.current();
      if (!rw) return;
      const bounds = domainBoundsM(grid);
      if (!bounds) { disposeBox(); return; }
      if (boxRef.current?.owner !== rw) disposeBox();
      if (!boxRef.current) {
        const actor = makeDomainBoxActor(grid, color);
        if (!actor) return;
        rw.getRenderer().addActor(actor);
        boxRef.current = { actor, owner: rw };
      } else {
        const data = boxRef.current.actor.getMapper()!.getInputData() as ReturnType<typeof vtkPolyData.newInstance>;
        const points = data.getPoints().getData();
        for (let corner = 0; corner < 8; corner++) {
          for (let axis = 0; axis < 3; axis++) points[3 * corner + axis] = (corner & (1 << axis)) ? bounds.upper[axis]! : bounds.lower[axis]!;
        }
        data.getPoints().modified();
        data.modified();
      }
      const box = boxRef.current.actor;
      box.getProperty().setColor(...color);
      box.setScale(scale.xy, scale.xy, scale.z);
      rw.getRenderWindow().render();
    };
    // Preserve the registered actor across frame/bounds changes; only initial
    // mount waits for the render-window effect that follows this hook.
    if (boxRef.current) { update(); return; }
    const frame = requestAnimationFrame(update);
    return () => cancelAnimationFrame(frame);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gridKey, lifecycleGeneration, colorKey, scale.xy, scale.z]);
}
