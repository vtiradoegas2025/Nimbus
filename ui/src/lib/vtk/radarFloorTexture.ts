/* Textured radar floor. Uniform grids use two triangles; terrain-following
 * grids use a separately capped (at most 512²) draped mesh while the RGBA
 * texture retains the floor data's independent display resolution. */

import vtkActor from "@kitware/vtk.js/Rendering/Core/Actor";
import vtkMapper from "@kitware/vtk.js/Rendering/Core/Mapper";
import vtkDataArray from "@kitware/vtk.js/Common/Core/DataArray";
import vtkPolyData from "@kitware/vtk.js/Common/DataModel/PolyData";
import vtkTexture from "@kitware/vtk.js/Rendering/Core/Texture";
import type { VtkDisposable } from "./vtkLifecycle";
import { releaseVtkPointDataArrays } from "./vtkArrayOwnership";
import type { TerrainFloorMesh } from "@/lib/plot/radarFloor";

// ---------- Types ----------

// vtkTexture's .d.ts omits delete()/modified() (declared by macro.obj, not vtkAlgorithm).
interface VtkObjectLike extends VtkDisposable {
  modified?: () => void;
}

function asVtkObject(value: object): VtkObjectLike {
  return value as VtkObjectLike;
}

type VtkActor = ReturnType<typeof vtkActor.newInstance>;
type VtkMapper = ReturnType<typeof vtkMapper.newInstance>;
type VtkTexture = ReturnType<typeof vtkTexture.newInstance>;

/** [x, y, z] coordinate triple. */
type Point3 = [number, number, number];

/**
 * The assembled floor: a plane-source actor with a bilinear-filtered texture
 * carrying the RGBA floor data. Flat mode is two triangles; terrain mode
 * replaces the plane points with the physical bottom-interface lattice.
 *
 * Call `setFloorData` whenever new RGBA data arrives from `encodeFloorRgba`.
 * Call `setPlaneGeometry` to reposition or resize the plane in world coordinates.
 * The `actor` can be added to any vtk.js renderer; visibility management is the
 * caller's responsibility (the plane starts invisible — caller must opt in).
 * Call `dispose` on teardown to avoid leaking vtk.js objects.
 */
export interface RadarFloorTexture {
  /**
   * Push new RGBA8 pixel data into the texture. `rgba` must be a Uint8ClampedArray
   * of length `shape[0] * shape[1] * 4`. Triggers a texture update (modified()).
   *
   * `shape` is [nx, ny] in the same (x-slowest) layout convention as
   * `encodeFloorRgba`: the caller controls layout coherence.
   */
  setFloorData(rgba: Uint8ClampedArray<ArrayBuffer>, shape: [number, number]): void;

  /**
   * Update the plane geometry in world coordinates.
   * `origin`, `point1`, and `point2` follow the vtkPlaneSource convention:
   *   origin → corner at (u=0, v=0)
   *   point1 → corner at (u=1, v=0)  (x-axis edge)
   *   point2 → corner at (u=0, v=1)  (y-axis edge)
   */
  setPlaneGeometry(origin: Point3, point1: Point3, point2: Point3): void;

  /** Drape the texture over one physical terrain point per x/y sample. Arrays
   * use the field convention (x slowest, y fastest). */
  setTerrainMesh(
    mesh: TerrainFloorMesh,
    scale: { xy: number; z: number },
  ): void;

  /** The actor to add to a renderer. Starts invisible. */
  readonly actor: VtkActor;

  /**
   * Release all vtk.js objects. Safe to call multiple times (idempotent after
   * the first call). Matches the disposal discipline in vtkLifecycle.ts: actors
   * → mappers → sources; texture resources released before image data.
   */
  dispose(): void;
}

// ---------- Factory ----------

/** Construct a RadarFloorTexture. Actor starts invisible; caller manages visibility.
 *  Texture is resizable so it can be re-uploaded with different dimensions. */
export function createRadarFloorTexture(): RadarFloorTexture {
  // ----- Pipeline objects -----

  const data = vtkPolyData.newInstance();
  const mapper: VtkMapper = vtkMapper.newInstance();
  mapper.setInputData(data);

  // resizable: true — the GPU texture can be reallocated when `setFloorData`
  // is called with a different shape. Without this flag vtk.js caches the
  // original dimensions and silently clips or wraps on resize.
  const texture: VtkTexture = vtkTexture.newInstance({ resizable: true });
  texture.setInterpolate(true); // bilinear — the primary anti-pixelation control
  texture.setEdgeClamp(true); // no wraparound at domain boundary

  const actor: VtkActor = vtkActor.newInstance();
  actor.setMapper(mapper);
  actor.addTexture(texture);
  // The texture is the scientific color mapping. Scene lights may shape cloud
  // and geometry, but must never tint or dim the radar palette.
  actor.getProperty().setLighting(false);
  // vtkActor.getIsOpaque consults `model.texture` (singular), which addTexture
  // never sets, so a texture's alpha cannot classify this actor. Left opaque it
  // renders with depthMask on and its clear-air cells occlude the volume behind.
  actor.setForceTranslucent(true);
  actor.setVisibility(false);

  // ----- Disposal guard -----

  let disposed = false;

  // ----- Helpers -----

  function uploadImageData(
    rgba: Uint8ClampedArray<ArrayBuffer>,
    nx: number,
    ny: number,
  ): void {
    // width=nx, height=ny: encodeFloorRgba is y-slowest/x-fastest, so u↔x, v↔y.
    // Swapping produces a transposed floor.
    const imageData = new ImageData(rgba, nx, ny);
    texture.setJsImageData(imageData);
    asVtkObject(texture).modified?.();
  }

  // ----- Public API -----

  function setFloorData(
    rgba: Uint8ClampedArray<ArrayBuffer>,
    shape: [number, number],
  ): void {
    if (disposed) throw new Error("RadarFloorTexture has been disposed");
    const [nx, ny] = shape;
    if (!(nx > 0) || !(ny > 0)) {
      throw new Error(`RadarFloorTexture.setFloorData: invalid shape [${nx}, ${ny}]`);
    }
    const expectedLength = nx * ny * 4;
    if (rgba.length !== expectedLength) {
      throw new Error(
        `RadarFloorTexture.setFloorData: rgba length ${rgba.length} `
        + `does not match shape [${nx}, ${ny}] × 4 = ${expectedLength}`,
      );
    }
    uploadImageData(rgba, nx, ny);
  }

  function setPlaneGeometry(origin: Point3, point1: Point3, point2: Point3): void {
    if (disposed) throw new Error("RadarFloorTexture has been disposed");
    const point3: Point3 = [
      point1[0] + point2[0] - origin[0],
      point1[1] + point2[1] - origin[1],
      point1[2] + point2[2] - origin[2],
    ];
    setGeometry(
      Float32Array.from([...origin, ...point1, ...point2, ...point3]),
      Uint32Array.from([3, 0, 1, 2, 3, 1, 3, 2]),
      Float32Array.from([0, 0, 1, 0, 0, 1, 1, 1]),
    );
    actor.setScale(1, 1, 1);
  }

  function setGeometry(
    points: Float32Array,
    polys: Uint32Array,
    tcoords: Float32Array,
  ): void {
    releaseVtkPointDataArrays(data.getPointData());
    data.getPoints().setData(points, 3);
    data.getPolys().setData(polys);
    data.getPointData().setTCoords(vtkDataArray.newInstance({
      name: "TextureCoordinates",
      values: tcoords,
      numberOfComponents: 2,
    }));
    data.modified();
  }

  function setTerrainMesh(
    mesh: TerrainFloorMesh,
    scale: { xy: number; z: number },
  ): void {
    if (disposed) throw new Error("RadarFloorTexture has been disposed");
    if (mesh.points.length < 12 || mesh.points.length % 3 !== 0
        || mesh.polys.length < 8 || mesh.polys.length % 4 !== 0
        || mesh.tcoords.length !== mesh.points.length / 3 * 2
        || !(Number.isFinite(scale.xy) && scale.xy > 0)
        || !(Number.isFinite(scale.z) && scale.z > 0)) {
      throw new Error("RadarFloorTexture.setTerrainMesh: invalid terrain mesh");
    }
    setGeometry(mesh.points, mesh.polys, mesh.tcoords);
    actor.setScale(scale.xy, scale.xy, scale.z);
  }

  function dispose(): void {
    if (disposed) return;
    disposed = true;

    // Detach textures first to prevent stale pointer deref in actor's destructor.
    const textures = actor.getTextures?.() ?? [];
    for (const t of textures) {
      actor.removeTexture?.(t);
    }
    // Delete pipeline order: actor → mapper → data → texture (texture last).
    for (const resource of [actor, mapper, data, texture]) {
      asVtkObject(resource).delete?.();
    }
  }

  return { setFloorData, setPlaneGeometry, setTerrainMesh, actor, dispose };
}
