/*
 * Behavioural contract for the tube plots (Streamlines3D, Trajectories3D),
 * exercised against the REAL vtk.js classes rather than the component sources.
 *
 * These plots have twice been broken by vtk.js dropping state that its API
 * implies it keeps, and both failures were invisible to source-text assertions:
 * the code read exactly as intended and rendered nothing (or rendered grey).
 * Everything below therefore runs the actual filters and inspects their output.
 *
 * vtk.js is imported directly here. That is deliberate and confined to this
 * file: the lib/ modules under test stay vtk-free so the rest of the
 * deterministic runner keeps importing them without a rendering stack.
 */

import vtkPolyData from "@kitware/vtk.js/Common/DataModel/PolyData.js";
import vtkDataArray from "@kitware/vtk.js/Common/Core/DataArray.js";
import vtkTubeFilter from "@kitware/vtk.js/Filters/General/TubeFilter.js";
import vtkTransformPolyDataFilter from "@kitware/vtk.js/Filters/General/TransformPolyDataFilter.js";
import vtkTransform from "@kitware/vtk.js/Common/Transform/Transform.js";
import vtkMatrixBuilder from "@kitware/vtk.js/Common/Core/MatrixBuilder.js";
import vtkMapper from "@kitware/vtk.js/Rendering/Core/Mapper.js";
import vtkColorTransferFunction from "@kitware/vtk.js/Rendering/Core/ColorTransferFunction.js";
import vtkColorMaps from "@kitware/vtk.js/Rendering/Core/ColorTransferFunction/ColorMaps.js";
import { ScalarMode } from "@kitware/vtk.js/Rendering/Core/Mapper/Constants.js";

import { VTK_SCALAR_SLOT } from "../../../../ui/src/lib/vtk/vtkArrayOwnership.ts";
import {
  applyScaledGeometry,
  buildScaledGeometry,
} from "../../../../ui/src/lib/vtk/vtkScaledPolyData.ts";

let passed = 0;
let failed = 0;
const failures: string[] = [];
function check(condition: boolean, label: string): void {
  if (condition) passed++;
  else {
    failed++;
    failures.push(`FAIL ${label}`);
  }
}

/** One three-point polyline carrying a per-point scalar, in true coordinates. */
function makePaths(): ReturnType<typeof vtkPolyData.newInstance> {
  const polyData = vtkPolyData.newInstance();
  polyData.getPoints().setData(Float32Array.from([0, 0, 100, 1000, 0, 900, 2000, 0, 1783]), 3);
  polyData.getLines().setData(Uint32Array.from([3, 0, 1, 2]));
  polyData.getPointData().setScalars(vtkDataArray.newInstance({
    name: VTK_SCALAR_SLOT,
    values: Float32Array.from([100, 900, 1783]),
    numberOfComponents: 1,
  }));
  return polyData;
}

/* ---------- The owned scaled copy survives repeated re-execution ----------
 * UI-19: a vtkTransformPolyDataFilter between the paths and the tube filter
 * emptied the polylines on its SECOND execution, so streamlines drew nothing at
 * all and trajectories went blank on the first vertical-scale change. Rebuild at
 * several scales in a row -- as dragging the vertical-scale slider does -- and
 * require real tube geometry every time. */
{
  const paths = makePaths();
  const scaledPaths = vtkPolyData.newInstance();
  const tube = vtkTubeFilter.newInstance({ radius: 50, numberOfSides: 6, capping: true });
  tube.setInputData(scaledPaths);

  const tubePointCounts: number[] = [];
  for (const z of [1, 1.5, 3.5, 6, 1]) {
    const geometry = buildScaledGeometry(paths, { xy: 1, z });
    check(geometry !== null, `scale ${z}x still yields tubeable geometry`);
    applyScaledGeometry(scaledPaths, geometry!);
    tube.modified();
    const out = tube.getOutputData();
    tubePointCounts.push(out.getPoints().getNumberOfPoints());
    check(
      paths.getLines().getNumberOfValues() === 4,
      `scale ${z}x leaves the source polylines intact`,
    );
  }
  check(
    tubePointCounts.every((n) => n > 0 && n === tubePointCounts[0]),
    "every re-tube produces the same non-empty geometry",
  );

  // The scale must reach the drawn points, and only the z axis.
  const flat = buildScaledGeometry(paths, { xy: 1, z: 1 })!;
  const tall = buildScaledGeometry(paths, { xy: 1, z: 3.5 })!;
  check(tall.points[2] === flat.points[2] * 3.5, "vertical exaggeration reaches the drawn points");
  check(tall.points[0] === flat.points[0], "vertical exaggeration leaves x alone");
  check(tall.points[3] === flat.points[3], "vertical exaggeration leaves the next point's x alone");
}

/* ---------- vtkTubeFilter does not mark its output scalars active ----------
 * It interpolates every input point array onto the tube points but leaves the
 * output's activeScalars at -1, so a mapper on the default scalar mode finds no
 * scalars and falls back to the actor's flat colour: the trajectory paths
 * rendered uniform grey while the legend advertised a Height ramp. */
{
  const paths = makePaths();
  const tube = vtkTubeFilter.newInstance({ radius: 50, numberOfSides: 6, capping: true });
  tube.setInputData(paths);
  const out = tube.getOutputData();

  check(out.getPoints().getNumberOfPoints() > 0, "the tube filter produces geometry");
  check(
    out.getPointData().getArrayByName(VTK_SCALAR_SLOT) !== null,
    "the tube filter interpolates the colour array onto the tube points",
  );
  check(
    out.getPointData().getScalars() === null,
    "the tube filter leaves no ACTIVE scalars -- colour must be selected by name",
  );
}

/* ---------- The mapper colours the tubes by array name ----------
 * The end-to-end property the user sees: a Height ramp actually produces a
 * ramp of colours, not one flat colour. */
{
  const paths = makePaths();
  const tube = vtkTubeFilter.newInstance({ radius: 50, numberOfSides: 6, capping: true });
  tube.setInputData(paths);

  const ctf = vtkColorTransferFunction.newInstance();
  const preset = vtkColorMaps.getPresetByName("Viridis (matplotlib)");
  check(Boolean(preset), "the Viridis preset resolves in this vtk.js build");
  ctf.applyColorMap(preset);
  ctf.setMappingRange(100, 1783);

  const mapper = vtkMapper.newInstance();
  mapper.setInputConnection(tube.getOutputPort());
  mapper.setLookupTable(ctf);
  mapper.setScalarVisibility(true);
  mapper.setUseLookupTableScalarRange(true);
  mapper.setScalarMode(ScalarMode.USE_POINT_FIELD_DATA);
  mapper.setColorByArrayName(VTK_SCALAR_SLOT);

  mapper.mapScalars(tube.getOutputData(), 1.0);
  // vtk.js types getColorMapColors() as a Uint8Array, but it returns the RGBA
  // vtkDataArray -- read the buffer through the shape it actually has.
  const colors = mapper.getColorMapColors() as unknown as
    { getData: () => Uint8Array } | null;
  check(Boolean(colors), "the mapper resolves per-point colours for the tubes");
  if (!colors) {
    // Exactly the production symptom: no mapped colours means the actor's flat
    // colour is drawn instead of the ramp. Report it rather than throwing.
    check(false, "the tubes carry a colour RAMP, not one flat colour");
  } else {
    const rgba = colors.getData();
    const first = [rgba[0], rgba[1], rgba[2]].join(",");
    const last = [rgba.at(-4), rgba.at(-3), rgba.at(-2)].join(",");
    check(first !== last, "the tubes carry a colour RAMP, not one flat colour");
    // Viridis runs dark blue/purple -> yellow across the mapped range.
    check(rgba[2] > rgba[0], "the low end of the ramp is the blue end of Viridis");
    check(Number(rgba.at(-4)) > Number(rgba.at(-2)), "the high end of the ramp is the yellow end");
  }
}

/* ---------- Canary: the vtk.js aliasing bug that forced the owned copy -------
 * vtkTransformPolyDataFilter shares its input's cell arrays into its output
 * (output.setLines(input.getLines())) and vtkPolyData.initialize() then clears
 * them in place, so its second execution destroys its own input's topology.
 * If a vtk.js upgrade ever fixes this, THIS test fails -- and the scaled-copy
 * indirection in lib/vtk/vtkScaledPolyData.ts could be reconsidered. Until then
 * it documents exactly why that indirection exists. */
{
  const paths = makePaths();
  const transform = vtkTransform.newInstance();
  const filter = vtkTransformPolyDataFilter.newInstance();
  filter.setTransform(transform);
  filter.setInputData(paths);

  transform.setMatrix(vtkMatrixBuilder.buildFromRadian().scale(1, 1, 1).getMatrix());
  filter.getOutputData();
  check(
    paths.getLines().getNumberOfValues() === 4,
    "vtkTransformPolyDataFilter's FIRST execution leaves its input intact",
  );

  transform.setMatrix(vtkMatrixBuilder.buildFromRadian().scale(1, 1, 3.5).getMatrix());
  filter.modified();
  filter.getOutputData();
  check(
    paths.getLines().getNumberOfValues() === 0,
    "vtkTransformPolyDataFilter's SECOND execution still empties its own input "
      + "(if this fails, vtk.js fixed the aliasing bug -- see vtkScaledPolyData.ts)",
  );
}

console.log(`\ntubePipeline tests: ${passed} passed, ${failed} failed.`);
if (failures.length) {
  console.log(failures.join("\n"));
  process.exit(1);
}
