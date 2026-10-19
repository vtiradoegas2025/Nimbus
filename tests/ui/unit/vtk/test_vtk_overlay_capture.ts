import assert from "node:assert/strict";
import { vtkColorbarCaptureStops } from "../../../../ui/src/lib/vtk/vtkOverlayCapture.ts";
import { DBZ_VTK_STOPS_VOLUME } from "../../../../ui/src/lib/plot/plotColormaps.ts";

assert.deepEqual(vtkColorbarCaptureStops("Cool to Warm"), [
  { offset: 0, color: "#3b4cc0" }, { offset: 0.5, color: "#b4b4b4" }, { offset: 1, color: "#b40426" },
]);
for (const name of ["Inferno", "Viridis", "Plasma", "RdBu", "Greys", "dBZ", "dBZ floor", "dBZ volume"]) {
  const stops = vtkColorbarCaptureStops(name);
  assert.equal(stops[0].offset, 0);
  assert.equal(stops.at(-1)!.offset, 1);
  assert.ok(stops.every((stop, index) => index === 0 || stop.offset >= stops[index - 1].offset));
}
const volume = vtkColorbarCaptureStops("dBZ volume");
assert.equal(volume.length, DBZ_VTK_STOPS_VOLUME.length);
DBZ_VTK_STOPS_VOLUME.forEach((stop, index) => {
  assert.ok(Math.abs(volume[index].offset - (stop.scalar + 30) / 105) < 0.000001);
  assert.equal(volume[index].color, `rgb(${[stop.r, stop.g, stop.b].map(value => Math.round(value * 255)).join(",")})`);
});
console.log("VTK capture legends preserve physical palette anchors and RGB stops.");
