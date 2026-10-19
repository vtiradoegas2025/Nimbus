import assert from "node:assert/strict";
import factory from "@kitware/vtk.js/Rendering/OpenGL/ViewNodeFactory.js";
import viewNode from "@kitware/vtk.js/Rendering/SceneGraph/ViewNode.js";
import { installOrderedRenderer } from "../../../../ui/src/lib/vtk/orderedRenderer.ts";

installOrderedRenderer();
const renderer = factory.newInstance().createNode({
  isDeleted: () => false, getClassName: () => "vtkRenderer",
}) as any;
renderer.setMyFactory({ createNode: () => viewNode.newInstance() });
const camera = { name: "camera" };
const props = Array.from({ length: 6 }, (_, index) => ({ index }));
renderer.addMissingNode(camera);
renderer.addMissingNodes(props, true);
const originals = renderer.getChildren().slice();
for (const order of [props.slice().reverse(), [props[3], props[1], props[5], props[0], props[4], props[2]], props]) {
  renderer.prepareNodes();
  renderer.addMissingNode(camera);
  renderer.addMissingNodes(order, true);
  renderer.removeUnusedNodes();
  assert.deepEqual(renderer.getChildren().map((node: any) => node.getRenderable()), [camera, ...order]);
  assert.ok(renderer.getChildren().every((node: any) => originals.includes(node)), "reordering reuses GPU owners");
}
renderer.prepareNodes();
renderer.addMissingNode(camera);
renderer.addMissingNodes([props[5], props[1]], true);
renderer.removeUnusedNodes();
assert.deepEqual(renderer.getChildren().map((node: any) => node.getRenderable()), [camera, props[5], props[1]]);
renderer.delete();
console.log("ordered renderer: cached traversal follows prop order across reversals and removal");
