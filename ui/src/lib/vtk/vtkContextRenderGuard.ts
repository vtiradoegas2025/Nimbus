import { installOwnedRenderWindow } from "./ownedRenderWindow";
export { guardVtkContextRendering, holdVtkSceneRendering, protectVtkTraversal } from "./vtkTraversalGuard";

export function installVtkContextRenderGuard(): void {
  installOwnedRenderWindow();
}
