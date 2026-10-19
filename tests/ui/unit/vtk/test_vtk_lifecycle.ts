import {
  disposeVtkScene,
  reduceVtkContextState,
  type VtkContextState,
} from "../../../../ui/src/lib/vtk/vtkLifecycle.ts";
import { readFileSync } from "node:fs";

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

{
  const calls: string[] = [];
  const errors = disposeVtkScene({
    clearScene: () => calls.push("clear"),
    cleanupSteps: [
      () => {
        calls.push("detach-listener");
        throw new Error("already detached");
      },
      () => calls.push("remove-actor"),
    ],
    resources: [
      {
        delete: () => {
          calls.push("actor-delete");
          throw new Error("deleted");
        },
      },
      { delete: () => calls.push("mapper-delete") },
    ],
    renderWindow: {
      getApiSpecificRenderWindow: () => ({
        releaseGraphicsResources: () => calls.push("graphics-release"),
        getCanvas: () => ({
          dataset: {},
          getContext: () => ({
            getExtension: () => ({ loseContext: () => calls.push("context-release") }),
          }),
        }),
        setContainer: (container) =>
          calls.push(container === null ? "detach-container" : "bad"),
      }),
      delete: () => calls.push("window-delete"),
    },
  });
  check(calls[0] === "clear", "owner ref clears before VTK cleanup");
  check(calls.indexOf("graphics-release") > 0
    && calls.indexOf("graphics-release") < calls.indexOf("actor-delete")
    && calls.indexOf("graphics-release") < calls.indexOf("context-release"),
  "graphics allocations retire while pipeline owners and the GL context are alive");
  check(
    calls.includes("remove-actor"),
    "a throwing listener does not skip renderer detach",
  );
  check(
    calls.includes("mapper-delete"),
    "a throwing resource does not skip later resources",
  );
  check(
    calls.includes("context-release") &&
      calls.indexOf("context-release") > calls.indexOf("window-delete"),
    "WebGL context remains valid until mapper and framebuffer deletion completes",
  );
  check(
    calls.indexOf("detach-container") < calls.indexOf("window-delete"),
    "render window detaches from its DOM container",
  );
  check(calls.includes("window-delete"), "render window deletion is attempted after cleanup failures");
  check(errors.length === 2, "individual cleanup failures are captured");
}

{
  let resourceDeleted = false;
  let windowDeleted = false;
  const failure = new Error("partially initialized graphics resource");
  const errors = disposeVtkScene({
    clearScene: () => undefined,
    resources: [{ delete: () => { resourceDeleted = true; } }],
    renderWindow: {
      getApiSpecificRenderWindow: () => ({ releaseGraphicsResources: () => { throw failure; } }),
      delete: () => { windowDeleted = true; },
    },
  });
  check(resourceDeleted && windowDeleted, "graphics-release failure cannot strand pipeline or window owners");
  check(errors.length === 1 && errors[0] === failure, "graphics-release errors remain observable");
}

for (const failurePoint of ["point-data", "array-list"] as const) {
  const calls: string[] = [];
  const failure = new Error(`unavailable ${failurePoint}`);
  const errors = disposeVtkScene({
    clearScene: () => undefined,
    resources: [{
      getPointData: () => {
        if (failurePoint === "point-data") throw failure;
        return { getArrays: () => { throw failure; }, removeAllArrays: () => { calls.push("arrays-remove"); } };
      },
      delete: () => { calls.push("resource-delete"); },
    }, { delete: () => { calls.push("next-delete"); } }],
    renderWindow: { delete: () => { calls.push("window-delete"); } },
  });
  check(errors.length === 1 && errors[0] === failure, `${failurePoint} failure remains observable`);
  check(calls.includes("resource-delete") && calls.includes("next-delete") && calls.includes("window-delete"),
    `${failurePoint} failure does not strand remaining owners`);
  if (failurePoint === "array-list") check(calls.includes("arrays-remove"), "array-list failure still detaches point data");
}

{
  let windowDeleted = false;
  let contextRetired = false;
  const canvas = {
    width: 1280, height: 840, dataset: {} as Record<string, string>,
    getContext: () => ({ getExtension: () => ({ loseContext: () => {
      check(windowDeleted, "drawing context retires after its VTK window");
      check(canvas.width === 1280 && canvas.height === 840,
        "drawing buffer remains intact through graphics resource deletion");
      contextRetired = true;
    } }) }),
  };
  const errors = disposeVtkScene({
    clearScene: () => undefined,
    renderWindow: {
      getApiSpecificRenderWindow: () => ({ getCanvas: () => canvas }),
      delete: () => { windowDeleted = true; },
    },
  });
  check(errors.length === 0 && contextRetired, "drawing-buffer retirement preserves context cleanup");
  check(canvas.width === 0 && canvas.height === 0,
    "retired canvases release their native backing dimensions before garbage collection");
}

/*
 * The interactor.
 *
 * `vtkGenericRenderWindow.delete()` chains setContainer, the OpenGL window's
 * delete and its own window `resize` removal — and never touches
 * `model.interactor`. The interactor's CONSTRUCTOR registers a
 * `visibilitychange` listener on the DOCUMENT, which only
 * `RenderWindowInteractor.delete()` removes, so every scene left one more
 * listener on a target that outlives it. A listener is a GC root, and that one
 * closes over the interactor's model, which reaches the view and through it the
 * whole renderer graph: nothing a scene allocated was ever reclaimed.
 */
{
  const calls: string[] = [];
  let windowDeleted = false;
  const errors = disposeVtkScene({
    clearScene: () => calls.push("clear"),
    renderWindow: {
      // Throws once the window is gone, the way a deleted vtk.js object does,
      // so this also pins that the handle is taken up front.
      getInteractor: () => {
        if (windowDeleted) throw new Error("render window already deleted");
        calls.push("get-interactor");
        return { delete: () => calls.push("interactor-delete") };
      },
      delete: () => {
        windowDeleted = true;
        calls.push("window-delete");
      },
    },
  });
  check(errors.length === 0, "the interactor path adds no errors of its own");
  check(
    calls.includes("interactor-delete"),
    "the interactor is deleted -- nothing else releases it",
  );
  check(
    calls.indexOf("get-interactor") < calls.indexOf("window-delete"),
    "the handle is taken before the window is deleted, while it can still be had",
  );
  check(
    calls.indexOf("interactor-delete") > calls.indexOf("window-delete"),
    "and deleted after it, because the window's teardown unbinds events on it",
  );
}

{
  const calls: string[] = [];
  let wrapperDeleted = false;
  const coreRenderer = {
    delete: () => { calls.push("renderer-delete"); },
    getActiveCamera: () => { throw new Error("borrowed camera must not be inspected or deleted"); },
  };
  const coreWindow = { delete: () => { calls.push("core-window-delete"); } };
  const errors = disposeVtkScene({
    clearScene: () => undefined,
    renderWindow: {
      getRenderer: () => {
        if (wrapperDeleted) throw new Error("renderer no longer available");
        return coreRenderer;
      },
      getRenderWindow: () => {
        if (wrapperDeleted) throw new Error("core window no longer available");
        return coreWindow;
      },
      getInteractor: () => ({ delete: () => { calls.push("interactor-delete"); } }),
      delete: () => { wrapperDeleted = true; calls.push("wrapper-delete"); },
    },
  });
  check(errors.length === 0, "owned cores are captured before wrapper deletion without inspecting borrowed cameras");
  check(calls.join(",") === "wrapper-delete,interactor-delete,renderer-delete,core-window-delete",
    "API wrapper and interactor retire before the owned renderer and core window");
}

for (const failurePoint of ["renderer-get", "core-get", "renderer-delete", "core-delete"] as const) {
  const calls: string[] = [];
  const failure = new Error(failurePoint);
  const errors = disposeVtkScene({
    clearScene: () => undefined,
    renderWindow: {
      getRenderer: () => {
        if (failurePoint === "renderer-get") throw failure;
        return { delete: () => { calls.push("renderer-delete"); if (failurePoint === "renderer-delete") throw failure; } };
      },
      getRenderWindow: () => {
        if (failurePoint === "core-get") throw failure;
        return { delete: () => { calls.push("core-delete"); if (failurePoint === "core-delete") throw failure; } };
      },
      getInteractor: () => ({ delete: () => { calls.push("interactor-delete"); } }),
      delete: () => { calls.push("wrapper-delete"); },
    },
  });
  check(errors.length === 1 && errors[0] === failure, `${failurePoint} is reported without throwing`);
  check(calls.includes("wrapper-delete") && calls.includes("interactor-delete")
    && (failurePoint === "renderer-get" || calls.includes("renderer-delete"))
    && (failurePoint === "core-get" || calls.includes("core-delete")),
  `${failurePoint} does not skip remaining owner teardown`);
}

{
  let deletions = 0;
  const owner = { delete: () => { deletions++; } };
  disposeVtkScene({ clearScene: () => undefined,
    resources: [owner], renderWindow: { getRenderer: () => owner, getRenderWindow: () => owner } });
  check(deletions === 1, "aliased core owners already present in resources are deleted once");
}

{
  // Absent, null and throwing: three shapes of "no interactor here", none of
  // which may cost the rest of the teardown.
  const calls: string[] = [];
  disposeVtkScene({
    clearScene: () => undefined,
    renderWindow: { delete: () => calls.push("window-delete") },
  });
  check(calls.at(-1) === "window-delete", "a render window with no interactor still disposes");

  const nullCalls: string[] = [];
  disposeVtkScene({
    clearScene: () => undefined,
    renderWindow: {
      getInteractor: () => null,
      delete: () => nullCalls.push("window-delete"),
    },
  });
  check(nullCalls.at(-1) === "window-delete", "a null interactor is skipped rather than thrown on");

  const throwCalls: string[] = [];
  const throwErrors = disposeVtkScene({
    clearScene: () => undefined,
    renderWindow: {
      getInteractor: () => { throw new Error("no interactor"); },
      delete: () => throwCalls.push("window-delete"),
    },
  });
  check(
    throwCalls.includes("window-delete"),
    "a throwing getInteractor does not strand the render window",
  );
  check(throwErrors.length === 1, "and that failure is reported rather than swallowed");
}

{
  const scenes: Array<[string, number]> = [
    ["VolumeRender3D.tsx", 1],
    ["Isosurface3D.tsx", 1],
    ["Streamlines3D.tsx", 1],
    ["Trajectories3D.tsx", 1],
    ["SupercellRender3D.tsx", 1],
  ];
  for (const [file, expectedScenes] of scenes) {
    const source = readFileSync(
      new URL(`../../../../ui/src/components/plots/${file}`, import.meta.url),
      "utf8",
    );
    const disposals = source.match(/disposeVtkScene\(\{/g)?.length ?? 0;
    const recoveryMounts =
      source.match(
        /\}, \[[^\]]*ctxLoss\.available[^\]]*ctxLoss\.generation[^\]]*vtkBackend\.backend[^\]]*vtkBackend\.confirmActiveApi[^\]]*\]\);/g,
      )?.length ?? 0;
    check(
      disposals === expectedScenes,
      `${file} uses shared disposal for every scene`,
    );
    check(
      !source.includes("renderWindow.delete()"),
      `${file} has no fallible ad-hoc window cleanup`,
    );
    check(
      recoveryMounts >= expectedScenes,
      `${file} rebuilds every scene after context restoration or backend fallback`,
    );
    check(
      source.includes("if (!ctxLoss.available) return;"),
      `${file} does not rebuild against a context that is still lost`,
    );
    check(
      source.includes("markFrameRendered"),
      `${file} acknowledges a successful recovery frame`,
    );
    check(
      source.includes("const vtkLifecycleGeneration = contextGeneration * 2")
        && (source.match(/vtkLifecycleGeneration/g)?.length ?? 0) >= 5,
      `${file} recommits scene data and auxiliary actors after an explicit backend switch`,
    );
  }
}

{
  let deletes = 0;
  const options = {
    clearScene: () => {},
    resources: [
      {
        delete: () => {
          deletes++;
          if (deletes > 1) throw new Error("double");
        },
      },
    ],
    renderWindow: {
      delete: () => {
        deletes++;
        if (deletes > 2) throw new Error("double");
      },
    },
  };
  disposeVtkScene(options);
  let threw = false;
  try {
    disposeVtkScene(options);
  } catch {
    threw = true;
  }
  check(!threw, "repeated scene disposal never throws");
  check(deletes === 4, "repeated disposal still attempts final render-window release");
}

{
  const calls: string[] = [];
  const array = { delete: () => calls.push("array-delete") };
  disposeVtkScene({
    clearScene: () => {},
    resources: [{
      getPointData: () => ({
        getArrays: () => [array],
        removeAllArrays: () => calls.push("arrays-detach"),
      }),
      delete: () => calls.push("dataset-delete"),
    }],
    renderWindow: {},
  });
  check(calls.join(",") === "arrays-detach,array-delete,dataset-delete", "data arrays detach and delete before their data set");
}

{
  let state: VtkContextState = { lost: false, available: true, generation: 0 };
  state = reduceVtkContextState(state, { type: "lost" });
  check(
    state.lost && !state.available && state.generation === 1,
    "context loss blocks and retires the stale renderer generation",
  );
  state = reduceVtkContextState(state, { type: "restored" });
  check(
    state.lost && state.available && state.generation === 2,
    "browser restoration requests a new renderer generation",
  );
  state = reduceVtkContextState(state, { type: "frame_rendered", generation: 1 });
  check(state.lost, "an obsolete renderer cannot clear the recovery overlay");
  state = reduceVtkContextState(state, { type: "frame_rendered", generation: 2 });
  check(!state.lost, "the rebuilt generation clears recovery only after rendering");
}

{
  const [{ default: vtkRenderWindow }, { default: vtkRenderer }, { default: vtkCamera }] = await Promise.all([
    import("@kitware/vtk.js/Rendering/Core/RenderWindow.js"),
    import("@kitware/vtk.js/Rendering/Core/Renderer.js"),
    import("@kitware/vtk.js/Rendering/Core/Camera.js"),
  ]);
  const core = vtkRenderWindow.newInstance();
  const renderer = vtkRenderer.newInstance();
  const borrowedCamera = vtkCamera.newInstance();
  renderer.setActiveCamera(borrowedCamera);
  core.addRenderer(renderer);
  const errors = disposeVtkScene({ clearScene: () => undefined, renderWindow: {
    getRenderer: () => renderer, getRenderWindow: () => core,
  } });
  check(errors.length === 0 && renderer.isDeleted() && core.isDeleted(),
    "actual VTK renderer and core window are deterministically deleted");
  check(!borrowedCamera.isDeleted(), "deleting owned core objects leaves an externally supplied camera intact");
  borrowedCamera.delete();
}

console.log(`\nvtkLifecycle tests: ${passed} passed, ${failed} failed.`);
if (failures.length) {
  console.log(failures.join("\n"));
  process.exit(1);
}
