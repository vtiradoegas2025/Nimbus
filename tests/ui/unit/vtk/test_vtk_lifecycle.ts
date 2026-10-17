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
      calls.indexOf("context-release") < calls.indexOf("detach-container"),
    "WebGL context retires before its canvas detaches",
  );
  check(
    calls.at(-2) === "detach-container",
    "render window detaches from its DOM container",
  );
  check(calls.at(-1) === "window-delete", "render window deletion is always attempted last");
  check(errors.length === 2, "individual cleanup failures are captured");
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

console.log(`\nvtkLifecycle tests: ${passed} passed, ${failed} failed.`);
if (failures.length) {
  console.log(failures.join("\n"));
  process.exit(1);
}
