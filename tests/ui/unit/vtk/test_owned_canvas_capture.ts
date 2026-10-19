import assert from "node:assert/strict";
import { captureOwnedCanvasImage } from "../../../../ui/src/lib/vtk/ownedCanvasCapture.ts";

const original = Object.getOwnPropertyDescriptor(globalThis, "document");
const source = { width: 1280, height: 840, getBoundingClientRect: () => ({ x: 20, y: 30 }) };
const child = { getBoundingClientRect: () => ({ x: 35, y: 55 }) };
const renderers = [{ getViewProps: () => [{ getContainer: () => ({ getElementsByTagName: () => [child] }) }] }];
try {
  for (const failure of [null, "encoding", "context"] as const) {
    const draws: unknown[][] = [];
    let removed = false;
    const canvas = {
      width: 0, height: 0,
      getContext: () => failure === "context" ? null : { drawImage: (...args: unknown[]) => draws.push(args) },
      toDataURL: (format: string) => {
        assert.equal(format, "image/png");
        assert.equal(canvas.width, source.width);
        assert.equal(canvas.height, source.height);
        if (failure === "encoding") throw new Error("encoding failed");
        return "data:image/png;base64,encoded";
      },
      remove: () => { removed = true; },
    };
    Object.defineProperty(globalThis, "document", {
      configurable: true, value: { createElement: () => canvas },
    });
    const capture = () => captureOwnedCanvasImage(source as HTMLCanvasElement,
      renderers as unknown as Parameters<typeof captureOwnedCanvasImage>[1]);
    if (failure) assert.throws(capture, failure === "encoding" ? /encoding failed/ : /canvas is unavailable/);
    else assert.equal(capture(), "data:image/png;base64,encoded");
    if (failure !== "context") assert.deepEqual(draws, [[source, 0, 0], [child, 15, 25]]);
    assert.equal(canvas.width, 0);
    assert.equal(canvas.height, 0);
    assert.equal(removed, true);
    assert.equal(source.width, 1280);
    assert.equal(source.height, 840);
  }
} finally {
  if (original) Object.defineProperty(globalThis, "document", original);
  else Reflect.deleteProperty(globalThis, "document");
}
console.log("Owned VTK snapshots preserve composition and retire temporary backing stores: passed");
