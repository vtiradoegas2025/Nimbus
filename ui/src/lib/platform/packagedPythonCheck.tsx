import { flushSync } from "react-dom";
import { createWorkspace, serializeWorkspace, pythonWorkspaceStorageKey } from "../studio/pythonWorkspace";
import { createRoot } from "react-dom/client";
import { EditorView } from "@codemirror/view";
import { PythonPlot } from "@/components/plots/PythonPlot";
import { listSteps } from "../run/runOutputs";
import { readSession, sessionKey } from "../studio/pythonSession";
import { disposePythonRuntimeClient, getPythonRuntimeClient } from "./pyodideRuntime";
import { readBinaryFile, readPySnippet, writeBinaryFile, writePySnippet } from "./tauri";
import { invokeCommand } from "./appInvoke";

export async function checkPackagedPython() {
  const started = performance.now();
  const projectId = `native-python-check-${Date.now()}`;
  const key = sessionKey(projectId, "primary");
  localStorage.setItem(pythonWorkspaceStorageKey(projectId, "primary"), serializeWorkspace(createWorkspace("")));
  const checks: Array<{ name: string; passed: boolean; detail: string }> = [];
  const check = (name: string, passed: boolean, detail: string) => {
    checks.push({ name: `python-${name}`, passed, detail });
    if (!passed) throw new Error(`Native Python ${name}: ${detail}`);
  };
  const until = async (predicate: () => boolean, label: string, milliseconds = 60_000) => {
    const limit = performance.now() + milliseconds;
    while (!predicate()) {
      const error = readSession(key).fatalError ?? readSession(key).loadErr;
      if (error) throw new Error(`Native Python ${label}: ${error}`);
      if (performance.now() > limit) throw new Error(`Native Python timed out: ${label}`);
      await new Promise((resolve) => setTimeout(resolve, 25));
    }
  };
  const fixture = await invokeCommand<{ runDir: string }>("create_visualization_brick_qualification_fixture");
  const host = document.createElement("div");
  host.style.cssText = "position:fixed;inset:0;background:var(--color-bg);padding:16px";
  document.body.append(host);
  const root = createRoot(host);
  const click = (label: string) => {
    const button = [...host.querySelectorAll<HTMLButtonElement>("button")].find((node) => node.textContent?.trim() === label);
    if (!button || button.disabled) throw new Error(`Native Python control unavailable: ${label}`);
    button.click();
  };
  const run = async (source: string) => {
    const editor = EditorView.findFromDOM(host.querySelector(".cm-editor")!);
    if (!editor) throw new Error("Native Python editor did not mount");
    flushSync(() => editor.dispatch({ changes: { from: 0, to: editor.state.doc.length, insert: source } }));
    // WKWebView may suspend animation frames when occluded; the edit is
    // already flushed, so only an event-loop yield is needed before clicking.
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
    const count = readSession(key).completedRuns;
    click("Run script");
    await until(() => readSession(key).completedRuns > count, "script completion", 120_000);
    const completed = [...readSession(key).cellRuns.values()].at(-1);
    if (completed?.code.trim() !== source.trim()) throw new Error(`Native Python editor/run mismatch: expected ${source.length} characters, ran ${completed?.code.length ?? 0}`);
    const result = completed.result;
    if (!result || result.raised) throw new Error(`Native Python run: ${result?.stderr ?? readSession(key).ioStatus ?? "no output"}`);
    return result;
  };
  try {
    root.render(<PythonPlot runDir={fixture.runDir} steps={await listSteps(fixture.runDir)} stepIdx={0} projectId={projectId} />);
    await until(() => Boolean(host.querySelector(".cm-editor")) && readSession(key).datasetToken != null, "editor startup");
    await invokeCommand("record_visualization_qualification_progress", { stage: "native-python-startup" });
    const script = "import numpy as np\nimport matplotlib.pyplot as plt\nsentinel = 42\nfield = await load_field('scalar')\nassert np.isfinite(field).any()\na = np.array([9007199254740993], dtype=np.int64)\nnp.save('/nimbus/out/exact.npy', a)\nfig, ax = plt.subplots(figsize=(3,2), dpi=160)\nax.plot([0,1],[0,1])\ndisplay(fig)\nplt.close(fig)\nprint('native field and figure OK')\n";
    const path = `${fixture.runDir}/native-check.py`;
    await writePySnippet(path, script);
    check("script-file", await readPySnippet(path) === script, "script survives native write/read");
    const result = await run(await readPySnippet(path));
    await invokeCommand("record_visualization_qualification_progress", { stage: "native-python-script-complete" });
    check("run-field", result.stdout.includes("native field and figure OK"), `real interpreter loads the native terrain fixture; stdout=${JSON.stringify(result.stdout)}, stderr=${JSON.stringify(result.stderr)}`);
    check("figure", result.outputs.some((output) => output.kind === "image" && output.svg?.includes("<svg")), "PNG and SVG figure capture complete");
    const produced = result.produced.find((file) => file.name === "exact.npy");
    check("produced-file", Boolean(produced), "NumPy output is listed");
    const client = getPythonRuntimeClient(key);
    const bytes = await client.readProducedFile(await client.getPyodide(), produced!);
    const exported = `${fixture.runDir}/native-exact.npy`;
    await writeBinaryFile(exported, bytes);
    const reread = await readBinaryFile(exported);
    const view = new DataView(reread.buffer, reread.byteOffset, reread.byteLength);
    check("exact-export", view.getBigInt64(10 + view.getUint16(8, true), true) === 9007199254740993n,
      "worker output survives the real native binary-file bridge without rounding");
    const editor = EditorView.findFromDOM(host.querySelector(".cm-editor")!)!;
    flushSync(() => editor.dispatch({ changes: { from: 0, to: editor.state.doc.length, insert: "print('native loop started', flush=True)\nwhile True:\n    pass\n" } }));
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
    click("Run script");
    await until(() => readSession(key).liveStream?.stdout.includes("native loop started") === true, "loop startup");
    click("Stop");
    await until(() => !readSession(key).running, "Stop", 15_000);
    check("stop", !readSession(key).running, "Stop ends a running infinite loop");
    click("Restart");
    await until(() => readSession(key).ioStatus === "Interpreter restarted.", "Restart");
    const restarted = await run("assert 'sentinel' not in globals()\nprint('restart OK')\n");
    check("restart", restarted.stdout.includes("restart OK"), "fresh namespace executes after Restart");
    return { checks, elapsedMs: performance.now() - started, crossOriginIsolated };
  } finally {
    root.unmount(); host.remove(); disposePythonRuntimeClient(key);
    for (const storageKey of Object.keys(localStorage)) if (storageKey.includes(projectId)) localStorage.removeItem(storageKey);
    await invokeCommand("remove_visualization_brick_qualification_fixture");
  }
}
