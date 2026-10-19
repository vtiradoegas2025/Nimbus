import { invokeCommand } from "@/lib/platform/appInvoke";
import { getHardwareInfo, getSchemeCatalog, getSimulatorStatus, isTauriRuntime } from "@/lib/platform/tauri";

export async function runPackagedStartupCheck(root: HTMLElement): Promise<void> {
  const started = performance.now();
  const errors: string[] = [];
  const onError = (event: ErrorEvent) => { errors.push(event.message); };
  const onRejection = (event: PromiseRejectionEvent) => { errors.push(String(event.reason)); };
  window.addEventListener("error", onError);
  window.addEventListener("unhandledrejection", onRejection);
  const checks: Array<{ name: string; passed: true }> = [];
  const requireCheck = (name: string, condition: boolean) => {
    if (!condition) throw new Error(`Packaged startup check failed: ${name}`);
    checks.push({ name, passed: true });
  };
  const report: Record<string, unknown> = {
    schema: "nimbus.studio.packaged-startup-check",
    schemaVersion: 1,
    profile: "startup",
    scope: "build-and-open",
    graphicsValidation: "pending",
    scienceQualification: "pending",
    checks,
  };
  try {
    await invokeCommand("record_visualization_qualification_progress", { stage: "startup-shell" });
    requireCheck("native-runtime", isTauriRuntime());
    const hardware = await getHardwareInfo();
    requireCheck("native-bridge", Number.isSafeInteger(hardware.coreCount) && (hardware.coreCount ?? 0) > 0);
    const simulator = await getSimulatorStatus();
    requireCheck("simulator-discovery", simulator.state === "ready" && Boolean(simulator.binaryPath));
    const catalog = await getSchemeCatalog();
    requireCheck("live-catalog", catalog.modules.length > 0);
    while (!root.querySelector(".nav-rail__item") && performance.now() - started < 30_000) {
      await new Promise(resolve => setTimeout(resolve, 50));
    }
    await new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve())));
    const shell = root.querySelector<HTMLElement>(".app");
    requireCheck("rendered-shell", Boolean(shell?.getClientRects().length) && document.visibilityState === "visible");
    const navigation = [...root.querySelectorAll(".nav-rail__item")].map(item => item.textContent?.trim());
    requireCheck("navigation", ["Projects", "Configure", "Runs", "Plots", "Settings"].every(name => navigation.includes(name)));
    requireCheck("no-startup-errors", errors.length === 0 && root.querySelector(".error-boundary") === null);
    report.status = "passed";
  } catch (error) {
    report.status = "failed";
    report.error = error instanceof Error ? error.message : String(error);
    report.pageErrors = errors;
  } finally {
    window.removeEventListener("error", onError);
    window.removeEventListener("unhandledrejection", onRejection);
  }
  report.elapsedMs = performance.now() - started;
  await invokeCommand("finish_visualization_qualification", { report });
}
