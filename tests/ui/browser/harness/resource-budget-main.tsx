import { createRoot } from "react-dom/client";
import { ResourceBudgetPanel } from "@nimbus/components/studio/ResourceBudgetPanel";
import { installTauriFixture } from "../support/fixtures";
import "@nimbus/styles/theme.css";
import "@nimbus/styles/base.css";
import "@nimbus/components/studio/PlotStudio.css";

/*
 * Resource-budget harness. The panel reports which native read paths the
 * desktop runtime supports, and a null capability record used to mean three
 * different things at once: browser mode, probe in flight, and probe threw.
 *
 * `?desktop=1` installs the Tauri fixture, which by default has no handler for
 * `native_field_read_capabilities` and therefore REJECTS it — the desktop
 * probe-failure state, reached without needing a real runtime. Without the
 * flag the page stays in genuine browser mode.
 *
 * `?desktop=1&capabilities=1` additionally answers that probe with the backend's
 * real record, giving the third state: a desktop runtime that reports its
 * contract, where the per-format table is the thing under test.
 */

const params = new URLSearchParams(window.location.search);
if (params.get("desktop") === "1") {
  installTauriFixture({ nativeCapabilities: params.get("capabilities") === "1" });
}

const root = document.getElementById("root");
if (!root) throw new Error("Resource-budget harness root missing");

createRoot(root).render(
  <ResourceBudgetPanel onClose={() => undefined} onClearCache={() => undefined} />,
);
