import { useState } from "react";
import { createRoot } from "react-dom/client";
import { InvestigationWorkspace } from "@nimbus/components/studio/InvestigationWorkspace";
import { createLocalDatasetSource } from "@nimbus/lib/dataset/datasetSources";
import { fixtureControl, installTauriFixture, steps } from "../support/fixtures";
import "@nimbus/styles/theme.css";
import "@nimbus/styles/base.css";
import "@nimbus/components/studio/PlotStudio.css";

/*
 * Investigation-workspace harness. "Build probe history" reads every frame in
 * the run one at a time; against the fixture that loop completes within a
 * single animation frame, so `?readDelay=<ms>` slows each read enough for a
 * spec to observe the build while it is still in flight -- which is the only
 * way to assert on the busy label, the cancel affordance, and the progress
 * counter that used to be left frozen after the loop ended.
 */

const params = new URLSearchParams(window.location.search);
installTauriFixture();
fixtureControl.setFieldReadDelay(Number(params.get("readDelay") ?? "0"));

const dataset = createLocalDatasetSource("nimbus_run", "/__nimbus_browser_fixture__");
const historySteps = params.get("timeUnknown") === "1"
  ? steps.map((step, index) => index === 1 ? { ...step, simulationTimeS: null } : step)
  : steps;

function Harness() {
  const [stepIdx, setStepIdx] = useState(0);
  return (
    <div className="plot-studio" style={{ height: "100vh", overflow: "auto" }}>
      <InvestigationWorkspace
        dataset={dataset}
        steps={historySteps}
        stepIdx={stepIdx}
        onStepChange={setStepIdx}
      />
    </div>
  );
}

const root = document.getElementById("root");
if (!root) throw new Error("Investigation harness root missing");
createRoot(root).render(<Harness />);
