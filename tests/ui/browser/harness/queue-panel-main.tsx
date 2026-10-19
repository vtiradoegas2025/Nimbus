import { useState } from "react";
import { createRoot } from "react-dom/client";
import { QueuePanel } from "@nimbus/components/shell/QueuePanel";
import { RunActionsHarness, installRunActionsFixture } from "./run-actions";
import type { QueuedRun, SimulatorStatus } from "@nimbus/lib/core/types";
import "@nimbus/styles/theme.css";
import "@nimbus/styles/base.css";

/*
 * Queue-panel harness. The App-level run-library harness draws its rows from
 * mockQueue, which holds one running and two queued runs and therefore never
 * populates the "Recent" archive section. This harness renders QueuePanel
 * directly against a synthetic archive large enough to exercise the collapse
 * boundary, so the truncation contract can be pinned without perturbing the
 * shared mock data that other specs assert against.
 */

const ARCHIVED_RUN_COUNT = 20;

const archivedRuns: QueuedRun[] = Array.from({ length: ARCHIVED_RUN_COUNT }, (_, index) => ({
  id: `archive-${String(index).padStart(3, "0")}`,
  caseName: `cart_cgrid_archive_${String(index).padStart(3, "0")}`,
  scheme: "klemp_wilhelmson_split_cartesian_cgrid",
  status: "completed",
  startedAtMs: 1_700_000_000_000 + index * 60_000,
  endedAtMs: 1_700_000_030_000 + index * 60_000,
  simTimeS: 600,
  step: 4200,
  velMax: 31.5,
  durationS: 600,
  outputDir: `/__nimbus_browser_fixture__/archive-${index}`,
}));

const liveRun: QueuedRun = {
  id: "archive-live",
  caseName: "cart_cgrid_live",
  scheme: "klemp_wilhelmson_split_cartesian_cgrid",
  status: "running",
  startedAtMs: 1_700_000_000_000,
  endedAtMs: null,
  simTimeS: 120,
  step: 900,
  velMax: 18.2,
  durationS: 1200,
  outputDir: "/__nimbus_browser_fixture__/live",
};

const simulator: SimulatorStatus = {
  binaryPath: "/__nimbus_browser_fixture__/bin/nimbus_sim",
  state: "ready",
  gpu: "Fixture GPU",
};

/** What App.tsx substitutes when `sim.data` is null — the shape that used to
 *  make a failed probe look identical to "no simulator configured". */
const UNKNOWN_SIMULATOR: SimulatorStatus = { binaryPath: null, state: "unknown", gpu: null };

/** Distinctive strings so a spec can prove the REAL message reached the DOM
 *  rather than a fixed banner that discarded it. */
const FIXTURE_ERROR_MESSAGE = "cannot delete a running run; abort it first";
const FIXTURE_SIM_ERROR = "get_simulator_status: fixture transport failure";

/* Query-driven state selection. Each of these renders a distinct failure or
 * transition that used to be indistinguishable from a neighbouring state, so
 * the specs need to reach them independently. */
const params = new URLSearchParams(window.location.search);
const errorScope = params.get("error");
const runsError = errorScope
  ? { scope: errorScope as "library" | "live" | "action", message: FIXTURE_ERROR_MESSAGE }
  : null;
const loading = params.get("loading") === "1";
const simulatorError = params.get("simError") === "1" ? FIXTURE_SIM_ERROR : null;
const deletable = params.get("deletable") === "1";
// `empty=1` exercises the loading-vs-empty pair; the archive is only populated
// when we are not testing that distinction.
const runs = params.get("empty") === "1" ? [] : [liveRun, ...archivedRuns];

function Harness() {
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [visibleRuns, setVisibleRuns] = useState(runs);
  return (
    <div style={{ display: "flex", height: "100vh" }}>
      <QueuePanel
        runs={visibleRuns}
        simulator={simulatorError ? UNKNOWN_SIMULATOR : simulator}
        activePage="runs"
        selectedRunId={selectedRunId}
        onSelectRun={setSelectedRunId}
        onDeleteRun={deletable
          ? async (id) => setVisibleRuns((current) => current.filter((run) => run.id !== id))
          : undefined}
        runsError={runsError}
        loading={loading}
        simulatorError={simulatorError}
      />
    </div>
  );
}

const root = document.getElementById("root");
if (!root) throw new Error("Queue-panel harness root missing");
if (params.get("mode") === "run-actions") installRunActionsFixture();
createRoot(root).render(params.get("mode") === "run-actions" ? <RunActionsHarness /> : <Harness />);
