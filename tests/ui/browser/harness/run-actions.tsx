import { useState } from "react";
import { RunDetail } from "@nimbus/components/runs/RunDetail";
import { mockQueue } from "@nimbus/lib/core/mockData";
import { installWorkflowFixture } from "../support/workflowFixture";
import type { QueuedRun } from "@nimbus/lib/core/types";

const remote = new URLSearchParams(window.location.search).get("remote") === "1";
const runs: QueuedRun[] = [0, 1].map(index => ({ ...mockQueue[0], id: `action-${index}`, caseName: `Action run ${index}`,
  outputDir: `/__workflow_fixture__/run-${index}`, provenance: mockQueue[1].provenance }));
if (remote) for (const run of runs) Object.assign(run, { status: "failed", endedAtMs: Date.now(),
  executionTarget: { id: "fixture", name: "Linux node", host: "fixture-host", binaryPath: "/opt/nimbus/nimbus_sim", workDirectory: "/home/research/nimbus" },
  remoteReceive: { state: "failed", error: "Prior receive transport failed", retryAvailable: true, updatedUnixMs: 1 } });
const outcome = new URLSearchParams(window.location.search).get("outcome");
if (outcome) for (const run of runs) {
  run.status = outcome === "empty" ? "completed" : "failed";
  run.frameCount = outcome === "partial" ? 2 : 0;
  run.provenance = { ...mockQueue[1].provenance!, exitCode: outcome === "empty" ? 0 : -4, completeFrameCount: run.frameCount };
}

export function RunActionsHarness() {
  const [index, setIndex] = useState(0);
  const [opened, setOpened] = useState("");
  return <><nav>{runs.map((run, position) => <button key={run.id} onClick={() => setIndex(position)}>Select {run.caseName}</button>)}</nav>
    <RunDetail run={runs[index]} goToPlots={() => setOpened(runs[index].id)} /><output aria-label="Opened plot run">{opened}</output></>;
}

export function installRunActionsFixture() { installWorkflowFixture(); }
