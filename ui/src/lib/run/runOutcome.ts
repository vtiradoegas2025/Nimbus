import type { QueuedRun } from "@/lib/core/types";

const SIGNALS: Record<number, string> = {
  4: "SIGILL: illegal instruction",
  6: "SIGABRT: process aborted",
  9: "SIGKILL: process killed",
  11: "SIGSEGV: invalid memory access",
  15: "SIGTERM: termination requested",
};

export function processExitSummary(code: number | null | undefined, remoteLinux = false): string {
  if (code == null || !Number.isInteger(code)) return "Process exit details are unavailable.";
  if (code < 0 && code >= -64) return `Process ended on signal ${-code}${SIGNALS[-code] ? ` (${SIGNALS[-code]})` : ""}.`;
  if (remoteLinux && code > 128 && SIGNALS[code - 128]) return `Exit code ${code} (Linux signal-style exit for ${SIGNALS[code - 128]}).`;
  return code === 0 ? "Process exited successfully." : `Process exited with code ${code}.`;
}

export function publishedFrameCount(run: QueuedRun): number | null {
  const count = run.frameCount ?? (run.provenance?.runtimeHealth === "verified" ? run.provenance.completeFrameCount : null);
  return count == null ? null : Number.isInteger(count) && count >= 0 ? count : 0;
}

export function runPlotBlockReason(run: QueuedRun, allowLivePreview = false): string | null {
  if (!run.outputDir) return "This run has no output directory.";
  if (run.remoteTerminationUnconfirmed) return "Confirm remote termination before opening saved results.";
  if (run.remoteTransferInProgress || run.remoteReceive?.state === "receiving") return "Wait for the current result transfer to finish.";
  if (run.libraryHealth === "invalid") return "The saved output failed validation. Inspect the run evidence and output directory.";
  if (run.executionTarget && run.origin !== "disk" && run.remoteReceive?.state !== "complete") return "Receive and verify the remote results before opening plots.";
  const frames = publishedFrameCount(run);
  if (frames === 0) return "No complete saved frames are available yet.";
  if (frames != null && frames > 0) return null;
  if (run.origin === "disk") return "No complete saved frames have been indexed.";
  if (run.status === "completed" || allowLivePreview && ["running", "failed", "aborted"].includes(run.status)) return null;
  return "Wait for a complete saved frame before opening plots.";
}

export function runFailureSummary(run: QueuedRun): { title: string; detail: string } | null {
  if (run.status !== "failed") return null;
  const code = run.provenance?.runtimeHealth === "verified" ? run.provenance.exitCode : null;
  if (run.executionTarget && code === 0 && run.remoteReceive?.state === "invalid") {
    return { title: "Remote process finished; saved results failed validation", detail: "Inspect the received output and run evidence before retrying the transfer or rerunning the simulation." };
  }
  if (run.executionTarget && code === 0 && run.remoteReceive?.state === "failed") {
    return { title: "Remote process finished; results were not received", detail: "Retry receiving the saved output after restoring the connection. A transfer failure does not require rerunning the simulation." };
  }
  return {
    title: publishedFrameCount(run) === 0 ? "Run failed without a complete saved frame" : "Run failed",
    detail: `${processExitSummary(code, !!run.executionTarget)} Review the run evidence${run.origin === "disk" ? " and output directory" : " and console log"} before retrying.`,
  };
}
