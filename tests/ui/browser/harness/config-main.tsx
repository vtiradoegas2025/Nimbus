import { createRoot } from "react-dom/client";
import { ConfigEditor } from "@nimbus/components/config/ConfigEditor";
import { RemoteWorkspaceSettings } from "@nimbus/components/settings/RemoteWorkspaceSettings";
import { RemoteComputeSettings } from "@nimbus/components/settings/RemoteComputeSettings";
import { ProjectsHarness } from "./projects";
import { Settings } from "@nimbus/components/settings/Settings";
import { installSettingsFixture } from "../support/settingsFixture";
import { installStudioAppearance } from "@nimbus/lib/studio/studioAppearance";
import { installRemoteWorkspaceFixture } from "../support/remoteWorkspaceFixture";
import { installWorkflowFixture } from "../support/workflowFixture";
import "@nimbus/styles/theme.css";
import "@nimbus/styles/base.css";
import "@nimbus/components/settings/Settings.css";

const root = document.getElementById("root");
if (!root) throw new Error("Config harness root missing");
if (new URLSearchParams(window.location.search).get("mode") === "settings") {
  await installSettingsFixture();
  installStudioAppearance();
}
if (new URLSearchParams(window.location.search).get("mode") === "workflow") installWorkflowFixture();
if (new URLSearchParams(window.location.search).get("mode") === "remote-workspace") installRemoteWorkspaceFixture();
createRoot(root).render(new URLSearchParams(window.location.search).get("mode") === "projects" ? <ProjectsHarness /> : new URLSearchParams(window.location.search).get("mode") === "settings" ? <Settings simulator={{ loading: false, error: null, data: { state: "ready", binaryPath: "/fixture/nimbus_sim", gpu: "Fixture GPU" } }} /> : new URLSearchParams(window.location.search).get("mode") === "remote-workspace" ? <RemoteWorkspaceSettings /> : new URLSearchParams(window.location.search).get("mode") === "remote-settings"
  ? <RemoteComputeSettings /> : new URLSearchParams(window.location.search).get("mode") === "workspace-settings" ? <RemoteWorkspaceSettings /> : <ConfigEditor />);
