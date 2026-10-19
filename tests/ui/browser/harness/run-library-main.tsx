import { createRoot } from "react-dom/client";
import { App } from "@nimbus/App";
import { createPackagedAtmosphericExampleSource } from "@nimbus/lib/dataset/packagedAtmosphericExamples";
import { createLocalDatasetSource, type DatasetSourceKind } from "@nimbus/lib/dataset/datasetSources";
import { installStudioAppearance } from "@nimbus/lib/studio/studioAppearance";
import { installTauriFixture } from "../support/fixtures";
import { mockQueue } from "@nimbus/lib/core/mockData";
import "@nimbus/styles/theme.css";
import "@nimbus/styles/base.css";

const root = document.getElementById("root");
if (!root) throw new Error("Run-library harness root missing");

installStudioAppearance();
if (new URLSearchParams(window.location.search).get("startupBeast") === "1") mockQueue.push({
  ...mockQueue[0], id: "startup-beast", caseName: "beast_xl", status: "completed", startedAtMs: 1,
  endedAtMs: 1000, outputDir: "/__nimbus_browser_fixture__/startup-beast",
});

const pickerMode = new URLSearchParams(window.location.search).get("datasetPicker");
let repeatOpenCount = 0;
const fixtureDatasetOpener = pickerMode === "packaged-cm1"
  ? async (kind: DatasetSourceKind) => {
      if (kind !== "cm1_netcdf") throw new Error(`Fixture expected a CM1 source, received ${kind}`);
      // Install the native-data fixture only after App's browser fallbacks have
      // initialized; Plot Studio then exercises the real CM1 adapter path.
      installTauriFixture();
      return {
        ...createLocalDatasetSource("cm1_netcdf", "/__nimbus_browser_fixture__/cm1out.nc"),
        displayName: "CM1 command-palette fixture",
      };
    }
  : pickerMode === "packaged-nimbus" || pickerMode === "repeat-nimbus"
  ? async (kind: DatasetSourceKind) => {
      if (kind !== "nimbus_run") throw new Error(`Fixture expected a Nimbus source, received ${kind}`);
      const source = createPackagedAtmosphericExampleSource();
      if (pickerMode !== "repeat-nimbus") return source;
      repeatOpenCount += 1;
      return { ...source, displayName: `Repeat-open fixture · request ${repeatOpenCount}` };
    }
    : pickerMode === "error"
    ? async () => { throw new Error("The operating system picker is unavailable"); }
    : undefined;

createRoot(root).render(<App openDatasetSource={fixtureDatasetOpener} />);
