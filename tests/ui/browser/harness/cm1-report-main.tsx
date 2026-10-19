import { createRoot } from "react-dom/client";
import { Cm1CompatibilityReport } from "@nimbus/components/cm1/Cm1CompatibilityReport";
import { createLocalDatasetSource, type LocalDatasetSource } from "@nimbus/lib/dataset/datasetSources";
import { installTauriFixture } from "../support/fixtures";
import "@nimbus/styles/theme.css";
import "@nimbus/styles/base.css";
import "@nimbus/components/studio/PlotStudio.css";

installTauriFixture();

const longToken = "case_segment_".repeat(24);
const source = createLocalDatasetSource("cm1_netcdf", `/research/${longToken}/cm1out_000001.nc`);
const dataset: LocalDatasetSource = {
  ...source,
  displayName: `CM1_${longToken}_review`,
  caseInventory: {
    schema: "nimbus.studio.cm1-case-inventory",
    schemaVersion: 2,
    directory: `/research/${longToken}`,
    historyFiles: [source.locator.path],
    namelistInput: `/research/${longToken}/namelist.input`,
    inputSounding: `/research/${longToken}/input_sounding`,
    restartFiles: [`/research/${longToken}/cm1rst_000001`],
    statisticsFiles: [],
    otherFileCount: 0,
    nestedDirectoryCount: 0,
  },
  namelistReport: {
    schema: "nimbus.studio.cm1-namelist-report",
    schemaVersion: 1,
    path: `/research/${longToken}/namelist.input`,
    groups: [{
      name: "param1",
      assignments: [
        { key: "nx", rawValue: "64" },
        { key: "ztop", rawValue: "18000" },
        { key: "stretch_z", rawValue: ".true." },
        { key: "irdamp", rawValue: "1" },
        { key: "cm1setup", rawValue: "4" },
        { key: "irst", rawValue: "1" },
        { key: "isnd", rawValue: "7" },
        { key: `unmapped_${longToken}`, rawValue: longToken },
      ],
    }],
    assignmentCount: 8,
    values: {
      "param1.nx": "64",
      "param1.ztop": "18000",
      "param1.stretch_z": ".true.",
      "param1.irdamp": "1",
      "param1.cm1setup": "4",
      "param1.irst": "1",
      "param1.isnd": "7",
    },
    warnings: [],
  },
  soundingReport: {
    schema: "nimbus.studio.cm1-sounding-report",
    schemaVersion: 1,
    path: `/research/${longToken}/input_sounding`,
    surface: { pressureHpa: 1000, potentialTemperatureK: 300, mixingRatioGkg: 12 },
    levels: [
      { sourceLine: 2, heightM: 250, potentialTemperatureK: 301, mixingRatioGkg: 11, uMs: 5, vMs: 2 },
      { sourceLine: 3, heightM: 1_000, potentialTemperatureK: 305, mixingRatioGkg: 8, uMs: 12, vMs: 6 },
      { sourceLine: 4, heightM: 17_000, potentialTemperatureK: 410, mixingRatioGkg: 0, uMs: 20, vMs: 10 },
    ],
    topHeightM: 17_000,
    warnings: [],
  },
};

const root = document.getElementById("root");
if (!root) throw new Error("CM1 report harness root missing");
const application = createRoot(root);
const render = (current: LocalDatasetSource) => application.render(
  <main
    className="cm1-qualification-harness"
    style={{ width: "100%", height: "100%", minWidth: 0, padding: 12, overflow: "auto" }}
  >
    <Cm1CompatibilityReport dataset={current} />
  </main>,
);
Object.assign(window, { __nimbusCm1ReportHarness: { switchSource: () => render({
  ...dataset, locator: { kind: "local_path", path: "/research/new-case/cm1out.nc" },
  displayName: "New CM1 source", soundingReport: { ...dataset.soundingReport!, path: "/research/new-case/input_sounding" },
}) } });
render(dataset);
