import { expect, test } from "@playwright/test";

test("raw checkpoint inspection follows the active YAML and stale validation cannot replace newer evidence", async ({ page }) => {
  test.setTimeout(45_000);
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto("/harness/config.html?mode=workflow");
  await expect(page.getByRole("button", { name: "Validate", exact: true })).toBeEnabled();
  await page.getByRole("radio", { name: "YAML", exact: true }).click();
  const editor = page.getByRole("textbox", { name: "Config YAML editor", exact: true });
  await editor.fill("schema: nimbus.v1\ntime:\n  run_time: 30\n  output_interval: 10\nextensions:\n  checkpoint:\n    resume_from: /__workflow_fixture__/saved.nimbus\n");
  await page.waitForFunction(() => window.__nimbusWorkflowFixture!.snapshot().calls.some(value => value.command === "inspect_checkpoint"));
  await expect(page.getByRole("button", { name: "Run simulation", exact: true })).toBeEnabled();
  await page.evaluate(() => window.__nimbusWorkflowFixture!.holdCheckpoint(true));
  await page.getByRole("button", { name: "Validate", exact: true }).click();
  await page.waitForFunction(() => window.__nimbusWorkflowFixture!.snapshot().inspections.length === 1);
  await expect(page.getByRole("button", { name: "Validating…", exact: true })).toBeDisabled();
  const fresh = "schema: nimbus.v1\ntime:\n  run_time: 40\n  output_interval: 10\n";
  await editor.fill(fresh);
  await expect(page.getByRole("button", { name: "Validate", exact: true })).toBeEnabled();
  await page.getByRole("button", { name: "Validate", exact: true }).click();
  await expect(page.locator(".action-dock-evidence").filter({ hasText: "Current exact-binary fixture rejection" })).toBeVisible();
  await page.evaluate(() => window.__nimbusWorkflowFixture!.settleCheckpoint("Old checkpoint result must stay stale"));
  await expect(page.locator("main")).not.toContainText("Old checkpoint result must stay stale");
  await expect(page.getByRole("button", { name: "Validate", exact: true })).toBeEnabled();
  const snapshot = await page.evaluate(() => window.__nimbusWorkflowFixture!.snapshot());
  expect(snapshot.calls.filter(value => value.command === "validate_config_contents").map(value => value.args.contents)).toEqual([fresh]);
  expect(snapshot.calls.filter(value => value.command === "spawn_sim")).toHaveLength(0);
  expect(errors).toEqual([]);
});

test("a removed compute profile remains explicit and local recovery requires selection", async ({ page }) => {
  test.setTimeout(45_000);
  await page.addInitScript(() => localStorage.setItem("nimbus-studio:remote-compute:v1", JSON.stringify({ version: 1, profiles: [{
    id: "connection-fixture", name: "Linux workstation", host: "fixture-host", binaryPath: "/opt/nimbus/nimbus_sim", workDirectory: "/home/research/nimbus",
  }] })));
  await page.goto("/harness/config.html");
  const target = page.getByRole("combobox", { name: "Compute target", exact: true });
  await target.selectOption("connection-fixture");
  await page.evaluate(() => {
    localStorage.setItem("nimbus-studio:remote-compute:v1", JSON.stringify({ version: 1, profiles: [] }));
    window.dispatchEvent(new CustomEvent("nimbus-studio:remote-compute-change"));
  });
  await expect(target).toHaveValue("connection-fixture");
  await expect(target.locator("option:checked")).toHaveText("Unavailable saved connection");
  await expect(page.getByRole("alert").filter({ hasText: "selected remote connection is unavailable" })).toBeVisible();
  await target.selectOption("local");
  await expect(target).toHaveValue("local");
  await expect(target.locator("option")).toHaveCount(1);
});

test("remote compute profiles persist while each configuration opens on this computer", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto("/harness/config.html?mode=remote-settings");
  await page.getByRole("textbox", { name: "Connection name", exact: true }).fill("Research workstation");
  await page.getByRole("textbox", { name: "SSH host", exact: true }).fill("workstation");
  const binary = page.getByRole("textbox", { name: "Remote simulator path", exact: true });
  await binary.fill("relative/nimbus_sim");
  await page.getByRole("textbox", { name: "Remote work directory", exact: true }).fill("/home/research/nimbus");
  await page.getByRole("button", { name: "Save connection", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("absolute Linux paths");
  await binary.fill("/opt/nimbus/bin/nimbus_sim");
  await page.locator(".settings-remote__inputs > summary").first().click();
  await page.getByRole("button", { name: "Add input file", exact: true }).click();
  await page.getByRole("textbox", { name: "Local input file 1", exact: true }).fill("/data/input_sounding");
  await page.getByRole("textbox", { name: "Remote input file 1", exact: true }).fill("inputs/input_sounding");
  await page.getByRole("button", { name: "Save connection", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("Connection saved");
  await expect(page.getByRole("button", { name: "Check connection", exact: true })).toBeDisabled();
  await page.reload();
  await expect(page.getByRole("textbox", { name: "SSH host", exact: true })).toHaveValue("workstation");
  await page.locator(".settings-remote__inputs > summary").first().click();
  await expect(page.getByRole("textbox", { name: "Local input file 1", exact: true })).toHaveValue("/data/input_sounding");
  await page.getByRole("textbox", { name: "Connection name", exact: true }).fill("Linux workstation");
  await page.getByRole("button", { name: "Save connection", exact: true }).click();
  const id = await page.getByRole("combobox", { name: "Saved connection", exact: true }).inputValue();
  await page.goto("/harness/config.html");
  const target = page.getByRole("combobox", { name: "Compute target", exact: true });
  await expect(target).toHaveValue("local");
  await expect(target.locator(`option[value="${id}"]`)).toHaveText("Linux workstation · workstation");
  await target.selectOption(id);
  await expect(page.getByText("Runs on workstation; plots remain on this computer.")).toBeVisible();
  await expect(page.getByText("Validation and simulation runs are available in the desktop app.", { exact: true })).toBeVisible();
  await expect(page.getByText("Remote compute requires Nimbus Studio desktop.", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Run simulation", exact: true })).toBeDisabled();
  await expect(target).toHaveValue(id);
  await page.reload();
  await expect(page.getByRole("combobox", { name: "Compute target", exact: true })).toHaveValue("local");
  await page.goto("/harness/config.html?mode=remote-settings");
  await page.getByRole("button", { name: "Remove connection", exact: true }).click();
  await expect(page.getByRole("status")).toHaveText("Connection removed.");
  await page.goto("/harness/config.html");
  await expect(page.getByRole("combobox", { name: "Compute target", exact: true }).locator("option")).toHaveCount(1);
  expect(errors).toEqual([]);
});

test("reviewed CM1 handoff restores the bounded smoke baseline without clipboard work", async ({ page }) => {
  await page.addInitScript(() => {
    const now = "2026-07-20T00:00:00.000Z";
    const yaml = "schema: nimbus.v1\ntime:\n  run_time: 30\n  output_interval: 10\ninitial_conditions:\n  sounding:\n    type: file\n    file_path: /work/cm1-sounding.txt\n";
    let hash = 0xcbf29ce484222325n;
    for (const byte of new TextEncoder().encode(yaml)) { hash ^= BigInt(byte); hash = BigInt.asUintN(64, hash * 0x100000001b3n); }
    localStorage.setItem("nimbus:cm1-guided-handoff:v2", JSON.stringify({
      schema: "nimbus.studio.cm1-guided-handoff",
      schemaVersion: 2,
      id: "cm1-handoff-browser",
      updatedAt: now,
      cm1Source: {
        schema: "nimbus.studio.dataset-source",
        schemaVersion: 1,
        kind: "cm1_netcdf",
        sourceModel: "cm1",
        nativeFormat: "netcdf",
        locator: { kind: "local_paths", path: "/case/cm1out_000001.nc", paths: ["/case/cm1out_000001.nc", "/case/cm1out_000002.nc"] },
        displayName: "2 netCDF history files",
        readOnly: true,
      },
      evidence: {
        historyFileCount: 2,
        soundingSourcePath: "/case/input_sounding",
        projectedLevelCount: 72,
        mappingCounts: { ready: 9, needsDecision: 3, different: 2, unsupported: 4 },
        mappingBlockers: ["3 mapping decisions remain"],
        mappingBlockerTotal: 1,
        mappingBlockersTruncated: false,
        reviewedAt: now,
      },
      smoke: {
        soundingPath: "/work/cm1-sounding.txt",
        configPath: "/work/cm1-smoke.yaml",
        yaml,
        configurationDigest: `fnv1a64:${hash.toString(16).padStart(16, "0")}`,
        durationS: 30,
        outputIntervalS: 10,
        catalogFingerprint: "browser-mock-catalog",
        executablePath: "/bin/nimbus_sim",
        initializationExecutablePath: "/bin/nimbus_sim",
        stagedAt: now,
        consumedAt: null,
      },
      run: { status: "not_started", runId: null, outputDir: null, pendingReason: null },
      comparisonOpenedAt: null,
    }));
  });
  await page.goto("/harness/config.html");
  const guide = page.locator(".config-editor__cm1-handoff");
  await expect(guide.getByText("Reviewed sounding · short Nimbus smoke test")).toBeVisible();
  await expect(guide.getByText("30 s run · 10 s output")).toBeVisible();
  await expect(page.getByLabel("Experiment configuration")).toHaveValue("cm1-sounding-smoke");
  await expect(guide.getByText(/2 CM1 history files retained for comparison/)).toBeVisible();
  await expect(guide.getByRole("button", { name: "Restore staged baseline" })).toBeVisible();
});

test("configuration viewport exposes the scientific-instrument visual contract", async ({ page }) => {
  await page.goto("/harness/config.html");
  await expect(page.getByText("Loading simulator schemes…")).toBeHidden();

  await expect(page.getByText("Experiment configuration")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Domain and run envelope" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Configuration decisions" })).toBeVisible();
  const draftBar = page.getByRole("region", { name: "Configuration drafts" });
  await expect(draftBar.getByLabel("Configuration draft")).toHaveValue(/draft_/);
  await expect(draftBar.getByText("File has unsaved changes", { exact: true })).toBeVisible();

  const basicDetail = page.getByRole("button", { name: "Basic", exact: true });
  const advancedDetail = page.getByRole("button", { name: "Advanced", exact: true });
  await expect(basicDetail).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator(".config-editor__sections .collapsible")).toHaveCount(12);
  await expect(page.locator(".collapsible", { hasText: "Numerical diffusion" })).toHaveCount(0);
  await advancedDetail.click();
  await expect(advancedDetail).toHaveAttribute("aria-pressed", "true");
  // One collapsible per SCHEMA_SECTIONS entry (lib/config/configSchema.ts).
  await expect(page.locator(".config-editor__sections .collapsible")).toHaveCount(24);
  await expect(page.locator(".collapsible", { hasText: "Numerical diffusion" })).toHaveCount(1);
  // Terrain is offered with its non-flat shapes gated by the binary catalog,
  // not hidden -- the Cartesian solvers reject a non-flat terrain.scheme.
  await expect(page.locator(".collapsible", { hasText: "Terrain" })).toHaveCount(1);
  const radar = page.locator(".collapsible", { hasText: "Radar products" });
  await expect(radar).toHaveCount(1);
  await radar.locator(".collapsible__head").click();
  await radar.getByRole("button", { name: "Enable radar diagnostics" }).click();
  await expect(radar.getByRole("checkbox", { name: "Reflectivity (dBZ)" })).toBeChecked();
  await expect(radar.getByRole("checkbox", { name: "Radial velocity" })).toBeChecked();
  await radar.getByRole("checkbox", { name: "Specific differential phase (KDP)" }).check();
  await expect(radar.getByRole("combobox", { name: "Polarimetric operator" }))
    .toHaveValue("zhang2021_rain_mp");

  const launchDock = page.locator(".config-editor__launch-dock");
  await expect(launchDock.getByText("Configuration editing ready", { exact: true })).toBeVisible();
  await expect(launchDock.getByRole("definition")).toHaveCount(3);
  await expect(launchDock.getByRole("button", { name: "Validate" })).toBeDisabled();
  await expect(launchDock.getByRole("button", { name: "Run simulation" })).toBeVisible();
  const dockBeforeScroll = await launchDock.boundingBox();
  await page.locator(".config-editor__body").evaluate((body) => { body.scrollTop = body.scrollHeight; });
  const dockAfterScroll = await launchDock.boundingBox();
  expect(dockBeforeScroll).not.toBeNull();
  expect(dockAfterScroll).not.toBeNull();
  expect(dockAfterScroll!.y).toBe(dockBeforeScroll!.y);

  const cm1Guide = page.locator(".cm1-orientation-guide");
  await expect(cm1Guide).toHaveCount(0);
  await page.getByRole("button", { name: "CM1 crosswalk" }).click();
  await expect(cm1Guide.getByText("Coming from CM1?", { exact: true })).toBeVisible();
  await expect(cm1Guide.getByText("Familiar controls now lead each maintained crosswalk.")).toBeVisible();
  await cm1Guide.locator("summary").click();
  await expect(cm1Guide.getByText(/do not claim solver or scheme equivalence/)).toBeVisible();
  await expect(cm1Guide.getByText(/CM1 execution, restart consumption/)).toBeVisible();

  const gridSection = page.locator(".collapsible", { hasText: "Domain and mesh" }).first();
  await expect(gridSection.getByText("CM1 Domain and mesh", { exact: true })).toBeVisible();
  await expect(gridSection.locator(".collapsible__canonical-id")).toHaveText("grid");
  await expect(gridSection.getByText("nx · ny · nz · dx · dy · dz · ztop · stretch_z")).toBeVisible();
  const nxRow = gridSection.locator(".schema-form__row", { hasText: "x cells (NX)" });
  await expect(nxRow.getByRole("note", { name: /CM1 nx: x grid points/ })).toBeVisible();
  await expect(nxRow.getByText("CM1 nx", { exact: true })).toBeVisible();
  await expect(nxRow.getByText("same scalar", { exact: true })).toBeVisible();
  await expect(nxRow.getByText("Nimbus grid.nx", { exact: true })).toBeVisible();

  const hierarchy = await page.evaluate(() => {
    const evidence = document.querySelector<HTMLElement>(".diagnostics");
    const decisions = document.querySelector<HTMLElement>(".config-editor__sections");
    const rootStyle = getComputedStyle(document.documentElement);
    const bodyStyle = getComputedStyle(document.body);
    if (!evidence || !decisions) throw new Error("configuration hierarchy missing");
    return {
      evidenceTop: evidence.getBoundingClientRect().top,
      decisionsTop: decisions.getBoundingClientRect().top,
      bodyFont: bodyStyle.fontFamily,
      focusRing: rootStyle.getPropertyValue("--color-focus-ring").trim(),
      background: bodyStyle.backgroundColor,
    };
  });
  expect(hierarchy.evidenceTop).toBeLessThan(hierarchy.decisionsTop);
  expect(hierarchy.bodyFont).toMatch(/-apple-system|system-ui|Segoe UI/);
  expect(hierarchy.focusRing).toContain("oklch");
  expect(hierarchy.background).not.toBe("rgb(0, 0, 0)");

  const experimentName = page.getByLabel("Experiment configuration");
  await experimentName.focus();
  const focusStyle = await experimentName.evaluate((element) => {
    const style = getComputedStyle(element);
    return { width: style.outlineWidth, style: style.outlineStyle };
  });
  expect(focusStyle).toEqual({ width: "2px", style: "solid" });

  const executionPolicy = page.locator(".collapsible", {
    hasText: "Execution policy",
  });
  await executionPolicy.locator(".collapsible__head").click();
  await expect(
    executionPolicy.getByText(
      "Choose whether Nimbus may recover for exploration or must fail closed for qualified evidence.",
    ),
  ).toBeVisible();
  await executionPolicy
    .getByRole("button", { name: "More information about Run mode" })
    .click();
  await expect(
    executionPolicy.getByText(
      /Qualified mode aborts on scheme exceptions or state repairs/,
    ),
  ).toBeVisible();
  const runMode = executionPolicy.locator(".schema-form select").first();
  await runMode.selectOption("qualified");
  await expect(executionPolicy.getByText("I reviewed the scheme maturity limits"))
    .toHaveCount(0);
  await runMode.selectOption("exploratory");
  await runMode.selectOption("qualified");
  await expect(runMode).toHaveValue("qualified");

  const checkpoints = page.locator(".collapsible", {
    hasText: "Checkpoints and continuation",
  });
  await checkpoints.locator(".collapsible__head").click();
  const inspector = checkpoints.locator(".checkpoint-inspector");
  await expect(inspector.getByRole("heading", { name: "Checkpoint evidence" })).toBeVisible();
  await expect(inspector.getByText("No source", { exact: true })).toBeVisible();
  await expect(
    inspector.getByText("CM1 restart files are not opened here."),
  ).toBeVisible();
  await expect(
    inspector.getByRole("button", { name: "Inspect checkpoint…" }),
  ).toBeDisabled();
  await expect(
    inspector.getByText("Studio never deletes checkpoint or sidecar files."),
  ).toBeVisible();
  // Form mode has no unchecked resume-path text input: desktop picker + native
  // inspection own acceptance, while YAML remains the explicit advanced view.
  await expect(checkpoints.locator('input[placeholder*="checkpoint_step"]')).toHaveCount(0);

  await page.setViewportSize({ width: 520, height: 900 });
  const responsive = await page.evaluate(() => {
    const heading = document.querySelector<HTMLElement>(".config-editor__decision-heading");
    const row = document.querySelector<HTMLElement>(".schema-form__row");
    const path = document.querySelector<HTMLElement>(".config-editor__path");
    const cm1Legend = document.querySelector<HTMLElement>(".cm1-orientation-guide__body dl");
    if (!heading || !row || !path || !cm1Legend) throw new Error("responsive configuration elements missing");
    return {
      headingDirection: getComputedStyle(heading).flexDirection,
      fieldColumns: getComputedStyle(row).gridTemplateColumns.split(" ").length,
      pathAlignment: getComputedStyle(path).textAlign,
      cm1LegendColumns: getComputedStyle(cm1Legend).gridTemplateColumns.split(" ").length,
    };
  });
  expect(responsive).toEqual({
    headingDirection: "column",
    fieldColumns: 1,
    pathAlignment: "left",
    cm1LegendColumns: 1,
  });
  await expect(launchDock).toBeVisible();
  await expect(launchDock.getByRole("button", { name: "Validate" })).toBeVisible();
  await expect(launchDock.getByRole("button", { name: "Run simulation" })).toBeVisible();
});

test("configuration branches preserve independent working documents across reload", async ({ page }) => {
  await page.goto("/harness/config.html");
  await expect(page.getByText("Loading simulator schemes…")).toBeHidden();
  const draftBar = page.getByRole("region", { name: "Configuration drafts" });
  const picker = draftBar.getByLabel("Configuration draft");
  const experimentName = page.getByLabel("Experiment configuration");

  await experimentName.fill("workflow-control");
  await draftBar.getByRole("button", { name: "Branch draft" }).click();
  await expect(picker.locator("option")).toHaveCount(2);
  await expect(experimentName).toHaveValue("workflow-control-branch");
  await experimentName.fill("workflow-sensitivity");
  await expect(draftBar.getByText("2/16 local", { exact: true })).toBeVisible();

  const originalId = await picker.locator("option").filter({ hasText: "workflow-control" }).last().getAttribute("value");
  expect(originalId).not.toBeNull();
  await picker.selectOption(originalId!);
  await expect(experimentName).toHaveValue("workflow-control");

  await page.reload();
  await expect(page.getByLabel("Experiment configuration")).toHaveValue("workflow-control");
  await expect(page.getByLabel("Configuration draft").locator("option")).toHaveCount(2);
});

test("runtime-published module variants are selectable and serialize canonically", async ({ page }) => {
  test.setTimeout(60_000);
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto("/harness/config.html");

  await expect(page.getByText("Loading simulator schemes…")).toBeHidden();
  await expect(page.locator(".config-editor__launch-dock").getByText("Configuration editing ready")).toBeVisible();
  await expect(page.getByRole("button", { name: "Run simulation" })).toBeDisabled();
  await page.getByRole("button", { name: "Advanced", exact: true }).click();
  const openSection = async (name: string) => {
    const target = page.locator(".collapsible").filter({
      has: page.locator(".collapsible__label").filter({ hasText: name }),
    }).first();
    const heading = target.locator(".collapsible__head");
    if (await heading.getAttribute("aria-expanded") !== "true") await heading.click();
    return target;
  };

  for (const module of [
    { section: "Turbulence (SGS)", ids: ["none", "smagorinsky", "tke"] },
    { section: "Numerical diffusion", ids: ["explicit", "implicit"] },
    { section: "Boundary layer (PBL)", ids: ["none", "slab", "ysu", "mynn"] },
    { section: "Radiation", ids: ["none", "simple_grey", "rrtmg"] },
    { section: "Chaos / ensemble", ids: ["none", "initial_conditions", "boundary_layer", "full_stochastic"] },
  ]) {
    const moduleSection = await openSection(module.section);
    const moduleSelector = moduleSection.locator(".schema-form select").first();
    for (const id of module.ids) {
      await moduleSelector.selectOption(id);
      await expect(moduleSelector).toHaveValue(id);
    }
  }

  const radiation = await openSection("Radiation");
  const rrtmg = radiation.locator('select option[value="rrtmg"]');
  await expect(rrtmg).toBeEnabled();
  await expect(radiation.locator(".schema-form select").first()).toHaveValue("rrtmg");

  const output = await openSection("Output");
  const compute = await openSection("Compute backend");
  const computeBackend = compute.locator(".schema-form select").first();
  await computeBackend.selectOption("vulkan");
  const cuda = compute.locator(".schema-form__row").filter({ hasText: "NVIDIA CUDA assist" }).locator(".toggle");
  await cuda.click();
  await expect(cuda).toHaveAttribute("aria-pressed", "true");
  await computeBackend.selectOption("cpu");
  await expect(cuda).toHaveCount(0);
  await computeBackend.selectOption("vulkan");
  await expect(cuda).toHaveAttribute("aria-pressed", "false");
  await cuda.click();
  const outputFormat = output.locator(".schema-form select").nth(1);
  await outputFormat.selectOption("netcdf");
  await expect(outputFormat).toHaveValue("netcdf");
  await expect(output.getByText("Provided by libnetcdf in the exact simulator binary.")).toBeVisible();
  await output.getByLabel("Lossless compression", { exact: true }).fill("5");
  await output.getByLabel("Read chunk target", { exact: true }).fill("512");
  await output.locator(".schema-form__row", { hasText: "Improve compression" }).locator(".toggle").click();
  await outputFormat.selectOption("npy_3d");
  await expect(output.getByLabel("Lossless compression", { exact: true })).toHaveCount(0);
  await outputFormat.selectOption("netcdf");
  await expect(output.getByLabel("Lossless compression", { exact: true })).toHaveValue("5");
  await expect(output.getByLabel("Read chunk target", { exact: true })).toHaveValue("512");


  const turbulence = await openSection("Turbulence (SGS)");
  const turbulenceSelector = turbulence.locator(".schema-form select").first();
  await turbulenceSelector.selectOption("none");
  await turbulence.getByRole("button", { name: "Enable" }).click();
  await expect(turbulenceSelector).toHaveValue("smagorinsky");

  const section = await openSection("Microphysics");
  const selector = section.locator(".schema-form select");

  for (const id of ["none", "kessler", "lin", "thompson", "milbrandt"]) {
    await selector.selectOption(id);
    await expect(selector).toHaveValue(id);
  }

  const hailProcesses = section.locator(".schema-form__row", {
    // Catalog displayName for microphysics.milbrandt.hail_processes.
    hasText: "New hail production",
  });
  await expect(hailProcesses).toBeVisible();
  const hailProcessesToggle = hailProcesses.locator(".toggle");
  await hailProcessesToggle.click();
  await expect(hailProcessesToggle).toHaveAttribute("aria-pressed", "false");

  await selector.selectOption("kessler");
  const accretion = section.locator(".schema-form__row", {
    hasText: "Rain accretion coefficient",
  });
  await accretion.locator('input[type="number"]').fill("1.6");
  await selector.selectOption("lin");
  await expect(accretion.locator('input[type="number"]')).toHaveValue("2.2");

  await page.getByRole("radio", { name: "YAML" }).click();
  const yamlEditor = page.getByLabel("Config YAML editor");
  await expect(yamlEditor).toHaveValue(/microphysics:\n  scheme: lin/);
  await expect(yamlEditor).toHaveValue(/netcdf_deflate_level: 5/);
  await expect(yamlEditor).toHaveValue(/cuda_assist: true/);
  await expect(yamlEditor).toHaveValue(/netcdf_shuffle: false/);
  await expect(yamlEditor).toHaveValue(/netcdf_target_chunk_kib: 512/);

  await expect(yamlEditor).toHaveValue(/lin:\n    accretion_coefficient: 2\.2/);
  await expect(yamlEditor).not.toHaveValue(/\n  kessler:/);

  // Raw YAML preserves the exact physical scheme identity and removes
  // parameters belonging to the old scheme.
  const linYaml = await yamlEditor.inputValue();
  await yamlEditor.fill(linYaml.replace(
    "microphysics:\n  scheme: lin",
    "microphysics:\n  scheme: milbrandt",
  ));
  await page.getByRole("radio", { name: "Form" }).click();
  await expect(selector).toHaveValue("milbrandt");
  await expect(hailProcesses).toBeVisible();
  await page.getByRole("radio", { name: "YAML" }).click();
  await expect(yamlEditor).toHaveValue(/microphysics:\n  scheme: milbrandt/);
  await expect(yamlEditor).not.toHaveValue(/\n  lin:/);
  // Leaving milbrandt dropped its parameters, so returning re-seeds them from
  // the catalog rather than resurrecting the earlier toggle-off: the descriptor
  // default is true (include/microphysics/scheme_descriptor.hpp).
  await expect(yamlEditor).toHaveValue(/hail_processes: true/);

  // Browser preview persistence mirrors desktop save/read closely enough to
  // prove catalog selections survive a document reload. User saves are not
  // added to the immutable packaged-template catalog.
  await page.getByRole("radio", { name: "Form" }).click();
  await turbulenceSelector.selectOption("tke");
  await page.locator(".config-editor__name-input").fill("student");
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await expect(page.getByRole("button", { name: "Saved ✓" })).toBeVisible();

  await page.reload();
  await expect(page.getByText("Loading simulator schemes…")).toBeHidden();
  await expect(page.locator('.config-editor__template-picker option[value="student.yaml"]')).toHaveCount(0);
  const reopenedMicrophysics = await openSection("Microphysics");
  await expect(reopenedMicrophysics.locator(".schema-form select").first()).toHaveValue("milbrandt");
  const reopenedTurbulence = await openSection("Turbulence (SGS)");
  await expect(reopenedTurbulence.locator(".schema-form select").first()).toHaveValue("tke");
  const reopenedCompute = await openSection("Compute backend");
  await expect(reopenedCompute.getByRole("button", { name: "NVIDIA CUDA assist", exact: true }))
    .toHaveAttribute("aria-pressed", "true");

  expect(pageErrors).toEqual([]);
});
