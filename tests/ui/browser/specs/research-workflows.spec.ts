import {expect,test,type Page} from "@playwright/test";
let pageErrors:string[]=[];
test.beforeEach(async({page})=>{pageErrors=[];page.on("pageerror",error=>pageErrors.push(error.message));});
test.afterEach(()=>expect(pageErrors).toEqual([]));
async function open(page:Page,tab:string){await page.goto("/harness/index.html?mode=research");await page.getByRole("button",{name:tab,exact:true}).click();}
async function data(page:Page,id:string){return page.evaluate(id=>window.__nimbusResearch.panel(id),id);}
async function number(page:Page,label:string,value:string){const input=page.getByLabel(label,{exact:true});await input.fill(value);await input.press("Enter");}

test("linked sections preserve full height, native geometry and a shared probe without new reads",async({page})=>{
  await open(page,"Linked sections");
  await page.getByRole("combobox",{name:"Contour / second field",exact:true}).selectOption("theta_prime");
  await page.getByRole("button",{name:"Load linked sections",exact:true}).click();
  await expect(page.locator('[data-panel-id="section-0"]')).toBeVisible();
  const panels=await data(page,"section-0") as {z:number[][];y:number[]}[];
  expect(panels[0].z).toHaveLength(8);expect(panels[0].z[0]).toHaveLength(4);
  expect(panels[0].z[0][0]).toBeCloseTo(2.25,6);expect(panels[0].z[7][3]).toBeCloseTo(14.75,6);expect(panels[0].y).toEqual([.25,.75,1.25,1.75,2.25,2.75,3.25,3.75]);
  const locator=await data(page,"locator") as {z:number[][];x:number[];y:number[]}[];
  expect(locator[0].z).toHaveLength(3);expect(locator[0].z[0]).toHaveLength(4);expect(locator[0].x).toEqual([.5,1.5,2.5,3.5]);expect(locator[0].y).toEqual([.75,2.25,3.75]);
  const before=await page.evaluate(()=>window.__nimbusResearch.calls().length);
  await number(page,"Shared probe fraction","1");
  await expect.poll(async()=>((await data(page,"profile-0")) as {x:number[]}[])[0].x[7]).toBeCloseTo(14.75,6);
  expect(await page.evaluate(()=>window.__nimbusResearch.calls().length)).toBe(before);
  expect(await page.evaluate(()=>window.__nimbusResearch.calls().filter(c=>c.command==="read_field_bytes").length)).toBe(0);
});

test("environmental overlays retain saved profiles, explicit wind reference and numerical layer coverage",async({page})=>{
  await open(page,"Environment");await page.getByRole("combobox",{name:"Shared saved frame",exact:true}).selectOption("1");
  await page.getByLabel("Storm-relative winds",{exact:true}).check();await number(page,"Storm u (m/s, Earth)","10");await number(page,"Storm v (m/s, Earth)","-5");
  await page.getByRole("button",{name:"Load environmental profiles",exact:true}).click();
  await expect(page.locator('[data-panel-id="hodographs"]')).toBeVisible();
  const curves=await data(page,"hodographs") as {x:number[];y:number[]}[];
  expect(curves).toHaveLength(3);expect(curves[0].x[0]).toBeCloseTo(-9.5);expect(curves[2].x[0]).toBeCloseTo(-7.5);expect(curves[2].y[0]).toBeCloseTo(4.75);
  expect(((await data(page,"difference")) as {x:number[]}[])[0].x).toEqual(Array(8).fill(2));
  await expect(page.locator('[data-panel-id="soundings"]')).toBeVisible();
});

test("cold-pool footprint uses physical area and full-height context with native-time histories",async({page},info)=>{
  await open(page,"Cold pools");await page.getByLabel("Build saved-time histories",{exact:true}).check();
  await page.getByRole("button",{name:"Measure cold-pool footprint",exact:true}).click();
  await expect(page.locator('[data-panel-id="area-history"]')).toBeVisible();
  expect(((await data(page,"area-history")) as {y:number[]}[])[0].y).toEqual([9,9,9]);
  const anomaly=await data(page,"anomaly-history") as {y:number[]}[];expect(anomaly[0].y).toEqual([-3,-4,-5]);expect(anomaly[1].y).toEqual([-3,-4,-5]);
  expect(((await data(page,"vertical-section")) as {z:number[][]}[])[0].z).toHaveLength(8);
  await expect(page.locator(".research-notes")).toContainText("6 selected cells");
  await page.getByRole("button",{name:"Hide analysis controls",exact:true}).click();
  await page.locator(".research-panels").screenshot({path:info.outputPath("cold-pool-panels.png")});
});

test("hydrometeor CFADs, joint distributions, mass and precipitation stay tied to saved inputs",async({page})=>{
  await open(page,"Hydrometeors / radar");await page.getByRole("combobox",{name:"Primary distribution field",exact:true}).selectOption("qr");
  await number(page,"Value maximum","0.004");await page.getByRole("combobox",{name:"Saved dry-air density for mass",exact:true}).selectOption("rho_d");
  await page.getByRole("combobox",{name:"Saved precipitation history",exact:true}).selectOption("precip_rate");
  await page.getByRole("button",{name:"Build hydrometeor diagnostics",exact:true}).click();
  await expect(page.locator('[data-panel-id="cfad"]')).toBeVisible();
  const rows=((await data(page,"cfad")) as {z:(number|null)[][]}[])[0].z;
  const sums=rows.filter(row=>row.some(value=>value!=null&&Number.isFinite(value))).map(row=>row.reduce<number>((a,b)=>a+(b??0),0));
  expect(sums.length).toBeGreaterThan(0);for(const sum of sums)expect(sum).toBeCloseTo(100,8);
  const joint=((await data(page,"joint")) as {z:number[][]}[])[0].z;expect(joint.flat().reduce((a,b)=>a+b,0)).toBe(96);
  await expect(page.locator(".research-notes")).toContainText("hydrometeor mass:");
  const text=await page.locator(".research-notes").innerText();const mass=Number(/mass: ([\d.e+-]+) kg/.exec(text)![1]);expect(mass).toBeCloseTo(144e6,-1);
  expect(((await data(page,"precipitation")) as {y:number[]}[])[0].y).toEqual([1,2,3]);
});

test("parcel cohorts select actual paths and compare only the three saved tendencies",async({page})=>{
  await open(page,"Parcel cohorts");await page.getByRole("button",{name:"Analyze parcel cohort",exact:true}).click();
  await expect(page.locator('[data-panel-id="paths"]')).toBeVisible();
  expect((await data(page,"paths"))).toHaveLength(2);
  const attribution=await data(page,"attribution") as {y:number[]}[];expect(attribution[1].y[3]).toBeCloseTo(.018,7);expect(attribution[0].y[3]).toBeCloseTo(.018,7);
  await expect(page.locator(".research-notes")).toContainText("do not establish a complete vorticity budget");
  await number(page,"Origin maximum (km)","1.5");await page.getByRole("button",{name:"Analyze parcel cohort",exact:true}).click();
  await expect.poll(async()=>(await data(page,"paths")).length).toBe(1);
  expect(await page.evaluate(()=>window.__nimbusResearch.calls().filter(c=>c.command==="read_trajectories").length)).toBe(1);
});

test("native-grid studies admit different resolutions without resampling or losing physical volume",async({page})=>{
  await open(page,"Resolution studies");await page.getByRole("combobox",{name:"Comparison field",exact:true}).selectOption("qr");await number(page,"Threshold (field units)","0");await number(page,"Distribution maximum","0.004");
  await page.getByRole("button",{name:"Compare native-grid responses",exact:true}).click();await expect(page.locator('[data-panel-id="threshold"]')).toBeVisible();
  const volumes=((await data(page,"threshold")) as {y:number[]}[])[0].y;expect(volumes).toEqual([72,72]);
  const response=((await data(page,"response")) as {x:number[];y:number[]}[])[0];expect(response.x[0]).not.toBe(response.x[1]);for(const value of response.y)expect(value).toBeCloseTo(.002,8);
  await expect(page.locator(".research-notes")).toContainText("no unequal-grid cellwise subtraction");
});

test("pressure workflow exposes all saved diagnostics and linked profiles",async({page})=>{
  await open(page,"Pressure");await page.getByLabel("Include every available pressure / buoyancy / PGF / w diagnostic",{exact:true}).check();
  await page.getByRole("button",{name:"Load pressure panels",exact:true}).click();await expect(page.locator('[data-panel-id="section-5"]')).toBeVisible();
  expect(((await data(page,"section-0")) as {z:number[][]}[])[0].z[7][3]).toBeCloseTo(3.5,6);
  await expect(page.locator(".research-notes")).toContainText("Their sum is not asserted to equal compressible p′");
});

test("research cancellation and changed settings cannot publish stale results",async({page})=>{
  await open(page,"Cold pools");await page.getByLabel("Include linked vertical section",{exact:true}).uncheck();
  await page.getByRole("button",{name:"Measure cold-pool footprint",exact:true}).click();await expect(page.locator('[data-panel-id="footprint"]')).toBeVisible();
  await page.getByRole("combobox",{name:"Shared saved frame",exact:true}).selectOption("1");
  await expect(page.getByRole("button",{name:"Export research figure package",exact:true})).toBeDisabled();
  await page.evaluate(()=>window.__nimbusResearch.delay(300));await page.getByRole("button",{name:"Measure cold-pool footprint",exact:true}).click();await page.getByRole("button",{name:"Cancel analysis",exact:true}).click();
  await expect(page.getByRole("status").filter({hasText:"Cancelled"})).toBeVisible();
  await expect(page.getByRole("button",{name:"Export research figure package",exact:true})).toBeDisabled();
});

test("research figures bind every panel and numeric table, replay ranges, and reject changed sources",async({page},info)=>{
  await open(page,"Cold pools");await page.getByLabel("Include linked vertical section",{exact:true}).uncheck();
  await page.getByRole("button",{name:"Measure cold-pool footprint",exact:true}).click();await expect(page.locator('[data-panel-id="footprint"]')).toBeVisible();await data(page,"footprint");
  await page.getByLabel("Figure title",{exact:true}).fill("Cold pool evidence");await page.getByLabel("Citation / study definition",{exact:true}).fill("Manufactured native-cell validation fixture");
  await page.evaluate(()=>window.__nimbusResearch.savePath("/__nimbus_browser_fixture__/research-figure.zip"));
  await page.getByRole("button",{name:"Export research figure package",exact:true}).click();
  await expect(page.getByRole("status").filter({hasText:"Figure package saved"})).toBeVisible();
  const array=await page.evaluate(()=>window.__nimbusResearch.file("/__nimbus_browser_fixture__/research-figure.zip"));expect(array).not.toBeNull();
  const {readStoreZip}=await import("../../../../ui/src/lib/platform/storeZip");const {createHash}=await import("node:crypto");
  const entries=readStoreZip(Uint8Array.from(array!));const recipe=JSON.parse(new TextDecoder().decode(entries.get("recipe.json")!));const provenance=JSON.parse(new TextDecoder().decode(entries.get("provenance.json")!));
  expect(recipe.sourceBindings.frames[0].generation).toBe("research-native-0");expect(provenance.panels).toHaveLength(2);
  for(const artifact of provenance.artifacts)expect(createHash("sha256").update(entries.get(artifact.filename)!).digest("hex")).toBe(artifact.sha256);
  expect(new TextDecoder().decode(entries.get("tables/1-footprint-summary.csv")!)).toContain("0,0,-2,6,12,0,9,-3,-3");
  await info.attach("research-figure.zip",{body:Buffer.from(array!),contentType:"application/zip"});
  await number(page,"Cold threshold (field units)","-4");
  await page.locator('input[type="file"]').setInputFiles({name:"research-figure.zip",mimeType:"application/zip",buffer:Buffer.from(array!)});
  await expect(page.getByLabel("Cold threshold (field units)",{exact:true})).toHaveValue("-2");
  recipe.sourceBindings.frames[0].generation="wrong-publication";
  await page.locator('input[type="file"]').setInputFiles({name:"recipe.json",mimeType:"application/json",buffer:Buffer.from(JSON.stringify(recipe))});
  await expect(page.getByRole("status").filter({hasText:"publication changed"})).toBeVisible();
});

test("ordinary Studio exposes the research workspace without losing the current dataset",async({page})=>{
  await page.goto("/harness/index.html?mode=studio");await page.waitForFunction(()=>!!window.__nimbusStudioHarness);
  await page.locator(".plot-studio__tools-menu summary").click();await page.getByRole("button",{name:"Research workspace",exact:true}).click();
  await expect(page.getByRole("region",{name:"Research workspace",exact:true})).toBeVisible();
  await expect(page.getByRole("combobox",{name:"Shared saved frame",exact:true})).toBeVisible();
  await expect(page.locator(".plot-studio__frame-bar")).toHaveCount(0);
  await page.getByRole("button",{name:"Close research workspace",exact:true}).click();await expect(page.getByRole("region",{name:"Research workspace",exact:true})).toHaveCount(0);
  await expect(page.locator(".plot-studio__frame-bar")).toBeVisible();
});

test("section viewports stay linked and refine actual source sampling",async({page})=>{
  await open(page,"Linked sections");
  await page.getByRole("combobox",{name:"Contour / second field",exact:true}).selectOption("theta_prime");
  await page.getByRole("button",{name:"Load linked sections",exact:true}).click();
  await expect(page.locator('[data-panel-id="section-1"]')).toBeVisible();await data(page,"section-1");
  await page.locator('[data-panel-id="section-0"] .plot-canvas').evaluate(el=>(el as HTMLElement&{emit:(name:string,event:unknown)=>void}).emit("plotly_relayout",{"xaxis.range[0]":1,"xaxis.range[1]":3,"yaxis.range[0]":.5,"yaxis.range[1]":2.5}));
  await expect.poll(()=>page.locator('[data-panel-id="section-1"] .plot-canvas').evaluate(el=>(el as HTMLElement&{layout:{yaxis:{range:number[]}}}).layout.yaxis.range)).toEqual([.5,2.5]);
  await page.getByRole("button",{name:"Refine section to visible distance range",exact:true}).click();
  expect(Number(await page.getByLabel("A x (km)",{exact:true}).inputValue())).toBeCloseTo(.5+1/Math.sqrt(2),6);
  await page.getByRole("button",{name:"Load linked sections",exact:true}).click();
  await expect(page.getByRole("button",{name:"Export research figure package",exact:true})).toBeEnabled();
  expect(((await data(page,"section-0")) as {x:number[]}[])[0].x.at(-1)).toBeCloseTo(2,6);
});

test("CM1 netCDF research profiles use the normal bounded dataset adapter",async({page})=>{
  await page.goto("/harness/index.html?mode=studio&cm1=1");
  await page.waitForFunction(()=>!!window.__nimbusStudioHarness);
  await page.locator(".plot-studio__tools-menu summary").click();await page.getByRole("button",{name:"Research workspace",exact:true}).click();
  await page.getByRole("button",{name:"Environment",exact:true}).click();
  await page.getByRole("combobox",{name:"Profile field",exact:true}).selectOption("w");
  await page.getByRole("button",{name:"Load environmental profiles",exact:true}).click();
  await expect(page.locator('[data-panel-id="profiles"]')).toBeVisible();
  await expect(page.getByRole("button",{name:"Export research figure package",exact:true})).toBeEnabled();
  const traces=await page.locator('[data-panel-id="profiles"] .plot-canvas').evaluate(el=>(el as HTMLElement&{data:{x:number[];y:number[]}[]}).data);
  expect(traces).toHaveLength(3);expect(traces[0].x.length).toBeGreaterThan(10);expect(traces[0].y.every(Number.isFinite)).toBe(true);
});

test("research selection and controls survive named-project navigation and restart",async({page})=>{
  await page.goto("/harness/run-library.html");
  await page.getByRole("button",{name:"Projects",exact:true}).click();await page.getByRole("region",{name:"Projects",exact:true}).getByRole("button",{name:"Open included example"}).click();
  await page.locator(".plot-studio__tools-menu summary").click();await page.getByRole("button",{name:"Research workspace",exact:true}).click();
  await page.getByRole("button",{name:"Cold pools",exact:true}).click();await number(page,"Cold threshold (field units)","-4");
  await expect.poll(()=>page.evaluate(()=>{
    const store=JSON.parse(localStorage.getItem("nimbus:studio:projects:v1")??"{}");
    const project=store.projects?.find((entry:{id:string})=>entry.id===localStorage.getItem("nimbus:studio:active-project:v1"));
    const raw=project?.workspace?.rendererState?.researchRecipe;return raw?JSON.parse(raw).settings["cold-pools"]?.threshold:null;
  })).toBe(-4);
  await page.getByRole("button",{name:"Configure",exact:true}).click();await page.getByRole("button",{name:"Plots",exact:true}).click();
  await expect(page.getByRole("region",{name:"Research workspace",exact:true})).toBeVisible();await expect(page.getByLabel("Cold threshold (field units)",{exact:true})).toHaveValue("-4");
  await page.reload();await expect(page.getByRole("region",{name:"Research workspace",exact:true})).toBeVisible();await expect(page.getByLabel("Cold threshold (field units)",{exact:true})).toHaveValue("-4");
});

test("a new analysis owns its result when the saved frame changes during a slow read",async({page})=>{
  await open(page,"Cold pools");await page.getByLabel("Include linked vertical section",{exact:true}).uncheck();
  await page.evaluate(()=>window.__nimbusResearch.delay(150));
  await page.getByRole("button",{name:"Measure cold-pool footprint",exact:true}).click();
  await page.waitForFunction(()=>window.__nimbusResearch.calls().length>0);
  await page.getByRole("combobox",{name:"Shared saved frame",exact:true}).selectOption("1");
  await page.getByRole("button",{name:"Measure cold-pool footprint",exact:true}).click();
  await expect(page.getByRole("button",{name:"Export research figure package",exact:true})).toBeEnabled();
  await expect(page.locator(".research-notes")).toContainText("Minimum -4");
  await expect(page.locator(".research-header")).toContainText("saved t = 30 s");
});

test("research X and Y section switches retain the full saved vertical extent",async({page})=>{
  await open(page,"Linked sections");
  await page.getByRole("button",{name:"X section",exact:true}).click();await page.getByRole("button",{name:"Load linked sections",exact:true}).click();
  await expect(page.getByRole("button",{name:"Export research figure package",exact:true})).toBeEnabled();
  let section=((await data(page,"section-0")) as {z:number[][]}[])[0];expect(section.z).toHaveLength(8);expect(section.z[7].at(-1)).toBeCloseTo(8.75,6);
  await page.getByRole("button",{name:"Y section",exact:true}).click();await page.getByRole("button",{name:"Load linked sections",exact:true}).click();
  await expect(page.getByRole("button",{name:"Export research figure package",exact:true})).toBeEnabled();
  section=((await data(page,"section-0")) as {z:number[][]}[])[0];expect(section.z).toHaveLength(8);expect(section.z[7].at(-1)).toBeCloseTo(11.75,6);
});
