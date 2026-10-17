import {useState} from "react";
import {ResearchWorkspace} from "@nimbus/components/research/ResearchWorkspace";
import {registerListedStepMetadata} from "@nimbus/lib/run/runOutputs";
import type {LocalDatasetSource} from "@nimbus/lib/dataset/datasetSources";
import {RESEARCH_RUN,RESEARCH_COARSE,researchSteps,researchCalls,setResearchDelay} from "../support/researchFixture";
import {waitForPlotCanvasReady} from "@nimbus/lib/plot/plotCanvasReadiness";
import {fixtureControl} from "../support/fixtures";
const steps=researchSteps(),coarse=researchSteps(true);registerListedStepMetadata(RESEARCH_RUN,steps);registerListedStepMetadata(RESEARCH_COARSE,coarse);
const source=(path:string,name:string):LocalDatasetSource=>({schema:"nimbus.studio.dataset-source",schemaVersion:1,kind:"nimbus_run",sourceModel:"nimbus",nativeFormat:"nimbus_history",locator:{kind:"local_paths",path},displayName:name,readOnly:true});
const dataset=source(RESEARCH_RUN,"Research native fixture"),members=[{source:dataset,steps},{source:source(RESEARCH_COARSE,"Research coarse fixture"),steps:coarse}];
declare global{interface Window{__nimbusResearch:{calls:()=>typeof researchCalls;panel:(id:string)=>Promise<unknown[]>;delay:(milliseconds:number)=>void;file:(path:string)=>number[]|null;savePath:(path:string|null)=>void}}}
window.__nimbusResearch={panel:async id=>{const el=document.querySelector<HTMLElement>(`[data-panel-id="${id}"] .plot-canvas`);if(!el)throw new Error("Panel missing");await waitForPlotCanvasReady(el);return (el as HTMLElement&{data:unknown[]}).data;},calls:()=>researchCalls,delay:setResearchDelay,file:path=>{const bytes=fixtureControl.readFileBytes(path);return bytes?[...bytes]:null;},savePath:fixtureControl.setSaveDialogPath};
export function ResearchHarness(){const [step,setStep]=useState(0);return <main style={{height:"100vh",display:"flex",flexDirection:"column"}}><ResearchWorkspace dataset={dataset} steps={steps} stepIdx={step} onStepChange={setStep} members={members} onClose={()=>{}}/></main>;}
