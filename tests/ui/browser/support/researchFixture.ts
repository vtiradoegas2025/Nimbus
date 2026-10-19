import type {GridInfo,StepInfo} from "@nimbus/lib/run/runOutputs";
import type {NpyN} from "@nimbus/lib/platform/npy";
import {encodeNpy} from "@nimbus/lib/platform/pythonExport";
import {encodeFieldWindow} from "./fixtures";
export const RESEARCH_RUN="/__nimbus_browser_fixture__/research";
export const RESEARCH_COARSE="/__nimbus_browser_fixture__/research-coarse";
const fields=["w","u","v","theta","temperature","qv","p","theta_prime","p_prime","dynamic_pressure","buoyancy_pressure","pressure_gradient_force_z","buoyancy","qr","qc","rho_d","reflectivity_dbz","precip_rate"];
const units:Record<string,string>={w:"m/s",u:"m/s",v:"m/s",theta:"K",temperature:"K",qv:"kg/kg",p:"Pa",theta_prime:"K",p_prime:"Pa",dynamic_pressure:"Pa",buoyancy_pressure:"Pa",pressure_gradient_force_z:"m/s^2",buoyancy:"m/s^2",qr:"kg/kg",qc:"kg/kg",rho_d:"kg/m3",reflectivity_dbz:"dBZ",precip_rate:"mm/h"};
export const researchCalls:{command:string;field:string;step:number;run:string;origin?:number[];extent?:number[];stride?:number[]}[]=[];
let delay=0;
export function setResearchDelay(value:number){delay=value;}
export function researchGrid(coarse=false):GridInfo{return {coordinateSystem:"cartesian",geometryKind:"physical",nr:coarse?2:4,nth:coarse?2:3,nz:coarse?4:8,drM:coarse?2000:1000,dyM:coarse?2250:1500,dzM:coarse?1000:500,scalarOriginM:coarse?[1000,1125,500]:[500,750,250],physicalBoundsM:{lower:[0,0,0],upper:[4000,4500,4000]}};}
export function researchSteps(coarse=false):StepInfo[]{const grid=researchGrid(coarse);return [0,1,2].map(index=>({index,simulationTimeS:index*30,publicationState:"complete",publicationGeneration:`research-${coarse?"coarse":"native"}-${index}`,fields:[...fields],fieldUnits:{...units},fieldOrigins:Object.fromEntries(fields.map(field=>[field,"native" as const])),fieldShapes:Object.fromEntries(fields.map(field=>[field,field==="precip_rate"?[grid.nr!,grid.nth!]:[grid.nr!,grid.nth!,grid.nz!]])),grid}));}
export function researchSource(run:string,step:number,field:string):NpyN{
  const g=researchGrid(run===RESEARCH_COARSE),shape=field==="precip_rate"?[g.nr!,g.nth!]:[g.nr!,g.nth!,g.nz!];
  const data=new Float32Array(shape.reduce((a,b)=>a*b,1));
  for(let i=0;i<shape[0];i++)for(let j=0;j<shape[1];j++)for(let k=0;k<(shape[2]??1);k++){
    const x=(g.scalarOriginM![0]+i*g.drM!)/1000,y=(g.scalarOriginM![1]+j*g.dyM!)/1000,z=(g.scalarOriginM![2]+k*g.dzM!)/1000;
    const values:Record<string,number>={w:x+2*y+z+step,u:x+step*2,v:y-step,theta:300+z+2*step,temperature:290-z*6,qv:.01,p:100000*Math.exp(-z/8),theta_prime:x<2?-3-step:-1,p_prime:x-y+z,dynamic_pressure:x-y,buoyancy_pressure:z,pressure_gradient_force_z:.1*z,buoyancy:.01*z,qr:z*.001,qc:z*.0005,rho_d:1,reflectivity_dbz:z*10,precip_rate:1+step};
    data[(i*shape[1]+j)*(shape[2]??1)+k]=values[field]??NaN;
  }
  return {shape,data};
}
export async function researchInvoke(command:string,payload:unknown):Promise<{value:unknown}|null>{
  const args=(payload??{}) as Record<string,unknown>,run=String(args.runDir);if(run!==RESEARCH_RUN&&run!==RESEARCH_COARSE)return null;
  if(command==="list_nested_history_levels")return {value:[]};
  if(!["read_field_window","read_field_bytes","reduce_field","read_trajectories"].includes(command))return null;
  const step=Number(args.stepIndex??0),field=String(args.fieldName??"trajectories");
  const call:typeof researchCalls[number]={command,field,step,run};researchCalls.push(call);
  if(delay)await new Promise(resolve=>setTimeout(resolve,delay));
  if(command==="read_trajectories"){
    const data=new Float32Array(2*4*9);for(let p=0;p<2;p++)for(let s=0;s<4;s++)data.set([s*10,p*1000+s*100,p*500,1000+p*1000+s*100,10+p*10,.001+.0006*s*10,.0001,.0002,.0003],(p*4+s)*9);
    return {value:encodeNpy(data,[2,4,9]).buffer};
  }
  const source=researchSource(run,step,field);
  if(command==="read_field_bytes")return {value:encodeNpy(source.data,source.shape).buffer};
  if(command==="reduce_field"){const data=[...source.data];return {value:{min:Math.min(...data),max:Math.max(...data),mean:data.reduce((a,b)=>a+b,0)/data.length,rms:0,finite:data.length,total:data.length,countAbove:0,countBelow:0,countAbsAbove:0}};}
  const request=args.request as {requestId:string;origin:number[];extent:number[];stride:number[];aggregation:Parameters<typeof encodeFieldWindow>[5];maxOutputBytes:number;maxSourceBytes:number};
  Object.assign(call,{origin:request.origin,extent:request.extent,stride:request.stride});
  return {value:encodeFieldWindow(request.requestId,source,request.origin,request.extent,request.stride,request.aggregation,request.maxOutputBytes,request.maxSourceBytes,false,undefined,researchGrid(run===RESEARCH_COARSE))};
}
