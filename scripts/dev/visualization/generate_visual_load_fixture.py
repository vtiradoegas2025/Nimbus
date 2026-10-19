#!/usr/bin/env python3
"""Write full-size, explicitly manufactured Nimbus history for renderer load tests."""
from __future__ import annotations
import argparse, hashlib, json, math, os, shutil, struct, time
from contextlib import ExitStack
from pathlib import Path
import numpy as np

VOLUMES = ('buoyancy','cloud_condensate','cloud_frozen_condensate','cloud_liquid_condensate',
           'p','p_prime','qr','qv','rho','temperature','theta','theta_prime','total_condensate',
           'u','v','vorticity_magnitude','vorticity_z','w')
COLUMNS = ('cape','cin','cloud_base_height','cloud_top_height','el','lcl','lfc','pbl_height',
           'srh_0_1km','srh_0_3km','stp','surface_latent_heat_flux','surface_sensible_heat_flux',
           'updraft_helicity','vil')
SCHEMA = 'nimbus.manufactured-visual-load.v1'

def digest(data): return hashlib.sha256(data).hexdigest()
def header(shape):
    text = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + ', '.join(map(str,shape)) + "), }"
    payload = text.encode(); payload += b' ' * ((16-(10+len(payload)+1)%16)%16) + b'\n'
    return b'\x93NUMPY\x01\x00'+struct.pack('<H',len(payload))+payload

def atomic_json(path, value):
    data=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode();temp=path.with_suffix(path.suffix+'.pending');temp.write_bytes(data);temp.replace(path);return digest(data)

def history_index(root,generation,entries):
    payload = '{\n  "schema": "nimbus.history.index.segment",\n  "version": 1,\n  "publication_generation": "'+generation+'",\n  "sequence": 1,\n  "operation": "reset",\n  "previous_segment": null,\n  "previous_segment_sha256": null,\n  "entries": ['
    payload += ','.join('\n    '+json.dumps(entry,separators=(', ',': ')) for entry in entries)+'\n  ]\n}'
    encoded=(payload[:-1]+',\n  "checksum": "'+digest(payload.encode())+'"\n}\n').encode()
    path=root/'history_index/active_a.json';path.parent.mkdir();path.write_bytes(encoded)
    head='{\n  "schema": "nimbus.history.index.head",\n  "version": 1,\n  "publication_generation": "'+generation+'",\n  "sequence": 1,\n  "reset_sequence": 1,\n  "segment": "history_index/active_a.json",\n  "segment_sha256": "'+digest(encoded)+'"\n}'
    (root/'history_index.json').write_text(head[:-1]+',\n  "checksum": "'+digest(head.encode())+'"\n}\n')

def smooth_noise(x, y, z, length, seed):
    coordinates = [axis / np.float32(length) for axis in (x, y, z)]
    cells = [np.floor(axis).astype(np.int32) for axis in coordinates]
    fractions = [axis - cell for axis, cell in zip(coordinates, cells)]
    weights = [np.asarray(f*f*(3-2*f), dtype=np.float32) for f in fractions]
    result = np.zeros(np.broadcast_shapes(x.shape, y.shape, z.shape), dtype=np.float32)
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                bits = ((cells[0]+a).astype(np.uint32)*np.uint32(73856093)
                        ^ (cells[1]+b).astype(np.uint32)*np.uint32(19349663)
                        ^ (cells[2]+c).astype(np.uint32)*np.uint32(83492791) ^ np.uint32(seed))
                bits ^= bits >> 16
                bits *= np.uint32(2246822519)
                bits ^= bits >> 13
                value = (bits & np.uint32(0xFFFFFF)).astype(np.float32) / np.float32(0xFFFFFF)
                weight = (weights[0] if a else 1-weights[0]) * (weights[1] if b else 1-weights[1]) * (weights[2] if c else 1-weights[2])
                result += value * weight
    return result * 2 - 1


def fields(shape, spacing, i0, i1, clock):
    nx,ny,nz=shape;dx,dy,dz=spacing;lx,ly,lz=nx*dx,ny*dy,nz*dz
    x=((np.arange(i0,i1,dtype=np.float32)+0.5)*dx-lx*.40-clock*12)[:,None,None]
    y=((np.arange(ny,dtype=np.float32)+0.5)*dy-ly*.48-clock*5)[None,:,None]
    z=((np.arange(nz,dtype=np.float32)+0.5)*dz)[None,None,:]
    scale=min(lx,ly)/32000;vertical_scale=lz/18000;height=z/lz
    sx,sy,sz=x/scale,y/scale,z/vertical_scale
    minimum_length=3*max(dx/scale,dy/scale,dz/vertical_scale)
    large=smooth_noise(sx,sy,sz,max(1900,minimum_length),71)
    medium=smooth_noise(sx+731,sy-429,sz+197,max(720,minimum_length),103)
    fine=smooth_noise(sx-313,sy+821,sz-577,max(310,minimum_length),211)
    edge=large*650+medium*230+fine*85
    cx=sx-(2200*(sz/12000)**1.6+400*np.sin(sz/3100))
    cy=sy-(900*(sz/12000)**1.4-300*np.sin(sz/2700))
    r2=cx*cx+cy*cy
    radius=1250+1550*np.clip(sz/11000,0,1)+350*np.sin(sz/2100)**2
    tower=np.clip((radius-np.sqrt(r2)+edge)/250,0,1)
    tower*=np.clip((sz-1250+180*large)/200,0,1)*np.clip((12700-sz+edge)/450,0,1)
    tower*=np.clip(.85+.23*large+.14*medium,.25,1.2)
    downstream=np.clip((sx+2500)/15500,0,1)
    plan=np.sqrt(((sx-5600)/12000)**2+((sy-900)/(7000+1200*downstream))**2)
    half_depth=700-300*downstream+150*large
    mid=11600+350*downstream+200*large
    anvil=np.clip((1-plan+edge/15000)/.06,0,1)*np.clip((half_depth-np.abs(sz-mid)+140*medium)/180,0,1)
    anvil*=np.clip(.85-.4*downstream+.2*medium,.15,1)
    dome=np.clip((1-np.sqrt(((sx-2900)/2300)**2+((sy-1000)/2100)**2+((sz-12600)/1550)**2)+edge/9000)/.15,0,1)
    flank=np.zeros_like(large)
    for fx,fy,top,radius_xy in [(-2200,-1200,8500,1700),(-4100,-2000,6500,1450),(-5900,-2400,4500,1150),(-7400,-2500,3000,900)]:
        radial=np.sqrt((sx-fx-900*sz/top)**2+(sy-fy)**2)
        turret=np.clip((radius_xy-radial+edge*.6)/220,0,1)
        turret*=np.clip((sz-1350)/180,0,1)*np.clip((top-sz+edge*.5)/350,0,1)
        flank=np.maximum(flank,turret*.7)
    wall=np.clip((1-np.sqrt(((sx+350)/1900)**2+((sy+250)/1600)**2+((sz-1400)/800)**2)+large*.14)/.22,0,1)*.65
    clear_slot=np.exp(-((sx+1000)/1900)**2-((sy-2200)/2100)**2)*np.exp(-((sz-2100)/1900)**2)
    density=np.maximum.reduce([tower,anvil,dome,flank,wall])*(1-.8*clear_slot)
    cloud=np.asarray(.0032*np.maximum(density,0),dtype=np.float32)
    frozen_fraction=np.clip((sz-4500)/3500,0,1)
    frozen=np.asarray(cloud*frozen_fraction,dtype=np.float32)
    liquid=np.asarray(cloud-frozen,dtype=np.float32)
    cloud=liquid+frozen
    rain_radius=np.sqrt((sx-1300)**2+(sy-1700)**2)
    angle=np.arctan2(sy-1700,sx-1300)
    curtain=np.exp(-((rain_radius-(3300+300*np.sin(angle*2)))/(850+300*large))**2)
    curtain*=np.clip((np.sin(angle)+.75)/.7,0,1)
    rain_core=np.exp(-((sx-5500)/2500)**2-((sy-3600)/3100)**2)
    rain=np.asarray(.003*np.maximum(curtain,rain_core)*np.exp(-((sz-2100)/4200)**4)*np.clip(1+.35*medium,.3,1.5),dtype=np.float32)
    vortex=np.exp(-r2/2100**2)
    stem=np.exp(-r2/(radius*.8)**2)*np.sin(np.pi*np.clip(sz/14000,0,1))
    updraft=np.asarray(48*stem-13*(rain/.003)*np.exp(-((sz-1800)/3000)**2),dtype=np.float32)
    prime=np.asarray(2.7*stem-5.5*np.exp(-((sx-3500)**2+(sy-2000)**2)/9000**2)*np.exp(-(sz/900)**2),dtype=np.float32)
    pressure_prime=np.asarray(-260*vortex*np.exp(-((height-.3)/.26)**2),dtype=np.float32)
    pressure=np.asarray(100000*np.exp(-z/8200)+pressure_prime,dtype=np.float32)
    theta=np.asarray(300+z*.0038+prime,dtype=np.float32)
    rho=np.broadcast_to(np.asarray(1.15*np.exp(-z/8500),dtype=np.float32),cloud.shape)
    swirl=np.float32(.009)*vortex
    u=np.asarray(12+26*height-swirl*cy,dtype=np.float32)
    v=np.asarray(4+8*height+swirl*cx,dtype=np.float32)
    vort=np.asarray(.018*vortex*(1-r2/2100**2),dtype=np.float32)
    values=dict(buoyancy=np.asarray(9.81*prime/(300+z*.0038),dtype=np.float32),cloud_condensate=cloud,
                cloud_frozen_condensate=frozen,cloud_liquid_condensate=liquid,p=pressure,p_prime=pressure_prime,
                qr=rain,qv=np.asarray(np.broadcast_to(.015*np.exp(-z/2300),cloud.shape),dtype=np.float32),rho=rho,
                temperature=np.asarray(theta*np.power(pressure/100000,np.float32(287/1004)),dtype=np.float32),
                theta=theta,theta_prime=prime,total_condensate=cloud+rain,u=u,v=v,
                vorticity_magnitude=np.abs(vort),vorticity_z=vort,w=updraft)
    active=cloud>=np.float32(.00005);levels=np.broadcast_to(z,cloud.shape)
    base=np.min(np.where(active,levels,np.float32(lz)),axis=2);top=np.max(np.where(active,levels,np.float32(0)),axis=2)
    core=np.exp(-(x[:,:,0]**2+y[:,:,0]**2)/(6000*scale)**2)
    column_shape=(i1-i0,ny)
    maps=dict(cape=3000+250*core,cin=25+5*core,cloud_base_height=base,cloud_top_height=top,
              el=np.full(column_shape,lz*.72),lcl=800+100*core,lfc=1200+100*core,pbl_height=np.full(column_shape,1200),
              srh_0_1km=150+60*core,srh_0_3km=250+90*core,stp=1+core,
              surface_latent_heat_flux=np.zeros(column_shape),surface_sensible_heat_flux=np.zeros(column_shape),
              updraft_helicity=np.sum(updraft*vort*((z>=2000)&(z<=5000)),axis=2)*dz,
              vil=np.sum(rho*(cloud+rain),axis=2)*dz)
    values.update({name:np.asarray(value,dtype=np.float32) for name,value in maps.items()})
    return values

def generate(output,shape,spacing,frames,frame_seconds,reference,chunk_x):
    if output.exists(): raise ValueError('Output must be a new directory; existing data is never overwritten')
    if min(shape)<4 or max(shape)>2048 or min(spacing)<=0 or not all(math.isfinite(x) for x in spacing):raise ValueError('Invalid grid')
    if not 2<=frames<=32 or not 1<=chunk_x<=32 or not 0<frame_seconds<=60:raise ValueError('Invalid bounded fixture parameters')
    per_frame=sum(len(header(shape if f in VOLUMES else shape[:2]))+4*math.prod(shape if f in VOLUMES else shape[:2]) for f in VOLUMES+COLUMNS)
    output.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(output.parent).free < per_frame*frames+4*1024**3:raise ValueError('Insufficient space for the full, uncompressed fixture')
    specification=dict(schema=SCHEMA,synthetic=True,not_a_simulation=True,shape=shape,spacing_m=spacing,frames=frames,frame_seconds=frame_seconds,fields=list(VOLUMES+COLUMNS),bytes_per_frame=per_frame,source_function='advected_storm_morphology_v2',features=['tilted_tower','downwind_anvil','overshooting_top','flanking_turrets','lowered_cloud_base','precipitation_curtain'],minimum_noise_lattice_cells=3)
    generation=digest(json.dumps(specification,sort_keys=True).encode())[:32]
    descriptors=[f for f in reference['fields'] if f['field_id'] in reference['published_fields']]
    if set(reference['published_fields']) != set(VOLUMES+COLUMNS):raise ValueError('Reference field set differs from the 33-field presentation workload')
    output.mkdir();atomic_json(output/'visual_load_fixture.json',specification)
    (output/'README_NOT_A_SIMULATION.txt').write_text('SYNTHETIC VISUAL LOAD TEST — NOT A SIMULATION\nKnown analytic signals sampled directly on the stated grid. No storm evolution, scientific forecasts, or finer model detail are implied. Fields use the real presentation data layout.\n')
    root=dict(schema='nimbus.output',manifest_version=7,publication_generation=generation,publication_state='publishing',frame_completion='atomic_step_manifest',history_index=dict(schema='nimbus.history.index',version=1,checksum='sha256'),data_products=['history'],synthetic_fixture=specification)
    atomic_json(output/'nimbus_output.json',root);entries=[];receipt=[];started=time.monotonic()
    for index in range(frames):
        directory=output/f'step_{index:06}';directory.mkdir();stats={};hashes={f:hashlib.sha256() for f in VOLUMES+COLUMNS}
        with ExitStack() as stack:
            streams={f:stack.enter_context((directory/(f+'.npy')).open('xb')) for f in VOLUMES+COLUMNS}
            for f,stream in streams.items():h=header(shape if f in VOLUMES else shape[:2]);stream.write(h);hashes[f].update(h)
            for first in range(0,shape[0],chunk_x):
                last=min(shape[0],first+chunk_x);values=fields(shape,spacing,first,last,index*frame_seconds)
                for f,stream in streams.items():
                    array=np.ascontiguousarray(values[f],dtype='<f4');assert np.isfinite(array).all(),f
                    payload=memoryview(array).cast('B');stream.write(payload);hashes[f].update(payload)
                    lo,hi=float(array.min()),float(array.max());old=stats.get(f,[lo,hi]);stats[f]=[min(old[0],lo),max(old[1],hi)]
            for stream in streams.values():stream.flush();os.fsync(stream.fileno())
        manifest=dict(reference);manifest.update(publication_generation=generation,step_index=index,simulation_time_s=index*frame_seconds,step_dir=directory.name,publication_state='complete',synthetic_fixture=dict(schema=SCHEMA,not_a_simulation=True,time_semantics='analytic_fixture_time'))
        manifest['grid']=dict(coordinate_system='cartesian',axis_order=['x','y','z'],dimensions=dict(zip('xyz',shape)),spacing_m=dict(zip('xyz',spacing)),native_model_staggering='c_grid',publication_staggering='cell_centered',scalar_origin_m=[s/2 for s in spacing],grid_scalar_origin_m=[s/2 for s in spacing],physical_bounds_m=dict(lower=[spacing[0],spacing[1],0],upper=[(shape[0]-1)*spacing[0],(shape[1]-1)*spacing[1],shape[2]*spacing[2]]),valid_index_range=dict(lower=[1,1,0],upper=[shape[0]-2,shape[1]-2,shape[2]-1]),halo_width=dict(lower=[1,1,0],upper=[1,1,0]))
        manifest['data_product']=dict(kind='history',restart_complete=False,storage_format='npy_3d',producer=SCHEMA)
        manifest.pop('soundings',None)
        manifest['fields']=[dict(f,description='Manufactured visual test signal: '+f.get('description',f['field_id'])) for f in descriptors]
        h=atomic_json(directory/'manifest.json',manifest);entries.append(dict(step_index=index,step_dir=directory.name,manifest_sha256=h))
        files={f:dict(sha256=hashes[f].hexdigest(),bytes=(directory/(f+'.npy')).stat().st_size,range=stats[f]) for f in streams}
        assert sum(f['bytes'] for f in files.values())==per_frame
        receipt.append(dict(step=index,fields=files,manifest_sha256=h));print(f'frame {index+1}/{frames}: {per_frame} bytes, {time.monotonic()-started:.1f}s',flush=True)
    history_index(output,generation,entries);root['publication_state']='ready';atomic_json(output/'nimbus_output.json',root)
    atomic_json(output/'fixture_receipt.json',dict(specification=specification,generator_sha256=digest(Path(__file__).read_bytes()),reference_manifest_sha256=digest(json.dumps(reference,sort_keys=True).encode()),frames=receipt,total_field_bytes=per_frame*frames,elapsed_seconds=time.monotonic()-started))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference-manifest',type=Path,default=Path(__file__).resolve().parents[3]/'tests/fixtures/reference/visualization/visual_load_manifest_v7.json');p.add_argument('--shape',type=int,nargs=3,default=[640,640,180]);p.add_argument('--spacing',type=float,nargs=3,default=[50,50,100]);p.add_argument('--frames',type=int,default=4);p.add_argument('--frame-seconds',type=float,default=5);p.add_argument('--chunk-x',type=int,default=8);a=p.parse_args()
    generate(a.output.resolve(),tuple(a.shape),tuple(a.spacing),a.frames,a.frame_seconds,json.loads(a.reference_manifest.read_text()),a.chunk_x)
if __name__=='__main__':main()
