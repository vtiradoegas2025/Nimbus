#pragma once
namespace nimbus::metal::shaders
{
inline constexpr char kRrtmgLongwaveTransferBatchMsl[] = R"msl(
#include <metal_stdlib>
using namespace metal;
struct Push{int layers;int columns;int has_clouds;int p0;int p1;int p2;int p3;int p4;};
constant int ends[16]={10,22,38,52,68,76,88,96,108,114,122,130,134,136,138,140};
constant float a0[16]={1.66f,1.55f,1.58f,1.66f,1.54f,1.454f,1.89f,1.33f,1.668f,1.66f,1.66f,1.66f,1.66f,1.66f,1.66f,1.66f};
constant float a1[16]={0.0f,0.25f,0.22f,0.0f,0.13f,0.446f,-0.10f,0.40f,-0.006f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
constant float a2[16]={0.0f,-12.0f,-11.7f,0.0f,-0.72f,-0.243f,0.19f,-0.062f,0.414f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
inline int band_for(uint g){for(int b=0;b<16;++b)if(g<uint(ends[b]))return b;return 15;}
inline float secant_for(int b,float w){if(b==0||b==3||b>=9)return 1.66f;return clamp(a0[b]+a1[b]*exp(a2[b]*w),1.50f,1.80f);}
inline float2 clear_transfer(float d){if(d<=0.06f)return float2(d-0.5f*d*d,d/6.0f);float t=exp(-d);return float2(1.0f-t,1.0f-2.0f*(1.0f/d-t/(1.0f-t)));}
inline float lookup(float d){float c=d/(1.0f/0.278f+d);uint i=uint(10000.0f*c+0.5f);if(i>=10000u)return 1.0e10f;float f=float(i)/10000.0f;return(1.0f/0.278f)*f/(1.0f-f);}
inline float2 layer_transfer(float gas,float cloud){if(cloud>0.0f){if(gas>0.06f)gas=lookup(gas);float total=gas+cloud;if(total>0.06f)total=lookup(total);return clear_transfer(total);}return clear_transfer(gas);}
kernel void rrtmg_longwave_transfer_batch(
 device const float* absorption [[buffer(0)]],device const float* fraction [[buffer(1)]],
 device const float* layer_planck [[buffer(2)]],device const float* interface_planck [[buffer(3)]],
 device const float* surface_planck [[buffer(4)]],device const float* widths [[buffer(5)]],
 device const float* clouds [[buffer(6)]],device const float4* parameters [[buffer(7)]],
 device float* upward [[buffer(8)]],device float* downward [[buffer(9)]],
 constant Push& pc [[buffer(10)]],uint work [[thread_position_in_grid]]){
 uint total=uint(pc.columns)*140u;if(work>=total)return;uint column=work/140u,g=work%140u;int band=band_for(g);float4 p=parameters[column];
 float secant=secant_for(band,p.y);uint stride=uint(pc.layers+1),base=work*stride,sbase=column*uint(pc.layers)*140u,lb=column*uint(pc.layers)*16u,ib=column*uint(pc.layers+1)*16u,surface=column*16u;
 float down=0.0f;downward[base+uint(pc.layers)]=0.0f;for(int layer=pc.layers-1;layer>=0;--layer){uint ix=sbase+uint(layer)*140u+g;float cloud=pc.has_clouds!=0?clouds[lb+uint(layer)*16u+uint(band)]:0.0f;float2 tr=layer_transfer(secant*absorption[ix],secant*cloud);float planck=layer_planck[lb+uint(layer)*16u+uint(band)],diff=interface_planck[ib+uint(layer)*16u+uint(band)]-planck,source=fraction[ix]*(planck+tr.y*diff);down+=(source-down)*tr.x;downward[base+uint(layer)]=down;}
 float up=surface_planck[surface+uint(band)]*fraction[sbase+g]+(1.0f-p.x)*down;upward[base]=up;for(int layer=0;layer<pc.layers;++layer){uint ix=sbase+uint(layer)*140u+g;float cloud=pc.has_clouds!=0?clouds[lb+uint(layer)*16u+uint(band)]:0.0f;float2 tr=layer_transfer(secant*absorption[ix],secant*cloud);float planck=layer_planck[lb+uint(layer)*16u+uint(band)],diff=interface_planck[ib+uint(layer+1)*16u+uint(band)]-planck,source=fraction[ix]*(planck+tr.y*diff);up+=(source-up)*tr.x;upward[base+uint(layer+1)]=up;}
 float scale=3.14159265358979323846e4f*widths[uint(band)];for(int i=0;i<=pc.layers;++i){uint ix=base+uint(i);upward[ix]*=scale;downward[ix]*=scale;}}
)msl";
}
