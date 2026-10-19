#pragma once
namespace nimbus::metal::shaders
{
inline constexpr char kRrtmgShortwaveTransferBatchMsl[] = R"msl(
#include <metal_stdlib>
using namespace metal;
struct Push{int layers;int columns;int has_clouds;int p0;int p1;int p2;int p3;int p4;};
struct T{float rd;float rf;float td;float tf;float beam;};
constant int ends[14]={6,18,26,34,44,54,56,66,74,80,86,94,100,112};
inline T clear_transfer(){T r;r.rd=0;r.rf=0;r.td=1;r.tf=1;r.beam=1;return r;}
inline float transmission(float d){float b=min(d,40.0f);return b<=1e-4f?1.0f-b+0.5f*b*b:exp(-b);}
inline T pifm(float absorption,float scattering,float asym,float mu){T r=clear_transfer();float tau=absorption+scattering;if(tau<=0)return r;float omega=clamp(scattering/tau,0.0f,1.0f),g1=(8.0f-omega*(5.0f+3.0f*asym))*0.25f,g2=3.0f*omega*(1.0f-asym)*0.25f,g3=(2.0f-3.0f*asym*mu)*0.25f,g4=1.0f-g3;r.beam=transmission(tau/mu);if(omega>=0.9999995f){float gt=g1*tau;r.rd=(gt-(g1*mu-g3)*(1.0f-r.beam))/(1.0f+gt);r.td=1.0f-r.rd;r.rf=gt/(1.0f+gt);r.tf=1.0f-r.rf;return r.beam==1.0f?clear_transfer():r;}float x1=g1*g4+g2*g3,x2=g1*g3+g2*g4,k=sqrt(max(0.0f,g1*g1-g2*g2)),kp=k*mu,kp1=1.0f+kp,km1=1.0f-kp,two=2.0f*k,ok=1.0f-kp*kp,kg=k+g1;float r1=km1*(x2+k*g3),r2=kp1*(x2-k*g3),r3=two*(g3-x2*mu),r4=ok*kg,r5=ok*(k-g1),t1=kp1*(x1+k*g4),t2=km1*(x1-k*g4),t3=two*(g4+x1*mu),beta=(g1-k)/kg,e=transmission(k*tau),inv=1.0f/e,den=r4*inv+r5*e;if(fabs(den)<=1e-8f){r.rd=1e-8f;r.td=r.beam;}else{r.rd=omega*(r1*inv-r2*e-r3*r.beam)/den;r.td=r.beam-omega*(r.beam*(t1*inv-t2*e)-t3)/den;}float e2=e*e,id=1.0f/((1.0f-beta*e2)*kg);r.rf=g2*(1.0f-e2)*id;r.tf=two*e*id;return r;}
inline int band_for(uint g){for(int b=0;b<14;++b)if(g<uint(ends[b]))return b;return 13;}
kernel void rrtmg_shortwave_transfer_batch(
 device const float* absorption [[buffer(0)]],device const float* rayleigh [[buffer(1)]],device const float* solar [[buffer(2)]],
 device const float* cloud_depth [[buffer(3)]],device const float* cloud_omega [[buffer(4)]],device const float* cloud_asymmetry [[buffer(5)]],
 device const float4* parameters [[buffer(6)]],device float* upward [[buffer(7)]],device float* downward [[buffer(8)]],
 device float* dr [[buffer(9)]],device float* fr [[buffer(10)]],device float* dt [[buffer(11)]],device float* ft [[buffer(12)]],
 device float* beam [[buffer(13)]],device float* cumulative [[buffer(14)]],device float* up_direct [[buffer(15)]],
 device float* up_diffuse [[buffer(16)]],device float* down_total [[buffer(17)]],device float* down_diffuse [[buffer(18)]],
 constant Push& pc [[buffer(19)]],uint work [[thread_position_in_grid]]){
 uint total=uint(pc.columns)*112u;if(work>=total)return;uint column=work/112u,g=work%112u;float4 p=parameters[column];uint stride=uint(pc.layers+1),base=work*stride,sbase=column*uint(pc.layers)*112u,cbase=column*uint(pc.layers)*14u;if(p.x==0.0f||p.y==0.0f){for(int top=0;top<=pc.layers;++top){upward[base+uint(top)]=0.0f;downward[base+uint(top)]=0.0f;}return;}int band=band_for(g);cumulative[base]=1.0f;
 for(int top=0;top<pc.layers;++top){int layer=pc.layers-1-top;uint ix=sbase+uint(layer)*112u+g,sx=base+uint(top),cx=cbase+uint(layer)*14u+uint(band);float cd=pc.has_clouds!=0?cloud_depth[cx]:0.0f,cs=pc.has_clouds!=0?cd*cloud_omega[cx]:0.0f,scatter=rayleigh[ix]+cs,asym=scatter>0.0f&&pc.has_clouds!=0?cs*cloud_asymmetry[cx]/scatter:0.0f;T tr=pifm(absorption[ix]+cd-cs,scatter,asym,p.x);dr[sx]=tr.rd;fr[sx]=tr.rf;dt[sx]=tr.td;ft[sx]=tr.tf;beam[sx]=tr.beam;cumulative[base+uint(top+1)]=cumulative[base+uint(top)]*tr.beam;}
 up_direct[base+uint(pc.layers)]=p.z;up_diffuse[base+uint(pc.layers)]=p.z;int low=pc.layers-1;uint lx=base+uint(low);float coupling=1.0f/(1.0f-p.z*fr[lx]);up_direct[lx]=dr[lx]+ft[lx]*((dt[lx]-beam[lx])*p.z+beam[lx]*p.z)*coupling;up_diffuse[lx]=fr[lx]+ft[lx]*ft[lx]*p.z*coupling;for(int top=low-1;top>=0;--top){uint x=base+uint(top);coupling=1.0f/(1.0f-up_diffuse[x+1]*fr[x]);up_direct[x]=dr[x]+ft[x]*((dt[x]-beam[x])*up_diffuse[x+1]+beam[x]*up_direct[x+1])*coupling;up_diffuse[x]=fr[x]+ft[x]*ft[x]*up_diffuse[x+1]*coupling;}
 down_total[base]=1.0f;down_diffuse[base]=0.0f;down_total[base+1]=dt[base];down_diffuse[base+1]=fr[base];for(int top=1;top<pc.layers;++top){uint x=base+uint(top);coupling=1.0f/(1.0f-fr[x]*down_diffuse[x]);down_total[x+1]=cumulative[x]*dt[x]+ft[x]*((down_total[x]-cumulative[x])+cumulative[x]*dr[x]*down_diffuse[x])*coupling;down_diffuse[x+1]=fr[x]+ft[x]*ft[x]*down_diffuse[x]*coupling;}
 float incident=p.y*p.x*solar[column*112u+g]/p.w;for(int top=0;top<=pc.layers;++top){uint x=base+uint(top);coupling=1.0f/(1.0f-down_diffuse[x]*up_diffuse[x]);float u=(cumulative[x]*up_direct[x]+(down_total[x]-cumulative[x])*up_diffuse[x])*coupling,d=cumulative[x]+((down_total[x]-cumulative[x])+cumulative[x]*up_direct[x]*down_diffuse[x])*coupling;uint output_index=base+uint(pc.layers-top);upward[output_index]=incident*u;downward[output_index]=incident*d;}}
)msl";
}
