#pragma once

namespace nimbus::metal::shaders
{

inline constexpr const char* kAcousticBatchedCartesianCgridMsl = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct PressureConstants {
    int nx; int ny; int nz; float dx; float dy; float dz_val;
    float gamma_val; float dt_small; float rho_min; float rho_max;
    float p_min; float p_max; float p_ref_fallback; float mass_flux_weight;
    float frame_velocity_u_ms; float frame_velocity_v_ms;
    int preserve_unrepaired;
    float gamma_low;
};

kernel void acoustic_pressure_massflux_cartesian_cgrid(
    device const float* u [[buffer(0)]], device const float* v [[buffer(1)]],
    device const float* w [[buffer(2)]], device const float* rho [[buffer(3)]],
    device const float* p [[buffer(4)]], device const float* rho0 [[buffer(5)]],
    device const float* p0 [[buffer(6)]], device const float* theta [[buffer(7)]],
    device float* rho_out [[buffer(8)]], device float* p_out [[buffer(9)]],
    device float* mfx [[buffer(10)]], device float* mfy [[buffer(11)]],
    device float* mfz [[buffer(12)]], constant PressureConstants& c [[buffer(13)]],
    uint gid [[thread_position_in_grid]])
{
    uint total=uint(c.nx)*uint(c.ny)*uint(c.nz);if(gid>=total)return;
    int k=int(gid%uint(c.nz));uint ij=gid/uint(c.nz);int j=int(ij%uint(c.ny));int i=int(ij/uint(c.ny));
    float fxp=0.0f,fyp=0.0f,fzp=0.0f;
    if(i+1<c.nx){uint q=uint(((i+1)*c.ny+j)*c.nz+k);fxp=.5f*(rho[gid]+rho[q])*(u[gid]-c.frame_velocity_u_ms);}
    if(j+1<c.ny){uint q=uint((i*c.ny+j+1)*c.nz+k);fyp=.5f*(rho[gid]+rho[q])*(v[gid]-c.frame_velocity_v_ms);}
    if(k+1<c.nz){uint q=gid+1;fzp=.5f*(rho[gid]+rho[q])*w[gid];}else fzp=rho[gid]*w[gid];
    mfx[gid]+=c.mass_flux_weight*fxp;mfy[gid]+=c.mass_flux_weight*fyp;mfz[gid]+=c.mass_flux_weight*fzp;
    float drho=0.0f;
    if(i>=1&&i<=c.nx-2&&j>=1&&j<=c.ny-2){
        uint im=uint(((i-1)*c.ny+j)*c.nz+k),jm=uint((i*c.ny+j-1)*c.nz+k);
        float fxm=.5f*(rho[im]+rho[gid])*(u[im]-c.frame_velocity_u_ms);
        float fym=.5f*(rho[jm]+rho[gid])*(v[jm]-c.frame_velocity_v_ms);
        float fzm=0.0f;if(k>0)fzm=.5f*(rho[gid-1]+rho[gid])*w[gid-1];
        drho=-((fxp-fxm)/c.dx+(fyp-fym)/c.dy+(fzp-fzm)/c.dz_val);if(!isfinite(drho))drho=0.0f;
    }
    float rn=rho[gid]+drho*c.dt_small;if(!isfinite(rn)||rn<=0.0f)rn=max(.1f,rho0[k]);rn=clamp(rn,c.rho_min,c.rho_max);rho_out[gid]=rn;
    float pn=c.p_ref_fallback;if(theta[gid]>0.0f)pn=100000.0f*pow(rn*287.0f*theta[gid]/100000.0f,c.gamma_val);if(!isfinite(pn)||pn<=0.0f)pn=c.p_ref_fallback;p_out[gid]=clamp(pn,c.p_min,c.p_max);
    (void)p;(void)p0;
}

struct BcConstants {
    int nx; int ny; int nz; int face; int bc_west; int bc_east;
    int bc_south; int bc_north; int bc_bottom; int bc_top;
    float frame_velocity_u_ms; float frame_velocity_v_ms; float dt_small;
    float inv_dx; float inv_dy; float pad2;
};
inline uint bat(int i,int j,int k,constant BcConstants& c){return uint((i*c.ny+j)*c.nz+k);}
inline void copy_scalar_pair(device float* rho,device float* p,uint dst,uint src){rho[dst]=rho[src];p[dst]=p[src];}

)MSL";

} // namespace nimbus::metal::shaders
