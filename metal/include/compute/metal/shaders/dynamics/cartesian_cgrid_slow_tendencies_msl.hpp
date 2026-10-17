#pragma once

namespace nimbus::metal::shaders
{

inline constexpr const char* kCartesianCgridSlowTendenciesMsl = R"msl(
#include <metal_stdlib>
using namespace metal;

struct SlowConstants {
    int nr; int nth; int nz; float dx;
    float dy; float dz; float g; int has_moisture;
    int has_loading; float coriolis_f; float emit_advection; float coriolis_beta;
    float frame_u; float frame_v;
};

inline float safe_rho(float value)
{
    return isfinite(value) && value > 1.0e-6f ? value : 1.0f;
}
inline float sanitize(float value) { return isfinite(value) ? value : 0.0f; }
inline float upwind_deriv(int hw, float m3, float m2, float m1, float c,
                          float p1, float p2, float p3, float a, float inv_d)
{
    if (hw <= 0) return 0.0f;
    if (hw == 1) return a >= 0.0f ? (c-m1)*inv_d : (p1-c)*inv_d;
    if (hw == 2) return a >= 0.0f
        ? (2.0f*p1+3.0f*c-6.0f*m1+m2)*(inv_d/6.0f)
        : (-p2+6.0f*p1-3.0f*c-2.0f*m1)*(inv_d/6.0f);
    return a >= 0.0f
        ? (-2.0f*m3+15.0f*m2-60.0f*m1+20.0f*c+30.0f*p1-3.0f*p2)*(inv_d/60.0f)
        : (2.0f*p3-15.0f*p2+60.0f*p1-20.0f*c-30.0f*m1+3.0f*m2)*(inv_d/60.0f);
}

kernel void cartesian_cgrid_slow_tendencies(
    device const float* u [[buffer(0)]], device const float* v [[buffer(1)]],
    device const float* w [[buffer(2)]], device const float* rho [[buffer(3)]],
    device const float* pressure [[buffer(4)]], device const float* rho0 [[buffer(5)]],
    device const float* qv0 [[buffer(6)]], device const float* u0 [[buffer(7)]],
    device const float* v0 [[buffer(8)]], device const float* qv [[buffer(9)]],
    device const float* loading [[buffer(10)]], device float* du [[buffer(11)]],
    device float* dv [[buffer(12)]], device float* dw [[buffer(13)]],
    device float* drho [[buffer(14)]], device float* dp [[buffer(15)]],
    device float* du_adv [[buffer(16)]], device float* dv_adv [[buffer(17)]],
    device float* dw_adv [[buffer(18)]], constant SlowConstants& p [[buffer(19)]],
    uint gid [[thread_position_in_grid]])
{
#define IDX(ii,jj,kk) uint((((ii)*p.nth+(jj))*p.nz+(kk)))
    const uint total=uint(p.nr)*uint(p.nth)*uint(p.nz); if(gid>=total)return;
    const int k=int(gid%uint(p.nz));const uint ij=gid/uint(p.nz);
    const int j=int(ij%uint(p.nth));const int i=int(ij/uint(p.nth));
    du[gid]=0.0f;dv[gid]=0.0f;dw[gid]=0.0f;drho[gid]=0.0f;dp[gid]=0.0f;
    if(p.emit_advection!=0.0f){du_adv[gid]=0.0f;dv_adv[gid]=0.0f;dw_adv[gid]=0.0f;}
    const float idx=1.0f/p.dx, idy=1.0f/p.dy, idz=1.0f/p.dz;
    const float i2x=0.5f/p.dx, i2y=0.5f/p.dy, i2z=0.5f/p.dz;
    const bool horizontal=i>=1&&i<=p.nr-2&&j>=1&&j<=p.nth-2;
    const bool face=horizontal&&k>=0&&k<=p.nz-2;
    const bool cell=horizontal&&k>=1&&k<=p.nz-2;
    if(face){
        const float c=u[gid];
        const float va=0.25f*(v[IDX(i,j-1,k)]+v[IDX(i,j,k)]+v[IDX(i+1,j-1,k)]+v[IDX(i+1,j,k)]);
        const float wa=k==0?0.25f*(w[IDX(i,j,0)]+w[IDX(i+1,j,0)]):
            0.25f*(w[IDX(i,j,k-1)]+w[IDX(i,j,k)]+w[IDX(i+1,j,k-1)]+w[IDX(i+1,j,k)]);
        const int hx=min(i,p.nr-1-i),hy=min(j,p.nth-1-j),hz=min(k,p.nz-1-k);
        const float dxv=upwind_deriv(hx,hx>=3?u[IDX(i-3,j,k)]:0.0f,hx>=2?u[IDX(i-2,j,k)]:0.0f,
            u[IDX(i-1,j,k)],c,u[IDX(i+1,j,k)],hx>=2?u[IDX(i+2,j,k)]:0.0f,hx>=3?u[IDX(i+3,j,k)]:0.0f,c,idx);
        const float dyv=upwind_deriv(hy,hy>=3?u[IDX(i,j-3,k)]:0.0f,hy>=2?u[IDX(i,j-2,k)]:0.0f,
            u[IDX(i,j-1,k)],c,u[IDX(i,j+1,k)],hy>=2?u[IDX(i,j+2,k)]:0.0f,hy>=3?u[IDX(i,j+3,k)]:0.0f,va,idy);
        const float dzv=k==0?(wa>=0.0f?0.0f:(u[IDX(i,j,1)]-c)*idz):
            upwind_deriv(hz,hz>=3?u[IDX(i,j,k-3)]:0.0f,hz>=2?u[IDX(i,j,k-2)]:0.0f,
                u[IDX(i,j,k-1)],c,u[IDX(i,j,k+1)],hz>=2?u[IDX(i,j,k+2)]:0.0f,hz>=3?u[IDX(i,j,k+3)]:0.0f,wa,idz);
        const float adv=-(c-p.frame_u)*dxv-(va-p.frame_v)*dyv-wa*dzv;
        float cor=p.coriolis_f*(va-v0[k]);
        if(p.coriolis_beta!=0.0f){
            float flo=p.coriolis_f+p.coriolis_beta*(float(j)+0.25f)*p.dy;
            float fhi=p.coriolis_f+p.coriolis_beta*(float(j)+0.75f)*p.dy;
            cor=0.25f*(flo*(v[IDX(i,j-1,k)]+v[IDX(i+1,j-1,k)]-2.0f*v0[k])+
                       fhi*(v[IDX(i,j,k)]+v[IDX(i+1,j,k)]-2.0f*v0[k]));}
        du[gid]=sanitize(adv+cor);if(p.emit_advection!=0.0f)du_adv[gid]=sanitize(adv);
    }
    if(face){
        const float c=v[gid];
        const float ua=0.25f*(u[IDX(i-1,j,k)]+u[IDX(i,j,k)]+u[IDX(i-1,j+1,k)]+u[IDX(i,j+1,k)]);
        const float wa=k==0?0.25f*(w[IDX(i,j,0)]+w[IDX(i,j+1,0)]):
            0.25f*(w[IDX(i,j,k-1)]+w[IDX(i,j,k)]+w[IDX(i,j+1,k-1)]+w[IDX(i,j+1,k)]);
        const int hx=min(i,p.nr-1-i),hy=min(j,p.nth-1-j),hz=min(k,p.nz-1-k);
        const float dxv=upwind_deriv(hx,hx>=3?v[IDX(i-3,j,k)]:0.0f,hx>=2?v[IDX(i-2,j,k)]:0.0f,
            v[IDX(i-1,j,k)],c,v[IDX(i+1,j,k)],hx>=2?v[IDX(i+2,j,k)]:0.0f,hx>=3?v[IDX(i+3,j,k)]:0.0f,ua,idx);
        const float dyv=upwind_deriv(hy,hy>=3?v[IDX(i,j-3,k)]:0.0f,hy>=2?v[IDX(i,j-2,k)]:0.0f,
            v[IDX(i,j-1,k)],c,v[IDX(i,j+1,k)],hy>=2?v[IDX(i,j+2,k)]:0.0f,hy>=3?v[IDX(i,j+3,k)]:0.0f,c,idy);
        const float dzv=k==0?(wa>=0.0f?0.0f:(v[IDX(i,j,1)]-c)*idz):
            upwind_deriv(hz,hz>=3?v[IDX(i,j,k-3)]:0.0f,hz>=2?v[IDX(i,j,k-2)]:0.0f,
                v[IDX(i,j,k-1)],c,v[IDX(i,j,k+1)],hz>=2?v[IDX(i,j,k+2)]:0.0f,hz>=3?v[IDX(i,j,k+3)]:0.0f,wa,idz);
        const float adv=-(ua-p.frame_u)*dxv-(c-p.frame_v)*dyv-wa*dzv;
        float cor=p.coriolis_f*(ua-u0[k]);
        if(p.coriolis_beta!=0.0f){
            float flo=p.coriolis_f+p.coriolis_beta*(float(j)+0.75f)*p.dy;
            float fhi=p.coriolis_f+p.coriolis_beta*(float(j)+1.25f)*p.dy;
            cor=0.25f*(flo*(u[IDX(i-1,j,k)]+u[IDX(i,j,k)]-2.0f*u0[k])+
                       fhi*(u[IDX(i-1,j+1,k)]+u[IDX(i,j+1,k)]-2.0f*u0[k]));}
        dv[gid]=sanitize(adv-cor);if(p.emit_advection!=0.0f)dv_adv[gid]=sanitize(adv);
    }
    if(face){
        const float c=w[gid];
        const float ua=0.25f*(u[IDX(i-1,j,k)]+u[IDX(i,j,k)]+u[IDX(i-1,j,k+1)]+u[IDX(i,j,k+1)]);
        const float va=0.25f*(v[IDX(i,j-1,k)]+v[IDX(i,j,k)]+v[IDX(i,j-1,k+1)]+v[IDX(i,j,k+1)]);
        const int hx=min(i,p.nr-1-i),hy=min(j,p.nth-1-j),hz=min(k,p.nz-1-k);
        const float dxv=upwind_deriv(hx,hx>=3?w[IDX(i-3,j,k)]:0.0f,hx>=2?w[IDX(i-2,j,k)]:0.0f,
            w[IDX(i-1,j,k)],c,w[IDX(i+1,j,k)],hx>=2?w[IDX(i+2,j,k)]:0.0f,hx>=3?w[IDX(i+3,j,k)]:0.0f,ua,idx);
        const float dyv=upwind_deriv(hy,hy>=3?w[IDX(i,j-3,k)]:0.0f,hy>=2?w[IDX(i,j-2,k)]:0.0f,
            w[IDX(i,j-1,k)],c,w[IDX(i,j+1,k)],hy>=2?w[IDX(i,j+2,k)]:0.0f,hy>=3?w[IDX(i,j+3,k)]:0.0f,va,idy);
        const float dzv=k==0?(c>=0.0f?c*idz:(w[IDX(i,j,1)]-c)*idz):
            upwind_deriv(hz,hz>=3?w[IDX(i,j,k-3)]:0.0f,hz>=2?w[IDX(i,j,k-2)]:0.0f,
                w[IDX(i,j,k-1)],c,w[IDX(i,j,k+1)],hz>=2?w[IDX(i,j,k+2)]:0.0f,hz>=3?w[IDX(i,j,k+3)]:0.0f,c,idz);
        const float rf=0.5f*(rho[gid]+rho[IDX(i,j,k+1)]), r0=0.5f*(rho0[k]+rho0[k+1]);
        const float buoy=-p.g*(rf-r0)/safe_rho(rf);
        const float moist=p.has_moisture!=0?p.g*0.608f*(0.5f*(qv[gid]+qv[IDX(i,j,k+1)])-0.5f*(qv0[k]+qv0[k+1])):0.0f;
        const float load=p.has_loading!=0?-p.g*0.5f*(loading[gid]+loading[IDX(i,j,k+1)]):0.0f;
        const float adv=-(ua-p.frame_u)*dxv-(va-p.frame_v)*dyv-c*dzv;
        dw[gid]=sanitize(adv+buoy+moist+load);if(p.emit_advection!=0.0f)dw_adv[gid]=sanitize(adv);
    }
    if(cell){
        const float uc=0.5f*(u[gid]+u[IDX(i-1,j,k)]),vc=0.5f*(v[gid]+v[IDX(i,j-1,k)]),wc=0.5f*(w[gid]+w[IDX(i,j,k-1)]);
        const float px=(pressure[IDX(i+1,j,k)]-pressure[IDX(i-1,j,k)])*i2x;
        const float py=(pressure[IDX(i,j+1,k)]-pressure[IDX(i,j-1,k)])*i2y;
        const float pz=(pressure[IDX(i,j,k+1)]-pressure[IDX(i,j,k-1)])*i2z;
        dp[gid]=sanitize(-(uc-p.frame_u)*px-(vc-p.frame_v)*py-wc*pz);
    }
#undef IDX
}
)msl";

} // namespace nimbus::metal::shaders
