#pragma once

namespace nimbus::metal::shaders
{

inline constexpr const char* kCartesianAgridMsl = R"msl(
#include <metal_stdlib>
using namespace metal;

struct SlowConstants {
    int nr; int nth; int nz; float dx; float dy; float dz; float g;
    float gamma_val; float coriolis_f; float coriolis_beta;
    int bc_west; int bc_east; int bc_south; int bc_north; int top_bc; int has_moisture;
};

inline uint slow_index(int i,int j,int k,constant SlowConstants& c){
    return uint((i*c.nth+j)*c.nz+k);
}
inline bool closed_lateral(int bc){return bc==2||bc==3;}
inline float average_product(float ra,float rb,float va,float vb){return 0.25f*(ra+rb)*(va+vb);}
inline float clean(float value){return isfinite(value)?value:0.0f;}

kernel void cartesian_agrid_tendencies(
    device const float* u [[buffer(0)]], device const float* v [[buffer(1)]],
    device const float* w [[buffer(2)]], device const float* rho [[buffer(3)]],
    device const float* pressure [[buffer(4)]], device const float* theta [[buffer(5)]],
    device const float* p0 [[buffer(6)]], device const float* rho0 [[buffer(7)]],
    device float* du [[buffer(8)]], device float* dv [[buffer(9)]],
    device float* dw [[buffer(10)]], device float* drho [[buffer(11)]],
    device float* dp [[buffer(12)]], device const float* loading [[buffer(13)]],
    device const float* u0 [[buffer(14)]], device const float* v0 [[buffer(15)]],
    device const float* qv [[buffer(16)]], device const float* qv0 [[buffer(17)]],
    constant SlowConstants& c [[buffer(18)]], uint gid [[thread_position_in_grid]])
{
    uint total=uint(c.nr)*uint(c.nth)*uint(c.nz); if(gid>=total)return;
    int k=int(gid%uint(c.nz)); uint ij=gid/uint(c.nz); int j=int(ij%uint(c.nth)); int i=int(ij/uint(c.nth));
    if(i==0||i==c.nr-1||j==0||j==c.nth-1||k==0||k==c.nz-1){
        du[gid]=0.0f;dv[gid]=0.0f;dw[gid]=0.0f;drho[gid]=0.0f;dp[gid]=0.0f;return;
    }
    uint ip=slow_index(i+1,j,k,c),im=slow_index(i-1,j,k,c);
    uint jp=slow_index(i,j+1,k,c),jm=slow_index(i,j-1,k,c);
    uint kp=slow_index(i,j,k+1,c),km=slow_index(i,j,k-1,c);
    float ux=u[gid],uy=v[gid],wz=w[gid],rv=rho[gid],pv=pressure[gid];
    float rs=rv>1.0e-6f&&isfinite(rv)?rv:1.0f;
    float ix=1.0f/(2.0f*c.dx),iy=1.0f/(2.0f*c.dy),iz=1.0f/(2.0f*c.dz);
    float ux_x=(u[ip]-u[im])*ix,ux_y=(u[jp]-u[jm])*iy,ux_z=(u[kp]-u[km])*iz;
    float vy_x=(v[ip]-v[im])*ix,vy_y=(v[jp]-v[jm])*iy,vy_z=(v[kp]-v[km])*iz;
    float w_x=(w[ip]-w[im])*ix,w_y=(w[jp]-w[jm])*iy,w_z=(w[kp]-w[km])*iz;
    float p_x=(pressure[ip]-pressure[im])*ix,p_y=(pressure[jp]-pressure[jm])*iy;
    float p_z=(pressure[kp]-pressure[km])*iz;
    float local_f=c.coriolis_beta==0.0f?c.coriolis_f:
        c.coriolis_f+c.coriolis_beta*(float(j)+0.5f)*c.dy;
    bool split_slow_only=(c.has_moisture&2)!=0;
    du[gid]=clean(-ux*ux_x-uy*ux_y-wz*ux_z+local_f*(uy-v0[k])
                  +(split_slow_only?0.0f:-p_x/rs));
    dv[gid]=clean(-ux*vy_x-uy*vy_y-wz*vy_z-local_f*(ux-u0[k])
                  +(split_slow_only?0.0f:-p_y/rs));
    float moisture=(c.has_moisture&1)!=0?c.g*0.608f*(qv[gid]-qv0[k]):0.0f;
    dw[gid]=clean(-ux*w_x-uy*w_y-wz*w_z
                  +(split_slow_only?0.0f:-(p_z-(p0[k+1]-p0[k-1])*iz)/rs)
                  -c.g*(rv-rho0[k])/rs+moisture-c.g*loading[gid]);
    auto xface=[&](int face)->float{
        if(face==0){if(closed_lateral(c.bc_west))return 0.0f;
            if(c.bc_west==0)return average_product(rho[slow_index(c.nr-2,j,k,c)],rho[slow_index(1,j,k,c)],u[slow_index(c.nr-2,j,k,c)],u[slow_index(1,j,k,c)]);
            return rho[slow_index(0,j,k,c)]*u[slow_index(0,j,k,c)];}
        if(face==c.nr-2){if(closed_lateral(c.bc_east))return 0.0f;
            if(c.bc_east==0)return average_product(rho[slow_index(c.nr-2,j,k,c)],rho[slow_index(1,j,k,c)],u[slow_index(c.nr-2,j,k,c)],u[slow_index(1,j,k,c)]);
            return rho[slow_index(c.nr-1,j,k,c)]*u[slow_index(c.nr-1,j,k,c)];}
        return average_product(rho[slow_index(face,j,k,c)],rho[slow_index(face+1,j,k,c)],u[slow_index(face,j,k,c)],u[slow_index(face+1,j,k,c)]);
    };
    auto yface=[&](int face)->float{
        if(face==0||face==c.nth-2){int bc=face==0?c.bc_south:c.bc_north;if(closed_lateral(bc))return 0.0f;
            if(bc==0)return average_product(rho[slow_index(i,c.nth-2,k,c)],rho[slow_index(i,1,k,c)],v[slow_index(i,c.nth-2,k,c)],v[slow_index(i,1,k,c)]);
            int bj=face==0?0:c.nth-1;return rho[slow_index(i,bj,k,c)]*v[slow_index(i,bj,k,c)];}
        return average_product(rho[slow_index(i,face,k,c)],rho[slow_index(i,face+1,k,c)],v[slow_index(i,face,k,c)],v[slow_index(i,face+1,k,c)]);
    };
    auto zface=[&](int face)->float{
        if(face==0)return 0.0f;if(face==c.nz-2){if(c.top_bc!=2)return 0.0f;return rho[slow_index(i,j,c.nz-1,c)]*w[slow_index(i,j,c.nz-1,c)];}
        return average_product(rho[slow_index(i,j,face,c)],rho[slow_index(i,j,face+1,c)],w[slow_index(i,j,face,c)],w[slow_index(i,j,face+1,c)]);
    };
    drho[gid]=split_slow_only?0.0f:clean(-((xface(i)-xface(i-1))/c.dx+(yface(j)-yface(j-1))/c.dy+(zface(k)-zface(k-1))/c.dz));
    float divergence=ux_x+vy_y+w_z;
    dp[gid]=clean((split_slow_only?0.0f:-c.gamma_val*pv*divergence)-ux*p_x-uy*p_y-wz*p_z);
    (void)theta;
}
)msl";

inline constexpr const char* kAcousticAgridMsl = R"msl(
#include <metal_stdlib>
using namespace metal;
struct PressureConstants {int nr;int nth;int nz;float dx;float dy;float dz;float gamma_val;float dt;float rho_floor;float p_floor;};
struct MomentumConstants {int nr;int nth;int nz;float dx;float dy;float dz;float dt;float clamp_h;float clamp_v;float padding;};
inline uint at(int i,int j,int k,int ny,int nz){return uint((i*ny+j)*nz+k);}
inline float clean(float value){return isfinite(value)?value:0.0f;}
kernel void acoustic_pressure_agrid(device const float* u[[buffer(0)]],device const float* v[[buffer(1)]],
 device const float* w[[buffer(2)]],device const float* rho[[buffer(3)]],device const float* p[[buffer(4)]],
 device float* rho_out[[buffer(5)]],device float* p_out[[buffer(6)]],constant PressureConstants& c[[buffer(7)]],
 uint gid[[thread_position_in_grid]]){
 uint total=uint(c.nr)*uint(c.nth)*uint(c.nz);if(gid>=total)return;int k=int(gid%uint(c.nz));uint ij=gid/uint(c.nz);int j=int(ij%uint(c.nth));int i=int(ij/uint(c.nth));
 bool xb=i==0||i==c.nr-1,yb=j==0||j==c.nth-1,zb=k==0||k==c.nz-1;
 if(xb||yb||zb){int is=clamp(i,1,c.nr-2),js=clamp(j,1,c.nth-2),ks=clamp(k,1,c.nz-2);uint s=at(is,js,ks,c.nth,c.nz);float rs=rho[s];rho_out[gid]=rs;
   if(zb&&!xb&&!yb)p_out[gid]=p[s]+(k==0?1.0f:-1.0f)*max(rs,1.0e-3f)*9.81f*c.dz;else p_out[gid]=p[s];return;}
 float div=(u[at(i+1,j,k,c.nth,c.nz)]-u[at(i-1,j,k,c.nth,c.nz)])/(2.0f*c.dx)
          +(v[at(i,j+1,k,c.nth,c.nz)]-v[at(i,j-1,k,c.nth,c.nz)])/(2.0f*c.dy)
          +(w[at(i,j,k+1,c.nth,c.nz)]-w[at(i,j,k-1,c.nth,c.nz)])/(2.0f*c.dz);
 float rs=rho[gid]>1.0e-6f&&isfinite(rho[gid])?rho[gid]:1.0f;
 float rn=rho[gid]+clean(-rs*div)*c.dt;if(!isfinite(rn)||rn<=0.0f)rn=c.rho_floor;rho_out[gid]=max(rn,c.rho_floor);
 float pn=p[gid]+clean(-c.gamma_val*p[gid]*div)*c.dt;if(!isfinite(pn)||pn<=0.0f)pn=c.p_floor;p_out[gid]=max(pn,c.p_floor);
}
kernel void acoustic_momentum_agrid(device const float* rho[[buffer(0)]],device const float* p[[buffer(1)]],
 device const float* u[[buffer(2)]],device const float* v[[buffer(3)]],device const float* w[[buffer(4)]],
 device float* u_out[[buffer(5)]],device float* v_out[[buffer(6)]],device float* w_out[[buffer(7)]],
 constant MomentumConstants& c[[buffer(8)]],uint gid[[thread_position_in_grid]]){
 uint total=uint(c.nr)*uint(c.nth)*uint(c.nz);if(gid>=total)return;int k=int(gid%uint(c.nz));uint ij=gid/uint(c.nz);int j=int(ij%uint(c.nth));int i=int(ij/uint(c.nth));
 bool xb=i==0||i==c.nr-1,yb=j==0||j==c.nth-1,zb=k==0||k==c.nz-1;
 if(xb||yb||zb){int is=clamp(i,1,c.nr-2),js=clamp(j,1,c.nth-2),ks=clamp(k,1,c.nz-2);uint s=at(is,js,ks,c.nth,c.nz);u_out[gid]=u[s];v_out[gid]=v[s];w_out[gid]=zb?0.0f:w[s];return;}
 float rs=rho[gid]>1.0e-6f&&isfinite(rho[gid])?rho[gid]:1.0f;
 float px=(p[at(i+1,j,k,c.nth,c.nz)]-p[at(i-1,j,k,c.nth,c.nz)])/(2.0f*c.dx);
 float py=(p[at(i,j+1,k,c.nth,c.nz)]-p[at(i,j-1,k,c.nth,c.nz)])/(2.0f*c.dy);
 float pz=(p[at(i,j,k+1,c.nth,c.nz)]-p[at(i,j,k-1,c.nth,c.nz)])/(2.0f*c.dz);
 u_out[gid]=clamp(u[gid]+clean(-px/rs)*c.dt,-c.clamp_h,c.clamp_h);
 v_out[gid]=clamp(v[gid]+clean(-py/rs)*c.dt,-c.clamp_h,c.clamp_h);
 w_out[gid]=clamp(w[gid]+clean(-pz/rs)*c.dt,-c.clamp_v,c.clamp_v);
}
)msl";

} // namespace nimbus::metal::shaders
