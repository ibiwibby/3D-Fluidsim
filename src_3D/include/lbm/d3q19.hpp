#pragma once
#include "config.hpp"
#ifdef __CUDACC__
#define LBM_HD __host__ __device__
#else
#define LBM_HD
#endif
namespace lbm {
// Rest; +/- x,y,z; xy, xz, yz edge diagonals. No cube-corner directions.
LBM_HD inline int cx(int i) {
    constexpr int v[Q]={0,1,-1,0,0,0,0, 1,-1,1,-1, 1,-1,1,-1, 0,0,0,0};
    return v[i];
}
LBM_HD inline int cy(int i) {
    constexpr int v[Q]={0,0,0,1,-1,0,0, 1,-1,-1,1, 0,0,0,0, 1,-1,1,-1};
    return v[i];
}
LBM_HD inline int cz(int i) {
    constexpr int v[Q]={0,0,0,0,0,1,-1, 0,0,0,0, 1,-1,-1,1, 1,-1,-1,1};
    return v[i];
}
LBM_HD inline int opposite(int i) { return i==0?0:(i%2?i+1:i-1); }
LBM_HD inline Real weight(int i) { return i==0 ? 1.0/3 : (i<=6 ? 1.0/18 : 1.0/36); }
LBM_HD inline std::size_t index(int i,std::size_t cell,std::size_t n) { return std::size_t(i)*n+cell; }
LBM_HD inline std::size_t cell_index(int x,int y,int z,int nx,int ny) {
    return x + std::size_t(nx)*(y+std::size_t(ny)*z);
}
LBM_HD inline Real equilibrium(int i,Real rho,Real ux,Real uy,Real uz) {
    const Real cu=cx(i)*ux+cy(i)*uy+cz(i)*uz;
    return weight(i)*rho*(1+3*cu+4.5*cu*cu-1.5*(ux*ux+uy*uy+uz*uz));
}
// 0=fluid, 1=stationary channel wall, 2=central sphere (stair-step geometry).
LBM_HD inline int solid_type(int x,int y,int z,const Config& c) {
    if(!c.channel)return 0;
    if(y==0||y==c.ny-1||z==0||z==c.nz-1)return 1;
    const Real dx=x-(c.nx-1)*0.5,dy=y-(c.ny-1)*0.5,dz=z-(c.nz-1)*0.5;
    return c.obstacle && dx*dx+dy*dy+dz*dz<=c.obstacle_radius*c.obstacle_radius ? 2:0;
}
// Guo source for constant force density in +x, cs^2=1/3, dt=1.
LBM_HD inline Real forcing(int i,Real ux,Real uy,Real uz,Real fx,Real omega) {
    const Real cu=cx(i)*ux+cy(i)*uy+cz(i)*uz;
    return weight(i)*(1-0.5*omega)*(3*(cx(i)-ux)+9*cu*cx(i))*fx;
}
}
#undef LBM_HD
