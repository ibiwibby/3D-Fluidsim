#include "lbm/simulation.cuh"
#include "lbm/d3q19.hpp"
#include "lbm/cuda_utils.cuh"
#include <cstdio>
#include <utility>
namespace lbm {
namespace {
__device__ void invalid(std::size_t cell) {
    printf("Invalid D3Q19 state at cell %llu\n",static_cast<unsigned long long>(cell));
    asm volatile("trap;");
}
__global__ void initialize_kernel(Real* f,Config c,bool bump,Real ux,Real uy,Real uz) {
    const auto cell=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const auto n=std::size_t(c.nx)*c.ny*c.nz;
    if(cell>=n) return;
    const int x=cell%c.nx,y=(cell/c.nx)%c.ny,z=cell/(std::size_t(c.nx)*c.ny);
    const int x0=(c.nx-c.bump_width)/2,y0=(c.ny-c.bump_width)/2,z0=(c.nz-c.bump_width)/2;
    const bool inside=x>=x0&&x<x0+c.bump_width&&y>=y0&&y<y0+c.bump_width&&z>=z0&&z<z0+c.bump_width;
    const Real rho=1+((bump&&inside)?c.density_increase:0);
    const bool solid=solid_type(x,y,z,c)!=0;
    // Counteract the half-force correction so the physical initial velocity is zero.
    if(c.channel && !solid)ux-=0.5*c.force_x/rho;
    if(solid)ux=uy=uz=0;
    for(int i=0;i<Q;++i) f[index(i,cell,n)]=equilibrium(i,rho,ux,uy,uz);
}
__device__ bool moments(const Real* f,std::size_t cell,std::size_t n,
                        Real* local,Real& rho,Real& ux,Real& uy,Real& uz) {
    rho=ux=uy=uz=0;
    for(int i=0;i<Q;++i) {
        const auto v=f[index(i,cell,n)];
        if(!isfinite(v)) { invalid(cell); return false; }
        local[i]=v;rho+=v;ux+=v*cx(i);uy+=v*cy(i);uz+=v*cz(i);
    }
    if(!isfinite(rho)||rho<=0) { invalid(cell); return false; }
    ux/=rho;uy/=rho;uz/=rho;
    if(!isfinite(ux*ux+uy*uy+uz*uz)) { invalid(cell); return false; }
    return true;
}
__global__ void collide_kernel(Real* f,Config c) {
    const auto cell=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const auto n=std::size_t(c.nx)*c.ny*c.nz;
    if(cell>=n)return;
    const int x=cell%c.nx,y=(cell/c.nx)%c.ny,z=cell/(std::size_t(c.nx)*c.ny);
    if(solid_type(x,y,z,c))return;
    Real local[Q],rho,ux,uy,uz;
    if(!moments(f,cell,n,local,rho,ux,uy,uz))return;
    const Real omega=1/c.tau,fx=c.channel?c.force_x:0;
    ux+=0.5*fx/rho;
    for(int i=0;i<Q;++i) {
        local[i]+= -omega*(local[i]-equilibrium(i,rho,ux,uy,uz))
                   + forcing(i,ux,uy,uz,fx,omega);
        if(!isfinite(local[i])) { invalid(cell);return; }
    }
    for(int i=0;i<Q;++i)f[index(i,cell,n)]=local[i];
}
__device__ int wrap(int x,int n) { return x<0?n-1:(x>=n?0:x); }
__global__ void stream_kernel(const Real* f,Real* next,Config c) {
    const auto cell=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const auto n=std::size_t(c.nx)*c.ny*c.nz;
    if(cell>=n)return;
    const int x=cell%c.nx,y=(cell/c.nx)%c.ny,z=cell/(std::size_t(c.nx)*c.ny);
    if(solid_type(x,y,z,c)) {
        for(int i=0;i<Q;++i)next[index(i,cell,n)]=f[index(i,cell,n)];
        return;
    }
    for(int i=0;i<Q;++i) {
        const auto source=cell_index(wrap(x-cx(i),c.nx),wrap(y-cy(i),c.ny),wrap(z-cz(i),c.nz),c.nx,c.ny);
        const int sx=source%c.nx,sy=(source/c.nx)%c.ny,sz=source/(std::size_t(c.nx)*c.ny);
        // Halfway bounce-back: reflect the local outgoing opposite population.
        next[index(i,cell,n)]=solid_type(sx,sy,sz,c)
            ? f[index(opposite(i),cell,n)] : f[index(i,source,n)];
    }
}
__global__ void fields_kernel(const Real* f,Real* fields,Config c) {
    const auto cell=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const auto n=std::size_t(c.nx)*c.ny*c.nz;
    if(cell>=n)return;
    Real local[Q],rho,ux,uy,uz;
    if(!moments(f,cell,n,local,rho,ux,uy,uz))return;
    const int x=cell%c.nx,y=(cell/c.nx)%c.ny,z=cell/(std::size_t(c.nx)*c.ny);
    if(solid_type(x,y,z,c)) { rho=0;ux=uy=uz=0; }
    else if(c.channel)ux+=0.5*c.force_x/rho;
    fields[cell]=rho;
    fields[n+3*cell]=ux;fields[n+3*cell+1]=uy;fields[n+3*cell+2]=uz;
}
void finish() { CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); }
}
Simulation::Simulation(Config config):c_(config) {
    c_.validate();
    try {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&current_),c_.bytes()));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&next_),c_.bytes()));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&fields_),4*c_.cells()*sizeof(Real)));
    } catch(...) { cudaFree(current_);cudaFree(next_);cudaFree(fields_);throw; }
}
Simulation::~Simulation(){cudaFree(fields_);cudaFree(next_);cudaFree(current_);}
void Simulation::initialize(bool bump,Real ux,Real uy,Real uz) {
    if(!std::isfinite(ux)||!std::isfinite(uy)||!std::isfinite(uz))throw std::invalid_argument("Invalid velocity");
    ready_=false;
    initialize_kernel<<<static_cast<unsigned>((c_.cells()+127)/128),128>>>(current_,c_,bump,ux,uy,uz);
    finish();ready_=true;
}
void Simulation::upload(const std::vector<Real>& values) {
    if(values.size()!=c_.populations())throw std::invalid_argument("Incorrect upload size");
    ready_=false;
    CUDA_CHECK(cudaMemcpy(current_,values.data(),c_.bytes(),cudaMemcpyHostToDevice));ready_=true;
}
std::vector<Real> Simulation::download() const {
    if(!ready_)throw std::logic_error("No valid state");
    std::vector<Real> result(c_.populations());
    CUDA_CHECK(cudaMemcpy(result.data(),current_,c_.bytes(),cudaMemcpyDeviceToHost));return result;
}
Fields Simulation::fields() const {
    if(!ready_)throw std::logic_error("No valid state");
    fields_kernel<<<static_cast<unsigned>((c_.cells()+127)/128),128>>>(current_,fields_,c_);finish();
    Fields result;result.density.resize(c_.cells());result.velocity.resize(3*c_.cells());
    CUDA_CHECK(cudaMemcpy(result.density.data(),fields_,c_.cells()*sizeof(Real),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.velocity.data(),fields_+c_.cells(),3*c_.cells()*sizeof(Real),cudaMemcpyDeviceToHost));
    result.solid.resize(c_.cells());
    for(int z=0;z<c_.nz;++z)for(int y=0;y<c_.ny;++y)for(int x=0;x<c_.nx;++x)
        result.solid[cell_index(x,y,z,c_.nx,c_.ny)]=solid_type(x,y,z,c_);
    return result;
}
void Simulation::collide_only(){
    if(!ready_)throw std::logic_error("No valid state");
    ready_=false;
    collide_kernel<<<static_cast<unsigned>((c_.cells()+127)/128),128>>>(current_,c_);
    finish();ready_=true;
}
void Simulation::stream_only(){
    if(!ready_)throw std::logic_error("No valid state");
    ready_=false;
    stream_kernel<<<static_cast<unsigned>((c_.cells()+127)/128),128>>>(current_,next_,c_);
    finish();std::swap(current_,next_);ready_=true;
}
void Simulation::step(){collide_only();stream_only();}
}
