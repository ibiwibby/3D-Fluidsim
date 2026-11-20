#include "lbm/simulation.cuh"
#include "lbm/d3q19.hpp"
#include "lbm/cuda_utils.cuh"
#include "lbm/output.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
namespace {
void require(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
void check_equilibrium(const lbm::Simulation& sim,const lbm::Config& c,
                       lbm::Real ux=0,lbm::Real uy=0,lbm::Real uz=0) {
    const auto values=sim.download();double worst=0;
    for(int i=0;i<lbm::Q;++i)for(std::size_t cell=0;cell<c.cells();++cell) {
        const double error=std::abs(values[lbm::index(i,cell,c.cells())]-lbm::equilibrium(i,1,ux,uy,uz));
        require(std::isfinite(error),"Nonfinite equilibrium state");worst=std::max(worst,error);
    }
    require(worst<=c.tolerance,"Equilibrium population mismatch");
    const auto fields=sim.fields();
    for(std::size_t cell=0;cell<c.cells();++cell) {
        require(std::isfinite(fields.density[cell])&&std::abs(fields.density[cell]-1)<=c.tolerance,"Density mismatch");
        const double expected[3]={ux,uy,uz};
        for(int k=0;k<3;++k)require(std::isfinite(fields.velocity[3*cell+k])&&std::abs(fields.velocity[3*cell+k]-expected[k])<=c.tolerance,"Velocity mismatch");
    }
    std::cout<<"  maximum population error="<<worst<<'\n';
}
void checks() {
    lbm::Config c;c.nx=5;c.ny=6;c.nz=7;c.bump_width=2;
    lbm::Simulation sim(c);
    sim.initialize();check_equilibrium(sim,c);
    auto perturbed=sim.download();
    perturbed[lbm::index(1,0,c.cells())]+=.01;perturbed[lbm::index(2,0,c.cells())]+=.01;perturbed[0]-=.02;
    sim.upload(perturbed);sim.collide_only();check_equilibrium(sim,c);
    std::cout<<"Initialization and collision: PASS\n";
    // Independent explicit directions and forward-push oracle, versus GPU pull.
    const int directions[19][3]={{0,0,0},{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},
      {1,1,0},{-1,-1,0},{1,-1,0},{-1,1,0},{1,0,1},{-1,0,-1},{1,0,-1},{-1,0,1},
      {0,1,1},{0,-1,-1},{0,1,-1},{0,-1,1}};
    std::vector<double> expected(c.populations());
    for(std::size_t j=0;j<expected.size();++j)expected[j]=double(j+1);
    sim.upload(expected);
    for(int step=0;step<2;++step) {
        std::vector<double> next(expected.size(),-1);
        for(int i=0;i<19;++i)for(int z=0;z<c.nz;++z)for(int y=0;y<c.ny;++y)for(int x=0;x<c.nx;++x) {
            const int dx=(x+directions[i][0]+c.nx)%c.nx,dy=(y+directions[i][1]+c.ny)%c.ny,dz=(z+directions[i][2]+c.nz)%c.nz;
            const auto source=x+c.nx*(y+c.ny*z),dest=dx+c.nx*(dy+c.ny*dz);
            next[i*c.cells()+dest]=expected[i*c.cells()+source];
        }
        sim.stream_only();require(sim.download()==next,"Streaming label mismatch");expected.swap(next);
    }
    std::cout<<"3D streaming on 5x6x7, including two swaps: PASS\n";
    for(bool moving:{false,true}) {
        const double ux=moving?.02:0,uy=moving?-.01:0,uz=moving?.015:0;
        sim.initialize(false,ux,uy,uz);
        for(int step=0;step<1000;++step)sim.step();
        check_equilibrium(sim,c,ux,uy,uz);
        std::cout<<(moving?"Uniform 3D flow":"Rest state")<<" through 1000 steps: PASS\n";
    }
}
void run(lbm::Config c,const std::filesystem::path& directory,bool bump) {
    c.validate();
    // Avoid mixing new snapshots with old runs or silently replacing results.
    if(std::filesystem::exists(directory)&&!std::filesystem::is_empty(directory))
        throw std::runtime_error("Output directory is not empty; choose a new --output directory");
    std::filesystem::create_directories(directory);
    std::ofstream diagnostics;diagnostics.exceptions(std::ios::failbit|std::ios::badbit);
    diagnostics.open(directory/"diagnostics.csv");
    diagnostics<<std::setprecision(17)<<"step,mass,relative_mass_error,min_density,max_density,max_speed\n";
    std::ofstream metadata;metadata.exceptions(std::ios::failbit|std::ios::badbit);
    metadata.open(directory/"run.txt");
    metadata<<std::setprecision(17)<<"D3Q19 BGK; periodic xyz; double; lattice units\n"
            <<"grid="<<c.nx<<'x'<<c.ny<<'x'<<c.nz<<"\nsteps="<<c.steps<<"\noutput_every="<<c.output_every
            <<"\ntau="<<c.tau<<"\nbump="<<bump<<"\nbump_width="<<c.bump_width<<"\ndensity_increase="<<c.density_increase<<'\n';metadata.close();
    lbm::Simulation sim(c);sim.initialize(bump);
    const double expected_mass=c.cells()+(bump?double(c.bump_width)*c.bump_width*c.bump_width*c.density_increase:0);
    std::vector<std::pair<int,std::string>> frames;
    int step=0;
    while(true) {
        // Validate fields each step, but only persist full fields at output intervals.
        const auto f=sim.fields();double mass=0,minrho=1e300,maxrho=0,maxspeed=0;
        for(std::size_t cell=0;cell<c.cells();++cell) {
            const double rho=f.density[cell],u=f.velocity[3*cell],v=f.velocity[3*cell+1],w=f.velocity[3*cell+2];
            const double speed=std::sqrt(u*u+v*v+w*w);
            require(std::isfinite(rho)&&rho>0&&std::isfinite(speed),"Invalid field at step "+std::to_string(step));
            mass+=rho;minrho=std::min(minrho,rho);maxrho=std::max(maxrho,rho);maxspeed=std::max(maxspeed,speed);
        }
        const double error=std::abs(mass-expected_mass)/expected_mass;
        require(std::isfinite(mass)&&error<=c.tolerance,"Mass error at step "+std::to_string(step));
        if(!bump)check_equilibrium(sim,c);
        diagnostics<<step<<','<<mass<<','<<error<<','<<minrho<<','<<maxrho<<','<<maxspeed<<'\n';diagnostics.flush();
        if(step%c.output_every==0||step==c.steps) {
            std::ostringstream name;name<<"flow_"<<std::setw(6)<<std::setfill('0')<<step<<".vti";
            lbm::write_vti(directory/name.str(),c,f,step);frames.emplace_back(step,name.str());
            lbm::write_pvd(directory/"flow.pvd",frames);
            std::cout<<"Step "<<step<<": mass="<<std::setprecision(15)<<mass<<" relative_error="<<error<<" max_speed="<<maxspeed<<'\n';
        }
        if(step==c.steps)break;
        sim.step();++step;
    }
    diagnostics.close();std::cout<<"Saved "<<frames.size()<<" snapshots to "<<directory<<"; checks passed.\n";
}
int integer(const std::string& text){std::size_t used=0;int v=std::stoi(text,&used);if(used!=text.size())throw std::invalid_argument("Invalid integer");return v;}
}
int main(int argc,char** argv) {
 try {
    const std::string mode=argc>1?argv[1]:"--help";
    if(mode=="--help") {
        std::cout<<"D3Q19 CUDA: --device-info | --check | --rest | --disturbance\n"
                 <<"Run options: --steps N --output-every N --output DIRECTORY\n"
                 <<"Default: 32x32x32, 50 steps, central 4^3 cube at rho=1.10, background=1.\n";return 0;
    }
    require(mode=="--device-info"||mode=="--check"||mode=="--rest"||mode=="--disturbance","Unknown mode; use --help");
    lbm::Config c;std::filesystem::path output="output";
    for(int i=2;i<argc;++i) {
        require(mode=="--rest"||mode=="--disturbance","Options apply only to run modes");
        const std::string option=argv[i];require(i+1<argc,"Missing option value");const std::string value=argv[++i];
        if(option=="--steps")c.steps=integer(value);
        else if(option=="--output-every")c.output_every=integer(value);
        else if(option=="--output")output=value;
        else throw std::invalid_argument("Unknown option: "+option);
    }
    c.validate();int count=0;CUDA_CHECK(cudaGetDeviceCount(&count));require(count>0,"No CUDA GPU visible");
    CUDA_CHECK(cudaSetDevice(0));cudaDeviceProp prop{};CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    std::cout<<"GPU: "<<prop.name<<" compute capability "<<prop.major<<'.'<<prop.minor<<'\n';
    if(mode=="--device-info")return 0;
    if(mode=="--check")checks();else run(c,output,mode=="--disturbance");
    return 0;
 }catch(const std::exception& e){std::cerr<<"D3Q19 failed: "<<e.what()<<'\n';return 1;}
}
