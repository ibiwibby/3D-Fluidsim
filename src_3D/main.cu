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
// Channel-specific tests supplement the periodic-lattice regression suite.
void channel_checks() {
    lbm::Config c;c.nx=12;c.ny=10;c.nz=8;c.obstacle_radius=1.5;
    c.channel=true;c.force_x=0;c.validate();
    lbm::Simulation sim(c);
    std::vector<double> labels(c.populations());
    for(std::size_t j=0;j<labels.size();++j)labels[j]=double(j+1);
    sim.upload(labels);
    std::vector<double> expected(labels.size(),-1);
    // Forward routing with explicit reverse-direction search, independent of opposite().
    for(int z=0;z<c.nz;++z)for(int y=0;y<c.ny;++y)for(int x=0;x<c.nx;++x) {
        const auto cell=lbm::cell_index(x,y,z,c.nx,c.ny);
        for(int i=0;i<19;++i) {
            if(lbm::solid_type(x,y,z,c)){expected[i*c.cells()+cell]=labels[i*c.cells()+cell];continue;}
            const int dx=(x+lbm::cx(i)+c.nx)%c.nx,dy=(y+lbm::cy(i)+c.ny)%c.ny,dz=(z+lbm::cz(i)+c.nz)%c.nz;
            if(lbm::solid_type(dx,dy,dz,c)) {
                int opposite=-1;
                for(int j=0;j<19;++j)if(lbm::cx(j)==-lbm::cx(i)&&lbm::cy(j)==-lbm::cy(i)&&lbm::cz(j)==-lbm::cz(i))opposite=j;
                require(opposite>=0,"Missing reverse direction");
                expected[opposite*c.cells()+cell]=labels[i*c.cells()+cell];
            }else expected[i*c.cells()+lbm::cell_index(dx,dy,dz,c.nx,c.ny)]=labels[i*c.cells()+cell];
        }
    }
    sim.stream_only();require(sim.download()==expected,"Channel bounce-back routing mismatch");
    sim.initialize();for(int step=0;step<100;++step)sim.step();
    auto f=sim.fields();
    for(std::size_t j=0;j<c.cells();++j){
        require(std::abs(f.density[j]-(f.solid[j]?0:1))<c.tolerance,"Channel rest density mismatch");
        for(int k=0;k<3;++k)require(std::abs(f.velocity[3*j+k])<c.tolerance,"Channel rest velocity mismatch");
    }
    c.force_x=1e-5;lbm::Simulation forced(c);forced.initialize();
    const auto before=forced.download();forced.collide_only();const auto after=forced.download();
    for(int z=0;z<c.nz;++z)for(int y=0;y<c.ny;++y)for(int x=0;x<c.nx;++x){
        const auto cell=lbm::cell_index(x,y,z,c.nx,c.ny);
        if(lbm::solid_type(x,y,z,c))continue;
        double dm=0,dx=0,dy=0,dz=0;
        for(int i=0;i<19;++i){double d=after[i*c.cells()+cell]-before[i*c.cells()+cell];
            dm+=d;dx+=d*lbm::cx(i);dy+=d*lbm::cy(i);dz+=d*lbm::cz(i);}
        require(std::abs(dm)<c.tolerance&&std::abs(dx-c.force_x)<c.tolerance
            &&std::abs(dy)<c.tolerance&&std::abs(dz)<c.tolerance,"Force mass/momentum mismatch");
    }
    forced.initialize();double initial_mass=0;
    for(double rho:forced.fields().density)initial_mass+=rho;
    for(int step=0;step<200;++step)forced.step();
    f=forced.fields();double mass=0,mean=0;
    for(std::size_t j=0;j<c.cells();++j){
        require(std::isfinite(f.density[j]),"Invalid channel density");
        if(f.solid[j]){for(int k=0;k<3;++k)require(f.velocity[3*j+k]==0,"Moving solid");continue;}
        require(f.density[j]>0,"Nonpositive channel density");mass+=f.density[j];
        for(int k=0;k<3;++k)require(std::isfinite(f.velocity[3*j+k]),"Invalid channel velocity");
        mean+=f.velocity[3*j];
    }
    require(std::abs(mass-initial_mass)/initial_mass<c.tolerance&&mean>0,"Channel mass or flow-direction failure");
    std::cout<<"Channel reflection, zero-force rest, force moments and driven mass: PASS\n";
}
void run(lbm::Config c,const std::filesystem::path& directory,bool bump) {
    c.validate();
    // Avoid mixing new snapshots with old runs or silently replacing results.
    if(std::filesystem::exists(directory)&&!std::filesystem::is_empty(directory))
        throw std::runtime_error("Output directory is not empty; choose a new --output directory");
    std::filesystem::create_directories(directory);
    std::ofstream diagnostics;diagnostics.exceptions(std::ios::failbit|std::ios::badbit);
    diagnostics.open(directory/"diagnostics.csv");
    diagnostics<<std::setprecision(17)<<"step,mass,relative_mass_error,min_density,max_density,max_speed,mean_ux,max_mach\n";
    std::ofstream metadata;metadata.exceptions(std::ios::failbit|std::ios::badbit);
    metadata.open(directory/"run.txt");
    metadata<<std::setprecision(17)<<"D3Q19 BGK; double; lattice units\n"
            <<"channel="<<c.channel<<"\nobstacle="<<c.obstacle<<"\nobstacle_radius="<<c.obstacle_radius<<"\nforce_x="<<(c.channel?c.force_x:0)<<"\nboundaries="<<(c.channel?"periodic x, bounce-back y/z walls and obstacle":"periodic xyz")<<"\n"
            <<"grid="<<c.nx<<'x'<<c.ny<<'x'<<c.nz<<"\nsteps="<<c.steps<<"\noutput_every="<<c.output_every
            <<"\ntau="<<c.tau<<"\nbump="<<bump<<"\nbump_width="<<c.bump_width<<"\ndensity_increase="<<c.density_increase<<'\n';metadata.close();
    lbm::Simulation sim(c);sim.initialize(bump);
    std::size_t fluid_cells=0;
    for(int z=0;z<c.nz;++z)for(int y=0;y<c.ny;++y)for(int x=0;x<c.nx;++x)
        fluid_cells+=!lbm::solid_type(x,y,z,c);
    const double expected_mass=fluid_cells+(bump?double(c.bump_width)*c.bump_width*c.bump_width*c.density_increase:0);
    std::vector<std::pair<int,std::string>> frames;
    int step=0;
    while(true) {
        // Validate fields each step, but only persist full fields at output intervals.
        const auto f=sim.fields();double mass=0,minrho=1e300,maxrho=0,maxspeed=0,sumux=0;
        for(std::size_t cell=0;cell<c.cells();++cell) {
            if(f.solid[cell]!=0)continue;
            const double rho=f.density[cell],u=f.velocity[3*cell],v=f.velocity[3*cell+1],w=f.velocity[3*cell+2];
            const double speed=std::sqrt(u*u+v*v+w*w);
            require(std::isfinite(rho)&&rho>0&&std::isfinite(speed),"Invalid field at step "+std::to_string(step));
            sumux+=u;mass+=rho;minrho=std::min(minrho,rho);maxrho=std::max(maxrho,rho);maxspeed=std::max(maxspeed,speed);
        }
        const double error=std::abs(mass-expected_mass)/expected_mass;
        require(std::isfinite(mass)&&error<=c.tolerance,"Mass error at step "+std::to_string(step));
        if(!bump&&!c.channel)check_equilibrium(sim,c);
        const double mach=maxspeed*std::sqrt(3.0);
        if(c.channel)require(mach<0.1,"Channel exceeded chosen low-Mach limit 0.1; reduce force");
        diagnostics<<step<<','<<mass<<','<<error<<','<<minrho<<','<<maxrho<<','<<maxspeed<<','<<sumux/fluid_cells<<','<<mach<<'\n';diagnostics.flush();
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
        std::cout<<"D3Q19 CUDA: --device-info | --check | --rest | --disturbance | --channel\n"
                 <<"Run options: --steps N --output-every N --output DIRECTORY [--no-obstacle]\n"
                 <<"Default: 32x32x32, 50 steps, central 4^3 cube at rho=1.10, background=1.\n";return 0;
    }
    require(mode=="--device-info"||mode=="--check"||mode=="--rest"||mode=="--disturbance"||mode=="--channel","Unknown mode; use --help");
    lbm::Config c;std::filesystem::path output="output";
    if(mode=="--channel"){c.channel=true;c.steps=3000;c.output_every=50;output="channel_output";}
    for(int i=2;i<argc;++i) {
        require(mode=="--rest"||mode=="--disturbance"||mode=="--channel","Options apply only to run modes");
        const std::string option=argv[i];
        if(option=="--no-obstacle"){require(c.channel,"--no-obstacle requires --channel");c.obstacle=false;continue;}
        require(i+1<argc,"Missing option value");const std::string value=argv[++i];
        if(option=="--steps")c.steps=integer(value);
        else if(option=="--output-every")c.output_every=integer(value);
        else if(option=="--output")output=value;
        else throw std::invalid_argument("Unknown option: "+option);
    }
    c.validate();int count=0;CUDA_CHECK(cudaGetDeviceCount(&count));require(count>0,"No CUDA GPU visible");
    CUDA_CHECK(cudaSetDevice(0));cudaDeviceProp prop{};CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    std::cout<<"GPU: "<<prop.name<<" compute capability "<<prop.major<<'.'<<prop.minor<<'\n';
    if(mode=="--device-info")return 0;
    if(mode=="--check"){checks();channel_checks();}else run(c,output,mode=="--disturbance");
    return 0;
 }catch(const std::exception& e){std::cerr<<"D3Q19 failed: "<<e.what()<<'\n';return 1;}
}
