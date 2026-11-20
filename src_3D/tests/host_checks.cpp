#include "lbm/d3q19.hpp"
#include "lbm/output.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <set>
#include <tuple>
int main() {
    double sum=0,first[3]={},second[3][3]={},fourth[3][3]={};
    std::set<std::tuple<int,int,int>> directions;
    for(int i=0;i<lbm::Q;++i){
        const int v[3]={lbm::cx(i),lbm::cy(i),lbm::cz(i)};
        directions.emplace(v[0],v[1],v[2]);
        assert(std::abs(v[0])+std::abs(v[1])+std::abs(v[2])<=2);
        sum+=lbm::weight(i);
        for(int a=0;a<3;++a){first[a]+=lbm::weight(i)*v[a];for(int b=0;b<3;++b){
            second[a][b]+=lbm::weight(i)*v[a]*v[b];
            fourth[a][b]+=lbm::weight(i)*v[a]*v[a]*v[b]*v[b];
        }}
    }
    assert(directions.size()==19&&std::abs(sum-1)<1e-14);
    for(const auto& v:directions)assert(directions.count({-std::get<0>(v),-std::get<1>(v),-std::get<2>(v)}));
    for(int a=0;a<3;++a){assert(std::abs(first[a])<1e-14);for(int b=0;b<3;++b){
        assert(std::abs(second[a][b]-(a==b?1./3:0))<1e-14);
        assert(std::abs(fourth[a][b]-(a==b?1./3:1./9))<1e-14);
    }}
    double rho=0,mx=0,my=0,mz=0;
    for(int i=0;i<19;++i){double f=lbm::equilibrium(i,1.1,.02,-.01,.015);rho+=f;mx+=f*lbm::cx(i);my+=f*lbm::cy(i);mz+=f*lbm::cz(i);}
    assert(std::abs(rho-1.1)<1e-14&&std::abs(mx/rho-.02)<1e-14&&std::abs(my/rho+.01)<1e-14&&std::abs(mz/rho-.015)<1e-14);
    for(int i=0;i<19;++i){const int o=lbm::opposite(i);assert(lbm::opposite(o)==i);
        assert(lbm::cx(o)==-lbm::cx(i)&&lbm::cy(o)==-lbm::cy(i)&&lbm::cz(o)==-lbm::cz(i));}
    double source_mass=0,source_momentum=0;
    for(int i=0;i<19;++i){const auto f=lbm::forcing(i,.02,-.01,.015,1e-5,1.0);
        source_mass+=f;source_momentum+=f*lbm::cx(i);}
    assert(std::abs(source_mass)<1e-18&&std::abs(source_momentum-0.5e-5)<1e-18);
    lbm::Config channel;channel.channel=true;channel.validate();
    assert(lbm::solid_type(0,0,16,channel)==1&&lbm::solid_type(0,16,0,channel)==1);
    assert(lbm::solid_type(15,15,15,channel)==2&&lbm::solid_type(0,15,15,channel)==0);
    assert(lbm::solid_type(31,15,15,channel)==0);
    channel.obstacle=false;assert(lbm::solid_type(15,15,15,channel)==0);
    lbm::Config c;c.validate();assert(c.cells()==32768&&c.bytes()==4980736);
    c.nx=3;c.ny=4;c.nz=5;c.bump_width=2;c.validate();
    lbm::Fields fields;fields.density.resize(c.cells());fields.velocity.resize(3*c.cells());
    for(int z=0;z<c.nz;++z)for(int y=0;y<c.ny;++y)for(int x=0;x<c.nx;++x){
        const auto cell=lbm::cell_index(x,y,z,c.nx,c.ny);
        fields.density[cell]=1+x+10*y+100*z;
        fields.velocity[3*cell]=x;fields.velocity[3*cell+1]=y;fields.velocity[3*cell+2]=z;
    }
    lbm::write_vti("build/host/sample.vti",c,fields,7);
    lbm::write_pvd("build/host/sample.pvd",{{7,"sample.vti"}});
    c.nx=0;bool rejected=false;try{c.validate();}catch(const std::invalid_argument&){rejected=true;}assert(rejected);
    std::cout<<"PASS: D3Q19 symmetry/moments, equilibrium, dimensions and output fixture\n";
}
