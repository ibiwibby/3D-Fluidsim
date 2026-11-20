#include "lbm/output.hpp"
#include <cstdint>
#include <fstream>
#include <sstream>
namespace lbm {
namespace {
std::ofstream file(const std::filesystem::path& path) {
    std::ofstream out;out.exceptions(std::ios::failbit|std::ios::badbit);
    out.open(path,std::ios::binary|std::ios::trunc);return out;
}
void block(std::ofstream& out,const std::vector<Real>& data) {
    const std::uint64_t bytes=data.size()*sizeof(Real);
    out.write(reinterpret_cast<const char*>(&bytes),sizeof(bytes));
    out.write(reinterpret_cast<const char*>(data.data()),static_cast<std::streamsize>(bytes));
}
}
void write_vti(const std::filesystem::path& path,const Config& c,const Fields& f,int step) {
    static_assert(sizeof(Real)==8,"VTK Float64 output requires doubles");
    c.validate();
    if(f.density.size()!=c.cells()||f.velocity.size()!=3*c.cells())throw std::invalid_argument("Invalid field lengths");
    if(!f.solid.empty()&&f.solid.size()!=c.cells())throw std::invalid_argument("Invalid solid mask size");
    const std::vector<Real> empty_mask(f.solid.empty()?c.cells():0,0);
    const auto& mask=f.solid.empty()?empty_mask:f.solid;
    const std::uint16_t one=1;
    if(*reinterpret_cast<const unsigned char*>(&one)!=1)throw std::runtime_error("Writer requires little-endian host");
    std::ostringstream extent;extent<<"0 "<<c.nx-1<<" 0 "<<c.ny-1<<" 0 "<<c.nz-1;
    auto out=file(path);
    out<<"<?xml version=\"1.0\"?>\n<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n"
       <<"<ImageData WholeExtent=\""<<extent.str()<<"\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n"
       <<"<FieldData><DataArray type=\"Float64\" Name=\"TimeValue\" NumberOfTuples=\"1\" format=\"ascii\">"<<step<<"</DataArray></FieldData>\n"
       <<"<Piece Extent=\""<<extent.str()<<"\"><PointData Scalars=\"density\" Vectors=\"velocity\">\n"
       <<"<DataArray type=\"Float64\" Name=\"density\" NumberOfComponents=\"1\" format=\"appended\" offset=\"0\"/>\n"
       <<"<DataArray type=\"Float64\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"appended\" offset=\""
       <<sizeof(std::uint64_t)+f.density.size()*sizeof(Real)<<"\"/>\n"
       <<"<DataArray type=\"Float64\" Name=\"solid\" NumberOfComponents=\"1\" format=\"appended\" offset=\""
       <<2*sizeof(std::uint64_t)+(f.density.size()+f.velocity.size())*sizeof(Real)<<"\"/>\n"
       <<"</PointData><CellData/></Piece></ImageData>\n<AppendedData encoding=\"raw\">_";
    block(out,f.density);block(out,f.velocity);block(out,mask);
    out<<"\n</AppendedData></VTKFile>\n";out.close();
}
void write_pvd(const std::filesystem::path& path,const std::vector<std::pair<int,std::string>>& frames) {
    auto out=file(path);
    out<<"<?xml version=\"1.0\"?>\n<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\"><Collection>\n";
    for(const auto& frame:frames) {
        // Filenames are generated internally, not arbitrary XML input.
        if(frame.second.find_first_of("\"<>&")!=std::string::npos)throw std::invalid_argument("Invalid snapshot filename");
        out<<"<DataSet timestep=\""<<frame.first<<"\" group=\"\" part=\"0\" file=\""<<frame.second<<"\"/>\n";
    }
    out<<"</Collection></VTKFile>\n";out.close();
}
}
