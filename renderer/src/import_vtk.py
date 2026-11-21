"""Convert the project's uncompressed VTI/PVD output into an offline WebGL viewer."""
import argparse
import base64
import json
import re
import struct
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TYPES = {"Float64":"f8", "Float32":"f4", "UInt8":"u1", "Int32":"i4"}


def read_vti(path):
    data = path.read_bytes()
    marker = re.search(rb'<AppendedData\s+encoding=["\x27]raw["\x27]\s*>\s*_', data)
    payload = None
    if marker:
        payload = data[marker.end():]
        root = ET.fromstring(data[:marker.start()]+b"</VTKFile>")
    else:
        root = ET.fromstring(data)
    if root.attrib.get("type")!="ImageData" or root.attrib.get("compressor"):
        raise ValueError("Expected uncompressed VTK ImageData")
    image = root.find("ImageData")
    pieces = image.findall("Piece")
    if len(pieces)!=1 or pieces[0].attrib["Extent"]!=image.attrib["WholeExtent"]:
        raise ValueError("Expected one complete VTI piece")
    extent = list(map(int,image.attrib["WholeExtent"].split()))
    if len(extent)!=6:
        raise ValueError("Invalid grid extent")
    shape = tuple(extent[2*i+1]-extent[2*i]+1 for i in range(3))
    if min(shape)<2:
        raise ValueError("Expected a three-dimensional point grid")
    spacing = list(map(float,image.attrib.get("Spacing","1 1 1").split()))
    origin = list(map(float,image.attrib.get("Origin","0 0 0").split()))
    if len(spacing)!=3 or len(origin)!=3 or not np.isfinite(spacing+origin).all() or min(spacing)<=0:
        raise ValueError("Invalid grid coordinates")
    endian = "<" if root.attrib.get("byte_order","LittleEndian")=="LittleEndian" else ">"
    header = root.attrib.get("header_type","UInt32")
    if header not in ("UInt32","UInt64"):
        raise ValueError("Unsupported binary header type")
    header_format = endian+("Q" if header=="UInt64" else "I")
    header_bytes = struct.calcsize(header_format)
    fields = {}
    for array in pieces[0].findall("PointData/DataArray"):
        kind = array.attrib["type"]
        if kind not in TYPES:
            continue
        dtype = np.dtype(endian+TYPES[kind])
        components = int(array.attrib.get("NumberOfComponents",1))
        count = int(np.prod(shape))*components
        fmt = array.attrib.get("format","ascii")
        if fmt=="ascii":
            values = np.fromstring(array.text or "",sep=" ",dtype=dtype)
        elif fmt=="appended" and payload is not None:
            offset = int(array.attrib["offset"])
            if offset<0 or offset+header_bytes>len(payload):
                raise ValueError("Invalid appended array offset")
            size = struct.unpack_from(header_format,payload,offset)[0]
            if size!=count*dtype.itemsize or offset+header_bytes+size>len(payload):
                raise ValueError("Invalid appended array length")
            values = np.frombuffer(payload,dtype=dtype,count=count,offset=offset+header_bytes)
        else:
            raise ValueError("Supported arrays use ASCII or uncompressed raw appended data")
        if values.size!=count or not np.isfinite(values).all():
            raise ValueError("Invalid point array")
        fields[array.attrib["Name"]] = values.reshape(shape[2],shape[1],shape[0],components)
    if not fields:
        raise ValueError("No supported point fields")
    geometry = dict(shape=shape,spacing=spacing,origin=origin,extent=extent)
    return geometry,fields


def encode(array):
    values = np.asarray(array,dtype="<f4",order="C")
    if not np.isfinite(values).all():
        raise ValueError("Field overflows browser Float32 storage")
    return base64.b64encode(zlib.compress(values.tobytes(),6)).decode()


def convert(source,destination,fill_field=None):
    if source.suffix.lower()==".pvd":
        root = ET.parse(source).getroot()
        if root.attrib.get("type")!="Collection":
            raise ValueError("Expected a PVD Collection")
        series = [(float(d.attrib["timestep"]),source.parent/d.attrib["file"])
                  for d in root.findall("Collection/DataSet")]
    elif source.suffix.lower()==".vti":
        series = [(0.,source)]
    else:
        raise ValueError("Select a .pvd series or .vti snapshot")
    if not series or not np.isfinite([t for t,_ in series]).all() or any(a[0]>=b[0] for a,b in zip(series,series[1:])):
        raise ValueError("Expected strictly increasing finite timestamps")
    frames = []
    geometry = None
    ranges = {"density":[float("inf"),-float("inf")],"speed":[0.,0.]}
    solid_presence = None
    has_solids = False
    for time,path in series:
        grid,fields = read_vti(path)
        if geometry is None:
            geometry = grid
        if grid!=geometry:
            raise ValueError("Grid changes between snapshots")
        rho,vel = fields.get("density"),fields.get("velocity")
        if rho is None or rho.shape[-1]!=1 or vel is None or vel.shape[-1]!=3:
            raise ValueError("Each frame requires point density and three-component velocity")
        solid = fields.get("solid")
        present = solid is not None
        if solid_presence is not None and present!=solid_presence:
            raise ValueError("Solid mask availability changes between snapshots")
        solid_presence = present
        if present and (solid.shape[-1]!=1 or not np.isin(solid,[0,1,2]).all()):
            raise ValueError("Expected solid codes 0=fluid, 1=wall, 2=obstacle")
        fluid = solid[...,0]==0 if present else np.ones(rho.shape[:3],dtype=bool)
        if not fluid.any():
            raise ValueError("Frame contains no fluid sites")
        ranges["density"][0] = min(ranges["density"][0],float(rho[...,0][fluid].min()))
        ranges["density"][1] = max(ranges["density"][1],float(rho[...,0][fluid].max()))
        ranges["speed"][1] = max(ranges["speed"][1],float(np.linalg.norm(vel[fluid],axis=-1).max()))
        frame = dict(time=time,density=encode(rho),velocity=encode(vel))
        if present:
            frame["solid"]=encode(solid)
            has_solids |= bool((solid>0).any())
        if fill_field:
            fraction = fields.get(fill_field)
            if fraction is None or fraction.shape[-1]!=1 or fraction.min()<0 or fraction.max()>1:
                raise ValueError("Liquid fraction must be a scalar point array in [0,1]")
            frame["fill"] = encode(fraction)
        frames.append(frame)
    packet = dict(grid=geometry,frames=frames,ranges=ranges,fillField=fill_field,source=source.name,hasSolids=has_solids)
    metadata=source.parent/"run.txt"
    if metadata.exists():
        settings=dict(line.split("=",1) for line in metadata.read_text().splitlines() if "=" in line)
        closed=settings.get("closed_ends")=="1"
        if settings.get("channel")=="1":
            packet["label"]="Closed box" if closed else "Periodic channel"
            packet["closedEnds"]=closed
        if "steps" in settings:
            requested=int(settings["steps"])
            packet["requestedSteps"]=requested
            packet["partial"]=frames[-1]["time"]<requested
    template = (ROOT/"src/vtk_view.html").read_text(encoding="utf8")
    library = (ROOT/"vendor/three.min.js").read_text(encoding="utf8")
    surface = (ROOT/"src/volume_replay.js").read_text(encoding="utf-8-sig")
    tracers = (ROOT/"src/vtk_tracers.js").read_text(encoding="utf8")
    html = template.replace("/* THREE */",library).replace("/* SURFACE */",surface).replace("/* TRACERS */",tracers)
    html = html.replace("/* DATA */",json.dumps(packet,separators=(",",":")).replace("</","<\\/"))
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(html,encoding="utf8")
    print(f"Imported {len(frames)} frames, grid {geometry['shape']}, times {frames[0]['time']}..{frames[-1]['time']}")
    print(f"{destination}: {destination.stat().st_size:,} bytes")


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("destination",type=Path)
    p.add_argument("--fill-field",help="Optional liquid fraction point array in [0,1]")
    args = p.parse_args()
    convert(args.source,args.destination,args.fill_field)
