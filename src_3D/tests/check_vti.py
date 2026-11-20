"""Independently inspect binary framing, field order and series metadata."""
from pathlib import Path
import struct
import sys
import xml.etree.ElementTree as ET
path = Path(sys.argv[1])
data = path.read_bytes()
prefix, payload = data.split(b'<AppendedData encoding="raw">_', 1)
root = ET.fromstring(prefix + b'</VTKFile>')
image = root.find('ImageData')
assert image.attrib['WholeExtent'] == '0 2 0 3 0 4'
assert image.find('FieldData/DataArray').text == '7'
arrays = image.findall('Piece/PointData/DataArray')
assert [a.attrib['Name'] for a in arrays] == ['density', 'velocity']
for a in arrays:
    offset = int(a.attrib['offset'])
    length, = struct.unpack_from('<Q', payload, offset)
    n = 60 * int(a.attrib['NumberOfComponents'])
    assert length == n * 8
    actual = struct.unpack_from(f'<{n}d', payload, offset + 8)
    expected = []
    for z in range(5):
        for y in range(4):
            for x in range(3):
                expected.extend([1+x+10*y+100*z] if a.attrib['Name']=='density' else [x,y,z])
    assert actual == tuple(expected)
assert payload[8+60*8+8+180*8:] == b'\n</AppendedData></VTKFile>\n'
pvd = ET.parse(path.with_suffix('.pvd')).getroot()
assert pvd.find('Collection/DataSet').attrib['timestep'] == '7'
print('PASS: VTI binary offsets, sizes, xyz ordering, vector components, time and PVD')
