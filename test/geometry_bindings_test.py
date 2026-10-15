#!/usr/bin/env python3
"""Compare the generated Python codec against a corpus emitted by geometry_json_test."""
import json
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bindings/python'))
from xgc2_math.geometry import encode_geometry, decode_geometry
corpus=json.loads(subprocess.check_output([sys.argv[1]],text=True))
for entry in corpus:
    kind,data=encode_geometry(entry['geometry'])
    assert kind==entry['type'] and list(data)==entry['parameters']
    assert decode_geometry(kind,data)==entry['geometry']
    for broken in [data[:-1],data+b'\0']:
        try:decode_geometry(kind,broken)
        except ValueError:pass
        else:raise AssertionError('malformed payload accepted')
print(f'Python/C++: {len(corpus)} layouts byte-identical; JSON flat triangles and malformed lengths pass')
