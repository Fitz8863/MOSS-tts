#!/usr/bin/env python3
import argparse, json, os
from collections import Counter
import onnx
from onnx import TensorProto

def dtype_name(t):
    return TensorProto.DataType.Name(t.elem_type if hasattr(t, 'elem_type') else t.data_type)

def inspect(path):
    print(f'\n=== {path} ({os.path.getsize(path)/1024/1024:.2f} MiB) ===')
    m=onnx.load(path, load_external_data=True)
    print('ir',m.ir_version,'opset',[(x.domain or 'ai.onnx',x.version) for x in m.opset_import])
    print('inputs',[(x.name,[d.dim_value if d.dim_value else d.dim_param for d in x.type.tensor_type.shape.dim],dtype_name(x.type.tensor_type)) for x in m.graph.input])
    print('outputs',[(x.name,[d.dim_value if d.dim_value else d.dim_param for d in x.type.tensor_type.shape.dim],dtype_name(x.type.tensor_type)) for x in m.graph.output])
    print('nodes',len(m.graph.node),Counter(n.op_type for n in m.graph.node))
    init=Counter(dtype_name(x) for x in m.graph.initializer)
    print('initializers',len(m.graph.initializer),init)
    print('external',sum(1 for x in m.graph.initializer if x.data_location==onnx.TensorProto.EXTERNAL))
    print('metadata',{x.key:x.value for x in m.metadata_props})

ap=argparse.ArgumentParser(); ap.add_argument('paths',nargs='+'); a=ap.parse_args()
for p in a.paths: inspect(p)
