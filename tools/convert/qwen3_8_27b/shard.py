"""Stream exact packed TP2 shards into one NInfer v2 container.

Usage: python -m tools.convert.qwen3_8_27b.shard --input SOURCE.ninfer
       --output DESTINATION.ninfer --tensor-parallel 2
No model inference, tensor library, dequantization, or quantization is involved.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from math import prod
import os
from pathlib import Path
import tempfile
from typing import Iterable

from tools.artifact.container import (Artifact, ArtifactIdentity, ArtifactWriter,
    ResourceObject, ResourceSpec, TensorObject, plan_objects)
from tools.artifact.layouts import row_split_geometry
from tools.artifact.numeric import get_format, QuantFormat
from . import inventory as source
from .tp2_inventory import (MODEL_ID, WEIGHTS_ID, RESOURCE_NAME, TensorPlan,
                            build_plan, manifest_bytes)

COPY_CHUNK_BYTES = 1024 * 1024


def _intervals(intervals, limit):
    checked = tuple(intervals)
    if not checked:
        raise ValueError("selection must not be empty")
    previous = 0
    for begin,end in checked:
        if type(begin) is not int or type(end) is not int or not 0 <= begin < end <= limit or begin < previous:
            raise ValueError("selection intervals must be ordered, disjoint and in bounds")
        previous = end
    return checked


def packed_chunks(payload, format: str, shape: tuple[int,int], rows=None, columns=None) -> Iterable[bytes]:
    """Copy plane-major rows/groups with compact strides and zero new padding.

    columns is one group-aligned half-open K interval. No numerical interpretation
    occurs, including for signed codes and FP16 scale words. chunks own bytes so
    consumers cannot retain exports of the source mmap.
    """
    numeric = get_format(format)
    if not isinstance(numeric,QuantFormat):
        raise ValueError("packed selection requires grouped integer storage")
    old = row_split_geometry(numeric,shape)
    if len(payload) != old.payload_bytes:
        raise ValueError("packed source payload byte length mismatch")
    rows = _intervals(((0,shape[0]),) if rows is None else rows,shape[0])
    selected_n = sum(end-begin for begin,end in rows)
    if columns is None:
        begin,end = 0,old.k_pad
        selected_k = shape[1]
    else:
        ((begin,end),) = _intervals((columns,),shape[1])
        if begin % numeric.group_size or end % numeric.group_size:
            raise ValueError("K cuts must align to quantization groups")
        selected_k = end-begin
    new = row_split_geometry(numeric,(selected_n,selected_k))
    group_begin = begin//numeric.group_size
    group_count = (end-begin)//numeric.group_size
    cursor = 0
    for old_offset,old_stride,new_offset,new_stride,unit in (
        (old.base_offset,old.base_row_bytes,new.base_offset,new.base_row_bytes,old.base_bytes_per_group),
        (old.high_offset,old.high_row_bytes,new.high_offset,new.high_row_bytes,old.high_bytes_per_group),
        (old.scale_offset,old.scale_row_bytes,new.scale_offset,new.scale_row_bytes,2)):
        if new_offset > cursor:
            yield bytes(new_offset-cursor)
            cursor = new_offset
        if not unit:
            continue
        count = group_count*unit
        for row_begin,row_end in rows:
            for row in range(row_begin,row_end):
                offset = old_offset+row*old_stride+group_begin*unit
                yield bytes(payload[offset:offset+count])
                if new_stride > count:
                    yield bytes(new_stride-count)
                cursor += new_stride
    if cursor != new.payload_bytes:
        raise ValueError("internal packed selection length mismatch")


def direct_chunks(payload, shape, word_bytes, rows=None, columns=None):
    if len(payload) != prod(shape)*word_bytes:
        raise ValueError("direct source payload byte length mismatch")
    selected_rows = _intervals(((0,shape[0]),) if rows is None else rows,shape[0])
    width = prod(shape[1:])
    selected_columns = _intervals(((0,width),) if columns is None else columns,width)
    for begin,end in selected_rows:
        for row in range(begin,end):
            for col_begin,col_end in selected_columns:
                offset = (row*width+col_begin)*word_bytes
                count = (col_end-col_begin)*word_bytes
                for start in range(offset,offset+count,COPY_CHUNK_BYTES):
                    yield bytes(payload[start:min(start+COPY_CHUNK_BYTES,offset+count)])


def validate_source(artifact: Artifact) -> bool:
    """Require the complete exact source profile before any destination exists."""
    if artifact.identity != ArtifactIdentity(MODEL_ID,source.WEIGHTS_ID):
        raise ValueError("source must be qwen3.8-27b/groupwise-int NInfer v2")
    names = {obj.name for obj in artifact.objects}
    dflash2 = any(name.startswith("dflash2/") for name in names)
    tensors = source.BASE_TENSOR_SPECS + (source.DFLASH2_TENSOR_SPECS if dflash2 else ())
    expected_names = {s.name for s in tensors} | {s.name for s in source.RESOURCE_SPECS}
    if names != expected_names:
        raise ValueError(f"source inventory mismatch: missing={sorted(expected_names-names)}, extra={sorted(names-expected_names)}")
    for spec in tensors:
        obj = artifact.find(spec.name)
        if not isinstance(obj,TensorObject) or (obj.shape,obj.format,obj.layout) != (spec.shape,spec.format,spec.layout):
            raise ValueError(f"source tensor contract mismatch: {spec.name}")
    for spec in source.RESOURCE_SPECS:
        obj = artifact.find(spec.name)
        if not isinstance(obj,ResourceObject) or obj.encoding != spec.encoding:
            raise ValueError(f"source resource contract mismatch: {spec.name}")
    return dflash2


def _copy_payload(artifact, name):
    with artifact.payload(name) as payload:
        for start in range(0,len(payload),COPY_CHUNK_BYTES):
            yield bytes(payload[start:start+COPY_CHUNK_BYTES])


def tensor_chunks(artifact: Artifact, item: TensorPlan):
    if item.rows is None and item.columns is None:
        yield from _copy_payload(artifact,item.source_name)
        return
    with artifact.payload(item.source_name) as payload:
        if item.spec.layout == "row-split-k128-v1":
            columns = None if item.columns is None else item.columns[0]
            if item.columns is not None and len(item.columns) != 1:
                raise ValueError("packed K selection requires a single interval")
            yield from packed_chunks(payload,item.spec.format,item.source_shape,item.rows,columns)
        else:
            yield from direct_chunks(payload,item.source_shape,get_format(item.spec.format).word_bytes,item.rows,item.columns)


def convert(input_path: str | Path, output_path: str | Path, tensor_parallel: int = 2) -> None:
    if type(tensor_parallel) is not int or tensor_parallel != 2:
        raise ValueError("only tensor-parallel 2 is supported")
    input_path,output_path = Path(input_path),Path(output_path)
    if input_path.resolve() == output_path.resolve() or (output_path.exists() and input_path.samefile(output_path)):
        raise ValueError("input and output must be different files")
    temporary = None
    with Artifact(input_path) as artifact:
        optional = validate_source(artifact)
        plan = build_plan(optional)
        resources = tuple(ResourceSpec(s.name,s.encoding,artifact.find(s.name).bytes) for s in source.RESOURCE_SPECS)
        metadata = manifest_bytes(optional)
        specs = resources + (ResourceSpec(RESOURCE_NAME,"raw-bytes-v1",len(metadata)),) + tuple(p.spec for p in plan)
        plan_objects(specs)  # All destination contracts checked before mkstemp.
        try:
            descriptor,temporary = tempfile.mkstemp(prefix=f".{output_path.name}.",suffix=".tmp",dir=output_path.parent)
            os.close(descriptor)
            with ArtifactWriter(temporary,ArtifactIdentity(MODEL_ID,WEIGHTS_ID),specs) as writer:
                for resource in resources:
                    with closing(_copy_payload(artifact,resource.name)) as chunks:
                        writer.write(resource.name,chunks)
                writer.write(RESOURCE_NAME,metadata)
                for item in plan:
                    with closing(tensor_chunks(artifact,item)) as chunks:
                        writer.write(item.spec.name,chunks)
            os.replace(temporary,output_path)
            temporary = None
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--tensor-parallel",required=True,type=int,choices=(2,))
    args = parser.parse_args(argv)
    try:
        convert(args.input,args.output,args.tensor_parallel)
    except (ValueError,OSError) as exc:
        parser.exit(2,f"error: {exc}\n")


if __name__ == "__main__":
    main()
