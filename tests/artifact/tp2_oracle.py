"""Independent, bounded-memory full-byte qualification of a Qwen3.8 TP2 artifact.

Run with Python 3.11 and explicit --source SOURCE --destination DESTINATION.
No converter transform or plan is imported. The recipe below derives selections
from the architectural head/channel geometry. All source code/scale/direct words
are compared against both rank shards, and every destination padding byte is
checked. Boundary code decoding uses an independent scalar integer/FP16 oracle.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from math import prod
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.artifact.container import (
    Artifact, ArtifactIdentity, ResourceObject, TensorObject, TensorSpec,
)
from tools.convert.qwen3_8_27b import inventory as source_inventory

CHUNK = 1024 * 1024
MANIFEST_NAME = "parallel/tp2_inventory.json"
QUANT = {
    "Q4G64_F16S": (4, 64),
    "Q5G64_F16S": (5, 64),
    "Q6G64_F16S": (6, 64),
    "W8G32_F16S": (8, 32),
}


@dataclass(frozen=True)
class ExpectedTensor:
    source_name: str
    spec: TensorSpec
    rows: tuple | None = None
    columns: tuple | None = None


def decoder_selection(role, rank):
    """Fixed source geometry: Q24/KV4 D256, GDN QK16/V48 D128."""
    def part(start, local):
        return start + rank * local, start + (rank + 1) * local

    if role in {"input_norm", "post_attention_norm", "attention/query_norm",
                "attention/key_norm", "gdn/norm"}:
        return None, None
    if role in {"attention/query_key", "attention/gate_value"}:
        return (part(0, 12 * 256), part(24 * 256, 2 * 256)), None
    if role == "gdn/query_key":
        return (part(0, 8 * 128), part(16 * 128, 8 * 128)), None
    if role == "gdn/value_z":
        return (part(0, 24 * 128), part(48 * 128, 24 * 128)), None
    if role == "mlp/gate_up":
        return (part(0, 8704), part(17408, 8704)), None
    if role in {"gdn/a_log", "gdn/dt_bias", "gdn/a_projection", "gdn/b_projection"}:
        return (part(0, 24),), None
    if role == "gdn/convolution":
        return None, (part(0, 8 * 128), part(16 * 128, 8 * 128), part(32 * 128, 24 * 128))
    if role in {"attention/output", "gdn/output"}:
        return None, (part(0, 3072),)
    if role == "mlp/down":
        return None, (part(0, 8704),)
    raise ValueError(f"unknown decoder role: {role}")


def source_tensors(optional):
    return source_inventory.BASE_TENSOR_SPECS + (
        source_inventory.DFLASH2_TENSOR_SPECS if optional else ()
    )


def expected_tensors(optional):
    result = []
    for original in source_tensors(optional):
        if not original.name.startswith("text/layers/"):
            result.append(ExpectedTensor(original.name, TensorSpec(
                original.name, original.shape, original.format, original.layout)))
            continue
        role = original.name.split("/", 3)[3]
        for rank in range(2):
            rows, columns = decoder_selection(role, rank)
            shape = list(original.shape)
            if rows is not None:
                shape[0] = sum(end - begin for begin, end in rows)
            if columns is not None:
                shape[1] = sum(end - begin for begin, end in columns)
            result.append(ExpectedTensor(original.name, TensorSpec(
                f"rank{rank}/{original.name}", tuple(shape), original.format, original.layout), rows, columns))
    return tuple(result)


def expected_manifest(optional):
    plans = expected_tensors(optional)
    aliases = [
        {"name": "mtp/token_embedding", "objects": ["text/token_embedding"], "axis_order": None},
        {"name": "mtp/full_output_head", "objects": ["text/output_head"], "axis_order": None},
        {"name": "mtp/optimized_proposal_head", "objects": ["text/draft_head", "text/draft_head_token_ids"], "axis_order": None},
    ]
    for layer in range(64):
        if layer % 4 == 3:
            continue
        for rank in range(2):
            prefix = f"rank{rank}/text/layers/{layer}/gdn/"
            aliases.append({"name": prefix + "channel_major_convolution",
                            "objects": [prefix + "convolution"], "axis_order": [1, 0]})
    return {
        "schema": "qwen3.8-27b-tp2-inventory-v1",
        "source_identity": {"model_id": "qwen3.8-27b", "weights_id": "groupwise-int"},
        "world_size": 2,
        "owner_rank": 0,
        "rank_objects": [{"rank": rank, "objects": [p.spec.name for p in plans
                         if p.spec.name.startswith(f"rank{rank}/")]} for rank in range(2)],
        "owner_objects": [p.spec.name for p in plans if not p.spec.name.startswith("rank")],
        "owner_resources": [r.name for r in source_inventory.RESOURCE_SPECS],
        "aliases": aliases,
        "optional_features": {"dflash2": "preserved-disabled" if optional else "absent"},
    }


def compare_span(left, left_start, right, right_start, length, label):
    for offset in range(0, length, CHUNK):
        count = min(CHUNK, length - offset)
        a = bytes(left[left_start + offset:left_start + offset + count])
        b = bytes(right[right_start + offset:right_start + offset + count])
        if len(a) != count or len(b) != count or a != b:
            raise ValueError(f"{label}: byte mismatch at destination offset {right_start + offset}")


def zero_span(payload, start, length, label):
    for offset in range(start, start + length, CHUNK):
        count = min(CHUNK, start + length - offset)
        data = bytes(payload[offset:offset + count])
        if len(data) != count or data != bytes(count):
            raise ValueError(f"{label}: nonzero destination padding at {offset}")


def plane_geometry(shape, bits, group):
    n, k = shape
    groups = ((k + 127) // 128 * 128) // group
    base_unit = group if bits == 8 else group // 2
    high_unit = 0 if bits in (4, 8) else group * (bits - 4) // 8
    high = ((n * groups * base_unit + 255) // 256) * 256
    scales = high + ((n * groups * high_unit + 255) // 256) * 256
    return ((0, groups * base_unit, base_unit),
            (high, groups * high_unit, high_unit),
            (scales, groups * 2, 2))


def scalar_decode(payload, shape, bits, group, row, column):
    planes = plane_geometry(shape, bits, group)
    g, within = divmod(column, group)
    base, stride, unit = planes[0]
    if bits == 8:
        code = payload[base + row * stride + g * unit + within]
    else:
        packed = payload[base + row * stride + g * unit + within // 2]
        code = (packed >> (4 * (within % 2))) & 15
        if bits > 4:
            high, high_stride, high_unit = planes[1]
            bit = within * (bits - 4)
            word = payload[high + row * high_stride + g * high_unit + bit // 8]
            code |= ((word >> (bit % 8)) & ((1 << (bits - 4)) - 1)) << 4
    if code & (1 << (bits - 1)):
        code -= 1 << bits
    scale_at, scale_stride, _ = planes[2]
    scale_bytes = bytes(payload[scale_at + row * scale_stride + g * 2:
                                scale_at + row * scale_stride + g * 2 + 2])
    # Keep words in the result too: equal floating values alone lose signed-zero
    # scale identity and cannot establish preservation of quantized codes.
    return code, scale_bytes, code * struct.unpack("<e", scale_bytes)[0]


def verify_tensor(original, result, source, destination, rows=None, columns=None):
    label = destination.name
    if rows is None and columns is None:
        if len(original) != len(result):
            raise ValueError(f"{label}: replica/owner byte length differs")
        compare_span(original, 0, result, 0, len(original), label)
        return
    rows = ((0, source.shape[0]),) if rows is None else rows
    selected_rows = sum(end - begin for begin, end in rows)
    if source.layout == "contiguous-le-v1":
        word = {"BF16": 2, "FP32": 4, "I32": 4}[source.format]
        width = prod(source.shape[1:])
        columns = ((0, width),) if columns is None else columns
        expected = selected_rows * sum(end - begin for begin, end in columns) * word
        if len(original) != prod(source.shape) * word or len(result) != expected:
            raise ValueError(f"{label}: direct byte length mismatch")
        destination_offset = 0
        for begin, end in rows:
            for row in range(begin, end):
                for first, last in columns:
                    count = (last - first) * word
                    compare_span(original, (row * width + first) * word,
                                 result, destination_offset, count, label)
                    destination_offset += count
        return
    bits, group = QUANT[source.format]
    source_planes = plane_geometry(source.shape, bits, group)
    target_planes = plane_geometry(destination.shape, bits, group)
    if len(original) != source_planes[2][0] + source.shape[0] * source_planes[2][1]:
        raise ValueError(f"{label}: source byte length mismatch")
    if len(result) != target_planes[2][0] + selected_rows * target_planes[2][1]:
        raise ValueError(f"{label}: target byte length mismatch")
    logical_columns = ((0, source.shape[1]),) if columns is None else columns
    if columns is None:
        # Row selection preserves original K128 physical padding groups too.
        columns = ((0, (source.shape[1] + 127) // 128 * 128),)
    for first, last in columns:
        if first % group or last % group:
            raise ValueError(f"{label}: selection crosses a quantization group")
    cursor = 0
    for source_plane, target_plane in zip(source_planes, target_planes):
        source_offset, source_stride, unit = source_plane
        target_offset, target_stride, _ = target_plane
        zero_span(result, cursor, target_offset - cursor, label)
        cursor = target_offset
        if not unit:
            continue
        destination_row = 0
        for begin, end in rows:
            for row in range(begin, end):
                selected_bytes = 0
                for first, last in columns:
                    count = (last - first) // group * unit
                    compare_span(original, source_offset + row * source_stride + first // group * unit,
                                 result, target_offset + destination_row * target_stride + selected_bytes,
                                 count, label)
                    selected_bytes += count
                zero_span(result, target_offset + destination_row * target_stride + selected_bytes,
                          target_stride - selected_bytes, label)
                destination_row += 1
        cursor = target_offset + selected_rows * target_stride
    # Independent scalar decoding on both sides of every semantic segment,
    # including quantization-group and selected K boundary coordinates.
    destination_row = 0
    for begin, end in rows:
        for row in sorted({begin, end - 1}):
            local_row = destination_row + row - begin
            destination_column = 0
            for first, last in logical_columns:
                candidates = {first, last - 1, min(first + group - 1, last - 1), min(first + group, last - 1)}
                for column in sorted(candidates):
                    a = scalar_decode(original, source.shape, bits, group, row, column)
                    b = scalar_decode(result, destination.shape, bits, group, local_row,
                                      destination_column + column - first)
                    if a != b:
                        raise ValueError(f"{label}: scalar decode mismatch at row {local_row}")
                destination_column += last - first
        destination_row += end - begin


def strict_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate manifest field: {key}")
        value[key] = item
    return value


def verify(source_path, destination_path):
    with Artifact(source_path) as original, Artifact(destination_path) as result:
        if original.identity != ArtifactIdentity("qwen3.8-27b", "groupwise-int"):
            raise ValueError("source identity mismatch")
        if result.identity != ArtifactIdentity("qwen3.8-27b", "groupwise-int-tp2"):
            raise ValueError("destination identity mismatch")
        optional = any(o.name.startswith("dflash2/") for o in original.objects)
        source_specs = {s.name: s for s in source_tensors(optional)}
        resources = {r.name: r for r in source_inventory.RESOURCE_SPECS}
        plans = expected_tensors(optional)
        if {o.name for o in original.objects} != set(source_specs) | set(resources):
            raise ValueError("source inventory coverage mismatch")
        if {o.name for o in result.objects} != {p.spec.name for p in plans} | set(resources) | {MANIFEST_NAME}:
            raise ValueError("destination inventory coverage mismatch")
        for name, spec in source_specs.items():
            obj = original.find(name)
            if not isinstance(obj, TensorObject) or (obj.shape, obj.format, obj.layout) != (spec.shape, spec.format, spec.layout):
                raise ValueError(f"source tensor metadata mismatch: {name}")
        for plan in plans:
            obj = result.find(plan.spec.name)
            if not isinstance(obj, TensorObject) or (obj.shape, obj.format, obj.layout) != (plan.spec.shape, plan.spec.format, plan.spec.layout):
                raise ValueError(f"destination tensor metadata mismatch: {plan.spec.name}")
        manifest_obj = result.find(MANIFEST_NAME)
        if not isinstance(manifest_obj, ResourceObject) or manifest_obj.encoding != "raw-bytes-v1":
            raise ValueError("manifest resource metadata mismatch")
        if manifest_obj.bytes > 1024 * 1024:
            raise ValueError("manifest exceeds the closed recipe size bound")
        with result.payload(MANIFEST_NAME) as payload:
            actual_manifest = json.loads(bytes(payload), object_pairs_hook=strict_object)
        if json.dumps(actual_manifest, sort_keys=True) != json.dumps(expected_manifest(optional), sort_keys=True):
            raise ValueError("manifest does not equal the independent closed recipe")
        for name, spec in resources.items():
            left, right = original.find(name), result.find(name)
            if not isinstance(left, ResourceObject) or not isinstance(right, ResourceObject) or left.encoding != spec.encoding or right.encoding != spec.encoding or left.bytes != right.bytes:
                raise ValueError(f"resource metadata mismatch: {name}")
            with original.payload(name) as a, result.payload(name) as b:
                compare_span(a, 0, b, 0, len(a), name)
        for index, plan in enumerate(plans):
            with original.payload(plan.source_name) as a, result.payload(plan.spec.name) as b:
                verify_tensor(a, b, source_specs[plan.source_name], plan.spec, plan.rows, plan.columns)
            if (index + 1) % 100 == 0:
                print(f"Verified {index + 1}/{len(plans)} tensors", flush=True)
        print(f"PASS: {len(plans)} tensors, {len(resources)} resources, full byte coverage, "
              "zero shard padding and independent boundary decoding", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    try:
        verify(args.source, args.destination)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"FAIL: {exc}\n")


if __name__ == "__main__":
    main()
