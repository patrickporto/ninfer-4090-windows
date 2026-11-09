"""Closed Qwen3.8-27B TP2 storage contract (no runtime model graph).

MANIFEST_FIELDS is the complete v1 schema. Arrays are ordered and compared to
manifest() exactly, including aliases. owner_resources covers original frontend
resources only; parallel/tp2_inventory.json is consumed separately by the binder.
Owner physical names stay unchanged; decoder physical names gain rank0/ or
rank1/. Aliases never create physical objects or require another consumption.
SEMANTIC_SEGMENTS defines source segment boundaries, independently halved within
each segment, in concatenation order. Shape/format/layout remain directory-only.
"""
from __future__ import annotations

from dataclasses import dataclass
import json

from tools.artifact.container import TensorSpec
from . import inventory as source

MODEL_ID = source.MODEL_ID
WEIGHTS_ID = "groupwise-int-tp2"
SCHEMA = "qwen3.8-27b-tp2-inventory-v1"
RESOURCE_NAME = "parallel/tp2_inventory.json"
MANIFEST_FIELDS = frozenset(("schema", "source_identity", "world_size", "owner_rank",
    "rank_objects", "owner_objects", "owner_resources", "aliases", "optional_features"))
# Axis, semantic name and half-open global interval. These are fixed package
# semantics, not arbitrary instructions accepted from an artifact.
SEMANTIC_SEGMENTS = {
    "attention/query_key": (0, (("query",0,6144),("key",6144,7168))),
    "attention/gate_value": (0, (("gate",0,6144),("value",6144,7168))),
    "gdn/query_key": (0, (("query",0,2048),("key",2048,4096))),
    "gdn/value_z": (0, (("value",0,6144),("z",6144,12288))),
    "mlp/gate_up": (0, (("gate",0,17408),("up",17408,34816))),
    "gdn/convolution": (1, (("query",0,2048),("key",2048,4096),("value",4096,10240))),
    "attention/output": (1, (("query",0,6144),)),
    "gdn/output": (1, (("value",0,6144),)),
    "mlp/down": (1, (("intermediate",0,17408),)),
    "gdn/a_projection": (0, (("value_heads",0,48),)),
    "gdn/b_projection": (0, (("value_heads",0,48),)),
    "gdn/a_log": (0, (("value_heads",0,48),)),
    "gdn/dt_bias": (0, (("value_heads",0,48),)),
}
REPLICATED = frozenset(("input_norm", "post_attention_norm", "attention/query_norm",
                       "attention/key_norm", "gdn/norm"))


@dataclass(frozen=True)
class TensorPlan:
    source_name: str
    source_shape: tuple[int, ...]
    spec: TensorSpec
    rows: tuple[tuple[int, int], ...] | None = None
    columns: tuple[tuple[int, int], ...] | None = None


def build_plan(dflash2: bool = False) -> tuple[TensorPlan, ...]:
    specs = source.BASE_TENSOR_SPECS + (source.DFLASH2_TENSOR_SPECS if dflash2 else ())
    result = []
    for item in specs:
        if not item.name.startswith("text/layers/"):
            result.append(TensorPlan(item.name,item.shape,TensorSpec(item.name,item.shape,item.format,item.layout)))
            continue
        suffix = item.name.split("/",3)[3]
        for rank in range(2):
            shape = list(item.shape)
            rows = columns = None
            if suffix in SEMANTIC_SEGMENTS:
                axis, segments = SEMANTIC_SEGMENTS[suffix]
                intervals = tuple((begin+(end-begin)//2*rank,begin+(end-begin)//2*(rank+1))
                                  for _,begin,end in segments)
                shape[axis] = sum(end-begin for begin,end in intervals)
                if axis == 0: rows = intervals
                else: columns = intervals
            elif suffix not in REPLICATED:
                raise ValueError(f"unclassified decoder role: {item.name}")
            result.append(TensorPlan(item.name,item.shape,
                TensorSpec(f"rank{rank}/{item.name}",tuple(shape),item.format,item.layout),rows,columns))
    return tuple(result)


def manifest(dflash2: bool = False) -> dict:
    plan = build_plan(dflash2)
    aliases = []
    for alias in source.ALIAS_SPECS:
        if alias.layers is None:
            aliases.append({"name":alias.role_pattern,"objects":list(alias.object_patterns),"axis_order":None})
        else:
            for layer in alias.layers:
                for rank in range(2):
                    aliases.append({"name":f"rank{rank}/"+alias.role_pattern.format(l=layer),
                        "objects":[f"rank{rank}/"+name.format(l=layer) for name in alias.object_patterns],
                        "axis_order":list(alias.axis_order) if alias.axis_order else None})
    return {"schema":SCHEMA,"source_identity":{"model_id":MODEL_ID,"weights_id":source.WEIGHTS_ID},
        "world_size":2,"owner_rank":0,
        "rank_objects":[{"rank":r,"objects":[p.spec.name for p in plan if p.spec.name.startswith(f"rank{r}/")]} for r in range(2)],
        "owner_objects":[p.spec.name for p in plan if not p.spec.name.startswith("rank")],
        "owner_resources":[r.name for r in source.RESOURCE_SPECS],
        "aliases":aliases,"optional_features":{"dflash2":"preserved-disabled" if dflash2 else "absent"}}


def manifest_bytes(dflash2: bool = False) -> bytes:
    return json.dumps(manifest(dflash2),ensure_ascii=False,separators=(",",":"),sort_keys=True).encode()


def validate_manifest(value: object, dflash2: bool = False) -> None:
    # Serialized equality distinguishes JSON booleans/floats from integer ranks
    # and rejects every extra/missing/nested field and reordered coverage array.
    try:
        actual = json.dumps(value,ensure_ascii=False,separators=(",",":"),sort_keys=True).encode()
    except (TypeError,ValueError) as exc:
        raise ValueError("invalid TP2 inventory") from exc
    if actual != manifest_bytes(dflash2):
        raise ValueError("TP2 inventory does not match the closed registered recipe")
