"""Exact CPU-only TP2 storage tests; no torch or model allocation."""
import copy
import json
import struct
import tempfile
import unittest
from pathlib import Path


class ParallelShardsTest(unittest.TestCase):
    def test_byte_only_import(self):
        from tools.artifact.layouts import row_split_geometry
        self.assertEqual(row_split_geometry("Q5G64_F16S", (3, 192)).k_pad, 256)

    def test_exact_group_selection_and_padding(self):
        from tools.convert.qwen3_8_27b.shard import packed_chunks
        # Independent encoder/decoder: plane offsets and signed codes are not
        # obtained from production geometry or codecs.
        for bits, group, fmt in [(4,64,"Q4G64_F16S"),(5,64,"Q5G64_F16S"),(8,32,"W8G32_F16S")]:
            raw = encode(bits, group, 6, 256)
            selected = b"".join(packed_chunks(raw, fmt, (6,256), ((1,3),(5,6)), (64,192)))
            source = decode(raw,bits,group,6,256)
            actual_words = decode(selected,bits,group,3,128)
            expected_words = [source[r][64:192] for r in (1,2,5)]
            self.assertEqual(actual_words, expected_words)
            def values(rows):
                return [[code * struct.unpack("<e",struct.pack("<H",word))[0]
                         for code,word in row] for row in rows]
            self.assertEqual(values(actual_words), values(expected_words))
            padded = b"".join(packed_chunks(raw,fmt,(6,256),((0,6),),(64,128)))
            actual = decode(padded,bits,group,6,64, padded=True)
            for r in range(6):
                self.assertEqual(actual[r][:64],source[r][64:128])
                self.assertEqual(actual[r][64:],[(0,0)]*64)
            with self.assertRaises(ValueError):
                list(packed_chunks(raw,fmt,(6,256),((0,6),),(1,129)))

    def test_closed_plan_manifest(self):
        from tools.convert.qwen3_8_27b.tp2_inventory import build_plan, manifest_bytes, validate_manifest
        for optional in (False,True):
            plan=build_plan(optional)
            value=json.loads(manifest_bytes(optional))
            self.assertEqual(value["world_size"],2)
            self.assertEqual(value["optional_features"]["dflash2"], "preserved-disabled" if optional else "absent")
            names=[p.spec.name for p in plan]
            self.assertEqual(len(names),len(set(names)))
            self.assertNotIn("mtp/token_embedding",names)
            aliases=value["aliases"]
            self.assertEqual(len({a["name"] for a in aliases}),len(aliases))
            for alias in aliases:
                self.assertNotIn(alias["name"],names)
                self.assertTrue(set(alias["objects"]) <= set(names))
            covered=value["owner_objects"]+sum((r["objects"] for r in value["rank_objects"]),[])
            self.assertEqual(sorted(covered),sorted(names))
            self.assertEqual(manifest_bytes(optional),manifest_bytes(optional))
            validate_manifest(value,optional)
            for bad in ({**value,"extra":1},{**value,"world_size":True}):
                with self.assertRaises(ValueError): validate_manifest(bad,optional)
            bad=copy.deepcopy(value);bad["rank_objects"][1]["objects"].pop()
            with self.assertRaises(ValueError): validate_manifest(bad,optional)
            by_name={p.spec.name:p for p in plan}
            self.assertEqual(by_name["rank1/text/layers/3/attention/query_key"].rows,((3072,6144),(6656,7168)))
            self.assertEqual(by_name["rank1/text/layers/0/gdn/convolution"].columns,((1024,2048),(3072,4096),(7168,10240)))
            self.assertEqual(by_name["rank0/text/layers/0/mlp/down"].spec.shape,(5120,8704))


def encode(bits, group, n, k):
    groups=((k+127)//128*128)//group
    base_step=group if bits==8 else group//2
    high_step=group//8 if bits==5 else 0
    base_len=n*groups*base_step
    high_at=(base_len+255)//256*256
    scale_at=high_at+(n*groups*high_step+255)//256*256
    raw=bytearray(scale_at+n*groups*2)
    for r in range(n):
        for g in range(groups):
            struct.pack_into("<H",raw,scale_at+(r*groups+g)*2,0x3800+g+r)
            for i in range(group):
                code=(i+g*3+r*7)%(1<<bits)
                if bits==8:
                    if code==128:code=0
                    raw[(r*groups+g)*base_step+i]=code
                else:
                    raw[(r*groups+g)*base_step+i//2]|=(code&15)<<(4*(i%2))
                    if bits==5:raw[high_at+(r*groups+g)*high_step+i//8]|=(code>>4)<<(i%8)
    return bytes(raw)


def decode(raw,bits,group,n,k,padded=False):
    width=(k+127)//128*128;groups=width//group
    base_step=group if bits==8 else group//2;high_step=group//8 if bits==5 else 0
    high_at=(n*groups*base_step+255)//256*256
    scale_at=high_at+(n*groups*high_step+255)//256*256
    result=[]
    for r in range(n):
        row=[]
        for col in range(width if padded else k):
            g,i=divmod(col,group)
            if bits==8:code=raw[(r*groups+g)*base_step+i]
            else:
                code=(raw[(r*groups+g)*base_step+i//2]>>(4*(i%2)))&15
                if bits==5:code|=((raw[high_at+(r*groups+g)*high_step+i//8]>>(i%8))&1)<<4
            if code&(1<<(bits-1)):code-=1<<bits
            scale=struct.unpack_from("<H",raw,scale_at+(r*groups+g)*2)[0]
            row.append((code,scale))
        result.append(row)
    return result


class SourceAndWriterTest(unittest.TestCase):
    def test_strict_complete_source_metadata(self):
        from dataclasses import replace
        from types import SimpleNamespace
        from tools.artifact.container import ArtifactIdentity, ResourceSpec, plan_objects
        from tools.convert.qwen3_8_27b import inventory as inv
        from tools.convert.qwen3_8_27b.shard import validate_source
        from tools.artifact.container import TensorSpec
        for optional in (False,True):
            specs=[ResourceSpec(s.name,s.encoding,1) for s in inv.RESOURCE_SPECS]
            specs += [TensorSpec(s.name,s.shape,s.format,s.layout) for s in (inv.TENSOR_SPECS if optional else inv.BASE_TENSOR_SPECS)]
            objects=plan_objects(specs)
            index={o.name:o for o in objects}
            artifact=SimpleNamespace(identity=ArtifactIdentity(inv.MODEL_ID,inv.WEIGHTS_ID),objects=objects,find=index.__getitem__)
            self.assertEqual(validate_source(artifact),optional)
            for mutation in (objects[:-1],objects+(replace(objects[0],name="extra"),)):
                artifact.objects=mutation
                with self.assertRaises(ValueError):validate_source(artifact)
            artifact.objects=objects
            tensor=next(o for o in objects if o.kind=="tensor")
            for field,value in (("shape",(1,1)),("format","Q4G64_F16S"),("layout","contiguous-le-v1")):
                index[tensor.name]=replace(tensor,**{field:value})
                with self.assertRaises(ValueError):validate_source(artifact)
            index[tensor.name]=tensor
            artifact.identity=ArtifactIdentity(inv.MODEL_ID,"nvfp4")
            with self.assertRaises(ValueError):validate_source(artifact)

    def test_inverse_rows_and_k_exact_words(self):
        from tools.convert.qwen3_8_27b.shard import packed_chunks
        for bits,group,fmt in ((4,64,"Q4G64_F16S"),(5,64,"Q5G64_F16S"),(8,32,"W8G32_F16S")):
            raw=encode(bits,group,8,256)
            original=decode(raw,bits,group,8,256)
            shards=[]
            for rank in (0,1):
                selected=b"".join(packed_chunks(raw,fmt,(8,256),((rank*2,rank*2+2),(4+rank*2,6+rank*2))))
                shards.append(decode(selected,bits,group,4,256))
            self.assertEqual(shards[0][:2]+shards[1][:2]+shards[0][2:]+shards[1][2:],original)
            halves=[decode(b"".join(packed_chunks(raw,fmt,(8,256),columns=(r*128,(r+1)*128))),bits,group,8,128) for r in (0,1)]
            self.assertEqual([a+b for a,b in zip(*halves)],original)

    def test_direct_semantic_convolution_words(self):
        from tools.convert.qwen3_8_27b.shard import direct_chunks
        raw=b"".join(struct.pack("<H",i) for i in range(4*10240))
        for rank in (0,1):
            columns=((1024*rank,1024*(rank+1)),(2048+1024*rank,2048+1024*(rank+1)),(4096+3072*rank,4096+3072*(rank+1)))
            result=b"".join(direct_chunks(raw,(4,10240),2,columns=columns))
            words=struct.unpack("<"+"H"*(4*5120),result)
            expected=tuple(row*10240+col for row in range(4) for begin,end in columns for col in range(begin,end))
            self.assertEqual(words,expected)

    def test_tiny_container_atomic_stream_and_determinism(self):
        from unittest.mock import patch
        from tools.artifact.container import Artifact, ArtifactIdentity, TensorSpec, write_artifact
        from tools.convert.qwen3_8_27b import inventory as inv
        from tools.convert.qwen3_8_27b.shard import convert
        from tools.convert.qwen3_8_27b.tp2_inventory import TensorPlan
        src=TensorSpec("text/token_embedding",(6,256),"Q5G64_F16S","row-split-k128-v1")
        dst=TensorSpec("rank0/test",(3,128),src.format,src.layout)
        plan=(TensorPlan(src.name,src.shape,dst,((1,3),(5,6)),((64,192),)),)
        payload=encode(5,64,6,256)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);original=path/"source.ninfer";output=path/"result.ninfer"
            write_artifact(original,ArtifactIdentity(inv.MODEL_ID,inv.WEIGHTS_ID),[(src,payload)])
            with patch.object(inv,"BASE_TENSOR_SPECS",(src,)),patch.object(inv,"RESOURCE_SPECS",()),patch("tools.convert.qwen3_8_27b.shard.build_plan",return_value=plan):
                convert(original,output)
                first=output.read_bytes()
                convert(original,output)
                self.assertEqual(output.read_bytes(),first)
                with Artifact(output) as artifact:
                    self.assertEqual(artifact.identity.weights_id,"groupwise-int-tp2")
                    with artifact.payload(dst.name) as view:
                        got=decode(bytes(view),5,64,3,128)
                    self.assertEqual(got,[decode(payload,5,64,6,256)[r][64:192] for r in (1,2,5)])
                with patch("tools.convert.qwen3_8_27b.shard.tensor_chunks",side_effect=RuntimeError("injected")):
                    with self.assertRaisesRegex(RuntimeError,"injected"):convert(original,output)
                self.assertEqual(output.read_bytes(),first)
                self.assertEqual(sorted(p.name for p in path.iterdir()),["result.ninfer","source.ninfer"])
            with self.assertRaises(ValueError):convert(original,original)
            hardlink=path/"hardlink.ninfer";hardlink.hardlink_to(original)
            with self.assertRaises(ValueError):convert(original,hardlink)
            missing=path/"must-not-exist.ninfer"
            with self.assertRaises(ValueError):convert(original,missing)
            self.assertFalse(missing.exists())

    def test_tensor_codec_dependency_is_explicit(self):
        try:
            import torch
        except ImportError:
            from tools.artifact.layouts import decode_direct
            with self.assertRaisesRegex(ImportError,"PyTorch is required"):
                decode_direct(b"\0\0","BF16",(1,))


class ProductionRecipeTest(unittest.TestCase):
    def test_every_production_mapping_is_the_independent_tp2_recipe(self):
        from tools.convert.qwen3_8_27b.tp2_inventory import build_plan, manifest
        from tools.convert.qwen3_8_27b import inventory as source
        # Literal architectural geometry, independent of production segment tables.
        # Attention: Q12/KV2 x D256; GDN: QK8/V24 x D128; MLP: 8704.
        by_name={p.spec.name:p for p in build_plan(True)}
        expected_rank_names=[[],[]]
        for layer in range(64):
            for rank in range(2):
                prefix=f"rank{rank}/text/layers/{layer}/"
                rows=lambda start,count: (start+rank*count,start+(rank+1)*count)
                cases={
                    "input_norm":((5120,),None,None),
                    "post_attention_norm":((5120,),None,None),
                    "mlp/gate_up":((17408,5120),(rows(0,8704),rows(17408,8704)),None),
                    "mlp/down":((5120,8704),None,(rows(0,8704),)),
                }
                if layer % 4 == 3:
                    cases.update({
                        "attention/query_key":((3584,5120),(rows(0,12*256),rows(24*256,2*256)),None),
                        "attention/gate_value":((3584,5120),(rows(0,12*256),rows(24*256,2*256)),None),
                        "attention/query_norm":((256,),None,None),
                        "attention/key_norm":((256,),None,None),
                        "attention/output":((5120,3072),None,(rows(0,12*256),)),
                    })
                else:
                    cases.update({
                        "gdn/query_key":((2048,5120),(rows(0,8*128),rows(16*128,8*128)),None),
                        "gdn/value_z":((6144,5120),(rows(0,24*128),rows(48*128,24*128)),None),
                        "gdn/a_projection":((24,5120),(rows(0,24),),None),
                        "gdn/b_projection":((24,5120),(rows(0,24),),None),
                        "gdn/a_log":((24,),(rows(0,24),),None),
                        "gdn/dt_bias":((24,),(rows(0,24),),None),
                        "gdn/convolution":((4,5120),None,(rows(0,8*128),rows(16*128,8*128),rows(32*128,24*128))),
                        "gdn/norm":((128,),None,None),
                        "gdn/output":((5120,3072),None,(rows(0,24*128),)),
                    })
                for role,(shape,selected_rows,columns) in cases.items():
                    name=prefix+role
                    expected_rank_names[rank].append(name)
                    plan=by_name[name]
                    self.assertEqual((plan.spec.shape,plan.rows,plan.columns),(shape,selected_rows,columns),name)
                    self.assertEqual(plan.source_name,f"text/layers/{layer}/{role}")
                    original=next(s for s in source.TENSOR_SPECS if s.name==plan.source_name)
                    self.assertEqual((plan.source_shape,plan.spec.format,plan.spec.layout),(original.shape,original.format,original.layout))
        expected_owner={s.name for s in source.TENSOR_SPECS if not s.name.startswith("text/layers/")}
        self.assertEqual(set(by_name),expected_owner|set(expected_rank_names[0])|set(expected_rank_names[1]))
        for name in expected_owner:
            plan=by_name[name]
            self.assertEqual(plan.spec.name,plan.source_name)
            self.assertEqual(plan.spec.shape,plan.source_shape)
            self.assertIsNone(plan.rows)
            self.assertIsNone(plan.columns)
        self.assertTrue({"text/token_embedding","text/output_head","text/draft_head","text/draft_head_token_ids","text/final_norm"} <= expected_owner)
        self.assertTrue(all(s.name in expected_owner for s in source.MTP_TENSOR_SPECS+source.VISION_TENSOR_SPECS+source.DFLASH2_TENSOR_SPECS))
        value=manifest(True)
        self.assertEqual(set(value["owner_objects"]),expected_owner)
        self.assertEqual(value["owner_resources"],[
            "frontend/tokenizer.json","frontend/tokenizer_config.json",
            "frontend/chat_template.jinja","frontend/generation_config.json",
            "frontend/preprocessor_config.json","frontend/video_preprocessor_config.json"])
        for rank in range(2):
            self.assertEqual(set(value["rank_objects"][rank]["objects"]),set(expected_rank_names[rank]))
        expected_aliases={
            "mtp/token_embedding":(["text/token_embedding"],None),
            "mtp/full_output_head":(["text/output_head"],None),
            "mtp/optimized_proposal_head":(["text/draft_head","text/draft_head_token_ids"],None),
        }
        for layer in range(64):
            if layer%4 != 3:
                for rank in range(2):
                    prefix=f"rank{rank}/text/layers/{layer}/gdn/"
                    expected_aliases[prefix+"channel_major_convolution"]=([prefix+"convolution"],[1,0])
        self.assertEqual({a["name"]:(a["objects"],a["axis_order"]) for a in value["aliases"]},expected_aliases)
        self.assertEqual(len(value["aliases"]),len(expected_aliases))



class IndependentValidatorTest(unittest.TestCase):
    def test_independent_recipe_matches_all_qualified_plans(self):
        from tests.artifact import tp2_oracle as validator
        from tools.convert.qwen3_8_27b.tp2_inventory import build_plan,manifest
        for optional in (False,True):
            expected=validator.expected_tensors(optional)
            production=build_plan(optional)
            self.assertEqual(len(expected),len(production))
            for independent,actual in zip(expected,production):
                self.assertEqual((independent.source_name,independent.spec,independent.rows,independent.columns),
                                 (actual.source_name,actual.spec,actual.rows,actual.columns))
            self.assertEqual(validator.expected_manifest(optional),manifest(optional))

    def test_verifier_detects_corrupt_codes_scales_and_padding(self):
        from tests.artifact import tp2_oracle as validator
        from tools.artifact.container import TensorSpec
        from tools.convert.qwen3_8_27b.shard import packed_chunks
        for bits,group,fmt in ((4,64,"Q4G64_F16S"),(5,64,"Q5G64_F16S"),(8,32,"W8G32_F16S")):
            source=encode(bits,group,6,256)
            rows=((1,3),(5,6));columns=((64,128),)
            result=b"".join(packed_chunks(source,fmt,(6,256),rows,columns[0]))
            src=TensorSpec("source",(6,256),fmt,"row-split-k128-v1")
            dst=TensorSpec("destination",(3,64),fmt,src.layout)
            validator.verify_tensor(source,result,src,dst,rows,columns)
            # Every output byte is covered: codes, scales, new padded groups,
            # and intra-tensor alignment gaps. Each corruption must be detected.
            for offset in range(len(result)):
                corrupt=bytearray(result);corrupt[offset]^=1
                with self.assertRaises(ValueError,msg=f"{fmt}: unchecked byte {offset}"):
                    validator.verify_tensor(source,corrupt,src,dst,rows,columns)

    def test_verifier_detects_direct_and_replica_corruption(self):
        from tests.artifact import tp2_oracle as validator
        from tools.artifact.container import TensorSpec
        from tools.convert.qwen3_8_27b.shard import direct_chunks
        source=bytes(range(128))
        src=TensorSpec("source",(4,16),"BF16","contiguous-le-v1")
        dst=TensorSpec("destination",(4,8),"BF16",src.layout)
        columns=((0,4),(12,16))
        result=b"".join(direct_chunks(source,src.shape,2,columns=columns))
        validator.verify_tensor(source,result,src,dst,None,columns)
        for offset in range(len(result)):
            corrupt=bytearray(result);corrupt[offset]^=1
            with self.assertRaises(ValueError):validator.verify_tensor(source,corrupt,src,dst,None,columns)
        validator.verify_tensor(source,source,src,src,None,None)
        corrupt=bytearray(source);corrupt[-1]^=1
        with self.assertRaises(ValueError):validator.verify_tensor(source,corrupt,src,src,None,None)


class MidstreamFailureTest(unittest.TestCase):
    def test_original_io_failure_survives_suspended_mmap_generator(self):
        from unittest.mock import patch
        from tools.artifact.container import Artifact, ArtifactIdentity, ArtifactWriter, ResourceSpec, TensorSpec, write_artifact
        from tools.convert.qwen3_8_27b import inventory as inv
        from tools.convert.qwen3_8_27b.shard import convert
        from tools.convert.qwen3_8_27b.tp2_inventory import TensorPlan
        original_write=ArtifactWriter.write
        src=TensorSpec("text/token_embedding",(6,256),"Q5G64_F16S","row-split-k128-v1")
        resource=inv.ResourceSpec("frontend/tokenizer.json")
        for failure_kind in ("resource","owner_tensor","sharded_tensor"):
            with self.subTest(failure_kind=failure_kind),tempfile.TemporaryDirectory() as folder:
                source=Path(folder)/"source.ninfer";output=Path(folder)/"output.ninfer"
                write_artifact(source,ArtifactIdentity(inv.MODEL_ID,inv.WEIGHTS_ID),[
                    (ResourceSpec(resource.name,resource.encoding,2),b"{}"),(src,encode(5,64,6,256))])
                output.write_bytes(b"old destination")
                if failure_kind=="sharded_tensor":
                    target=TensorSpec("rank0/test",(3,128),src.format,src.layout)
                    plan=(TensorPlan(src.name,src.shape,target,((1,3),(5,6)),((64,192),)),)
                else:
                    target=src
                    plan=(TensorPlan(src.name,src.shape,src),)
                failing_name=resource.name if failure_kind=="resource" else target.name
                def fail_midstream(writer,name,payload):
                    if name==failing_name:
                        first_chunk=next(payload)
                        self.assertGreater(len(first_chunk),0)
                        raise OSError("injected failure after first chunk")
                    return original_write(writer,name,payload)
                with patch.object(inv,"BASE_TENSOR_SPECS",(src,)),patch.object(inv,"RESOURCE_SPECS",(resource,)),patch("tools.convert.qwen3_8_27b.shard.build_plan",return_value=plan),patch.object(ArtifactWriter,"write",fail_midstream):
                    with self.assertRaisesRegex(OSError,"after first chunk"):
                        convert(source,output)
                self.assertEqual(output.read_bytes(),b"old destination")
                self.assertEqual(sorted(p.name for p in Path(folder).iterdir()),["output.ninfer","source.ninfer"])
                with Artifact(source):
                    pass
