#include "targets/qwen3_6_27b/impl/load/tensor_weights.h"
#include "artifact/typed_binding.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <set>
#include <vector>

namespace ninfer::targets::qwen3_6_27b::detail {
namespace {
using Json                                = nlohmann::json;
using F                                   = artifact::NumericFormat;
using P                                   = TensorObjectPlacement;
constexpr std::string_view inventory_name = "parallel/tp2_inventory.json";
enum class OwnerGroup { Core, Proposal, Mtp, Vision, Dflash2 };

struct OwnerRole {
    std::string name;
    OwnerGroup group;
};

// Closed physical owner inventory in converter order. Binding and descriptor
// validation remain shared with Single/PP in bind_owner_objects.
std::vector<OwnerRole> owner_roles(bool dflash2) {
    std::vector<OwnerRole> roles;
    auto add = [&](std::string prefix, std::initializer_list<const char*> suffixes,
                   OwnerGroup group) {
        for (auto suffix : suffixes) roles.push_back({prefix + suffix, group});
    };
    add("text/", {"token_embedding", "final_norm", "output_head"}, OwnerGroup::Core);
    add("text/", {"draft_head", "draft_head_token_ids"}, OwnerGroup::Proposal);
    add("mtp/",
        {"input_projection", "embedding_norm", "hidden_norm", "layer/input_norm",
         "layer/attention/query_key_gate_value", "layer/attention/query_norm",
         "layer/attention/key_norm", "layer/attention/output", "layer/post_attention_norm",
         "layer/mlp/gate_up", "layer/mlp/down", "final_norm"},
        OwnerGroup::Mtp);
    add("vision/", {"patch_embedding", "patch_embedding_bias", "position_embedding"},
        OwnerGroup::Vision);
    for (int layer = 0; layer < 27; ++layer)
        add("vision/layers/" + std::to_string(layer) + "/",
            {"attention/qkv", "attention/qkv_bias", "attention/output", "attention/output_bias",
             "mlp/fc1", "mlp/fc1_bias", "mlp/fc2", "mlp/fc2_bias", "norm1/weight", "norm1/bias",
             "norm2/weight", "norm2/bias"},
            OwnerGroup::Vision);
    add("vision/merger/", {"fc1", "fc1_bias", "fc2", "fc2_bias", "norm/weight", "norm/bias"},
        OwnerGroup::Vision);
    if (dflash2) {
        add("dflash2/", {"feature_projection", "context_norm"}, OwnerGroup::Dflash2);
        for (int layer = 0; layer < 5; ++layer)
            add("dflash2/layers/" + std::to_string(layer) + "/",
                {"input_norm", "attention_conv/base_kernel", "attention_conv/kernel_projection",
                 "attention/query_key_value", "attention/query_norm", "attention/key_norm",
                 "attention/output", "post_attention_norm", "mlp_conv/base_kernel",
                 "mlp_conv/kernel_projection", "mlp/gate_up", "mlp/down"},
                OwnerGroup::Dflash2);
        add("dflash2/",
            {"final_norm", "candidate_selector/hidden_projection",
             "candidate_selector/predecessor_codebook", "candidate_selector/successor_codebook"},
            OwnerGroup::Dflash2);
    }
    return roles;
}

constexpr std::array<const char*, 6> resources{
    "frontend/tokenizer.json",           "frontend/tokenizer_config.json",
    "frontend/chat_template.jinja",      "frontend/generation_config.json",
    "frontend/preprocessor_config.json", "frontend/video_preprocessor_config.json"};

Json expected_inventory(bool dflash2) {
    Json ranks = Json::array();
    for (int rank = 0; rank < 2; ++rank) {
        Json names = Json::array();
        for (int layer = 0; layer < 64; ++layer) {
            const auto prefix =
                "rank" + std::to_string(rank) + "/text/layers/" + std::to_string(layer) + "/";
            names.push_back(prefix + "input_norm");
            if (layer % 4 == 3) {
                for (auto suffix :
                     {"attention/query_key", "attention/gate_value", "attention/query_norm",
                      "attention/key_norm", "attention/output"})
                    names.push_back(prefix + suffix);
            } else {
                for (auto suffix :
                     {"gdn/a_log", "gdn/dt_bias", "gdn/convolution", "gdn/a_projection",
                      "gdn/b_projection", "gdn/query_key", "gdn/value_z", "gdn/norm", "gdn/output"})
                    names.push_back(prefix + suffix);
            }
            for (auto suffix : {"post_attention_norm", "mlp/gate_up", "mlp/down"})
                names.push_back(prefix + suffix);
        }
        ranks.push_back({{"rank", rank}, {"objects", names}});
    }
    Json owners = Json::array();
    for (const auto& role : owner_roles(dflash2)) owners.push_back(role.name);
    Json aliases = Json::array({{{"name", "mtp/token_embedding"},
                                 {"objects", {"text/token_embedding"}},
                                 {"axis_order", nullptr}},
                                {{"name", "mtp/full_output_head"},
                                 {"objects", {"text/output_head"}},
                                 {"axis_order", nullptr}},
                                {{"name", "mtp/optimized_proposal_head"},
                                 {"objects", {"text/draft_head", "text/draft_head_token_ids"}},
                                 {"axis_order", nullptr}}});
    for (int layer = 0; layer < 64; ++layer)
        if (layer % 4 != 3)
            for (int rank = 0; rank < 2; ++rank) {
                const auto prefix = "rank" + std::to_string(rank) + "/text/layers/" +
                                    std::to_string(layer) + "/gdn/";
                aliases.push_back({{"name", prefix + "channel_major_convolution"},
                                   {"objects", {prefix + "convolution"}},
                                   {"axis_order", {1, 0}}});
            }
    return {{"schema", "qwen3.8-27b-tp2-inventory-v1"},
            {"source_identity", {{"model_id", "qwen3.8-27b"}, {"weights_id", "groupwise-int"}}},
            {"world_size", 2},
            {"owner_rank", 0},
            {"rank_objects", ranks},
            {"owner_objects", owners},
            {"owner_resources", resources},
            {"aliases", aliases},
            {"optional_features", {{"dflash2", dflash2 ? "preserved-disabled" : "absent"}}}};
}

P owner_placement(OwnerGroup group, qwen3_6::StartupFeatures features) {
    switch (group) {
    case OwnerGroup::Core:
        return P::Rank0Device;
    case OwnerGroup::Proposal:
        return features.optimized_proposal() ? P::Rank0Device : P::ValidateOnly;
    case OwnerGroup::Mtp:
        return features.mtp() ? P::Rank0Device : P::ValidateOnly;
    case OwnerGroup::Vision:
        return features.vision ? P::Rank0Device : P::ValidateOnly;
    case OwnerGroup::Dflash2:
        return P::ValidateOnly;
    }
    throw artifact::ArtifactError("invalid TP owner group");
}
} // namespace

std::string tensor_inventory_json(bool dflash2) { return expected_inventory(dflash2).dump(); }

void validate_tensor_inventory(std::string_view text, bool dflash2) {
    try {
        // JSON parsers normally discard duplicate keys; reject them explicitly.
        std::vector<std::set<std::string>> keys;
        auto callback = [&](int, Json::parse_event_t event, Json& value) {
            if (event == Json::parse_event_t::object_start)
                keys.emplace_back();
            else if (event == Json::parse_event_t::key) {
                if (!keys.back().insert(value.get<std::string>()).second)
                    throw artifact::ArtifactError("duplicate TP inventory key");
            } else if (event == Json::parse_event_t::object_end)
                keys.pop_back();
            return true;
        };
        const auto parsed = Json::parse(text.begin(), text.end(), callback);
        // dump comparison preserves integer versus float/bool distinctions while
        // accepting insignificant whitespace and object-key order.
        if (parsed.dump() != tensor_inventory_json(dflash2))
            throw artifact::ArtifactError("TP inventory differs from registered closed recipe");
    } catch (const Json::exception& error) {
        throw artifact::ArtifactError(std::string("invalid TP inventory JSON: ") + error.what());
    }
}

std::array<artifact::MaterializationPlan, 2>
tensor_materialization_plans(const artifact::MaterializationPlan& full,
                             std::span<const TensorObjectAssignment> assignments) {
    if (assignments.size() != full.object_count)
        throw artifact::ArtifactError("TP ownership does not cover all objects");
    std::vector<int> owner(full.object_count, -1), residency(full.object_count, 0);
    for (const auto& a : assignments) {
        if (a.object.index >= owner.size() || owner[a.object.index] != -1)
            throw artifact::ArtifactError("duplicate or out-of-range TP ownership");
        switch (a.placement) {
        case P::Rank0Device:
            owner[a.object.index] = 0;
            break;
        case P::Rank1Device:
            owner[a.object.index] = 1;
            break;
        case P::OwnerHost:
            owner[a.object.index] = 2;
            break;
        case P::ValidateOnly:
            owner[a.object.index] = 3;
            break;
        default:
            throw artifact::ArtifactError("unknown TP placement");
        }
    }
    std::array<std::vector<artifact::ObjectHandle>, 2> selected;
    for (const auto& d : full.device_objects) {
        if (d.object.index >= owner.size() || owner[d.object.index] > 1 ||
            owner[d.object.index] < 0 || residency[d.object.index]++)
            throw artifact::ArtifactError("TP device ownership mismatch");
        selected[owner[d.object.index]].push_back(d.object);
    }
    for (const auto& h : full.host_objects) {
        if (h.object.index >= owner.size() || owner[h.object.index] != 2 ||
            residency[h.object.index]++)
            throw artifact::ArtifactError("TP host ownership mismatch");
        selected[0].push_back(h.object);
    }
    for (std::size_t i = 0; i < owner.size(); ++i)
        if ((owner[i] == 3) != (residency[i] == 0))
            throw artifact::ArtifactError("TP assigned object missing from materialization");
    return {artifact::filter_materialization_plan(full, selected[0]),
            artifact::filter_materialization_plan(full, selected[1])};
}

TensorArtifactLoadPlan bind_tensor_artifact(const artifact::Reader& reader,
                                            qwen3_6::StartupFeatures features) {
    if (reader.identity() != artifact::ArtifactIdentity{"qwen3.8-27b", "groupwise-int-tp2"})
        throw artifact::ArtifactError("TP binding requires qwen3.8-27b/groupwise-int-tp2");
    if (features.masked_draft()) throw artifact::ArtifactError("TP DFlash/DFlash2 is unsupported");
    artifact::Binder binder(reader);
    const bool dflash2 = binder.contains("dflash2/feature_projection");
    const auto inventory =
        binder.require_resource(inventory_name, artifact::ResourceEncoding::RawBytesV1);
    const auto bytes = binder.payload(inventory).data;
    validate_tensor_inventory({reinterpret_cast<const char*>(bytes.data()), bytes.size()}, dflash2);
    binder.validate_only(inventory);
    std::vector<TensorObjectAssignment> assignments{{inventory, P::ValidateOnly}};
    TensorArtifactLoadPlan out;
    out.owner = bind_owner_objects(binder, WeightsProfile::Qwen38GroupwiseInt, features);
    // Names are compiled roles, not an arbitrary artifact prefix classification.
    auto owner_handle = [&](std::string_view name) {
        const auto* object = reader.find(name);
        if (!object) throw artifact::ArtifactError("missing TP owner object");
        return artifact::ObjectHandle{static_cast<std::size_t>(object - reader.objects().data())};
    };
    for (const auto& role : owner_roles(dflash2))
        assignments.push_back({owner_handle(role.name), owner_placement(role.group, features)});
    for (auto name : resources) assignments.push_back({owner_handle(name), P::OwnerHost});

    for (std::size_t rank = 0; rank < 2; ++rank) {
        auto& local = out.ranks[rank];
        local.rank  = rank;
        for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
            const auto prefix =
                "rank" + std::to_string(rank) + "/text/layers/" + std::to_string(layer) + "/";
            auto tensor = [&](const char* suffix, F format,
                              std::initializer_list<std::uint64_t> shape) {
                const auto handle =
                    artifact::bind_device_tensor(binder, prefix + suffix, format, shape);
                assignments.push_back({handle, rank == 0 ? P::Rank0Device : P::Rank1Device});
                return handle;
            };
            auto weight = [&](const char* suffix, F format,
                              std::initializer_list<std::uint64_t> shape) {
                return WeightPlan{.object = tensor(suffix, format, shape), .format = format};
            };
            auto& target               = local.layers[layer];
            target.input_norm          = tensor("input_norm", F::BF16, {5120});
            target.post_attention_norm = tensor("post_attention_norm", F::BF16, {5120});
            target.mlp                 = {weight("mlp/gate_up", F::Q4G64_F16S, {17408, 5120}),
                                          weight("mlp/down", F::Q5G64_F16S, {5120, 8704})};
            if (layer % 4 == 3) {
                TensorAttentionPlan a;
                a.query_key  = weight("attention/query_key", F::Q4G64_F16S, {3584, 5120});
                a.gate_value = weight("attention/gate_value", F::Q5G64_F16S, {3584, 5120});
                a.output     = weight("attention/output", F::Q5G64_F16S, {5120, 3072});
                a.query_norm = tensor("attention/query_norm", F::BF16, {256});
                a.key_norm   = tensor("attention/key_norm", F::BF16, {256});
                target.mixer = a;
            } else {
                TensorGdnPlan g;
                g.query_key    = weight("gdn/query_key", F::Q4G64_F16S, {2048, 5120});
                g.value_z      = weight("gdn/value_z", F::Q5G64_F16S, {6144, 5120});
                g.a_projection = weight("gdn/a_projection", F::BF16, {24, 5120});
                g.b_projection = weight("gdn/b_projection", F::BF16, {24, 5120});
                g.output       = weight("gdn/output", F::Q5G64_F16S, {5120, 3072});
                g.a_log        = tensor("gdn/a_log", F::FP32, {24});
                g.dt_bias      = tensor("gdn/dt_bias", F::FP32, {24});
                g.convolution  = tensor("gdn/convolution", F::BF16, {4, 5120});
                g.norm         = tensor("gdn/norm", F::BF16, {128});
                target.mixer   = g;
            }
        }
    }
    const auto full     = binder.finish();
    out.materialization = tensor_materialization_plans(full, assignments);
    return out;
}
} // namespace ninfer::targets::qwen3_6_27b::detail
