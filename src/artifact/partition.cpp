#include "artifact/binder.h"
#include <algorithm>
#include <limits>
#include <utility>

namespace ninfer::artifact {
MaterializationPlan filter_materialization_plan(const MaterializationPlan& source,
                                                std::span<const ObjectHandle> selected) {
    std::vector<bool> present(source.object_count), keep(source.object_count);
    auto mark = [&](ObjectHandle h) {
        if (h.index >= present.size() || present[h.index])
            throw ArtifactError("duplicate or out-of-range materialization handle");
        present[h.index] = true;
    };
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    for (const auto& p : source.device_objects) {
        mark(p.object);
        if (!p.alignment || (p.alignment & (p.alignment - 1)) || p.offset % p.alignment ||
            !p.bytes || p.offset > source.device_capacity_bytes ||
            p.bytes > source.device_capacity_bytes - p.offset)
            throw ArtifactError("invalid materialization alignment or range");
        ranges.emplace_back(p.offset, p.offset + p.bytes);
    }
    std::sort(ranges.begin(), ranges.end());
    for (std::size_t i = 1; i < ranges.size(); ++i)
        if (ranges[i].first < ranges[i - 1].second)
            throw ArtifactError("overlapping materialization ranges");
    for (const auto& p : source.host_objects) mark(p.object);
    for (auto h : selected) {
        if (h.index >= present.size() || !present[h.index] || keep[h.index])
            throw ArtifactError("selected handle is absent, out of range, or duplicate");
        keep[h.index] = true;
    }
    MaterializationPlan out;
    out.object_count = source.object_count;
    for (auto p : source.device_objects) {
        if (!keep[p.object.index]) continue;
        const auto mask = p.alignment - 1;
        if (out.device_capacity_bytes > std::numeric_limits<std::uint64_t>::max() - mask)
            throw ArtifactError("materialization alignment overflows u64");
        p.offset = (out.device_capacity_bytes + mask) & ~mask;
        if (p.bytes > std::numeric_limits<std::uint64_t>::max() - p.offset)
            throw ArtifactError("materialization capacity overflows u64");
        out.device_capacity_bytes = p.offset + p.bytes;
        out.device_objects.push_back(p);
    }
    for (auto p : source.host_objects)
        if (keep[p.object.index]) out.host_objects.push_back(p);
    return out;
}
} // namespace ninfer::artifact
