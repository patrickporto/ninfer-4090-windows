#pragma once

#include "ninfer/types.h"

#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::perplexity {

inline KvCacheStorage parse_kv_storage(std::string_view text) {
    if (text == "bf16") { return KvCacheStorage::BFloat16; }
    if (text == "int8") { return KvCacheStorage::Int8Group64; }
    if (text == "fp8") { return KvCacheStorage::Fp8E4M3Row256; }
    if (text == "nvfp4") { return KvCacheStorage::Nvfp4Group16; }
    if (text == "k8v4") { return KvCacheStorage::Fp8KeyNvfp4Value; }
    if (text == "rk8v4") { return KvCacheStorage::RotatedInt8KeyInt4ValueGroup64; }
    if (text == "rk4v4") { return KvCacheStorage::RotatedInt4KeyInt4ValueGroup64; }
    if (text == "rk4v4-e8") { return KvCacheStorage::RK4V4E8; }
    if (text == "rk2v4-e8") { return KvCacheStorage::RK2V4E8; }
    throw std::invalid_argument("--kv-dtype must be bf16, int8, fp8, nvfp4, k8v4, rk8v4, "
                                "rk4v4, rk4v4-e8, or rk2v4-e8");
}

inline std::string kv_name(KvCacheStorage value) {
    switch (value) {
    case KvCacheStorage::BFloat16:
        return "bf16";
    case KvCacheStorage::Int8Group64:
        return "int8-g64";
    case KvCacheStorage::Fp8E4M3Row256:
        return "fp8-e4m3-r256";
    case KvCacheStorage::Nvfp4Group16:
        return "nvfp4";
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return "k8v4";
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
        return "rk8v4";
    case KvCacheStorage::RotatedInt4KeyInt4ValueGroup64:
        return "rk4v4";
    case KvCacheStorage::RK4V4E8:
        return "rk4v4-e8";
    case KvCacheStorage::RK2V4E8:
        return "rk2v4-e8";
    }
    throw std::logic_error("unknown KV dtype");
}

} // namespace ninfer::perplexity
