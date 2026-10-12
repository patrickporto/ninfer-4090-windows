#pragma once

#include <cstdint>
#include <utility>

namespace ninfer::runtime {

// Exact 64x64 -> 128-bit product as (low, high) halves. MSVC has no unsigned
// __int128, so saturating arithmetic uses this portable limb decomposition.
[[nodiscard]] constexpr std::pair<std::uint64_t, std::uint64_t>
uint128_product(std::uint64_t left, std::uint64_t right) noexcept {
    const std::uint64_t left_lo  = left & 0xffffffffULL;
    const std::uint64_t left_hi  = left >> 32U;
    const std::uint64_t right_lo = right & 0xffffffffULL;
    const std::uint64_t right_hi = right >> 32U;
    const std::uint64_t ll       = left_lo * right_lo;
    const std::uint64_t lh       = left_lo * right_hi;
    const std::uint64_t hl       = left_hi * right_lo;
    const std::uint64_t hh       = left_hi * right_hi;
    const std::uint64_t mid      = (ll >> 32U) + (lh & 0xffffffffULL) + (hl & 0xffffffffULL);
    const std::uint64_t low      = (ll & 0xffffffffULL) | (mid << 32U);
    const std::uint64_t high     = hh + (lh >> 32U) + (hl >> 32U) + (mid >> 32U);
    return {low, high};
}

} // namespace ninfer::runtime
