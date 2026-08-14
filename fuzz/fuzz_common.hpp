#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace aldous_tsp::fuzz {

class Reader {
public:
    Reader(const std::uint8_t* data, std::size_t size) noexcept
        : data_(data), size_(size) {}

    [[nodiscard]] bool empty() const noexcept { return offset_ >= size_; }
    [[nodiscard]] std::size_t remaining() const noexcept { return size_ - offset_; }

    std::uint8_t byte() noexcept {
        return empty() ? 0U : data_[offset_++];
    }

    std::uint64_t u64() noexcept {
        std::uint64_t value = 0;
        const std::size_t count = std::min<std::size_t>(remaining(), sizeof(value));
        if (count != 0U) {
            std::memcpy(&value, data_ + offset_, count);
            offset_ += count;
        }
        return value;
    }

    int bounded_int(int low, int high) noexcept {
        if (high <= low) {
            return low;
        }
        const std::uint64_t span = static_cast<std::uint64_t>(high - low + 1);
        return low + static_cast<int>(u64() % span);
    }

    double unit() noexcept {
        constexpr double denominator = 1.0 / static_cast<double>(std::uint64_t{1} << 53U);
        return static_cast<double>(u64() >> 11U) * denominator;
    }

    double finite(double scale = 1.0) noexcept {
        const double signed_unit = 2.0 * unit() - 1.0;
        return signed_unit * scale;
    }

private:
    const std::uint8_t* data_ = nullptr;
    std::size_t size_ = 0;
    std::size_t offset_ = 0;
};

} // namespace aldous_tsp::fuzz
