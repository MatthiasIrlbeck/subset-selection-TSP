#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>

namespace aldous_tsp {

class Rng {
public:
    Rng() { seed(1); }
    explicit Rng(std::uint64_t value) { seed(value); }

    void seed(std::uint64_t value) noexcept {
        for (std::uint64_t& state_word : state_) {
            state_word = splitmix64(value);
        }
    }

    std::uint64_t next_u64() noexcept {
        const std::uint64_t result = rotl(state_[1] * 5ULL, 7) * 9ULL;
        const std::uint64_t t = state_[1] << 17;

        state_[2] ^= state_[0];
        state_[3] ^= state_[1];
        state_[1] ^= state_[2];
        state_[0] ^= state_[3];
        state_[2] ^= t;
        state_[3] = rotl(state_[3], 45);
        return result;
    }

    double uniform() noexcept {
        return static_cast<double>(next_u64() >> 11U) * 0x1.0p-53;
    }

    int randint(int exclusive_upper) noexcept {
        if (exclusive_upper <= 1) {
            return 0;
        }
        const auto bound = static_cast<std::uint64_t>(exclusive_upper);
        const auto threshold = (std::uint64_t{0} - bound) % bound;
        for (;;) {
            const std::uint64_t x = next_u64();
            if (x >= threshold) {
                return static_cast<int>(x % bound);
            }
        }
    }

    template <class It>
    void partial_shuffle(It begin, It end, std::size_t count) noexcept {
        const auto n = static_cast<std::size_t>(end - begin);
        count = std::min(count, n);
        for (std::size_t i = 0; i < count; ++i) {
            const std::size_t j = i + static_cast<std::size_t>(randint(static_cast<int>(n - i)));
            std::iter_swap(begin + static_cast<std::ptrdiff_t>(i), begin + static_cast<std::ptrdiff_t>(j));
        }
    }

private:
    std::uint64_t state_[4]{};

    static std::uint64_t rotl(std::uint64_t x, int k) noexcept {
        return (x << k) | (x >> (64 - k));
    }

    static std::uint64_t splitmix64(std::uint64_t& x) noexcept {
        std::uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31U);
    }
};

std::uint64_t mix_hash64(std::uint64_t value) noexcept;
std::uint64_t make_stream_seed(std::uint64_t base_seed, std::uint64_t instance_index, std::uint64_t tag) noexcept;

} // namespace aldous_tsp
