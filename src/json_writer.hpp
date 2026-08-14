#pragma once

#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace aldous_tsp {

inline void append_json_replacement_character(std::string& out) {
    // Keep the emitted JSON ASCII-only at malformed input boundaries. This is
    // deterministic across platforms and makes the replacement visible to
    // downstream provenance tooling.
    out += "\\ufffd";
}

inline std::string json_escape_text(const std::string& input) {
    std::string out;
    out.reserve(input.size() + 8U);
    std::size_t i = 0;
    while (i < input.size()) {
        const auto c = static_cast<unsigned char>(input[i]);
        if (c < 0x80U) {
            switch (c) {
                case '"': out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\b': out += "\\b"; break;
                case '\f': out += "\\f"; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                default:
                    if (c < 0x20U) {
                        char buf[7];
                        std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
                        out += buf;
                    } else {
                        out.push_back(static_cast<char>(c));
                    }
                    break;
            }
            ++i;
            continue;
        }

        std::size_t length = 0;
        std::uint32_t code_point = 0;
        std::uint32_t minimum = 0;
        if (c >= 0xC2U && c <= 0xDFU) {
            length = 2U;
            code_point = static_cast<std::uint32_t>(c & 0x1FU);
            minimum = 0x80U;
        } else if (c >= 0xE0U && c <= 0xEFU) {
            length = 3U;
            code_point = static_cast<std::uint32_t>(c & 0x0FU);
            minimum = 0x800U;
        } else if (c >= 0xF0U && c <= 0xF4U) {
            length = 4U;
            code_point = static_cast<std::uint32_t>(c & 0x07U);
            minimum = 0x10000U;
        } else {
            append_json_replacement_character(out);
            ++i;
            continue;
        }

        bool valid = i + length <= input.size();
        for (std::size_t j = 1U; valid && j < length; ++j) {
            const auto continuation = static_cast<unsigned char>(input[i + j]);
            if ((continuation & 0xC0U) != 0x80U) {
                valid = false;
                break;
            }
            code_point = (code_point << 6U)
                | static_cast<std::uint32_t>(continuation & 0x3FU);
        }
        if (!valid || code_point < minimum || code_point > 0x10FFFFU
            || (code_point >= 0xD800U && code_point <= 0xDFFFU)) {
            append_json_replacement_character(out);
            ++i;
            continue;
        }
        out.append(input, i, length);
        i += length;
    }
    return out;
}

inline std::string json_number_text(double value) {
    if (!std::isfinite(value)) {
        return "null";
    }
    std::array<char, 64> buffer{};
    const auto result = std::to_chars(
        buffer.data(), buffer.data() + buffer.size(), value,
        std::chars_format::general, std::numeric_limits<double>::max_digits10);
    if (result.ec != std::errc{}) {
        throw std::runtime_error("failed to format a JSON number");
    }
    return std::string(buffer.data(), result.ptr);
}

class JsonWriter {
public:
    explicit JsonWriter(std::ostream& out) : out_(out) {}

    void string(const std::string& value) { out_ << '"' << json_escape_text(value) << '"'; }

    void number(double value) { out_ << json_number_text(value); }

    void double_array(const std::vector<double>& values) {
        out_ << '[';
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (i != 0U) out_ << ", ";
            number(values[i]);
        }
        out_ << ']';
    }

private:
    std::ostream& out_;
};

} // namespace aldous_tsp
