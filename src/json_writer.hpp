#pragma once

#include <cmath>
#include <cstdio>
#include <iomanip>
#include <ostream>
#include <string>
#include <vector>

namespace aldous_tsp {

inline std::string json_escape_text(const std::string& input) {
    std::string out;
    out.reserve(input.size() + 8U);
    for (char raw : input) {
        const auto c = static_cast<unsigned char>(raw);
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
    }
    return out;
}

class JsonWriter {
public:
    explicit JsonWriter(std::ostream& out) : out_(out) {}

    void string(const std::string& value) { out_ << '"' << json_escape_text(value) << '"'; }

    void number(double value) {
        if (!std::isfinite(value)) {
            out_ << "null";
            return;
        }
        out_ << std::setprecision(17) << value;
    }

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
