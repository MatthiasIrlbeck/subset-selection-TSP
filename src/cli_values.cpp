#include "cli_internal.hpp"

#include "generated_options.hpp"

namespace aldous_tsp {
namespace {

constexpr std::size_t kMaxPValues = 10000U;
constexpr std::uintmax_t kMaxPFileBytes = 8U * 1024U * 1024U;

} // namespace

bool parse_int(const std::string& text, int& out) {
    try {
        std::size_t pos = 0;
        const long long value = std::stoll(text, &pos, 10);
        if (pos != text.size()
            || value < static_cast<long long>(std::numeric_limits<int>::min())
            || value > static_cast<long long>(std::numeric_limits<int>::max())) {
            return false;
        }
        out = static_cast<int>(value);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool parse_double(const std::string& text, double& out) {
    try {
        std::size_t pos = 0;
        const double value = std::stod(text, &pos);
        if (pos != text.size() || !std::isfinite(value)) {
            return false;
        }
        out = value;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

std::vector<std::string> split(const std::string& text, const char delim) {
    std::vector<std::string> parts;
    std::string current;
    std::istringstream input(text);
    while (std::getline(input, current, delim)) {
        parts.push_back(current);
    }
    return parts;
}

bool parse_p_values(const std::string& text, std::vector<double>& out) {
    std::vector<double> values;
    for (const std::string& part : split(text, ',')) {
        if (part.empty()) {
            continue;
        }
        double p = 0.0;
        if (!parse_double(part, p) || !(p > 0.0 && p <= 1.0)) {
            return false;
        }
        values.push_back(p);
        if (values.size() > kMaxPValues) {
            return false;
        }
    }
    if (values.empty()) {
        return false;
    }
    out = std::move(values);
    return true;
}

bool parse_p_range(const std::string& text, std::vector<double>& out) {
    const auto parts = split(text, ':');
    if (parts.size() != 3U) {
        return false;
    }
    double first = 0.0;
    double last = 0.0;
    int count = 0;
    if (!parse_double(parts[0], first)
        || !parse_double(parts[1], last)
        || !parse_int(parts[2], count)
        || !(first > 0.0 && first <= 1.0)
        || !(last > 0.0 && last <= 1.0)
        || count < 1
        || static_cast<std::size_t>(count) > kMaxPValues) {
        return false;
    }
    std::vector<double> values;
    values.reserve(static_cast<std::size_t>(count));
    if (count == 1) {
        values.push_back(first);
    } else {
        for (int i = 0; i < count; ++i) {
            const double t = static_cast<double>(i) / static_cast<double>(count - 1);
            values.push_back(first + (last - first) * t);
        }
    }
    out = std::move(values);
    return true;
}

bool read_p_file(const std::string& path, std::vector<double>& out, std::string& error) {
    namespace fs = std::filesystem;
    std::error_code filesystem_error;
    const fs::path canonical_path = fs::canonical(fs::path(path), filesystem_error);
    if (filesystem_error || !canonical_path.is_absolute()
        || !fs::is_regular_file(canonical_path, filesystem_error)
        || filesystem_error) {
        error = "p-file must be a readable regular file: " + path;
        return false;
    }
    const std::uintmax_t bytes = fs::file_size(canonical_path, filesystem_error);
    if (filesystem_error || bytes > kMaxPFileBytes) {
        error = "p-file exceeds the 8 MiB safety limit: " + path;
        return false;
    }
    // Reading a user-selected local file is the explicit contract of --p-file.
    // The path has been canonicalized and restricted to a bounded regular file.
    // codeql[cpp/path-injection]
    std::ifstream input(canonical_path, std::ios::binary);
    if (!input) {
        error = "failed to open p-file: " + path;
        return false;
    }
    std::string text(static_cast<std::size_t>(bytes), '\0');
    input.read(text.data(), static_cast<std::streamsize>(text.size()));
    if (input.bad()) {
        error = "failed while reading p-file: " + path;
        return false;
    }
    text.resize(static_cast<std::size_t>(input.gcount()));
    char extra = '\0';
    if (input.get(extra)) {
        error = "p-file exceeds the 8 MiB safety limit: " + path;
        return false;
    }
    for (char& ch : text) {
        if (ch == ',') {
            ch = ' ';
        }
    }
    std::istringstream tokens(text);
    std::vector<double> values;
    std::string token;
    while (tokens >> token) {
        double p = 0.0;
        if (!parse_double(token, p) || !(p > 0.0 && p <= 1.0)) {
            error = "invalid p-value in p-file: " + token;
            return false;
        }
        values.push_back(p);
        if (values.size() > kMaxPValues) {
            error = "p-file contains more than 10000 values: " + path;
            return false;
        }
    }
    if (values.empty()) {
        error = "p-file contains no values: " + path;
        return false;
    }
    out = std::move(values);
    return true;
}

void canonicalize_p_values(std::vector<double>& values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

void apply_quick_preset(RunOptions& opt) {
    apply_generated_quick_preset(opt);
}

void print_help(const char* argv0) {
    print_generated_help(stdout, argv0);
}

bool validate_options(RunOptions& opt, std::string& error) {
    return validate_run_options(opt, error);
}

} // namespace aldous_tsp
