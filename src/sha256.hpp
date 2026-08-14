#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace aldous_tsp::detail {

// Small dependency-free SHA-256 implementation used only for executable and
// campaign provenance. The file helper returns false with an actionable error
// rather than throwing for ordinary I/O failures.
std::string sha256_hex(std::string_view bytes);
bool sha256_file(const std::filesystem::path& path,
                 std::string& digest,
                 std::string& error);

} // namespace aldous_tsp::detail
