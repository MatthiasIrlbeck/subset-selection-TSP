#include "run_cli_internal.hpp"

namespace run_cli_detail {

static std::string json_escape(const std::string& s){
    std::string out;
    out.reserve(s.size() + 8);
    for(unsigned char c : s){
        switch(c){
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if(c < 0x20){
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned)c);
                    out += buf;
                } else {
                    out.push_back((char)c);
                }
                break;
        }
    }
    return out;
}

static void write_json_string_or_null(std::ostream& out,const std::string& value){
    if(value.empty()) out << "null";
    else out << '"' << json_escape(value) << '"';
}

static inline bool is_shell_safe_char(char c){
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')
        || c == '_' || c == '@' || c == '%' || c == '+' || c == '=' || c == ':'
        || c == ',' || c == '.' || c == '/' || c == '-';
}

static std::string shell_quote_arg(const std::string& arg){
    if(arg.empty()) return "''";
    bool safe = true;
    for(char c : arg){
        if(!is_shell_safe_char(c)){
            safe = false;
            break;
        }
    }
    if(safe) return arg;
    std::string out;
    out.reserve(arg.size() + 2);
    out.push_back('\'');
    for(char c : arg){
        if(c == '\'') out += "'\\''";
        else out.push_back(c);
    }
    out.push_back('\'');
    return out;
}

static std::string build_command_line(int argc,char** argv){
    std::ostringstream oss;
    for(int i=0; i<argc; ++i){
        if(i) oss << ' ';
        oss << shell_quote_arg(argv[i] ? std::string(argv[i]) : std::string());
    }
    return oss.str();
}

static inline bool is_ascii_space(char c){
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

static std::string trim_ascii(std::string s){
    size_t begin = 0;
    while(begin < s.size() && is_ascii_space(s[begin])) ++begin;
    size_t end = s.size();
    while(end > begin && is_ascii_space(s[end - 1])) --end;
    return s.substr(begin, end - begin);
}

static bool read_first_line_trimmed(const std::filesystem::path& path,std::string& out){
    std::ifstream in(path);
    if(!in) return false;
    std::getline(in, out);
    out = trim_ascii(out);
    return true;
}

static bool is_hex_commit_hash(const std::string& s){
    if(s.size() != 40) return false;
    for(char c : s){
        bool hex = (c >= '0' && c <= '9')
                || (c >= 'a' && c <= 'f')
                || (c >= 'A' && c <= 'F');
        if(!hex) return false;
    }
    return true;
}

static std::filesystem::path normalized_path_or_empty(const std::filesystem::path& path){
    std::error_code ec;
    std::filesystem::path norm = std::filesystem::weakly_canonical(path, ec);
    if(ec) return path.lexically_normal();
    return norm;
}

static std::filesystem::path resolve_git_dir(const std::filesystem::path& dotgit){
    std::error_code ec;
    if(std::filesystem::is_directory(dotgit, ec)) return normalized_path_or_empty(dotgit);
    if(!std::filesystem::is_regular_file(dotgit, ec)) return {};
    std::string line;
    if(!read_first_line_trimmed(dotgit, line)) return {};
    constexpr const char* prefix = "gitdir:";
    if(line.compare(0, std::strlen(prefix), prefix) != 0) return {};
    std::filesystem::path gitdir = trim_ascii(line.substr(std::strlen(prefix)));
    if(gitdir.is_relative()) gitdir = dotgit.parent_path() / gitdir;
    return normalized_path_or_empty(gitdir);
}

static std::string lookup_packed_git_ref(const std::filesystem::path& git_dir,const std::string& refname){
    std::ifstream in(git_dir / "packed-refs");
    if(!in) return std::string();
    std::string line;
    while(std::getline(in, line)){
        line = trim_ascii(line);
        if(line.empty() || line[0] == '#' || line[0] == '^') continue;
        std::istringstream iss(line);
        std::string hash, ref;
        if(!(iss >> hash >> ref)) continue;
        if(ref == refname && is_hex_commit_hash(hash)) return hash;
    }
    return std::string();
}

static std::string read_git_ref_hash(const std::filesystem::path& git_dir,const std::string& refname){
    std::string line;
    if(read_first_line_trimmed(git_dir / refname, line) && is_hex_commit_hash(line)) return line;
    return lookup_packed_git_ref(git_dir, refname);
}

static std::string detect_git_commit_hash_from(const std::filesystem::path& start_dir){
    std::error_code ec;
    std::filesystem::path dir = normalized_path_or_empty(start_dir);
    if(dir.empty()) return std::string();
    while(true){
        std::filesystem::path dotgit = dir / ".git";
        if(std::filesystem::exists(dotgit, ec)){
            std::filesystem::path git_dir = resolve_git_dir(dotgit);
            if(!git_dir.empty()){
                std::string head;
                if(read_first_line_trimmed(git_dir / "HEAD", head)){
                    constexpr const char* ref_prefix = "ref: ";
                    if(head.compare(0, std::strlen(ref_prefix), ref_prefix) == 0){
                        std::string refname = trim_ascii(head.substr(std::strlen(ref_prefix)));
                        std::string hash = read_git_ref_hash(git_dir, refname);
                        if(!hash.empty()) return hash;
                    } else if(is_hex_commit_hash(head)) {
                        return head;
                    }
                }
            }
        }
        if(dir == dir.root_path()) break;
        std::filesystem::path parent = dir.parent_path();
        if(parent == dir) break;
        dir = parent;
    }
    return std::string();
}

// Walks up from CWD, not from the binary's source tree. The hash therefore
// reflects whichever repo contains the working directory at runtime.
static std::string detect_git_commit_hash(){
    std::error_code ec;
    std::string hash = detect_git_commit_hash_from(std::filesystem::current_path(ec));
    return hash;
}

static std::string detect_compiler_string(){
    std::ostringstream oss;
#if defined(__clang__)
    oss << "Clang " << __clang_version__;
#elif defined(__GNUC__)
    oss << "GCC " << __VERSION__;
#elif defined(_MSC_FULL_VER)
    oss << "MSVC " << _MSC_FULL_VER;
#else
    return std::string();
#endif
    oss << " (C++" << __cplusplus << ')';
    return oss.str();
}

ResultsMetadata build_results_metadata(int argc,char** argv,SolverMode mode){
    ResultsMetadata meta;
    meta.cli_command = build_command_line(argc, argv);
    meta.mode_name = solver_mode_name(mode);
    meta.git_commit = detect_git_commit_hash();
    meta.compiler = detect_compiler_string();
    return meta;
}

bool write_json(const char* fn,int N,const std::vector<double>& pv,const std::vector<std::vector<double>>& af,
                int done,int tgt,int rst,int sa,int seed,int threads,double ws,
                const OracleStats& oracle_stats,const std::string& oracle_status,
                const std::string& backend,const ResultsMetadata& meta,
                const std::array<int,4>& regime_counts,const std::array<double,4>& regime_seconds){
    std::ofstream f(fn);
    if(!f) return false;
    f.setf(std::ios::fixed);

    f << "{\n  \"schema_version\": " << meta.schema_version
      << ",\n  \"run_metadata\": {\n"
      << "    \"cli_command\": \"" << json_escape(meta.cli_command) << "\",\n"
      << "    \"git_commit\": ";
    write_json_string_or_null(f, meta.git_commit);
    f << ",\n    \"compiler\": ";
    write_json_string_or_null(f, meta.compiler);
    f << "\n  },\n  \"N\": " << N << ",\n  \"p_values\": [";
    for(int i=0;i<(int)pv.size();i++){
        if(i) f << ", ";
        f << std::setprecision(4) << pv[i];
    }
    f << std::setprecision(10)
      << "],\n  \"done\": " << done
      << ",\n  \"target\": " << tgt
      << ",\n  \"restarts\": " << rst
      << ",\n  \"sa_iters\": " << sa
      << ",\n  \"seed\": " << seed
      << ",\n  \"threads\": " << threads
      << ",\n  \"wall_seconds\": " << std::setprecision(2) << ws
      << std::setprecision(10)
      << ",\n  \"distance_backend\": \"" << json_escape(backend) << "\""
      << ",\n  \"mode\": \"" << json_escape(meta.mode_name) << "\""
      << ",\n  \"regime_counts\": {\"full\": " << regime_counts[0] << ", \"low\": " << regime_counts[1]
      << ", \"mid\": " << regime_counts[2] << ", \"high\": " << regime_counts[3] << "}"
      << ",\n  \"regime_seconds\": {\"full\": " << std::setprecision(2) << regime_seconds[0]
      << ", \"low\": " << regime_seconds[1] << ", \"mid\": " << regime_seconds[2]
      << ", \"high\": " << regime_seconds[3] << "}"
      << std::setprecision(10)
      << ",\n  \"oracle_status\": \"" << json_escape(oracle_status) << "\""
      << ",\n  \"oracle_summary\": {\n"
      << "    \"calls\": " << oracle_stats.calls << ",\n"
      << "    \"solved\": " << oracle_stats.solved << ",\n"
      << "    \"improved\": " << oracle_stats.improved << ",\n"
      << "    \"failed\": " << oracle_stats.failed << ",\n"
      << "    \"exact_gain\": " << oracle_stats.exact_gain << ",\n"
      << "    \"tsp_calls\": " << oracle_stats.tsp_calls << ",\n"
      << "    \"tsp_solved\": " << oracle_stats.tsp_solved << ",\n"
      << "    \"tsp_improved\": " << oracle_stats.tsp_improved << ",\n"
      << "    \"tsp_gain\": " << oracle_stats.tsp_gain << ",\n"
      << "    \"subset_calls\": " << oracle_stats.subset_calls << ",\n"
      << "    \"subset_solved\": " << oracle_stats.subset_solved << ",\n"
      << "    \"subset_improved\": " << oracle_stats.subset_improved << ",\n"
      << "    \"subset_gain\": " << oracle_stats.subset_gain << "\n"
      << "  },\n  \"summary\": {\n";

    bool first_summary = true;
    for(int pi=0; pi<(int)pv.size(); pi++){
        double p = pv[pi];
        const auto& v = af[pi];
        if(v.empty()) continue;
        int k = std::max(3, (int)std::round(p * N));
        double mean = 0.0;
        for(double x : v) mean += x;
        mean /= static_cast<double>(v.size());
        double var = 0.0;
        if(v.size() > 1){
            for(double x : v) var += (x - mean) * (x - mean);
            var /= static_cast<double>(v.size() - 1);
        }
        double sd = std::sqrt(var), se = v.size() > 1 ? sd / std::sqrt(static_cast<double>(v.size())) : 0.0;
        char pk[32]; std::snprintf(pk, sizeof(pk), "%.4f", p);
        char* dot = std::strchr(pk, '.');
        if(dot){
            char* e = pk + std::strlen(pk) - 1;
            while(e > dot + 1 && *e == '0'){ *e = 0; --e; }
        }
        if(!first_summary) f << ",\n";
        first_summary = false;
        f << "    \"" << pk << "\": {\n"
          << "      \"k\": " << k << ",\n"
          << "      \"mean\": " << mean << ",\n"
          << "      \"std\": " << sd << ",\n"
          << "      \"stderr\": " << se << ",\n"
          << "      \"n\": " << (int)v.size() << ",\n"
          << "      \"values\": [";
        for(int i=0;i<(int)v.size();i++){
            if(i) f << ", ";
            f << v[i];
        }
        f << "]\n    }";
    }
    f << "\n  }\n}\n";
    f.flush();
    return f.good();
}

} // namespace run_cli_detail
