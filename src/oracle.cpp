#include "aldous_tsp/oracle.hpp"

#include "aldous_tsp/solver.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <thread>
#include <utility>

#if !defined(_WIN32)
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include <random>

namespace aldous_tsp {
namespace {

std::string trim_ascii(std::string text) {
    auto is_space = [](unsigned char ch) {
        return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch == '\f' || ch == '\v';
    };
    while (!text.empty() && is_space(static_cast<unsigned char>(text.front()))) {
        text.erase(text.begin());
    }
    while (!text.empty() && is_space(static_cast<unsigned char>(text.back()))) {
        text.pop_back();
    }
    return text;
}

// Portable unique temporary working directory (no POSIX mkdtemp), used to stage
// the problem/parameter/tour files handed to the external solver. Works on both
// POSIX and Windows via std::filesystem.
struct TempWorkDir {
    std::filesystem::path path;
    bool ok = false;

    explicit TempWorkDir(const char* prefix) {
        namespace fs = std::filesystem;
        fs::path base;
        const char* tmpdir = std::getenv("TMPDIR");
        if (tmpdir != nullptr && *tmpdir != '\0') {
            base = tmpdir;
        } else {
            std::error_code ec;
            base = fs::temp_directory_path(ec);
            if (ec) { base = fs::path("."); }
        }
        std::mt19937_64 gen(std::random_device{}() ^ static_cast<std::uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count()));
        for (int attempt = 0; attempt < 64; ++attempt) {
            std::ostringstream name;
            name << prefix << '_' << std::hex << gen();
            const fs::path candidate = base / name.str();
            std::error_code ec;
            if (fs::create_directory(candidate, ec) && !ec) {
                path = candidate;
                ok = true;
                return;
            }
        }
    }

    TempWorkDir(const TempWorkDir&) = delete;
    TempWorkDir& operator=(const TempWorkDir&) = delete;

    ~TempWorkDir() {
        if (ok) {
            std::error_code ec;
            std::filesystem::remove_all(path, ec);
        }
    }
};

// Reads the external solver's captured console output and returns the part worth
// reporting: LKH prints its fatal diagnostics after a "*** Error ***" banner, so
// prefer that; otherwise fall back to the tail (the end is where failures show).
std::string summarize_child_output(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {};
    }
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (text.empty()) {
        return {};
    }
    constexpr std::size_t kMaxReport = 240U;
    const std::size_t marker = text.find("*** Error ***");
    std::string slice;
    if (marker != std::string::npos) {
        slice = text.substr(marker, kMaxReport);
    } else if (text.size() > kMaxReport) {
        slice = text.substr(text.size() - kMaxReport);
    } else {
        slice = text;
    }
    // Flatten to a single line so it fits in a JSON error field.
    std::string flat;
    flat.reserve(slice.size());
    bool prev_space = false;
    for (const char c : slice) {
        const bool is_space = (c == '\n' || c == '\r' || c == '\t' || c == ' ');
        if (is_space) {
            if (!prev_space && !flat.empty()) {
                flat += ' ';
            }
            prev_space = true;
        } else {
            flat += c;
            prev_space = false;
        }
    }
    return trim_ascii(flat);
}

#if !defined(_WIN32)

std::string resolve_exec_in_path(const std::string& program) {
    if (program.empty()) {
        return {};
    }
    if (program.find('/') != std::string::npos) {
        if (access(program.c_str(), X_OK) != 0) {
            return {};
        }
        std::error_code ec;
        const std::filesystem::path resolved = std::filesystem::weakly_canonical(program, ec);
        return ec ? std::filesystem::absolute(program).string() : resolved.string();
    }
    const char* path_env = std::getenv("PATH");
    if (path_env == nullptr) {
        return {};
    }
    const std::string path(path_env);
    std::size_t start = 0;
    while (start <= path.size()) {
        const std::size_t end = path.find(':', start);
        std::string dir = (end == std::string::npos) ? path.substr(start) : path.substr(start, end - start);
        if (dir.empty()) {
            dir = ".";
        }
        const std::string candidate = dir + "/" + program;
        if (access(candidate.c_str(), X_OK) == 0) {
            return candidate;
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1U;
    }
    return {};
}

int wait_for_process(pid_t pid, int timeout_sec) {
    int status = 0;
    if (timeout_sec <= 0) {
        while (waitpid(pid, &status, 0) < 0) {
            if (errno != EINTR) {
                return -1;
            }
        }
        if (WIFEXITED(status)) {
            return WEXITSTATUS(status);
        }
        return status == 0 ? -1 : status;
    }

    const auto start = std::chrono::steady_clock::now();
    for (;;) {
        const pid_t waited = waitpid(pid, &status, WNOHANG);
        if (waited == pid) {
            if (WIFEXITED(status)) {
                return WEXITSTATUS(status);
            }
            return status == 0 ? -1 : status;
        }
        if (waited < 0 && errno != EINTR) {
            return -1;
        }
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (elapsed > static_cast<double>(timeout_sec)) {
            kill(-pid, SIGKILL);
            (void)waitpid(pid, &status, 0);
            return 124;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

int run_external_process(const std::vector<std::string>& argv, const std::filesystem::path& cwd, int timeout_sec, bool verbose, std::string* captured) {
    if (argv.empty()) {
        return -1;
    }
    const std::filesystem::path log_path = cwd.empty()
        ? std::filesystem::path("oracle_output.txt")
        : (cwd / "oracle_output.txt");
    const std::string log_str = log_path.string();
    const pid_t pid = fork();
    if (pid < 0) {
        return -1;
    }
    if (pid == 0) {
        setpgid(0, 0);
        if (!cwd.empty() && chdir(cwd.string().c_str()) != 0) {
            _exit(126);
        }
        // Capture stdout+stderr so a failing solver's own diagnostics survive.
        const int fd = open(log_str.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
        if (fd >= 0) {
            dup2(fd, STDOUT_FILENO);
            dup2(fd, STDERR_FILENO);
            if (fd > STDERR_FILENO) {
                close(fd);
            }
        }
        std::vector<char*> args;
        args.reserve(argv.size() + 1U);
        for (const std::string& arg : argv) {
            args.push_back(const_cast<char*>(arg.c_str()));
        }
        args.push_back(nullptr);
        execvp(args[0], args.data());
        _exit(127);
    }
    setpgid(pid, pid);
    const int rc = wait_for_process(pid, timeout_sec);
    const std::string output = summarize_child_output(log_path);
    if (captured != nullptr) {
        *captured = output;
    }
    if (verbose && !output.empty()) {
        std::fprintf(stderr, "[oracle] %s\n", output.c_str());
    }
    return rc;
}

std::string capture_process_first_line(const std::vector<std::string>& argv, int timeout_sec) {
    if (argv.empty()) {
        return "unknown";
    }
    int pipefd[2] = {-1, -1};
    if (pipe(pipefd) != 0) {
        return "unknown";
    }
    const pid_t pid = fork();
    if (pid < 0) {
        close(pipefd[0]);
        close(pipefd[1]);
        return "unknown";
    }
    if (pid == 0) {
        setpgid(0, 0);
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        if (pipefd[1] > STDERR_FILENO) {
            close(pipefd[1]);
        }
        std::vector<char*> args;
        args.reserve(argv.size() + 1U);
        for (const std::string& arg : argv) {
            args.push_back(const_cast<char*>(arg.c_str()));
        }
        args.push_back(nullptr);
        execvp(args[0], args.data());
        _exit(127);
    }
    setpgid(pid, pid);
    close(pipefd[1]);
    const int flags = fcntl(pipefd[0], F_GETFL, 0);
    if (flags >= 0) {
        (void)fcntl(pipefd[0], F_SETFL, flags | O_NONBLOCK);
    }
    std::string output;
    std::array<char, 256> chunk{};
    const auto start = std::chrono::steady_clock::now();
    for (;;) {
        const ssize_t nread = read(pipefd[0], chunk.data(), chunk.size());
        if (nread > 0) {
            output.append(chunk.data(), static_cast<std::size_t>(nread));
            if (output.size() > 512U || output.find('\n') != std::string::npos) {
                break;
            }
        } else if (nread == 0) {
            break;
        } else if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK) {
            break;
        }
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (elapsed > static_cast<double>(timeout_sec)) {
            kill(-pid, SIGKILL);
            break;
        }
    }
    close(pipefd[0]);
    const int rc = wait_for_process(pid, timeout_sec);
    if (rc != 0 && output.empty()) {
        return "unknown";
    }
    const std::size_t newline = output.find('\n');
    if (newline != std::string::npos) {
        output.resize(newline);
    }
    output = trim_ascii(output);
    if (output.empty()) {
        return "unknown";
    }
    if (output.size() > 120U) {
        output.resize(120U);
    }
    return output;
}

#else  // _WIN32

std::string resolve_exec_in_path(const std::string& program) {
    if (program.empty()) {
        return {};
    }
    namespace fs = std::filesystem;
    std::error_code ec;
    auto resolve_variants = [&](const fs::path& base) -> std::string {
        if (fs::is_regular_file(base, ec)) {
            return fs::absolute(base, ec).string();
        }
        fs::path with_exe = base;
        with_exe += ".exe";
        if (fs::is_regular_file(with_exe, ec)) {
            return fs::absolute(with_exe, ec).string();
        }
        return {};
    };
    const bool looks_like_path = program.find('/') != std::string::npos
        || program.find('\\') != std::string::npos
        || (program.size() >= 2 && program[1] == ':');
    if (looks_like_path) {
        return resolve_variants(fs::path(program));
    }
    const char* path_env = std::getenv("PATH");
    if (path_env == nullptr) {
        return {};
    }
    const std::string path(path_env);
    std::size_t start = 0;
    while (start <= path.size()) {
        const std::size_t end = path.find(';', start);  // Windows PATH separator
        std::string dir = (end == std::string::npos) ? path.substr(start) : path.substr(start, end - start);
        if (!dir.empty()) {
            const std::string resolved = resolve_variants(fs::path(dir) / program);
            if (!resolved.empty()) {
                return resolved;
            }
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1U;
    }
    return {};
}

// Quote one argument per the Windows command-line parsing rules (CommandLineToArgvW).
std::string windows_quote_arg(const std::string& arg) {
    if (!arg.empty() && arg.find_first_of(" \t\n\v\"") == std::string::npos) {
        return arg;
    }
    std::string result = "\"";
    for (auto it = arg.begin();; ++it) {
        unsigned backslashes = 0;
        while (it != arg.end() && *it == '\\') {
            ++it;
            ++backslashes;
        }
        if (it == arg.end()) {
            result.append(static_cast<std::size_t>(backslashes) * 2U, '\\');
            break;
        }
        if (*it == '"') {
            result.append(static_cast<std::size_t>(backslashes) * 2U + 1U, '\\');
            result += '"';
        } else {
            result.append(static_cast<std::size_t>(backslashes), '\\');
            result += *it;
        }
    }
    result += '"';
    return result;
}

int run_external_process(const std::vector<std::string>& argv, const std::filesystem::path& cwd, int timeout_sec, bool verbose, std::string* captured) {
    if (argv.empty()) {
        return -1;
    }
    std::string command_line;
    for (std::size_t i = 0; i < argv.size(); ++i) {
        if (i != 0) {
            command_line += ' ';
        }
        command_line += windows_quote_arg(argv[i]);
    }
    std::vector<char> mutable_cmd(command_line.begin(), command_line.end());
    mutable_cmd.push_back('\0');

    const std::filesystem::path log_path = cwd.empty()
        ? std::filesystem::path("oracle_output.txt")
        : (cwd / "oracle_output.txt");
    const std::string log_str = log_path.string();

    SECURITY_ATTRIBUTES sa;
    ZeroMemory(&sa, sizeof(sa));
    sa.nLength = sizeof(sa);
    sa.lpSecurityDescriptor = nullptr;
    sa.bInheritHandle = TRUE;

    // Capture the child's console output so a failing solver's own diagnostics
    // survive; this also hands the child valid standard handles, which it would
    // otherwise lack under CREATE_NO_WINDOW.
    HANDLE log_handle = CreateFileA(log_str.c_str(), GENERIC_WRITE,
                                    FILE_SHARE_READ | FILE_SHARE_WRITE, &sa,
                                    CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    HANDLE null_in = CreateFileA("NUL", GENERIC_READ,
                                 FILE_SHARE_READ | FILE_SHARE_WRITE, &sa,
                                 OPEN_EXISTING, 0, nullptr);

    STARTUPINFOA startup;
    ZeroMemory(&startup, sizeof(startup));
    startup.cb = sizeof(startup);
    BOOL inherit_handles = FALSE;
    if (log_handle != INVALID_HANDLE_VALUE && null_in != INVALID_HANDLE_VALUE) {
        startup.dwFlags = STARTF_USESTDHANDLES;
        startup.hStdInput = null_in;
        startup.hStdOutput = log_handle;
        startup.hStdError = log_handle;
        inherit_handles = TRUE;
    }
    PROCESS_INFORMATION proc;
    ZeroMemory(&proc, sizeof(proc));

    const std::string cwd_str = cwd.string();
    const BOOL created = CreateProcessA(
        argv[0].c_str(),                              // exact executable (already resolved)
        mutable_cmd.data(),                           // command line (mutable buffer)
        nullptr, nullptr, inherit_handles,
        CREATE_NO_WINDOW,                             // no console window per invocation
        nullptr,
        cwd_str.empty() ? nullptr : cwd_str.c_str(),  // per-call working directory (thread-safe)
        &startup, &proc);
    if (log_handle != INVALID_HANDLE_VALUE) {
        CloseHandle(log_handle);
    }
    if (null_in != INVALID_HANDLE_VALUE) {
        CloseHandle(null_in);
    }
    if (!created) {
        return -1;
    }
    const DWORD wait_ms = (timeout_sec > 0) ? static_cast<DWORD>(timeout_sec) * 1000U : INFINITE;
    const DWORD wait_result = WaitForSingleObject(proc.hProcess, wait_ms);
    DWORD exit_code = 1;
    if (wait_result == WAIT_TIMEOUT) {
        TerminateProcess(proc.hProcess, 124U);
        WaitForSingleObject(proc.hProcess, 2000U);
        exit_code = 124;
    } else {
        GetExitCodeProcess(proc.hProcess, &exit_code);
    }
    CloseHandle(proc.hProcess);
    CloseHandle(proc.hThread);
    const std::string output = summarize_child_output(log_path);
    if (captured != nullptr) {
        *captured = output;
    }
    if (verbose && !output.empty()) {
        std::fprintf(stderr, "[oracle] %s\n", output.c_str());
    }
    return static_cast<int>(exit_code);
}

// The version probe is cosmetic; a real LKH exits non-zero on "--version"
// anyway, so we don't attempt to capture it on Windows.
std::string capture_process_first_line(const std::vector<std::string>&, int) { return "unknown"; }

#endif

// LKH stores edge costs in `int` and internally multiplies them by its PRECISION
// parameter, so a cost must satisfy  cost * PRECISION <= INT_MAX. We write
// PRECISION = 1 in the parameter file (our costs are already scaled integers, so
// LKH's default x100 buys nothing), and additionally cap the scale here so the
// largest cost stays well inside int even for very large instances -- leaving
// headroom for the node potentials LKH adds during its ascent.
//
// This only affects the fidelity of the *external solver's* optimization: we use
// the returned tour's node ORDER and recompute its true length in double
// precision, so the reported f(p) is never quantized by this scale.
constexpr long long kMaxOracleCost = 500000000LL;  // INT_MAX / ~4

int effective_oracle_scale(const Instance& inst, const std::vector<int>& nodes, int requested_scale) {
    double max_dist = 0.0;
    const int k = static_cast<int>(nodes.size());
    // The farthest pair bounds every cost; sampling the extremes is enough
    // because we only need an upper bound, so take the exact max over a bounded
    // number of pairs and fall back to the geometric bound for large k.
    if (k <= 256) {
        for (int i = 0; i < k; ++i) {
            for (int j = i + 1; j < k; ++j) {
                max_dist = std::max(max_dist, inst.dist(nodes[static_cast<std::size_t>(i)], nodes[static_cast<std::size_t>(j)]));
            }
        }
    } else {
        // Upper bound: the torus half-diagonal, or the full diagonal for the open
        // square. inst.side is the domain edge length.
        const double side = inst.side;
        max_dist = inst.periodic ? (std::sqrt(2.0) * side * 0.5) : (std::sqrt(2.0) * side);
    }
    if (!(max_dist > 0.0) || !std::isfinite(max_dist)) {
        return std::max(1, requested_scale);
    }
    const long long cap = static_cast<long long>(static_cast<double>(kMaxOracleCost) / max_dist);
    long long scale = static_cast<long long>(std::max(1, requested_scale));
    if (scale > cap) {
        scale = cap;
    }
    if (scale < 1) {
        scale = 1;
    }
    return static_cast<int>(scale);
}

bool write_tsplib_matrix(const std::filesystem::path& path, const Instance& inst, const std::vector<int>& nodes, int scale) {
    std::ofstream out(path);
    if (!out) {
        return false;
    }
    const int k = static_cast<int>(nodes.size());
    out << "NAME : aldous_oracle\nTYPE : TSP\nDIMENSION : " << k
        << "\nEDGE_WEIGHT_TYPE : EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX\nEDGE_WEIGHT_SECTION\n";
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            long long w = 0;
            if (i != j) {
                w = static_cast<long long>(std::llround(static_cast<double>(scale) * inst.dist(nodes[static_cast<std::size_t>(i)], nodes[static_cast<std::size_t>(j)])));
            }
            out << w << (j + 1 == k ? '\n' : ' ');
        }
    }
    out << "EOF\n";
    return out.good();
}

bool write_tsplib_euc2d(const std::filesystem::path& path, const Instance& inst, const std::vector<int>& nodes, int scale) {
    std::ofstream out(path);
    if (!out) {
        return false;
    }
    const int k = static_cast<int>(nodes.size());
    out << "NAME : aldous_oracle\nTYPE : TSP\nDIMENSION : " << k << "\nEDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n";
    for (int i = 0; i < k; ++i) {
        const Point& point = inst.points[static_cast<std::size_t>(nodes[static_cast<std::size_t>(i)])];
        const long long x = static_cast<long long>(std::llround(static_cast<double>(scale) * point.x));
        const long long y = static_cast<long long>(std::llround(static_cast<double>(scale) * point.y));
        out << (i + 1) << ' ' << x << ' ' << y << '\n';
    }
    out << "EOF\n";
    return out.good();
}

bool write_identity_tour(const std::filesystem::path& path, int k) {
    std::ofstream out(path);
    if (!out) {
        return false;
    }
    out << "NAME : init\nTYPE : TOUR\nDIMENSION : " << k << "\nTOUR_SECTION\n";
    for (int i = 1; i <= k; ++i) {
        out << i << '\n';
    }
    out << "-1\nEOF\n";
    return out.good();
}

std::vector<long long> extract_ints(const std::string& text) {
    std::vector<long long> values;
    const char* cursor = text.c_str();
    while (*cursor != '\0') {
        char* end = nullptr;
        const long long value = std::strtoll(cursor, &end, 10);
        if (end != cursor) {
            values.push_back(value);
            cursor = end;
        } else {
            ++cursor;
        }
    }
    return values;
}

bool parse_window(const std::vector<long long>& values, std::size_t start, int k, bool one_based, std::vector<int>& permutation) {
    if (start + static_cast<std::size_t>(k) > values.size()) {
        return false;
    }
    permutation.assign(static_cast<std::size_t>(k), -1);
    std::vector<unsigned char> seen(static_cast<std::size_t>(k), 0U);
    for (int i = 0; i < k; ++i) {
        const long long raw = values[start + static_cast<std::size_t>(i)];
        const long long v = one_based ? raw - 1LL : raw;
        if (v < 0 || v >= static_cast<long long>(k)) {
            return false;
        }
        const auto idx = static_cast<std::size_t>(v);
        if (seen[idx] != 0U) {
            return false;
        }
        seen[idx] = 1U;
        permutation[static_cast<std::size_t>(i)] = static_cast<int>(v);
    }
    return true;
}

bool parse_external_tour_file(const std::filesystem::path& path, int k, std::vector<int>& permutation) {
    std::ifstream in(path);
    if (!in) {
        return false;
    }
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return parse_external_tour_text(text, k, permutation);
}

bool external_oracle_polish_nodes(const Instance& inst,
                                  const OracleContext& oracle,
                                  const std::vector<int>& input_nodes,
                                  std::vector<int>& output_nodes,
                                  double& output_length,
                                  std::string* error_message) {
    auto fail = [&](const std::string& message) {
        if (error_message != nullptr) {
            *error_message = message;
        }
        output_length = std::numeric_limits<double>::infinity();
        return false;
    };
    const int k = static_cast<int>(input_nodes.size());
    if (k < 3) {
        return fail("tour has fewer than 3 nodes");
    }
    TempWorkDir tmp("aldous_oracle");
    if (!tmp.ok) {
        return fail("failed to create temporary working directory");
    }
    const std::filesystem::path problem = tmp.path / "problem.tsp";
    const std::filesystem::path init = tmp.path / "init.tour";
    const std::filesystem::path params = tmp.path / "run.par";
    const std::filesystem::path out_tour = tmp.path / "out.tour";
    const int safe_scale = effective_oracle_scale(inst, input_nodes, oracle.cfg.scale);
    const bool wrote_problem = oracle.cfg.problem_format == OracleProblemFormat::Matrix
        ? write_tsplib_matrix(problem, inst, input_nodes, safe_scale)
        : write_tsplib_euc2d(problem, inst, input_nodes, safe_scale);
    if (!wrote_problem) {
        return fail("failed to write TSPLIB problem file");
    }

    std::vector<std::string> argv;
    if (oracle.resolved == ResolvedOracleMode::Lkh) {
        if (!write_identity_tour(init, k)) {
            return fail("failed to write initial tour file");
        }
        std::ofstream par(params);
        if (!par) {
            return fail("failed to write LKH parameter file");
        }
        par << "PROBLEM_FILE = problem.tsp\n"
            << "INITIAL_TOUR_FILE = init.tour\n"
            << "TOUR_FILE = out.tour\n"
            // Our costs are already scaled integers; LKH's default PRECISION=100
            // would multiply them again in int arithmetic and overflow (LKH then
            // aborts with "PRECISION (= 100) is too large" and exit status 1).
            << "PRECISION = 1\n"
            << "RUNS = " << std::max(1, oracle.cfg.lkh_runs) << "\n"
            << "TRACE_LEVEL = " << (oracle.cfg.verbose ? 1 : 0) << "\n";
        if (oracle.cfg.lkh_max_trials > 0) {
            par << "MAX_TRIALS = " << oracle.cfg.lkh_max_trials << "\n";
        }
        if (oracle.cfg.time_limit_sec > 0) {
            par << "TIME_LIMIT = " << oracle.cfg.time_limit_sec << "\n";
        }
        par.close();
        argv = {oracle.exec_path, params.filename().string()};
    } else if (oracle.resolved == ResolvedOracleMode::Concorde) {
        argv = {oracle.exec_path, "-o", out_tour.filename().string(), problem.filename().string()};
    } else {
        return fail("no resolved external oracle executable");
    }

    std::string child_output;
    const int rc = run_external_process(argv, tmp.path, oracle.cfg.time_limit_sec, oracle.cfg.verbose, &child_output);
    // Accept whatever the solver actually produced: its contract is the tour
    // file, and exit-code conventions vary between solvers and versions. Only if
    // no usable tour comes back do we treat the call as failed -- and then we
    // report the solver's own diagnostics, which is what makes failures debuggable.
    std::vector<int> permutation;
    bool parsed = parse_external_tour_file(out_tour, k, permutation);
    if (!parsed) {
        const std::filesystem::path concorde_fallback = tmp.path / "problem.sol";
        parsed = parse_external_tour_file(concorde_fallback, k, permutation);
    }
    if (!parsed) {
        std::ostringstream oss;
        oss << "external solver returned no usable tour (exit status " << rc << ")";
        if (!child_output.empty()) {
            oss << "; solver said: " << child_output;
        }
        return fail(oss.str());
    }
    output_nodes.resize(static_cast<std::size_t>(k));
    for (int i = 0; i < k; ++i) {
        output_nodes[static_cast<std::size_t>(i)] = input_nodes[static_cast<std::size_t>(permutation[static_cast<std::size_t>(i)])];
    }
    output_length = cycle_length(inst, output_nodes);
    if (!std::isfinite(output_length)) {
        return fail("returned tour length is not finite");
    }
    if (error_message != nullptr) {
        error_message->clear();
    }
    return true;
}

} // namespace

bool parse_external_tour_text(const std::string& text, int k, std::vector<int>& permutation) {
    if (k < 1) {
        return false;
    }
    auto try_values = [&](const std::vector<long long>& values) {
        std::vector<int> tmp;
        for (std::size_t start = 0; start + static_cast<std::size_t>(k) <= values.size(); ++start) {
            if (parse_window(values, start, k, false, tmp) || parse_window(values, start, k, true, tmp)) {
                permutation = std::move(tmp);
                return true;
            }
        }
        return false;
    };
    const std::size_t tour_section = text.find("TOUR_SECTION");
    if (tour_section != std::string::npos && try_values(extract_ints(text.substr(tour_section)))) {
        return true;
    }
    return try_values(extract_ints(text));
}

bool external_oracle_applicable(const OracleContext& oracle, int k, bool full_tsp) noexcept {
    if (oracle.resolved == ResolvedOracleMode::None) {
        return false;
    }
    if (k <= kExactSmallTourLimit) {
        return false;
    }
    if (k < oracle.cfg.min_k || k > oracle.cfg.max_k) {
        return false;
    }
    if (full_tsp && !oracle.cfg.use_for_tsp) {
        return false;
    }
    if (!full_tsp && !oracle.cfg.use_for_subset) {
        return false;
    }
    return true;
}

bool build_oracle_context(const ExternalOracleConfig& cfg, OracleContext& oracle, std::string& error) {
    error.clear();
    const ExternalOracleConfig cfg_copy = cfg;
    oracle = OracleContext();
    oracle.cfg = cfg_copy;
    if (cfg_copy.mode == ExternalOracleMode::None) {
        oracle.status = "disabled";
        return true;
    }
    const std::string lkh = resolve_exec_in_path(cfg_copy.lkh_path);
    const std::string concorde = resolve_exec_in_path(cfg_copy.concorde_path);
    if (cfg_copy.mode == ExternalOracleMode::Auto) {
        if (!lkh.empty()) {
            oracle.resolved = ResolvedOracleMode::Lkh;
            oracle.exec_path = lkh;
        } else if (!concorde.empty()) {
            oracle.resolved = ResolvedOracleMode::Concorde;
            oracle.exec_path = concorde;
        } else {
            oracle.status = "auto: no supported external solver found on PATH";
            return true;
        }
    } else if (cfg_copy.mode == ExternalOracleMode::Lkh) {
        if (lkh.empty()) {
            error = "requested --oracle lkh, but executable was not found: " + cfg_copy.lkh_path;
            return false;
        }
        oracle.resolved = ResolvedOracleMode::Lkh;
        oracle.exec_path = lkh;
    } else if (cfg_copy.mode == ExternalOracleMode::Concorde) {
        if (concorde.empty()) {
            error = "requested --oracle concorde, but executable was not found: " + cfg_copy.concorde_path;
            return false;
        }
        oracle.resolved = ResolvedOracleMode::Concorde;
        oracle.exec_path = concorde;
    }

    oracle.version = capture_process_first_line({oracle.exec_path, "--version"}, 2);
    std::ostringstream status;
    status << resolved_oracle_mode_name(oracle.resolved)
           << " @ " << oracle.exec_path
           << " (version=" << oracle.version
           << ", format=" << oracle_problem_format_name(cfg_copy.problem_format)
           << ", scale=" << cfg_copy.scale
           << ", tsp-top=" << cfg_copy.tsp_top
           << ", subset-top=" << cfg_copy.subset_top
           << ", k-range=[" << cfg_copy.min_k << ',' << cfg_copy.max_k << ']'
           << (cfg_copy.inline_feedback ? ", inline-feedback" : ", posthoc-only")
           << (cfg_copy.verbose ? ", verbose" : "")
           << ')';
    oracle.status = status.str();
    return true;
}

bool external_oracle_polish_tour(Tour& candidate, const Instance& inst, const OracleContext& oracle, bool full_tsp, SearchStats* stats, bool enable_internal_two_opt) {
    candidate.ensure_edges(inst);
    if (!external_oracle_applicable(oracle, candidate.k, full_tsp)) {
        return false;
    }

    OracleCallRecord record;
    record.type = full_tsp ? "tsp" : "subset";
    record.k = candidate.k;
    record.solver = resolved_oracle_mode_name(oracle.resolved);
    record.format = oracle_problem_format_name(oracle.cfg.problem_format);
    record.exec_path = oracle.exec_path;
    record.status = "failed";
    record.error = "not run";
    record.before_length = candidate.length;
    record.after_length = std::numeric_limits<double>::quiet_NaN();
    record.gain = 0.0;

    if (stats != nullptr) {
        ++stats->oracle_calls;
        if (full_tsp) {
            ++stats->oracle_tsp_calls;
        } else {
            ++stats->oracle_subset_calls;
        }
    }

    const auto t0 = std::chrono::steady_clock::now();
    const double before = candidate.length;
    std::vector<int> out_nodes;
    double out_len = std::numeric_limits<double>::infinity();
    std::string failure_reason;
    const bool solved = external_oracle_polish_nodes(inst, oracle, candidate.nodes, out_nodes, out_len, &failure_reason);
    record.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    if (!solved) {
        if (stats != nullptr) {
            ++stats->oracle_failed;
            record.error = failure_reason.empty() ? "external oracle failed" : failure_reason;
            stats->oracle_call_records.push_back(std::move(record));
        }
        return false;
    }

    if (stats != nullptr) {
        ++stats->oracle_solved;
    }
    record.after_length = out_len;
    record.error = "";

    if (out_len + kImprovementEps < before) {
        candidate.set_tour(out_nodes, inst);
        if (enable_internal_two_opt && candidate.k <= 300) {
            (void)two_opt_descent(candidate, inst, 200, nullptr);
        }
        const double gain = before - candidate.length;
        record.status = "improved";
        record.after_length = candidate.length;
        record.gain = gain;
        if (stats != nullptr) {
            ++stats->oracle_improved;
            stats->oracle_gain += gain;
            stats->oracle_call_records.push_back(std::move(record));
        }
        return true;
    }

    record.status = "solved_no_improvement";
    record.gain = 0.0;
    if (stats != nullptr) {
        stats->oracle_call_records.push_back(std::move(record));
    }
    return false;
}

} // namespace aldous_tsp
