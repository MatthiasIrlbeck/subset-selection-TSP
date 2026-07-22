#include "aldous_tsp/oracle.hpp"
#include "aldous_tsp/memory.hpp"

#include "aldous_tsp/solver.hpp"

#include "sha256.hpp"

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
#include <optional>
#include <sstream>
#include <thread>
#include <utility>

#if !defined(_WIN32)
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
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

#if !defined(_WIN32)
extern char** environ;
#ifndef ALDOUS_TSP_HAVE_POSIX_SPAWN_CHDIR_NP
#define ALDOUS_TSP_HAVE_POSIX_SPAWN_CHDIR_NP 0
#endif
#endif

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

constexpr std::uintmax_t kMaxOracleConsoleBytes = 16U * 1024U * 1024U;
constexpr std::uintmax_t kMaxOracleTourBytes = 16U * 1024U * 1024U;
constexpr std::size_t kOracleConsoleReadBytes = 64U * 1024U;

// Reads a bounded tail of the external solver's captured console output and
// returns the part worth reporting. A malicious or broken solver must not make
// the parent allocate according to an untrusted output-file size.
std::string summarize_child_output(const std::filesystem::path& path) {
    std::error_code size_error;
    const std::uintmax_t file_bytes = std::filesystem::file_size(path, size_error);
    if (size_error) {
        return {};
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {};
    }
    const std::uintmax_t bounded = std::min<std::uintmax_t>(file_bytes, kOracleConsoleReadBytes);
    if (file_bytes > bounded) {
        in.seekg(static_cast<std::streamoff>(file_bytes - bounded), std::ios::beg);
    }
    std::string text(static_cast<std::size_t>(bounded), '\0');
    in.read(text.data(), static_cast<std::streamsize>(text.size()));
    text.resize(static_cast<std::size_t>(in.gcount()));
    if (text.empty()) {
        return file_bytes > kMaxOracleConsoleBytes
            ? "oracle console output exceeded the 16 MiB security limit"
            : std::string{};
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
    std::string summary = trim_ascii(flat);
    if (file_bytes > kMaxOracleConsoleBytes) {
        summary = "oracle console output exceeded the 16 MiB security limit; tail: " + summary;
    }
    return summary;
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

using ProcessClock = std::chrono::steady_clock;
using ProcessDeadline = ProcessClock::time_point;

class UniqueFd {
public:
    UniqueFd() = default;
    explicit UniqueFd(int fd) noexcept : fd_(fd) {}
    UniqueFd(const UniqueFd&) = delete;
    UniqueFd& operator=(const UniqueFd&) = delete;
    UniqueFd(UniqueFd&& other) noexcept : fd_(other.release()) {}
    UniqueFd& operator=(UniqueFd&& other) noexcept {
        if (this != &other) {
            reset(other.release());
        }
        return *this;
    }
    ~UniqueFd() { reset(); }

    int get() const noexcept { return fd_; }
    int release() noexcept {
        const int fd = fd_;
        fd_ = -1;
        return fd;
    }
    void reset(int fd = -1) noexcept {
        if (fd_ >= 0) {
            (void)close(fd_);
        }
        fd_ = fd;
    }

private:
    int fd_ = -1;
};

ProcessDeadline process_deadline(int timeout_sec) {
    if (timeout_sec <= 0) {
        return ProcessDeadline::max();
    }
    return ProcessClock::now() + std::chrono::seconds(timeout_sec);
}

int process_exit_code(int status) noexcept {
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    if (WIFSIGNALED(status)) {
        return 128 + WTERMSIG(status);
    }
    return -1;
}

void kill_process_group(pid_t pid) noexcept {
    if (pid <= 0) {
        return;
    }
    if (kill(-pid, SIGKILL) != 0) {
        // POSIX_SPAWN_SETPGROUP should make the negative-PID form sufficient.
        // Fall back to the direct child if the group disappeared or was not
        // established by a non-conforming implementation.
        (void)kill(pid, SIGKILL);
    }
}

void terminate_process_group_and_reap(pid_t pid) noexcept {
    if (pid <= 0) {
        return;
    }
    kill_process_group(pid);
    int status = 0;
    for (;;) {
        const pid_t waited = waitpid(pid, &status, 0);
        if (waited == pid || (waited < 0 && errno == ECHILD)) {
            return;
        }
        if (waited < 0 && errno != EINTR) {
            return;
        }
    }
}

class SpawnedProcess {
public:
    explicit SpawnedProcess(pid_t pid) noexcept : pid_(pid) {}
    SpawnedProcess(const SpawnedProcess&) = delete;
    SpawnedProcess& operator=(const SpawnedProcess&) = delete;
    ~SpawnedProcess() {
        if (active_) {
            terminate_process_group_and_reap(pid_);
        }
    }

    void mark_reaped() noexcept { active_ = false; }

private:
    pid_t pid_ = -1;
    bool active_ = true;
};

int wait_for_process_until(pid_t pid, ProcessDeadline deadline) {
    int status = 0;
    if (deadline == ProcessDeadline::max()) {
        for (;;) {
            const pid_t waited = waitpid(pid, &status, 0);
            if (waited == pid) {
                return process_exit_code(status);
            }
            if (waited < 0 && errno == ECHILD) {
                return -1;
            }
            if (waited < 0 && errno != EINTR) {
                terminate_process_group_and_reap(pid);
                return -1;
            }
        }
    }

    for (;;) {
        const pid_t waited = waitpid(pid, &status, WNOHANG);
        if (waited == pid) {
            return process_exit_code(status);
        }
        if (waited < 0 && errno == ECHILD) {
            return -1;
        }
        if (waited < 0 && errno != EINTR) {
            terminate_process_group_and_reap(pid);
            return -1;
        }
        const auto now = ProcessClock::now();
        if (now >= deadline) {
            terminate_process_group_and_reap(pid);
            return 124;
        }
        const auto remaining = deadline - now;
        auto sleep_time = std::chrono::duration_cast<std::chrono::milliseconds>(remaining);
        if (sleep_time <= std::chrono::milliseconds(0)) {
            sleep_time = std::chrono::milliseconds(1);
        }
        std::this_thread::sleep_for(
            std::min(std::chrono::milliseconds(100), sleep_time));
    }
}

std::vector<char*> spawn_argument_pointers(const std::vector<std::string>& argv) {
    std::vector<char*> pointers;
    pointers.reserve(argv.size() + 1U);
    for (const std::string& argument : argv) {
        pointers.push_back(const_cast<char*>(argument.c_str()));
    }
    pointers.push_back(nullptr);
    return pointers;
}

int spawn_in_new_process_group(const std::vector<std::string>& argv,
                               std::vector<char*>& arguments,
                               const posix_spawn_file_actions_t* file_actions,
                               pid_t& pid) noexcept {
    if (argv.empty() || arguments.size() != argv.size() + 1U) {
        return EINVAL;
    }
    posix_spawnattr_t attributes;
    int error = posix_spawnattr_init(&attributes);
    if (error != 0) {
        return error;
    }
    error = posix_spawnattr_setpgroup(&attributes, 0);
    if (error == 0) {
        error = posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETPGROUP);
    }
    if (error == 0) {
        // All callers provide an exact absolute executable (either the resolved
        // oracle or /bin/sh), so avoid a second PATH lookup at launch time.
        error = posix_spawn(&pid,
                            argv.front().c_str(),
                            file_actions,
                            &attributes,
                            arguments.data(),
                            environ);
    }
    (void)posix_spawnattr_destroy(&attributes);
    return error;
}

#if !ALDOUS_TSP_HAVE_POSIX_SPAWN_CHDIR_NP
std::vector<std::string> process_arguments_in_directory(
    const std::vector<std::string>& argv,
    const std::filesystem::path& cwd) {
    if (cwd.empty()) {
        return argv;
    }
    // POSIX has no standard working-directory file action before POSIX.1-2024.
    // On implementations without the common addchdir_np extension, spawn a
    // shell with a constant command and pass directory/program solely as argv;
    // no user-controlled text is interpolated. The shell immediately execs the
    // solver and retains the same PID/process group for timeout handling.
    std::vector<std::string> wrapped;
    wrapped.reserve(argv.size() + 5U);
    wrapped.emplace_back("/bin/sh");
    wrapped.emplace_back("-c");
    wrapped.emplace_back(
        "cd \"$1\" || exit 126; shift; "
        "[ -x \"$1\" ] || { echo 'oracle executable is not runnable' >&2; exit 127; }; "
        "exec \"$@\"");
    wrapped.emplace_back("aldous_tsp_spawn");
    wrapped.push_back(cwd.string());
    wrapped.insert(wrapped.end(), argv.begin(), argv.end());
    return wrapped;
}
#endif

int run_external_process(const std::vector<std::string>& argv,
                         const std::filesystem::path& cwd,
                         int timeout_sec,
                         bool verbose,
                         std::string* captured) {
    if (argv.empty()) {
        return -1;
    }
    const std::filesystem::path log_path = cwd.empty()
        ? std::filesystem::path("oracle_output.txt")
        : (cwd / "oracle_output.txt");
    const std::string log_str = log_path.string();
#if ALDOUS_TSP_HAVE_POSIX_SPAWN_CHDIR_NP
    const std::vector<std::string>& spawn_argv = argv;
#else
    const std::vector<std::string> spawn_argv =
        process_arguments_in_directory(argv, cwd);
#endif
    std::vector<char*> spawn_arguments = spawn_argument_pointers(spawn_argv);
    const std::string cwd_str = cwd.string();

    posix_spawn_file_actions_t actions;
    int error = posix_spawn_file_actions_init(&actions);
    if (error != 0) {
        if (captured != nullptr) {
            *captured = std::string("posix_spawn file-action initialization failed: ")
                + std::strerror(error);
        }
        return -1;
    }
    error = posix_spawn_file_actions_addopen(
        &actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
    if (error == 0) {
        error = posix_spawn_file_actions_addopen(
            &actions,
            STDOUT_FILENO,
            log_str.c_str(),
            O_WRONLY | O_CREAT | O_TRUNC,
            0600);
    }
    if (error == 0) {
        error = posix_spawn_file_actions_adddup2(
            &actions, STDOUT_FILENO, STDERR_FILENO);
    }
#if ALDOUS_TSP_HAVE_POSIX_SPAWN_CHDIR_NP
    if (error == 0 && !cwd_str.empty()) {
        error = posix_spawn_file_actions_addchdir_np(&actions, cwd_str.c_str());
    }
#endif

    pid_t pid = -1;
    if (error == 0) {
        error = spawn_in_new_process_group(
            spawn_argv, spawn_arguments, &actions, pid);
    }
    (void)posix_spawn_file_actions_destroy(&actions);
    if (error != 0) {
        if (captured != nullptr) {
            *captured = std::string("posix_spawn failed: ") + std::strerror(error);
        }
        return -1;
    }

    SpawnedProcess process(pid);
    const int rc = wait_for_process_until(pid, process_deadline(timeout_sec));
    process.mark_reaped();
    const std::string output = summarize_child_output(log_path);
    if (captured != nullptr) {
        *captured = output;
    }
    if (verbose && !output.empty()) {
        std::fprintf(stderr, "[oracle] %s\n", output.c_str());
    }
    return rc;
}

int poll_timeout_ms(ProcessDeadline deadline) noexcept {
    if (deadline == ProcessDeadline::max()) {
        return -1;
    }
    const auto now = ProcessClock::now();
    if (now >= deadline) {
        return 0;
    }
    const auto remaining = deadline - now;
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(remaining);
    const auto rounded = milliseconds + (milliseconds < remaining ? std::chrono::milliseconds(1)
                                                                   : std::chrono::milliseconds(0));
    const auto max_int = std::chrono::milliseconds(std::numeric_limits<int>::max());
    return static_cast<int>(std::min(rounded, max_int).count());
}

bool prepare_pipe_descriptor(UniqueFd& descriptor) noexcept {
    if (descriptor.get() < 0) {
        return false;
    }
    if (descriptor.get() <= STDERR_FILENO) {
        const int duplicate = fcntl(descriptor.get(), F_DUPFD, STDERR_FILENO + 1);
        if (duplicate < 0) {
            return false;
        }
        descriptor.reset(duplicate);
    }
    const int flags = fcntl(descriptor.get(), F_GETFD, 0);
    return flags >= 0
        && fcntl(descriptor.get(), F_SETFD, flags | FD_CLOEXEC) == 0;
}

std::string capture_process_first_line(const std::vector<std::string>& argv,
                                       int timeout_sec) {
    if (argv.empty()) {
        return "unknown";
    }
    // Build all C++ argument storage before acquiring OS resources. Once file
    // actions are initialized, the setup path below performs only non-throwing
    // POSIX calls until those actions have been destroyed.
    std::vector<char*> spawn_arguments = spawn_argument_pointers(argv);

    int pipefd[2] = {-1, -1};
    if (pipe(pipefd) != 0) {
        return "unknown";
    }
    UniqueFd read_end(pipefd[0]);
    UniqueFd write_end(pipefd[1]);
    if (!prepare_pipe_descriptor(read_end) || !prepare_pipe_descriptor(write_end)) {
        return "unknown";
    }
    const int read_flags = fcntl(read_end.get(), F_GETFL, 0);
    if (read_flags < 0
        || fcntl(read_end.get(), F_SETFL, read_flags | O_NONBLOCK) != 0) {
        return "unknown";
    }

    posix_spawn_file_actions_t actions;
    int error = posix_spawn_file_actions_init(&actions);
    if (error != 0) {
        return "unknown";
    }
    error = posix_spawn_file_actions_addopen(
        &actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
    if (error == 0) {
        error = posix_spawn_file_actions_addclose(&actions, read_end.get());
    }
    if (error == 0) {
        error = posix_spawn_file_actions_adddup2(
            &actions, write_end.get(), STDOUT_FILENO);
    }
    if (error == 0) {
        error = posix_spawn_file_actions_adddup2(
            &actions, write_end.get(), STDERR_FILENO);
    }
    if (error == 0) {
        error = posix_spawn_file_actions_addclose(&actions, write_end.get());
    }

    pid_t pid = -1;
    if (error == 0) {
        error = spawn_in_new_process_group(
            argv, spawn_arguments, &actions, pid);
    }
    (void)posix_spawn_file_actions_destroy(&actions);
    if (error != 0) {
        return "unknown";
    }

    SpawnedProcess process(pid);
    write_end.reset();
    const ProcessDeadline deadline = process_deadline(timeout_sec);
    std::string output;
    std::array<char, 256> chunk{};
    bool pipe_closed = false;
    while (output.size() <= 512U && output.find('\n') == std::string::npos) {
        pollfd descriptor{};
        descriptor.fd = read_end.get();
        descriptor.events = POLLIN | POLLHUP;
        const int polled = poll(&descriptor, 1, poll_timeout_ms(deadline));
        if (polled == 0) {
            break;
        }
        if (polled < 0) {
            if (errno == EINTR) {
                continue;
            }
            break;
        }
        if ((descriptor.revents & POLLNVAL) != 0) {
            break;
        }
        if ((descriptor.revents & (POLLIN | POLLHUP | POLLERR)) == 0) {
            continue;
        }
        for (;;) {
            const ssize_t nread = read(read_end.get(), chunk.data(), chunk.size());
            if (nread > 0) {
                output.append(chunk.data(), static_cast<std::size_t>(nread));
                if (output.size() > 512U || output.find('\n') != std::string::npos) {
                    break;
                }
                continue;
            }
            if (nread == 0) {
                pipe_closed = true;
            }
            if (nread < 0 && errno == EINTR) {
                continue;
            }
            break;
        }
        if (pipe_closed) {
            break;
        }
    }
    read_end.reset();

    const int rc = wait_for_process_until(pid, deadline);
    process.mark_reaped();
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
    std::error_code size_error;
    const std::uintmax_t file_bytes = std::filesystem::file_size(path, size_error);
    if (size_error || file_bytes > kMaxOracleTourBytes) {
        return false;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    std::string text(static_cast<std::size_t>(file_bytes), '\0');
    in.read(text.data(), static_cast<std::streamsize>(text.size()));
    if (!in && !in.eof()) {
        return false;
    }
    text.resize(static_cast<std::size_t>(in.gcount()));
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
    std::string launch_hash;
    std::string hash_error;
    if (!detail::sha256_file(oracle.exec_path, launch_hash, hash_error)) {
        return fail("oracle executable identity could not be verified before launch: " + hash_error);
    }
    if (launch_hash != oracle.exec_sha256) {
        return fail("oracle executable changed after resolution; refusing launch (expected sha256="
                    + oracle.exec_sha256 + ", actual sha256=" + launch_hash + ")");
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
    if (child_output.rfind("oracle console output exceeded the 16 MiB security limit", 0U) == 0U) {
        return fail(child_output);
    }
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

    std::string hash_error;
    if (!detail::sha256_file(oracle.exec_path, oracle.exec_sha256, hash_error)) {
        error = "failed to hash requested oracle executable: " + hash_error;
        oracle = OracleContext();
        oracle.cfg = cfg_copy;
        return false;
    }
    oracle.version = capture_process_first_line({oracle.exec_path, "--version"}, 2);
    std::ostringstream status;
    status << resolved_oracle_mode_name(oracle.resolved)
           << " @ " << oracle.exec_path
           << " (sha256=" << oracle.exec_sha256
           << ", version=" << oracle.version
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
    std::optional<ConcurrencyLimiter::Permit> resource_permit;
    if (oracle.concurrency_limiter != nullptr) {
        resource_permit.emplace(oracle.concurrency_limiter->acquire());
    }

    OracleCallRecord record;
    record.type = full_tsp ? "tsp" : "subset";
    record.k = candidate.k;
    record.solver = resolved_oracle_mode_name(oracle.resolved);
    record.format = oracle_problem_format_name(oracle.cfg.problem_format);
    record.exec_path = oracle.exec_path;
    record.exec_sha256 = oracle.exec_sha256;
    record.solver_version = oracle.version;
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
