#include "aldous_tsp/results.hpp"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <limits>
#include <sstream>
#include <string>
#include <thread>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace aldous_tsp {
namespace {

std::atomic<std::uint64_t> g_temp_counter{0};
using IoClock = std::chrono::steady_clock;

double elapsed_seconds(const IoClock::time_point start) {
    return std::chrono::duration<double>(IoClock::now() - start).count();
}

AtomicWriteResult failure(const std::string& message) {
    return {OutputCommitState::NotCommitted, message};
}

OutputCommitState committed_state(OutputDurability durability) {
    switch (durability) {
        case OutputDurability::None: return OutputCommitState::Committed;
        case OutputDurability::File: return OutputCommitState::FileDurable;
        case OutputDurability::Full: return OutputCommitState::FileDurable;
    }
    return OutputCommitState::Committed;
}

std::filesystem::path parent_directory(const std::filesystem::path& target) {
    const std::filesystem::path parent = target.parent_path();
    return parent.empty() ? std::filesystem::path(".") : parent;
}

std::filesystem::path unique_temp_candidate(const std::filesystem::path& target,
                                            std::uint64_t attempt) {
    const std::uint64_t serial = g_temp_counter.fetch_add(1, std::memory_order_relaxed);
    const std::uint64_t thread_id = static_cast<std::uint64_t>(
        std::hash<std::thread::id>{}(std::this_thread::get_id()));
#if defined(_WIN32)
    const std::uint64_t process_id = static_cast<std::uint64_t>(GetCurrentProcessId());
#else
    const std::uint64_t process_id = static_cast<std::uint64_t>(::getpid());
#endif
    std::ostringstream suffix;
    suffix << target.filename().string() << ".tmp." << process_id << '.'
           << thread_id << '.' << serial << '.' << attempt;
    return parent_directory(target) / suffix.str();
}

#if defined(_WIN32)

bool transient_windows_replace_error(DWORD code) noexcept {
    return code == ERROR_ACCESS_DENIED
        || code == ERROR_SHARING_VIOLATION
        || code == ERROR_LOCK_VIOLATION
        || code == ERROR_UNABLE_TO_MOVE_REPLACEMENT;
}

std::string windows_error_message(DWORD code) {
    LPSTR buffer = nullptr;
    const DWORD flags = FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM
        | FORMAT_MESSAGE_IGNORE_INSERTS;
    const DWORD size = FormatMessageA(flags, nullptr, code, 0,
                                      reinterpret_cast<LPSTR>(&buffer), 0, nullptr);
    std::string result = size != 0U && buffer != nullptr
        ? std::string(buffer, static_cast<std::size_t>(size))
        : std::string("Windows error ") + std::to_string(code);
    if (buffer != nullptr) {
        LocalFree(buffer);
    }
    return result;
}

AtomicWriteResult write_atomic_impl(const std::filesystem::path& target,
                                    const std::string& text,
                                    ReplacePolicy replace_policy,
                                    OutputDurability durability) {
    const auto total_start = IoClock::now();
    std::filesystem::path temp;
    HANDLE handle = INVALID_HANDLE_VALUE;
    for (std::uint64_t attempt = 0; attempt < 128U; ++attempt) {
        temp = unique_temp_candidate(target, attempt);
        handle = CreateFileW(temp.wstring().c_str(), GENERIC_WRITE, 0, nullptr,
                             CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (handle != INVALID_HANDLE_VALUE) {
            break;
        }
        const DWORD code = GetLastError();
        if (code != ERROR_FILE_EXISTS && code != ERROR_ALREADY_EXISTS) {
            return failure("failed to create temporary output file: "
                           + windows_error_message(code));
        }
    }
    if (handle == INVALID_HANDLE_VALUE) {
        return failure("failed to allocate a unique temporary output file");
    }

    std::string error;
    bool ok = true;
    const auto write_start = IoClock::now();
    std::size_t offset = 0;
    while (offset < text.size()) {
        const std::size_t remaining = text.size() - offset;
        const DWORD chunk = static_cast<DWORD>(std::min<std::size_t>(
            remaining, static_cast<std::size_t>((std::numeric_limits<DWORD>::max)())));
        DWORD written = 0;
        if (WriteFile(handle, text.data() + offset, chunk, &written, nullptr) == 0
            || written == 0U) {
            error = "failed to write temporary output file: "
                + windows_error_message(GetLastError());
            ok = false;
            break;
        }
        offset += static_cast<std::size_t>(written);
    }
    AtomicWriteResult result;
    result.write_seconds = elapsed_seconds(write_start);
    const auto sync_start = IoClock::now();
    if (ok && durability != OutputDurability::None && FlushFileBuffers(handle) == 0) {
        error = "failed to flush temporary output file: "
            + windows_error_message(GetLastError());
        ok = false;
    }
    if (CloseHandle(handle) == 0 && ok) {
        error = "failed to close temporary output file: "
            + windows_error_message(GetLastError());
        ok = false;
    }
    if (!ok) {
        DeleteFileW(temp.wstring().c_str());
        result.message = error;
        result.synchronization_seconds = elapsed_seconds(sync_start);
        result.total_seconds = elapsed_seconds(total_start);
        return result;
    }
    result.synchronization_seconds = elapsed_seconds(sync_start);

    const auto commit_start = IoClock::now();
    DWORD flags = durability == OutputDurability::Full ? MOVEFILE_WRITE_THROUGH : 0U;
    if (replace_policy == ReplacePolicy::ReplaceExisting) {
        flags |= MOVEFILE_REPLACE_EXISTING;
    }
    DWORD commit_error = ERROR_SUCCESS;
    bool committed = false;
    constexpr unsigned max_commit_attempts = 256U;
    for (unsigned attempt = 0; attempt < max_commit_attempts; ++attempt) {
        if (MoveFileExW(temp.wstring().c_str(), target.wstring().c_str(), flags) != 0) {
            committed = true;
            break;
        }
        commit_error = GetLastError();
        if (replace_policy != ReplacePolicy::ReplaceExisting
            || !transient_windows_replace_error(commit_error)) {
            break;
        }
        // Concurrent atomic replacers can briefly hold destination metadata.
        // Yield first, then back off by one millisecond under sustained contention.
        if (attempt < 15U) {
            (void)SwitchToThread();
        } else {
            Sleep(1U);
        }
    }
    if (!committed) {
        DeleteFileW(temp.wstring().c_str());
        if (replace_policy == ReplacePolicy::NoReplace
            && (commit_error == ERROR_FILE_EXISTS
                || commit_error == ERROR_ALREADY_EXISTS)) {
            result.message = "Refusing to overwrite existing output file "
                + target.string() + " (use --force or a different --output)";
            result.commit_seconds = elapsed_seconds(commit_start);
            result.total_seconds = elapsed_seconds(total_start);
            return result;
        }
        result.message = "failed to commit output file atomically: "
            + windows_error_message(commit_error);
        result.commit_seconds = elapsed_seconds(commit_start);
        result.total_seconds = elapsed_seconds(total_start);
        return result;
    }
    result.commit_seconds = elapsed_seconds(commit_start);
    if (durability == OutputDurability::Full) {
        result.state = OutputCommitState::FullyDurable;
    } else {
        result.state = committed_state(durability);
    }
    result.total_seconds = elapsed_seconds(total_start);
    return result;
}

#else

bool write_all(int fd, const std::string& text, std::string& error) {
    std::size_t offset = 0;
    while (offset < text.size()) {
        const std::size_t remaining = text.size() - offset;
        const std::size_t capped = std::min<std::size_t>(
            remaining, static_cast<std::size_t>((std::numeric_limits<ssize_t>::max)()));
        const ssize_t written = ::write(fd, text.data() + offset, capped);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            error = std::string("failed to write temporary output file: ")
                + std::strerror(errno);
            return false;
        }
        if (written == 0) {
            error = "failed to write temporary output file: zero-length write";
            return false;
        }
        offset += static_cast<std::size_t>(written);
    }
    return true;
}

bool sync_parent(const std::filesystem::path& target, std::string& error) {
    const std::filesystem::path parent = parent_directory(target);
    int flags = O_RDONLY;
#ifdef O_DIRECTORY
    flags |= O_DIRECTORY;
#endif
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
    const int fd = ::open(parent.c_str(), flags);
    if (fd < 0) {
        error = std::string("output was committed, but its directory could not be opened "
                            "for synchronization: ") + std::strerror(errno);
        return false;
    }
    bool ok = true;
    if (::fsync(fd) != 0) {
        error = std::string("output was committed, but its directory could not be "
                            "synchronized: ") + std::strerror(errno);
        ok = false;
    }
    if (::close(fd) != 0 && ok) {
        error = std::string("output was committed, but its directory handle could not be "
                            "closed: ") + std::strerror(errno);
        ok = false;
    }
    return ok;
}

AtomicWriteResult write_atomic_impl(const std::filesystem::path& target,
                                    const std::string& text,
                                    ReplacePolicy replace_policy,
                                    OutputDurability durability) {
    const auto total_start = IoClock::now();
    std::filesystem::path temp;
    int fd = -1;
    for (std::uint64_t attempt = 0; attempt < 128U; ++attempt) {
        temp = unique_temp_candidate(target, attempt);
        int flags = O_WRONLY | O_CREAT | O_EXCL;
#ifdef O_CLOEXEC
        flags |= O_CLOEXEC;
#endif
        fd = ::open(temp.c_str(), flags, static_cast<mode_t>(0600));
        if (fd >= 0) {
            break;
        }
        if (errno != EEXIST) {
            return failure(std::string("failed to create temporary output file: ")
                           + std::strerror(errno));
        }
    }
    if (fd < 0) {
        return failure("failed to allocate a unique temporary output file");
    }

    std::string error;
    const auto write_start = IoClock::now();
    bool ok = write_all(fd, text, error);
    AtomicWriteResult result;
    result.write_seconds = elapsed_seconds(write_start);
    const auto sync_start = IoClock::now();
    if (ok && durability != OutputDurability::None && ::fsync(fd) != 0) {
        error = std::string("failed to synchronize temporary output file: ")
            + std::strerror(errno);
        ok = false;
    }
    if (::close(fd) != 0 && ok) {
        error = std::string("failed to close temporary output file: ")
            + std::strerror(errno);
        ok = false;
    }
    if (!ok) {
        (void)::unlink(temp.c_str());
        result.message = error;
        result.synchronization_seconds = elapsed_seconds(sync_start);
        result.total_seconds = elapsed_seconds(total_start);
        return result;
    }
    result.synchronization_seconds = elapsed_seconds(sync_start);

    const auto commit_start = IoClock::now();
    bool committed = false;
    if (replace_policy == ReplacePolicy::ReplaceExisting) {
        committed = ::rename(temp.c_str(), target.c_str()) == 0;
    } else {
        // POSIX link(2) is an atomic create-if-absent operation when source and
        // target are on the same filesystem. The temporary file is deliberately
        // created in the target directory, so this is a portable no-clobber
        // commit without a time-of-check/time-of-use window.
        committed = ::link(temp.c_str(), target.c_str()) == 0;
    }
    if (!committed) {
        const int code = errno;
        (void)::unlink(temp.c_str());
        if (replace_policy == ReplacePolicy::NoReplace && code == EEXIST) {
            result.message = "Refusing to overwrite existing output file "
                + target.string() + " (use --force or a different --output)";
            result.commit_seconds = elapsed_seconds(commit_start);
            result.total_seconds = elapsed_seconds(total_start);
            return result;
        }
        result.message = std::string("failed to commit output file atomically: ")
            + std::strerror(code);
        result.commit_seconds = elapsed_seconds(commit_start);
        result.total_seconds = elapsed_seconds(total_start);
        return result;
    }

    std::string warning;
    if (replace_policy == ReplacePolicy::NoReplace && ::unlink(temp.c_str()) != 0) {
        warning = std::string("output was committed, but temporary-file cleanup failed: ")
            + std::strerror(errno);
    }

    result.state = committed_state(durability);
    result.message = warning;
    result.commit_seconds = elapsed_seconds(commit_start);
    if (durability == OutputDurability::Full) {
        const auto parent_sync_start = IoClock::now();
        std::string sync_error;
        if (!sync_parent(target, sync_error)) {
            result.synchronization_seconds += elapsed_seconds(parent_sync_start);
            if (!result.message.empty()) {
                result.message += "; ";
            }
            result.message += sync_error;
            result.total_seconds = elapsed_seconds(total_start);
            return result;
        }
        result.synchronization_seconds += elapsed_seconds(parent_sync_start);
        result.state = OutputCommitState::FullyDurable;
    }
    result.total_seconds = elapsed_seconds(total_start);
    return result;
}

#endif

} // namespace

AtomicWriteResult write_text_file_atomic(const std::string& path,
                                         const std::string& text,
                                         ReplacePolicy replace_policy,
                                         OutputDurability durability) {
    if (path.empty()) {
        return failure("output path must not be empty");
    }
    const std::filesystem::path target = std::filesystem::u8path(path);
    if (target.filename().empty()) {
        return failure("output path must name a file");
    }
    return write_atomic_impl(target, text, replace_policy, durability);
}

bool write_text_file_atomic(const std::string& path,
                            const std::string& text,
                            OutputDurability durability,
                            std::string* error) {
    const AtomicWriteResult result = write_text_file_atomic(
        path, text, ReplacePolicy::ReplaceExisting, durability);
    if (error != nullptr) {
        *error = result.message;
    }
    return result.satisfies(durability);
}

bool write_text_file_atomic(const std::string& path,
                            const std::string& text,
                            std::string* error) {
    return write_text_file_atomic(path, text, OutputDurability::Full, error);
}

} // namespace aldous_tsp
