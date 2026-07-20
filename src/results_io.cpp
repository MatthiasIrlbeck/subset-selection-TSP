#include "aldous_tsp/results.hpp"

#include <atomic>
#include <cerrno>
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

void set_error(std::string* error, const std::string& message) {
    if (error != nullptr) {
        *error = message;
    }
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

std::string windows_error_message(DWORD code) {
    LPSTR buffer = nullptr;
    const DWORD flags = FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                        FORMAT_MESSAGE_IGNORE_INSERTS;
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

bool write_atomic_impl(const std::filesystem::path& target,
                       const std::string& text,
                       OutputDurability durability,
                       std::string* error) {
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
            set_error(error, "failed to create temporary output file: " + windows_error_message(code));
            return false;
        }
    }
    if (handle == INVALID_HANDLE_VALUE) {
        set_error(error, "failed to allocate a unique temporary output file");
        return false;
    }

    bool ok = true;
    std::size_t offset = 0;
    while (offset < text.size()) {
        const std::size_t remaining = text.size() - offset;
        const DWORD chunk = static_cast<DWORD>(std::min<std::size_t>(
            remaining, static_cast<std::size_t>((std::numeric_limits<DWORD>::max)())));
        DWORD written = 0;
        if (WriteFile(handle, text.data() + offset, chunk, &written, nullptr) == 0 || written == 0U) {
            set_error(error, "failed to write temporary output file: " +
                             windows_error_message(GetLastError()));
            ok = false;
            break;
        }
        offset += static_cast<std::size_t>(written);
    }
    if (ok && durability != OutputDurability::None && FlushFileBuffers(handle) == 0) {
        set_error(error, "failed to flush temporary output file: " +
                         windows_error_message(GetLastError()));
        ok = false;
    }
    if (CloseHandle(handle) == 0 && ok) {
        set_error(error, "failed to close temporary output file: " +
                         windows_error_message(GetLastError()));
        ok = false;
    }
    if (!ok) {
        DeleteFileW(temp.wstring().c_str());
        return false;
    }

    DWORD flags = MOVEFILE_REPLACE_EXISTING;
    if (durability == OutputDurability::Full) {
        flags |= MOVEFILE_WRITE_THROUGH;
    }
    if (MoveFileExW(temp.wstring().c_str(), target.wstring().c_str(), flags) == 0) {
        const DWORD code = GetLastError();
        DeleteFileW(temp.wstring().c_str());
        set_error(error, "failed to replace output file atomically: " + windows_error_message(code));
        return false;
    }
    return true;
}

#else

bool write_all(int fd, const std::string& text, std::string* error) {
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
            set_error(error, std::string("failed to write temporary output file: ") +
                             std::strerror(errno));
            return false;
        }
        if (written == 0) {
            set_error(error, "failed to write temporary output file: zero-length write");
            return false;
        }
        offset += static_cast<std::size_t>(written);
    }
    return true;
}

bool sync_parent(const std::filesystem::path& target, std::string* error) {
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
        set_error(error, std::string("failed to open output directory for synchronization: ") +
                         std::strerror(errno));
        return false;
    }
    bool ok = true;
    if (::fsync(fd) != 0) {
        set_error(error, std::string("failed to synchronize output directory: ") +
                         std::strerror(errno));
        ok = false;
    }
    if (::close(fd) != 0 && ok) {
        set_error(error, std::string("failed to close output directory: ") +
                         std::strerror(errno));
        ok = false;
    }
    return ok;
}

bool write_atomic_impl(const std::filesystem::path& target,
                       const std::string& text,
                       OutputDurability durability,
                       std::string* error) {
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
            set_error(error, std::string("failed to create temporary output file: ") +
                             std::strerror(errno));
            return false;
        }
    }
    if (fd < 0) {
        set_error(error, "failed to allocate a unique temporary output file");
        return false;
    }

    bool ok = write_all(fd, text, error);
    if (ok && durability != OutputDurability::None && ::fsync(fd) != 0) {
        set_error(error, std::string("failed to synchronize temporary output file: ") +
                         std::strerror(errno));
        ok = false;
    }
    if (::close(fd) != 0 && ok) {
        set_error(error, std::string("failed to close temporary output file: ") +
                         std::strerror(errno));
        ok = false;
    }
    if (!ok) {
        (void)::unlink(temp.c_str());
        return false;
    }

    if (::rename(temp.c_str(), target.c_str()) != 0) {
        const int code = errno;
        (void)::unlink(temp.c_str());
        set_error(error, std::string("failed to replace output file atomically: ") +
                         std::strerror(code));
        return false;
    }
    if (durability == OutputDurability::Full && !sync_parent(target, error)) {
        return false;
    }
    return true;
}

#endif

} // namespace

bool write_text_file_atomic(const std::string& path,
                            const std::string& text,
                            OutputDurability durability,
                            std::string* error) {
    if (path.empty()) {
        set_error(error, "output path must not be empty");
        return false;
    }
    const std::filesystem::path target(path);
    if (target.filename().empty()) {
        set_error(error, "output path must name a file");
        return false;
    }
    return write_atomic_impl(target, text, durability, error);
}

bool write_text_file_atomic(const std::string& path,
                            const std::string& text,
                            std::string* error) {
    return write_text_file_atomic(path, text, OutputDurability::Full, error);
}

} // namespace aldous_tsp
