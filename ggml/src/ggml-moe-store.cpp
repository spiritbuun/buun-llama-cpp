#include "ggml-moe-store.h"
#include "ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

struct store_file {
    uint64_t size = 0;
    std::atomic<bool> direct{false};
#if defined(_WIN32)
    HANDLE handle = INVALID_HANDLE_VALUE;
    HANDLE buffered = INVALID_HANDLE_VALUE;
#else
    int fd = -1;
    int buffered = -1;
#endif
};

struct store_request {
    ggml_moe_store_done_t done = nullptr;
    void * user = nullptr;
    std::atomic<size_t> pieces_left{0};
    std::atomic<bool> failed{false};
};

struct store_piece {
    int priority = 0;
    uint64_t seq = 0;
    store_file * file = nullptr;
    uint64_t offset = 0;
    size_t size = 0;
    char * dst = nullptr;
    store_request * request = nullptr;

    bool operator<(const store_piece & other) const {
        // std::priority_queue pops the largest element first.
        return priority != other.priority ? priority > other.priority : seq > other.seq;
    }
};

}  // namespace

struct ggml_moe_store {
    size_t piece_bytes = 0;
    std::vector<std::unique_ptr<store_file>> files;
    std::vector<std::thread> threads;
    std::mutex mu;
    std::condition_variable cv;
    std::priority_queue<store_piece> queue;
    uint64_t next_seq = 0;
    bool stopping = false;
};

#if defined(_WIN32)
static HANDLE moe_store_open_handle(const char * path, bool direct) {
    const DWORD flags = FILE_FLAG_OVERLAPPED | (direct ? FILE_FLAG_NO_BUFFERING : FILE_FLAG_RANDOM_ACCESS);
    return CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, flags, nullptr);
}

// Reads up to size bytes; returns the count read, or -1.
static int64_t moe_store_pread(HANDLE handle, char * dst, size_t size, uint64_t offset, HANDLE event) {
    OVERLAPPED overlapped = {};
    overlapped.Offset = (DWORD)offset;
    overlapped.OffsetHigh = (DWORD)(offset >> 32);
    overlapped.hEvent = event;
    DWORD read = 0;
    if (!ReadFile(handle, dst, (DWORD)size, nullptr, &overlapped) && GetLastError() != ERROR_IO_PENDING) {
        return GetLastError() == ERROR_HANDLE_EOF ? 0 : -1;
    }
    if (!GetOverlappedResult(handle, &overlapped, &read, TRUE)) {
        return GetLastError() == ERROR_HANDLE_EOF ? 0 : -1;
    }
    return (int64_t)read;
}
#endif

// Reads a piece until it is complete or the file ends; the tail of the
// aligned window past the end of the file is zeroed.
static bool moe_store_read_piece(store_file & file, const store_piece & piece, void * event) {
    size_t done = 0;
    while (done < piece.size) {
        const bool direct = file.direct.load(std::memory_order_relaxed);
#if defined(_WIN32)
        const int64_t got = moe_store_pread(direct ? file.handle : file.buffered,
                piece.dst + done, piece.size - done, piece.offset + done, (HANDLE)event);
        if (got < 0 && direct) {
            // Some filters reject unbuffered reads only once a read is issued.
            file.direct.store(false);
            continue;
        }
#else
        (void)event;
        const ssize_t got = pread(direct ? file.fd : file.buffered,
                piece.dst + done, piece.size - done, (off_t)(piece.offset + done));
        if (got < 0 && errno == EINTR) {
            continue;
        }
        if (got < 0 && direct && (errno == EINVAL || errno == EFAULT)) {
            file.direct.store(false);
            continue;
        }
#endif
        if (got < 0) {
            return false;
        }
        if (got == 0) {
            if (piece.offset + done < file.size) {
                return false;
            }
            memset(piece.dst + done, 0, piece.size - done);
            break;
        }
        done += (size_t)got;
    }
#if defined(__linux__)
    if (!file.direct.load(std::memory_order_relaxed)) {
        // Buffered fallback: do not let streamed experts accumulate as page cache.
        (void)posix_fadvise(file.buffered, (off_t)piece.offset, (off_t)piece.size, POSIX_FADV_DONTNEED);
    }
#endif
    return true;
}

static void moe_store_finish(store_request * request, bool ok) {
    if (!ok) {
        request->failed.store(true);
    }
    if (request->pieces_left.fetch_sub(1) == 1) {
        if (request->done) {
            request->done(request->user, !request->failed.load());
        }
        delete request;
    }
}

static void moe_store_worker(ggml_moe_store * store) {
#if defined(_WIN32)
    HANDLE event = CreateEventA(nullptr, TRUE, FALSE, nullptr);
#else
    void * event = nullptr;
#endif
    for (;;) {
        store_piece piece;
        {
            std::unique_lock<std::mutex> lock(store->mu);
            store->cv.wait(lock, [&] { return store->stopping || !store->queue.empty(); });
            if (store->stopping) {
                break;
            }
            piece = store->queue.top();
            store->queue.pop();
        }
#if defined(_WIN32)
        const bool ok = event && moe_store_read_piece(*piece.file, piece, event);
#else
        const bool ok = moe_store_read_piece(*piece.file, piece, event);
#endif
        moe_store_finish(piece.request, ok);
    }
#if defined(_WIN32)
    if (event) {
        CloseHandle(event);
    }
#endif
}

struct ggml_moe_store * ggml_moe_store_init(int n_threads, size_t piece_bytes) {
    if (n_threads <= 0 || piece_bytes == 0) {
        return nullptr;
    }
    auto store = std::make_unique<ggml_moe_store>();
    store->piece_bytes = GGML_PAD(piece_bytes, (size_t)GGML_MOE_STORE_ALIGN);
    try {
        for (int i = 0; i < n_threads; i++) {
            store->threads.emplace_back(moe_store_worker, store.get());
        }
    } catch (...) {
        ggml_moe_store_free(store.release());
        return nullptr;
    }
    return store.release();
}

void ggml_moe_store_free(struct ggml_moe_store * store) {
    if (!store) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(store->mu);
        store->stopping = true;
    }
    store->cv.notify_all();
    for (std::thread & thread : store->threads) {
        thread.join();
    }
    while (!store->queue.empty()) {
        moe_store_finish(store->queue.top().request, false);
        store->queue.pop();
    }
    for (auto & file : store->files) {
#if defined(_WIN32)
        if (file->handle != INVALID_HANDLE_VALUE) CloseHandle(file->handle);
        if (file->buffered != INVALID_HANDLE_VALUE) CloseHandle(file->buffered);
#else
        if (file->fd >= 0) close(file->fd);
        if (file->buffered >= 0) close(file->buffered);
#endif
    }
    delete store;
}

int ggml_moe_store_open(struct ggml_moe_store * store, const char * path) {
    auto file = std::make_unique<store_file>();
#if defined(_WIN32)
    file->buffered = moe_store_open_handle(path, false);
    if (file->buffered == INVALID_HANDLE_VALUE) {
        return -1;
    }
    LARGE_INTEGER size = {};
    if (!GetFileSizeEx(file->buffered, &size)) {
        CloseHandle(file->buffered);
        return -1;
    }
    file->size = (uint64_t)size.QuadPart;
    file->handle = moe_store_open_handle(path, true);
    file->direct.store(file->handle != INVALID_HANDLE_VALUE);
#else
    file->buffered = open(path, O_RDONLY | O_CLOEXEC);
    if (file->buffered < 0) {
        return -1;
    }
    struct stat st = {};
    if (fstat(file->buffered, &st) != 0) {
        close(file->buffered);
        return -1;
    }
    file->size = (uint64_t)st.st_size;
#if defined(__linux__)
    file->fd = open(path, O_RDONLY | O_CLOEXEC | O_DIRECT);
    (void)posix_fadvise(file->buffered, 0, 0, POSIX_FADV_RANDOM);
#endif
    file->direct.store(file->fd >= 0);
#endif
    if (!file->direct.load()) {
        GGML_LOG_WARN("%s: direct I/O unavailable for %s; using buffered reads\n", __func__, path);
    }
    std::lock_guard<std::mutex> lock(store->mu);
    store->files.push_back(std::move(file));
    return (int)store->files.size() - 1;
}

size_t ggml_moe_store_span(size_t offset, size_t size) {
    return GGML_PAD((offset & (GGML_MOE_STORE_ALIGN - 1)) + size, (size_t)GGML_MOE_STORE_ALIGN);
}

void * ggml_moe_store_read(
        struct ggml_moe_store * store, int file, uint64_t offset, size_t size,
        void * dst, int priority, ggml_moe_store_done_t done, void * user) {
    const uint64_t start = offset & ~(uint64_t)(GGML_MOE_STORE_ALIGN - 1);
    const size_t span = ggml_moe_store_span((size_t)(offset - start), size);
    size_t n_pieces = 0;
    {
        std::lock_guard<std::mutex> lock(store->mu);
        if (!store->stopping && file >= 0 && file < (int)store->files.size() && size > 0 && dst &&
            ((uintptr_t)dst & (GGML_MOE_STORE_ALIGN - 1)) == 0 && offset + size <= store->files[file]->size) {
            n_pieces = (span + store->piece_bytes - 1) / store->piece_bytes;
            store_request * request = new store_request;
            request->done = done;
            request->user = user;
            request->pieces_left.store(n_pieces);
            for (size_t i = 0; i < n_pieces; i++) {
                const size_t begin = i * store->piece_bytes;
                store->queue.push({priority, store->next_seq++, store->files[file].get(), start + begin,
                        std::min(store->piece_bytes, span - begin), (char *)dst + begin, request});
            }
        }
    }
    if (n_pieces == 0) {
        if (done) {
            done(user, false);
        }
        return nullptr;
    }
    for (size_t i = 0; i < n_pieces && i < store->threads.size(); i++) {
        store->cv.notify_one();
    }
    return (char *)dst + (offset - start);
}

namespace {

// one open file shared by every live range of its mapping
struct store_mapped_file {
    int fd = -1;
    uint64_t id = 0;
    ~store_mapped_file() {
#if defined(__linux__)
        close(fd);
#endif
    }
};

struct store_mapping {
    size_t size = 0;
    uint64_t offset = 0;
    std::shared_ptr<store_mapped_file> file;
};

std::mutex g_mappings_mu;
std::map<uintptr_t, store_mapping> g_mappings;
uint64_t g_mapped_files = 0;

}  // namespace

void ggml_moe_store_map(const void * addr, size_t size, int fd) {
#if defined(__linux__)
    const int copy = fcntl(fd, F_DUPFD_CLOEXEC, 0);
    if (copy < 0) {
        return;
    }
    auto file = std::make_shared<store_mapped_file>();
    file->fd = copy;
    std::lock_guard<std::mutex> lock(g_mappings_mu);
    file->id = ++g_mapped_files;
    g_mappings[(uintptr_t)addr] = {size, 0, std::move(file)};
#else
    GGML_UNUSED(addr);
    GGML_UNUSED(size);
    GGML_UNUSED(fd);
#endif
}

void ggml_moe_store_unmap(const void * addr, size_t size) {
    const uintptr_t first = (uintptr_t)addr;
    const uintptr_t last = first + size;
    std::lock_guard<std::mutex> lock(g_mappings_mu);
    auto it = g_mappings.upper_bound(first);
    if (it != g_mappings.begin()) {
        --it;
    }
    while (it != g_mappings.end() && it->first < last) {
        const uintptr_t begin = it->first;
        const uintptr_t end = begin + it->second.size;
        if (end <= first) {
            ++it;
            continue;
        }
        const store_mapping mapping = it->second;
        it = g_mappings.erase(it);
        // keep the parts outside [first, last)
        if (begin < first) {
            g_mappings[begin] = {first - begin, mapping.offset, mapping.file};
        }
        if (end > last) {
            it = g_mappings.emplace(last, store_mapping{end - last, mapping.offset + (last - begin), mapping.file}).first;
            ++it;
        }
    }
}

bool ggml_moe_store_find(const void * ptr, size_t size, char * path, size_t path_size, uint64_t * file, uint64_t * offset) {
#if defined(__linux__)
    std::lock_guard<std::mutex> lock(g_mappings_mu);
    auto found = g_mappings.upper_bound((uintptr_t)ptr);
    if (found == g_mappings.begin()) {
        return false;
    }
    --found;
    const uint64_t position = (uintptr_t)ptr - found->first;
    if (position + size > found->second.size) {
        return false;
    }
    if (path) {
        // reopens unlinked files too, with flags of the store's choosing
        snprintf(path, path_size, "/proc/self/fd/%d", found->second.file->fd);
    }
    if (file) {
        *file = found->second.file->id;
    }
    if (offset) {
        *offset = found->second.offset + position;
    }
    return true;
#else
    GGML_UNUSED(ptr);
    GGML_UNUSED(size);
    GGML_UNUSED(path);
    GGML_UNUSED(path_size);
    GGML_UNUSED(file);
    GGML_UNUSED(offset);
    return false;
#endif
}

#if defined(__linux__)
static size_t moe_store_read_number(const std::string & path, const char * key) {
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
        if (!key) {
            return line == "max" ? SIZE_MAX : (size_t) strtoull(line.c_str(), nullptr, 10);
        }
        if (line.compare(0, strlen(key), key) == 0) {
            return (size_t) strtoull(line.c_str() + strlen(key), nullptr, 10);
        }
    }
    return SIZE_MAX;
}

// Headroom under every cgroup v2 limit above this process: its limit less the memory
// charged to it that cannot be reclaimed without swap: anonymous, shared (pinned host
// allocations land there) and kernel. Page cache, including the mapped weights, can be.
static size_t moe_store_cgroup_room() {
    std::ifstream file("/proc/self/cgroup");
    std::string line;
    size_t room = SIZE_MAX;
    while (std::getline(file, line)) {
        if (line.compare(0, 3, "0::") != 0) {
            continue;
        }
        std::string dir = "/sys/fs/cgroup" + line.substr(3);
        while (dir.size() > strlen("/sys/fs/cgroup")) {
            const size_t limit = moe_store_read_number(dir + "/memory.max", nullptr);
            if (limit != SIZE_MAX) {
                size_t used = 0;
                for (const char * key : {"anon ", "shmem ", "kernel "}) {
                    const size_t bytes = moe_store_read_number(dir + "/memory.stat", key);
                    used += bytes == SIZE_MAX ? 0 : bytes;
                }
                room = std::min(room, used < limit ? limit - used : 0);
            }
            dir.resize(dir.rfind('/'));
        }
    }
    return room;
}
#endif

size_t ggml_moe_store_host_room(void) {
#if defined(__linux__)
    const size_t available = moe_store_read_number("/proc/meminfo", "MemAvailable:");
    return std::min(moe_store_cgroup_room(), available == SIZE_MAX ? SIZE_MAX : available << 10);
#else
    return SIZE_MAX;
#endif
}
