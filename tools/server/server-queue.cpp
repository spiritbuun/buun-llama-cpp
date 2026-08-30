#include "server-task.h"
#include "server-queue.h"

#include "log.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <thread>

#define QUE_INF(fmt, ...) LOG_INF("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_WRN(fmt, ...) LOG_WRN("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_ERR(fmt, ...) LOG_ERR("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_DBG(fmt, ...) LOG_DBG("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)

#define RES_INF(fmt, ...) LOG_INF("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define RES_WRN(fmt, ...) LOG_WRN("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define RES_ERR(fmt, ...) LOG_ERR("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define RES_DBG(fmt, ...) LOG_DBG("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)

//
// server_queue
//

struct server_queue_idle_capture_state {
    std::atomic<uint64_t> active_generation { 0 };
    std::atomic<bool> cancelled { false };
};

server_queue::idle_capture_session::idle_capture_session(
        std::shared_ptr<server_queue_idle_capture_state> state,
        uint64_t generation) noexcept
    : state_(std::move(state)), generation_(generation) {
}

server_queue::idle_capture_session::idle_capture_session(
        idle_capture_session && other) noexcept
    : state_(std::move(other.state_)), generation_(other.generation_) {
    other.generation_ = 0;
}

server_queue::idle_capture_session &
server_queue::idle_capture_session::operator=(
        idle_capture_session && other) noexcept {
    if (this != &other) {
        reset();
        state_ = std::move(other.state_);
        generation_ = other.generation_;
        other.generation_ = 0;
    }
    return *this;
}

server_queue::idle_capture_session::~idle_capture_session() {
    reset();
}

server_queue::idle_capture_session::operator bool() const noexcept {
    return state_ && generation_ != 0 &&
        state_->active_generation.load(std::memory_order_acquire) ==
            generation_;
}

bool server_queue::idle_capture_session::continue_capture() const noexcept {
    return bool(*this) &&
        !state_->cancelled.load(std::memory_order_acquire);
}

void server_queue::idle_capture_session::cancel() noexcept {
    if (state_) {
        state_->cancelled.store(true, std::memory_order_release);
    }
}

void server_queue::idle_capture_session::reset() noexcept {
    if (state_ && generation_ != 0) {
        uint64_t expected = generation_;
        (void) state_->active_generation.compare_exchange_strong(
            expected, 0, std::memory_order_acq_rel,
            std::memory_order_acquire);
    }
    generation_ = 0;
    state_.reset();
}

server_queue::~server_queue() {
    {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        idle_capture_stopped = true;
        cancel_idle_capture_locked();
        running = false;
    }
    condition_tasks.notify_all();
    worker_stop();
}

void server_queue::cancel_idle_capture_locked() noexcept {
    if (idle_capture_state) {
        idle_capture_state->cancelled.store(true, std::memory_order_release);
    }
}

server_queue::idle_capture_session
server_queue::try_begin_idle_capture() noexcept {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    if (idle_capture_stopped || worker.busy || !queue_tasks.empty() ||
        !queue_tasks_deferred.empty() || !queue_tasks_unhandled.empty()) {
        return {};
    }
    try {
        if (!idle_capture_state) {
            idle_capture_state =
                std::make_shared<server_queue_idle_capture_state>();
        }
        if (idle_capture_state->active_generation.load(
                std::memory_order_acquire) != 0) {
            return {};
        }
        if (++idle_capture_generation == 0) {
            ++idle_capture_generation;
        }
        idle_capture_state->cancelled.store(
            false, std::memory_order_relaxed);
        idle_capture_state->active_generation.store(
            idle_capture_generation, std::memory_order_release);
        return idle_capture_session(
            idle_capture_state, idle_capture_generation);
    } catch (...) {
        return {};
    }
}

server_queue::idle_capture_session
server_queue::try_begin_prompt_boundary_capture() noexcept {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    const bool only_scheduler_wakes = std::all_of(
        queue_tasks.begin(), queue_tasks.end(), [](const server_task & task) {
            return task.type == SERVER_TASK_TYPE_NEXT_RESPONSE;
        });
    if (idle_capture_stopped || worker.busy || !only_scheduler_wakes ||
        !queue_tasks_deferred.empty() || !queue_tasks_unhandled.empty()) {
        return {};
    }
    try {
        if (!idle_capture_state) {
            idle_capture_state =
                std::make_shared<server_queue_idle_capture_state>();
        }
        if (idle_capture_state->active_generation.load(
                std::memory_order_acquire) != 0) {
            return {};
        }
        if (++idle_capture_generation == 0) {
            ++idle_capture_generation;
        }
        idle_capture_state->cancelled.store(false, std::memory_order_relaxed);
        idle_capture_state->active_generation.store(
            idle_capture_generation, std::memory_order_release);
        return idle_capture_session(
            idle_capture_state, idle_capture_generation);
    } catch (...) {
        return {};
    }
}

void server_queue::request_idle_maintenance() noexcept {
    idle_maintenance_requested.store(true, std::memory_order_release);
    condition_tasks.notify_one();
}

server_queue::diagnostic_snapshot server_queue::diagnostics(
        const std::unordered_set<int> & task_ids) noexcept {
    diagnostic_snapshot result;
    try {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        result.running = running;
        result.sleeping = sleeping;
        result.maintenance_requested = idle_maintenance_requested.load(
            std::memory_order_acquire);
        result.queued = queue_tasks.size() + queue_tasks_unhandled.size();
        result.deferred = queue_tasks_deferred.size();
        result.posts = diagnostic_posts_;
        result.dequeues = diagnostic_dequeues_;
        for (const auto & task : queue_tasks) {
            result.matching_queued += task_ids.count(task.id) != 0;
        }
        for (const auto & task : queue_tasks_unhandled) {
            result.matching_queued += task_ids.count(task.id) != 0;
        }
        for (const auto & task : queue_tasks_deferred) {
            result.matching_deferred += task_ids.count(task.id) != 0;
        }
    } catch (...) {
        // Diagnostics must never perturb request processing.
    }
    return result;
}

static bool task_resets_idle_timer(server_task_type type) {
    return type != SERVER_TASK_TYPE_METRICS;
}

int server_queue::post(server_task && task, bool front) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    cancel_idle_capture_locked();
    GGML_ASSERT(task.id != -1);
    const bool diagnostic_task =
        task.type != SERVER_TASK_TYPE_NEXT_RESPONSE;
    // if this is cancel task make sure to clean up pending tasks
    if (task.type == SERVER_TASK_TYPE_CANCEL) {
        cleanup_pending_task(task.id_target);
    }
    const int  task_id     = task.id;
    const bool reset_timer = task_resets_idle_timer(task.type);
    QUE_DBG("new task, id = %d, front = %d\n", task_id, front);
    if (front) {
        queue_tasks.push_front(std::move(task));
    } else {
        queue_tasks.push_back(std::move(task));
    }
    if (diagnostics_enabled() && diagnostic_task) {
        ++diagnostic_posts_;
        QUE_INF(
            "CACHE_QUEUE event=post task=%d count=1 front=%d queued=%zu "
            "deferred=%zu posts=%" PRIu64 " dequeues=%" PRIu64 "\n",
            task_id, int(front),
            queue_tasks.size() + queue_tasks_unhandled.size(),
            queue_tasks_deferred.size(), diagnostic_posts_,
            diagnostic_dequeues_);
    }
    if (reset_timer) {
        time_last_task = ggml_time_ms();
    }
    condition_tasks.notify_one();
    return task_id;
}

int server_queue::post(std::vector<server_task> && tasks, bool front) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    if (!tasks.empty()) {
        cancel_idle_capture_locked();
    }
    size_t diagnostic_task_count = 0;
    int first_task_id = -1;
    int last_task_id = -1;
    bool reset_timer = false;
    for (auto & task : tasks) {
        if (task.id == -1) {
            task.id = id++;
        }
        // if this is cancel task make sure to clean up pending tasks
        if (task.type == SERVER_TASK_TYPE_CANCEL) {
            cleanup_pending_task(task.id_target);
        }
        if (task.type != SERVER_TASK_TYPE_NEXT_RESPONSE) {
            if (first_task_id == -1) {
                first_task_id = task.id;
            }
            last_task_id = task.id;
            ++diagnostic_task_count;
        }
        reset_timer |= task_resets_idle_timer(task.type);
        QUE_DBG("new task, id = %d/%d, front = %d\n", task.id, (int) tasks.size(), front);
        if (front) {
            queue_tasks.push_front(std::move(task));
        } else {
            queue_tasks.push_back(std::move(task));
        }
    }
    if (diagnostics_enabled() && diagnostic_task_count != 0) {
        diagnostic_posts_ += diagnostic_task_count;
        QUE_INF(
            "CACHE_QUEUE event=post task_first=%d task_last=%d count=%zu "
            "front=%d queued=%zu deferred=%zu posts=%" PRIu64
            " dequeues=%" PRIu64 "\n",
            first_task_id, last_task_id, diagnostic_task_count, int(front),
            queue_tasks.size() + queue_tasks_unhandled.size(),
            queue_tasks_deferred.size(),
            diagnostic_posts_, diagnostic_dequeues_);
    }
    if (reset_timer) {
        time_last_task = ggml_time_ms();
    }
    condition_tasks.notify_one();
    return 0;
}

void server_queue::defer(server_task && task) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    cancel_idle_capture_locked();
    QUE_DBG("defer task, id = %d\n", task.id);
    queue_tasks_deferred.push_back(std::move(task));
    time_last_task = ggml_time_ms();
    condition_tasks.notify_one();
}

int server_queue::get_new_id() {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    int new_id = id++;
    return new_id;
}

void server_queue::pop_deferred_task(int id_slot) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    if (!queue_tasks_deferred.empty()) {
        // try to find a task that uses the specified slot
        bool found = false;
        for (auto it = queue_tasks_deferred.begin(); it != queue_tasks_deferred.end(); ++it) {
            if (it->id_slot == id_slot) {
                QUE_DBG("pop deferred task (use slot %d), id_task = %d\n", id_slot, it->id);
                queue_tasks.emplace_front(std::move(*it));
                queue_tasks_deferred.erase(it);
                found = true;
                break;
            }
        }
        // if not tasks found using the slot, just pop the first deferred task (default behavior)
        if (!found) {
            QUE_DBG("pop deferred task, id_task = %d\n", queue_tasks_deferred.front().id);
            queue_tasks.emplace_front(std::move(queue_tasks_deferred.front()));
            queue_tasks_deferred.pop_front();
        }
    }
    time_last_task = ggml_time_ms();
    condition_tasks.notify_one();
}

void server_queue::wait_until_no_sleep() {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    if (!sleeping) {
        return;
    } else {
        if (!req_stop_sleeping) {
            QUE_DBG("%s", "requesting to stop sleeping\n");
            req_stop_sleeping = true;
            condition_tasks.notify_one(); // only main thread is waiting on this
        }
        QUE_DBG("%s", "waiting until no sleep\n");
        condition_tasks.wait(lock, [&]{
            return !sleeping;
        });
    }
}

void server_queue::terminate() {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    idle_capture_stopped = true;
    cancel_idle_capture_locked();
    running = false;
    condition_tasks.notify_all();
}

bool server_queue::process_new_tasks(bool is_yielding) {
    while (true) {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        if (!running) {
            QUE_DBG("%s", "terminate\n");
            return true;
        }
        if (queue_tasks.empty()) {
            return false;
        }
        server_task task = std::move(queue_tasks.front());
        queue_tasks.pop_front();
        if (diagnostics_enabled() &&
            task.type != SERVER_TASK_TYPE_NEXT_RESPONSE) {
            ++diagnostic_dequeues_;
            QUE_INF(
                "CACHE_QUEUE event=dequeue task=%d queued=%zu deferred=%zu "
                "posts=%" PRIu64 " dequeues=%" PRIu64 "\n",
                task.id, queue_tasks.size() + queue_tasks_unhandled.size(),
                queue_tasks_deferred.size(), diagnostic_posts_,
                diagnostic_dequeues_);
        }
        lock.unlock();

        QUE_DBG("processing task, id = %d\n", task.id);
        if (!callback_new_task(std::move(task), is_yielding)) {
            // set it aside, do not put it back in the queue, else we offer it again in a loop
            GGML_ASSERT(is_yielding && "a task can only be declined while yielding");
            QUE_DBG("task declined, id = %d\n", task.id);
            lock.lock();
            queue_tasks_unhandled.push_back(std::move(task));
        }
    }
}

void server_queue::worker_loop() {
    while (true) {
        {
            std::unique_lock<std::mutex> lock(mutex_tasks);
            // wait on busy instead of yielding - busy stays set even when the yield already ended
            worker.cv.wait(lock, [&]{
                return worker.stop || worker.busy;
            });
            if (worker.stop) {
                return;
            }
        }

        // process tasks while the yield is active
        while (true) {
            bool terminated = false;
            try {
                // note: do not hold any lock here, the callback may post new tasks
                terminated = process_new_tasks(true);
            } catch (...) {
                std::unique_lock<std::mutex> lock(mutex_tasks);
                worker.exception = std::current_exception();
                break;
            }

            std::unique_lock<std::mutex> lock(mutex_tasks);
            if (terminated || worker.stop || !worker.yielding) {
                break;
            }
            if (!queue_tasks.empty()) {
                continue; // a new task arrived in the meantime
            }
            condition_tasks.wait(lock, [&]{
                return worker.stop || !running || !worker.yielding || !queue_tasks.empty();
            });
        }

        // signal to yield_to_queue() that no more tasks will be processed
        {
            std::unique_lock<std::mutex> lock(mutex_tasks);
            worker.busy = false;
        }
        condition_tasks.notify_all();
    }
}

void server_queue::worker_stop() {
    if (!worker.thread.joinable()) {
        return;
    }
    {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        worker.stop = true;
    }
    worker.cv.notify_one();
    condition_tasks.notify_all();
    worker.thread.join();
}

void server_queue::yield_to_queue(std::function<void()> && work) {
    GGML_ASSERT(worker.thread.joinable() && "yield_to_queue() requires start_loop() to be running");

    QUE_DBG("%s", "yielding to queue\n");

    {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        GGML_ASSERT(!worker.busy && "yield_to_queue() cannot be nested");
        worker.busy     = true;
        worker.yielding = true;
    }
    worker.cv.notify_one();

    // run the work on the current thread, so that all ggml compute stays on the same thread
    std::exception_ptr exception;
    try {
        work();
    } catch (...) {
        exception = std::current_exception();
    }

    {
        std::unique_lock<std::mutex> lock(mutex_tasks);

        // the yield is over, wait for the worker to finish its current task
        worker.yielding = false;
        condition_tasks.notify_all();
        condition_tasks.wait(lock, [&]{
            return !worker.busy;
        });

        // put the declined tasks back, keeping their order
        while (!queue_tasks_unhandled.empty()) {
            queue_tasks.push_front(std::move(queue_tasks_unhandled.back()));
            queue_tasks_unhandled.pop_back();
        }

        // make sure to avoid idle timeout here
        time_last_task = ggml_time_ms();

        // an exception from work() takes precedence over the one from the worker
        if (!exception) {
            std::swap(exception, worker.exception);
        } else {
            worker.exception = nullptr;
        }
    }

    QUE_DBG("%s", "done yielding to queue\n");

    // note: rethrow only after the declined tasks are back in the queue, so they are not lost
    if (exception) {
        std::rethrow_exception(exception);
    }
}

std::exception_ptr server_queue::yield_to_queue_capture_exception(
        std::function<void()> && work,
        const std::exception_ptr & delayed_work_exception) noexcept {
    try {
        yield_to_queue(std::move(work));
    } catch (...) {
        return delayed_work_exception ? delayed_work_exception : std::current_exception();
    }

    return delayed_work_exception;
}

void server_queue::start_loop(int64_t idle_sleep_ms) {
    {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        if (idle_capture_stopped) {
            return;
        }
        running = true;
        time_last_task = ggml_time_ms();
    }

    // spawn the worker thread used by yield_to_queue()
    GGML_ASSERT(!worker.thread.joinable() && "start_loop() is already running");
    worker.stop     = false;
    worker.busy     = false;
    worker.yielding = false;
    worker.thread = std::thread([this]() { worker_loop(); });

    constexpr auto max_wait_time = std::chrono::seconds(1);
    auto should_sleep = [&]() -> bool {
        // caller must hold mutex_tasks
        if (idle_sleep_ms < 0) {
            return false;
        }
        int64_t now = ggml_time_ms();
        return (now - time_last_task) >= idle_sleep_ms;
    };

    while (true) {
        QUE_DBG("%s", "processing new tasks\n");
        if (process_new_tasks(false)) {
            break; // terminate
        }

        // all tasks in the current loop is processed, slots data is now ready
        QUE_DBG("%s", "update slots\n");

        // this will run the main inference process for all slots
        const int64_t t_update_slots = ggml_time_ms();
        callback_update_slots();
        {
            // update_slots() may take a while to finish, we need to make sure it's not counted as idle
            // shift instead of reset, so that non-task_resets_idle_timer tasks do not delay the sleep
            std::unique_lock<std::mutex> lock(mutex_tasks);
            const int64_t now = ggml_time_ms();
            time_last_task = std::min(now, time_last_task + (now - t_update_slots));
        }

        QUE_DBG("%s", "waiting for new tasks\n");
        while (true) {
            std::unique_lock<std::mutex> lock(mutex_tasks);
            if (!running || !queue_tasks.empty()) {
                break; // go back to process new tasks or terminate
            }

            // no tasks, check for sleeping state
            if (should_sleep()) {
                QUE_INF("%s", "entering sleeping state\n");
                sleeping = true;
                req_stop_sleeping = false;
                // Call order cb0 -> cb1 -> cb{N}
                lock.unlock();
                for (auto & cb : callback_sleeping_state) {
                    cb(true);
                }
                lock.lock();
                // wait until we are requested to exit sleeping state
                condition_tasks.wait(lock, [&]{
                    return (!running || req_stop_sleeping);
                });
                if (!running) { // may changed during sleep
                    break; // terminate
                }
                QUE_INF("%s", "exiting sleeping state\n");
                req_stop_sleeping = false;
                // Call order cb{N} -> cb1 -> cb0
                lock.unlock();
                for (size_t i = callback_sleeping_state.size(); i > 0; i--) {
                    callback_sleeping_state[i - 1](false);
                }
                lock.lock();
                sleeping = false;
                time_last_task = ggml_time_ms();
                condition_tasks.notify_all(); // notify wait_until_no_sleep()
                break; // process new tasks
            } else {
                // wait for new tasks or timeout for checking sleeping condition
                bool res = condition_tasks.wait_for(lock, max_wait_time, [&]{
                    return (!queue_tasks.empty() || !running ||
                            idle_maintenance_requested.load(
                                std::memory_order_acquire));
                });
                if (res) {
                    if (running && queue_tasks.empty() &&
                        idle_maintenance_requested.exchange(
                            false, std::memory_order_acq_rel)) {
                        if (callback_idle) {
                            lock.unlock();
                            callback_idle();
                            continue;
                        }
                    }
                    break; // new task arrived or terminate
                }
                // timeout with an empty queue: give the context a quiet moment for
                // deferred maintenance (memory breathing), outside the queue lock
                if (callback_idle) {
                    lock.unlock();
                    callback_idle();
                }
                // loop again to check sleeping condition
            }
        }
    }

    worker_stop();
}

void server_queue::cleanup_pending_task(int id_target) {
    // no need lock because this is called exclusively by post()
    auto rm_func = [id_target](const server_task & task) {
        return task.id == id_target;
    };
    queue_tasks.erase(
        std::remove_if(queue_tasks.begin(),           queue_tasks.end(),           rm_func),
        queue_tasks.end());
    queue_tasks_deferred.erase(
        std::remove_if(queue_tasks_deferred.begin(),  queue_tasks_deferred.end(),  rm_func),
        queue_tasks_deferred.end());
    // a task declined while yielding is not in queue_tasks yet, but it can still be cancelled
    queue_tasks_unhandled.erase(
        std::remove_if(queue_tasks_unhandled.begin(), queue_tasks_unhandled.end(), rm_func),
        queue_tasks_unhandled.end());
}

//
// server_response
//

void server_response::add_waiting_task_id(int id_task) {
    RES_DBG("add task %d to waiting list. current waiting = %d (before add)\n", id_task, (int) waiting_task_ids.size());

    std::unique_lock<std::mutex> lock(mutex_results);
    waiting_task_ids.insert(id_task);
}

void server_response::add_waiting_task_ids(const std::unordered_set<int> & id_tasks) {
    std::unique_lock<std::mutex> lock(mutex_results);

    for (const auto & id_task : id_tasks) {
        RES_DBG("add task %d to waiting list. current waiting = %d (before add)\n", id_task, (int) waiting_task_ids.size());
        waiting_task_ids.insert(id_task);
    }
}

void server_response::remove_waiting_task_id(int id_task) {
    RES_DBG("remove task %d from waiting list. current waiting = %d (before remove)\n", id_task, (int) waiting_task_ids.size());

    std::unique_lock<std::mutex> lock(mutex_results);
    waiting_task_ids.erase(id_task);
    // make sure to clean up all pending results
    queue_results.erase(
        std::remove_if(queue_results.begin(), queue_results.end(), [id_task](const server_task_result_ptr & res) {
            return res->id == id_task;
        }),
        queue_results.end());
}

void server_response::remove_waiting_task_ids(const std::unordered_set<int> & id_tasks) {
    std::unique_lock<std::mutex> lock(mutex_results);

    for (const auto & id_task : id_tasks) {
        RES_DBG("remove task %d from waiting list. current waiting = %d (before remove)\n", id_task, (int) waiting_task_ids.size());
        waiting_task_ids.erase(id_task);
    }
}

server_task_result_ptr server_response::recv(const std::unordered_set<int> & id_tasks) {
    while (true) {
        std::unique_lock<std::mutex> lock(mutex_results);
        condition_results.wait(lock, [&]{
            if (!running) {
                RES_DBG("%s : queue result stop\n", "recv");
                std::terminate(); // we cannot return here since the caller is HTTP code
            }
            return !queue_results.empty();
        });

        for (size_t i = 0; i < queue_results.size(); i++) {
            if (id_tasks.find(queue_results[i]->id) != id_tasks.end()) {
                server_task_result_ptr res = std::move(queue_results[i]);
                queue_results.erase(queue_results.begin() + i);
                return res;
            }
        }
    }

    // should never reach here
}

server_task_result_ptr server_response::recv_with_timeout(const std::unordered_set<int> & id_tasks, int timeout) {
    while (true) {
        std::unique_lock<std::mutex> lock(mutex_results);

        for (int i = 0; i < (int) queue_results.size(); i++) {
            if (id_tasks.find(queue_results[i]->id) != id_tasks.end()) {
                server_task_result_ptr res = std::move(queue_results[i]);
                queue_results.erase(queue_results.begin() + i);
                return res;
            }
        }

        std::cv_status cr_res = condition_results.wait_for(lock, std::chrono::seconds(timeout));
        if (!running) {
            RES_DBG("%s : queue result stop\n", __func__);
            std::terminate(); // we cannot return here since the caller is HTTP code
        }
        if (cr_res == std::cv_status::timeout) {
            return nullptr;
        }
    }

    // should never reach here
}

server_task_result_ptr server_response::recv(int id_task) {
    std::unordered_set<int> id_tasks = {id_task};
    return recv(id_tasks);
}

void server_response::send(server_task_result_ptr && result) {
    RES_DBG("sending result for task id = %d\n", result->id);

    std::unique_lock<std::mutex> lock(mutex_results);
    for (const auto & id_task : waiting_task_ids) {
        if (result->id == id_task) {
            RES_DBG("task id = %d pushed to result queue\n", result->id);

            queue_results.emplace_back(std::move(result));
            if (diagnostics_enabled_.load(std::memory_order_acquire) &&
                (queue_results.back()->is_stop() ||
                 queue_results.back()->is_error())) {
                RES_INF(
                    "CACHE_QUEUE event=result_terminal task=%d matched=1 waiting=%zu "
                    "results=%zu\n",
                    id_task, waiting_task_ids.size(), queue_results.size());
            }
            condition_results.notify_all();
            return;
        }
    }
    if (diagnostics_enabled_.load(std::memory_order_acquire)) {
        RES_INF(
            "CACHE_QUEUE event=result task=%d matched=0 waiting=%zu "
            "results=%zu\n",
            result ? result->id : -1, waiting_task_ids.size(),
            queue_results.size());
    }
}

void server_response::broadcast(server_task_result_ptr && result) {
    std::unique_lock<std::mutex> lock(mutex_results);
    for (const auto & id_task : waiting_task_ids) {
        RES_DBG("task id = %d pushed to result queue\n", id_task);
        server_task_result_ptr res_copy(result->clone());
        res_copy->id = id_task; // override id with target task id
        queue_results.emplace_back(std::move(res_copy));
    }
    condition_results.notify_all();
}

void server_response::terminate() {
    running = false;
    condition_results.notify_all();
}

//
// server_response_reader
//

void server_response_reader::post_task(server_task && task, bool front) {
    GGML_ASSERT(id_tasks.empty() && "post_task() can only be called once per reader");
    GGML_ASSERT(!task.is_parent() && "not supported, use post_tasks() instead");
    task.index = 0;
    id_tasks.insert(task.id);
    states.push_back(task.create_state());
    queue_results.add_waiting_task_id(task.id);
    queue_tasks.post(std::move(task), front);
}

void server_response_reader::post_tasks(std::vector<server_task> && tasks, bool front) {
    GGML_ASSERT(id_tasks.empty() && "post_tasks() can only be called once per reader");
    id_tasks = server_task::get_list_id(tasks);
    states.reserve(tasks.size());
    size_t index = 0;
    for (auto & task : tasks) {
        task.index = index++;
        states.push_back(task.create_state());
        // for child tasks
        for (auto & child_task : task.child_tasks) {
            child_task.index = index++;
            states.push_back(child_task.create_state());
        }
    }
    GGML_ASSERT(states.size() == id_tasks.size());
    queue_results.add_waiting_task_ids(id_tasks);
    queue_tasks.post(std::move(tasks), front);
}

bool server_response_reader::has_next() const {
    return !cancelled && received_count < id_tasks.size();
}

// return nullptr if should_stop() is true before receiving a result
// note: if one error is received, it will stop further processing and return error result
server_task_result_ptr server_response_reader::next(const std::function<bool()> & should_stop) {
    while (true) {
        server_task_result_ptr result = queue_results.recv_with_timeout(id_tasks, polling_interval_seconds);
        if (result == nullptr) {
            if (queue_tasks.diagnostics_enabled()) {
                diagnostic_wait_seconds += uint64_t(std::max(
                    polling_interval_seconds, 0));
                if (diagnostic_wait_seconds == 10 ||
                    (diagnostic_wait_seconds > 10 &&
                     diagnostic_wait_seconds % 30 == 0)) {
                    const auto snapshot = queue_tasks.diagnostics(id_tasks);
                    const int first_task = id_tasks.empty()
                        ? -1 : *id_tasks.begin();
                    RES_INF(
                        "CACHE_QUEUE event=response_wait task=%d count=%zu "
                        "wait_s=%" PRIu64 " queued=%zu deferred=%zu "
                        "matching_queued=%zu matching_deferred=%zu "
                        "posts=%" PRIu64 " dequeues=%" PRIu64
                        " running=%d sleeping=%d maintenance=%d\n",
                        first_task, id_tasks.size(), diagnostic_wait_seconds,
                        snapshot.queued, snapshot.deferred,
                        snapshot.matching_queued,
                        snapshot.matching_deferred, snapshot.posts,
                        snapshot.dequeues, int(snapshot.running),
                        int(snapshot.sleeping),
                        int(snapshot.maintenance_requested));
                }
            }
            // timeout, check stop condition
            if (should_stop()) {
                return nullptr;
            }
        } else {
            diagnostic_wait_seconds = 0;
            if (result->is_error()) {
                stop(); // cancel remaining tasks
                SRV_DBG("%s", "received error result, stopping further processing\n");
                return result;
            }
            if (!states.empty()) {
                // update the generation state if needed
                const size_t idx = result->index;
                GGML_ASSERT(idx < states.size());
                result->update(states[idx]);
            }
            if (result->is_stop()) {
                received_count++;
            }
            return result;
        }
    }

    // should not reach here
}

server_response_reader::batch_response server_response_reader::wait_for_all(const std::function<bool()> & should_stop) {
    batch_response batch_res;
    batch_res.results.clear();
    batch_res.results.resize(id_tasks.size());
    while (has_next()) {
        auto res = next(should_stop);
        if (res == nullptr) {
            batch_res.is_terminated = true;
            return batch_res;
        }
        if (res->is_error()) {
            batch_res.error = std::move(res);
            return batch_res;
        }
        const size_t idx = res->index;
        GGML_ASSERT(idx < batch_res.results.size() && "index out of range");
        GGML_ASSERT(batch_res.results[idx] == nullptr && "duplicate result received");
        batch_res.results[idx] = std::move(res);
    }
    return batch_res;
}

void server_response_reader::stop() {
    queue_results.remove_waiting_task_ids(id_tasks);
    if (has_next() && !cancelled) {
        // if tasks is not finished yet, cancel them
        cancelled = true;
        std::vector<server_task> cancel_tasks;
        cancel_tasks.reserve(id_tasks.size());
        for (const auto & id_task : id_tasks) {
            SRV_WRN("cancel task, id_task = %d\n", id_task);
            server_task task(SERVER_TASK_TYPE_CANCEL);
            task.id_target = id_task;
            queue_results.remove_waiting_task_id(id_task);
            cancel_tasks.push_back(std::move(task));
        }
        // push to beginning of the queue, so it has highest priority
        queue_tasks.post(std::move(cancel_tasks), true);
    } else {
        SRV_DBG("%s", "all tasks already finished, no need to cancel\n");
    }
}
