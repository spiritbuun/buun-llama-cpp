#pragma once

#include "server-task.h"

#include <condition_variable>
#include <atomic>
#include <cstdint>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <unordered_set>

struct server_queue_idle_capture_state;

// struct for managing server tasks
// in most cases, use server_response_reader to post new tasks and retrieve results
struct server_queue {
public:
    struct diagnostic_snapshot {
        bool running = false;
        bool sleeping = false;
        bool maintenance_requested = false;
        size_t queued = 0;
        size_t deferred = 0;
        size_t matching_queued = 0;
        size_t matching_deferred = 0;
        uint64_t posts = 0;
        uint64_t dequeues = 0;
    };

    // Move-only cancellation session for one synchronous idle capture. Task
    // arrivals cancel it; they never wait for or reject normal queue work.
    class idle_capture_session {
    public:
        idle_capture_session() noexcept = default;
        idle_capture_session(idle_capture_session && other) noexcept;
        idle_capture_session & operator=(
            idle_capture_session && other) noexcept;
        ~idle_capture_session();

        idle_capture_session(const idle_capture_session &) = delete;
        idle_capture_session & operator=(
            const idle_capture_session &) = delete;

        explicit operator bool() const noexcept;
        bool continue_capture() const noexcept;
        void cancel() noexcept;

    private:
        std::shared_ptr<server_queue_idle_capture_state> state_;
        uint64_t generation_ = 0;
        idle_capture_session(
            std::shared_ptr<server_queue_idle_capture_state> state,
            uint64_t generation) noexcept;
        void reset() noexcept;
        friend struct server_queue;
    };

private:
    int id = 0;
    bool running  = false;
    bool sleeping = false;
    bool req_stop_sleeping = false;
    int64_t time_last_task = 0;

    // queues
    std::deque<server_task> queue_tasks;
    std::deque<server_task> queue_tasks_deferred;
    // tasks declined while yielding, put back in queue_tasks once the yield is done
    // note: kept as a member so that cleanup_pending_task() can also reach them
    std::deque<server_task> queue_tasks_unhandled;

    std::mutex mutex_tasks;
    std::condition_variable condition_tasks;
    std::shared_ptr<server_queue_idle_capture_state> idle_capture_state;
    uint64_t idle_capture_generation = 0;
    bool idle_capture_stopped = false;
    std::atomic<bool> idle_maintenance_requested { false };
    std::atomic<bool> diagnostics_enabled_ { false };
    uint64_t diagnostic_posts_ = 0;
    uint64_t diagnostic_dequeues_ = 0;

    // used by yield_to_queue, all fields are guarded by mutex_tasks
    struct worker_t {
        std::thread             thread;
        std::condition_variable cv;        // the worker sleeps on this until a yield starts
        std::exception_ptr      exception; // exception thrown while processing tasks, if any
        bool stop     = false;
        bool busy     = false; // set by yield_to_queue(), cleared by the worker once it is done processing tasks
        bool yielding = false; // work() is still running on the start_loop() thread
    };
    worker_t worker;

    // callback functions
    std::function<bool(server_task &&, bool)> callback_new_task;
    std::function<void(void)>                 callback_update_slots;
    std::vector<std::function<void(bool)>>    callback_sleeping_state;
    std::function<void(void)>                 callback_idle;

public:
    ~server_queue();

    // Add a new task to the end of the queue
    int post(server_task && task, bool front = false);

    // multi-task version of post()
    int post(std::vector<server_task> && tasks, bool front = false);

    // Add a new task, but defer until one slot is available
    void defer(server_task && task);

    // Get the next id for creating a new task
    int get_new_id();

    // Call when the state of one slot is changed, it will move one task from deferred to main queue
    // prioritize tasks that use the specified slot (otherwise, pop the first deferred task)
    void pop_deferred_task(int id_slot);

    // true if any deferred task explicitly pins id_slot — a reclaim sweep must not clear the
    // cache that task deferred itself to come back for
    bool has_deferred_for_slot(int id_slot) {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        for (const auto & task : queue_tasks_deferred) {
            if (task.id_slot == id_slot) {
                return true;
            }
        }
        return false;
    }

    // if sleeping, request exiting sleep state and wait until it is done
    // returns immediately if not sleeping
    void wait_until_no_sleep();

    bool is_sleeping() {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        return sleeping;
    }

    // end the start_loop routine
    void terminate();

    /**
     * Main loop consists of these steps:
     * - Wait until a new task arrives
     * - Process the task (i.e. maybe copy data into slot)
     * - Check if multitask is finished
     * - Update all slots
     *
     * Sleeping procedure (disabled if idle_sleep_ms < 0):
     * - If there is no task after idle_sleep_ms, enter sleeping state
     *   note: metrics tasks are processed as usual, but do not reset the idle timer
     * - Call callback_sleeping_state(true)
     * - Wait until req_stop_sleeping is set to true
     * - Call callback_sleeping_state(false)
     * - Exit sleeping state
     */
    void start_loop(int64_t idle_sleep_ms = -1);

    // while waiting for work() to finish, run process_new_tasks on the worker thread
    // returns once work() is done (may throw exceptions)
    // must be called from start_loop() thread (ideally inside callback_update_slots)
    // use case: return metrics while encode/decode is running
    // ref: https://github.com/ggml-org/llama.cpp/pull/27041
    //
    // tasks declined by callback_new_task are put back in the queue once this returns
    void yield_to_queue(std::function<void()> && work);

    // Variant for work that must publish failure at a later transaction
    // boundary. It completes queue cleanup and returns the selected exception
    // instead of throwing it. A delayed work exception takes precedence over a
    // concurrent queue-callback exception, matching yield_to_queue().
    std::exception_ptr yield_to_queue_capture_exception(
        std::function<void()> && work,
        const std::exception_ptr & delayed_work_exception) noexcept;

    // for metrics
    size_t queue_tasks_deferred_size() {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        return queue_tasks_deferred.size();
    }

    bool has_pending_tasks() {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        return !queue_tasks.empty() || !queue_tasks_deferred.empty() ||
            !queue_tasks_unhandled.empty();
    }

    // Acquires only while both queues are empty and no prior session is live.
    // Allocation occurs only on the first enabled automatic-capture attempt.
    idle_capture_session try_begin_idle_capture() noexcept;

    // Scheduler-only prompt-boundary variant. update_slots() keeps one
    // synthetic NEXT_RESPONSE wake queued while it is evaluating a live
    // request; that wake is not external work and may coexist with the
    // synchronous SWA frontier capture. Any real/deferred task still wins.
    idle_capture_session try_begin_prompt_boundary_capture() noexcept;

    // Worker-safe scheduler wake used only to run the registered idle
    // maintenance callback. It never fabricates a task or bypasses task
    // priority; a concurrently queued task wins and cancels the capture.
    void request_idle_maintenance() noexcept;

    // Cache-debug-only lifecycle tracing for a response waiter that has not
    // heard from the scheduler. The snapshot never changes queue state.
    void set_diagnostics(bool enabled) noexcept {
        diagnostics_enabled_.store(enabled, std::memory_order_release);
    }
    bool diagnostics_enabled() const noexcept {
        return diagnostics_enabled_.load(std::memory_order_acquire);
    }
    diagnostic_snapshot diagnostics(
        const std::unordered_set<int> & task_ids) noexcept;

    //
    // Functions below are not thread-safe, must only be used before start_loop() is called
    //

    // Register function to process a new task
    // the second argument tells whether the queue is currently yielding (see yield_to_queue)
    // only then may the callback return false to decline the task, and it must leave it
    // untouched, so that it can be put back in the queue later
    // note: while yielding, the callback runs on worker thread, not main thread
    void on_new_task(std::function<bool(server_task &&, bool)> callback) {
        callback_new_task = std::move(callback);
    }

    // Register the function to be called when all slots data is ready to be processed
    void on_update_slots(std::function<void(void)> callback) {
        callback_update_slots = std::move(callback);
    }

    // Register the function to be called on the loop's wait timeout while the queue is
    // empty (~1s cadence, same thread as callback_update_slots; not called while sleeping)
    void on_idle(std::function<void(void)> callback) {
        callback_idle = std::move(callback);
    }

    // Register callback for sleeping state change; multiple callbacks are allowed
    // for example: register order cb0, cb1, cb2
    // entering sleep: queue.sleeping = true --> cb0(true) --> cb1(true) --> cb2(true)
    // leaving sleep: cb2(false) --> cb1(false) --> cb0(false) --> queue.sleeping = false
    // Callbacks run in the documented order without mutex_tasks held.
    void on_sleeping_state(std::function<void(bool)> callback) {
        callback_sleeping_state.push_back(std::move(callback));
    }

private:
    void cancel_idle_capture_locked() noexcept;
    void cleanup_pending_task(int id_target);

    // process all pending tasks in the queue
    // returns true if the queue is terminated, false if there is no more task to process
    // while yielding, declined tasks are moved to queue_tasks_unhandled
    bool process_new_tasks(bool is_yielding);

    // for worker_t
    void worker_loop();
    void worker_stop();
};

// struct for managing server responses
// in most cases, use server_response_reader to retrieve results
struct server_response {
private:
    bool running = true;

    // for keeping track of all tasks waiting for the result
    std::unordered_set<int> waiting_task_ids;

    // the main result queue (using ptr for polymorphism)
    std::vector<server_task_result_ptr> queue_results;

    std::mutex mutex_results;
    std::condition_variable condition_results;
    std::atomic<bool> diagnostics_enabled_ { false };

public:
    void set_diagnostics(bool enabled) noexcept {
        diagnostics_enabled_.store(enabled, std::memory_order_release);
    }

    // add the id_task to the list of tasks waiting for response
    void add_waiting_task_id(int id_task);

    void add_waiting_task_ids(const std::unordered_set<int> & id_tasks);

    // when the request is finished, we can remove task associated with it
    void remove_waiting_task_id(int id_task);

    // remove multiple tasks from waiting list
    void remove_waiting_task_ids(const std::unordered_set<int> & id_tasks);

    // This function blocks the thread until there is a response for one of the id_tasks
    server_task_result_ptr recv(const std::unordered_set<int> & id_tasks);

    // same as recv(), but have timeout in seconds
    // if timeout is reached, nullptr is returned
    server_task_result_ptr recv_with_timeout(const std::unordered_set<int> & id_tasks, int timeout);

    // single-task version of recv()
    server_task_result_ptr recv(int id_task);

    // Send a new result to a waiting id_task
    void send(server_task_result_ptr && result);

    // broadcast a new result to all waiting tasks
    // (used by router mode)
    void broadcast(server_task_result_ptr && result);

    // terminate the waiting loop
    void terminate();
};

// RAII wrapper to make working with server_queue and server_response easier
// it provides a generator-like API for server responses
// support pooling connection state and aggregating multiple results
struct server_response_reader {
    std::unordered_set<int> id_tasks;
    server_queue & queue_tasks;
    server_response & queue_results;
    size_t received_count = 0;
    bool cancelled = false;
    int polling_interval_seconds;
    uint64_t diagnostic_wait_seconds = 0;

    // tracking generation state and partial tool calls
    // only used by streaming completions
    std::vector<task_result_state> states;

    // should_stop function will be called each polling_interval_seconds
    server_response_reader(server_queue & queue_tasks, server_response & queue_results, int polling_interval_seconds)
        : queue_tasks(queue_tasks), queue_results(queue_results), polling_interval_seconds(polling_interval_seconds) {}
    ~server_response_reader() {
        stop();
    }

    int get_new_id() {
        return queue_tasks.get_new_id();
    }

    // if front = true, the task will be posted to the front of the queue (high priority)
    void post_task(server_task && task, bool front = false);
    void post_tasks(std::vector<server_task> && tasks, bool front = false);
    bool has_next() const;

    // return nullptr if should_stop() is true before receiving a result
    // note: if one error is received, it will stop further processing and return error result
    server_task_result_ptr next(const std::function<bool()> & should_stop);

    struct batch_response {
        bool is_terminated = false; // if true, indicates that processing was stopped before all results were received
        std::vector<server_task_result_ptr> results;
        server_task_result_ptr error; // nullptr if no error
    };
    // aggregate multiple results
    batch_response wait_for_all(const std::function<bool()> & should_stop);

    void stop();
};
