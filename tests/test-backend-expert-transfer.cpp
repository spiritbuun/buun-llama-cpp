#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

// A CPU executor with a distinct, non-host buffer type exercises the scheduler's
// host-to-device transfers without requiring a GPU or a particular weight codec.
struct transfer_device {
    ggml_backend_ptr backend{ggml_backend_cpu_init()};
    ggml_backend_device device = *backend->device;
    ggml_backend_buffer_type buft = *ggml_backend_cpu_buffer_type();
    std::vector<std::pair<size_t, size_t>> copies;

    transfer_device() {
        device.context = this;
        device.iface.get_name = [](ggml_backend_dev_t) { return "test-transfer"; };
        device.iface.get_type = [](ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; };
        device.iface.get_buffer_type = [](ggml_backend_dev_t d) { return &self(d).buft; };
        device.iface.supports_op = [](ggml_backend_dev_t, const ggml_tensor * t) {
            return t->op == GGML_OP_NONE || t->op == GGML_OP_MUL_MAT_ID || t->op == GGML_OP_SCALE;
        };
        device.iface.supports_buft = [](ggml_backend_dev_t d, ggml_backend_buffer_type_t b) {
            return b == &self(d).buft;
        };
        device.iface.offload_op = [](ggml_backend_dev_t, const ggml_tensor *) { return true; };
        buft.device = &device;
        buft.iface.get_name = [](ggml_backend_buffer_type_t) { return "test-transfer"; };
        buft.iface.is_host = [](ggml_backend_buffer_type_t) { return false; };
        buft.iface.alloc_buffer = [](ggml_backend_buffer_type_t b, size_t size) {
            auto buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
            if (buffer) {
                buffer->buft = b;
            }
            return buffer;
        };
        backend->device = &device;
        backend->iface.set_tensor_async = [](ggml_backend_t b, ggml_tensor * t, const void * data,
                                             size_t offset, size_t size) {
            if (std::strstr(t->name, "experts")) {
                self(b->device).copies.emplace_back(offset, size);
            }
            ggml_backend_tensor_set(t, data, offset, size);
        };
        ggml_backend_cpu_set_n_threads(backend.get(), 1);
    }

    static transfer_device & self(ggml_backend_dev_t d) {
        return *static_cast<transfer_device *>(d->context);
    }
};

int main() {
    transfer_device device;
    ggml_backend_ptr cpu(ggml_backend_cpu_init());
    ggml_context_ptr weights_ctx(ggml_init({ggml_tensor_overhead(), nullptr, true}));
    auto * w = ggml_new_tensor_3d(weights_ctx.get(), GGML_TYPE_F32, 128, 32, 8);
    ggml_set_name(w, "experts");
    ggml_backend_buffer_ptr weights(ggml_backend_alloc_ctx_tensors(weights_ctx.get(), cpu.get()));
    GGML_ASSERT(weights);
    ggml_backend_buffer_set_usage(weights.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<float> values(128 * 32 * 8);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = float(i / (128 * 32) + 1);
    }
    ggml_backend_tensor_set(w, values.data(), 0, values.size() * sizeof(float));
    ggml_backend_t backends[] = {device.backend.get(), cpu.get()};
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 2, 128, false, true));

    for (const std::vector<int32_t> selected :
         {std::vector<int32_t>{}, {0, 1, 1, 7}, {2, 3, 4, 5}, {7, 7, 7, 7}, {}}) {
        ggml_context_ptr ctx(ggml_init({ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(128, false),
                                       nullptr, true}));
        const int tokens = selected.empty() ? 0 : 1;
        auto * x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 128, 1, tokens);
        auto * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 4, tokens);
        auto * probe = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        ggml_set_input(x);
        ggml_set_input(ids);
        ggml_set_input(probe);
        auto * y = ggml_mul_mat_id(ctx.get(), w, x, ids);
        auto * tail = ggml_scale(ctx.get(), probe, 2.f);
        ggml_set_output(y);
        ggml_set_output(tail);
        auto * graph = ggml_new_graph_custom(ctx.get(), 128, false);
        ggml_build_forward_expand(graph, y);
        ggml_build_forward_expand(graph, tail);
        ggml_backend_sched_set_tensor_backend(sched.get(), y, device.backend.get());
        ggml_backend_sched_set_tensor_backend(sched.get(), tail, device.backend.get());
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph));
        GGML_ASSERT(ggml_backend_sched_get_tensor_backend(sched.get(), y) == device.backend.get());
        if (tokens) {
            std::vector<float> input(128, 1.f);
            ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
            ggml_backend_tensor_set(ids, selected.data(), 0, selected.size() * sizeof(int32_t));
        }
        const float probe_value = 3.f;
        ggml_backend_tensor_set(probe, &probe_value, 0, sizeof(probe_value));

        std::vector<std::pair<size_t, size_t>> expected;
        auto unique = selected;
        std::sort(unique.begin(), unique.end());
        unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
        for (size_t i = 0; i < unique.size();) {
            const int first = unique[i];
            int last = first;
            while (++i < unique.size() && unique[i] == last + 1) {
                last = unique[i];
            }
            expected.emplace_back(first * w->nb[2], (last - first + 1) * w->nb[2] + (last < 7 ? 512 : 0));
        }
        for (int repeat = 0; repeat < 2; ++repeat) {
            device.copies.clear();
            GGML_ASSERT(ggml_backend_sched_graph_compute(sched.get(), graph) == GGML_STATUS_SUCCESS);
            GGML_ASSERT(device.copies == expected);
            std::vector<float> output(32 * selected.size());
            if (!output.empty()) {
                ggml_backend_tensor_get(y, output.data(), 0, output.size() * sizeof(float));
                for (size_t i = 0; i < output.size(); ++i) {
                    GGML_ASSERT(output[i] == 128.f * (selected[i / 32] + 1));
                }
            }
            float probe_output = 0;
            ggml_backend_tensor_get(tail, &probe_output, 0, sizeof(probe_output));
            GGML_ASSERT(probe_output == 6.f);
        }
        printf("PASS: selected=%zu transfer_ranges=%zu (two executions)\n", selected.size(), expected.size());
        ggml_backend_sched_reset(sched.get());
    }
}
