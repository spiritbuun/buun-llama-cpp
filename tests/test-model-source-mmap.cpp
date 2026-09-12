#include "llama-model-source.h"
#include "llama-model.h"
#include "llama-mmap.h"
#include "llama.h"
#include "llama-cpp.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "gguf.h"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#if !defined(_WIN32)
#include <sys/stat.h>
#endif

static void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

struct weight {
    std::array<int64_t, 4> ne;
    std::vector<float> data;
    size_t offset;
};

class source final : public llama_model_tensor_source {
public:
    std::string path;
    std::map<std::string, weight> weights;
    mutable size_t bound = 0, loaded = 0, mapped = 0, prepared = 0;
    mutable size_t prepared_bytes = 0;
    bool prepare = false;
    bool refuse_preparation = false;

    explicit source(const std::string & path) : path(path) {
        std::ofstream file(path, std::ios::binary);
        const uint64_t header = 0;
        file.write(reinterpret_cast<const char *>(&header), sizeof(header));
        const auto add = [&](const std::string & name, int64_t cols, int64_t rows) {
            weight w {{cols, rows, 1, 1}, std::vector<float>(cols * rows), size_t(file.tellp())};
            for (size_t i = 0; i < w.data.size(); ++i) w.data[i] = float(int(i % 17) - 8) / 16;
            file.write(reinterpret_cast<const char *>(w.data.data()), w.data.size() * sizeof(float));
            weights.emplace(name, std::move(w));
        };
        add("token_embd.weight", 32, 32); // deliberately not aligned like a GGUF header
        add("output_norm.weight", 32, 1);
        add("blk.0.attn_norm.weight", 32, 1);
        add("blk.0.ffn_norm.weight", 32, 1);
        for (const char * name : {"attn_q", "attn_k", "attn_v", "attn_output"})
            add(std::string("blk.0.") + name + ".weight", 32, 32);
        add("blk.0.ffn_gate.weight", 32, 64);
        add("blk.0.ffn_up.weight", 32, 64);
        add("blk.0.ffn_down.weight", 64, 32);
        require(bool(file), "writing test weights failed");
    }

    bool describe(const std::string & name, ggml_type & type, std::array<int64_t, 4> & ne) const override {
        const auto it = weights.find(name);
        if (it == weights.end()) return false;
        type = GGML_TYPE_F32;
        ne = it->second.ne;
        return true;
    }
    size_t tensor_capacity_hint() const override { return weights.size() + 1; }
    void bind(const std::string &) const override { ++bound; }
    std::optional<llama_model_tensor_file_region> file_region(const ggml_tensor * tensor) const override {
        // Keep norm tensors on the ordinary allocated path in the SAME CPU buft.
        if (tensor->ne[1] == 1 || prepare) return std::nullopt;
        return llama_model_tensor_file_region {path, weights.at(tensor->name).offset};
    }
    bool can_prepare_file(const ggml_tensor * tensor) const override {
        return prepare && tensor->ne[1] > 1;
    }
    std::unique_ptr<llama_file> prepare_file(const ggml_tensor * tensor, const std::function<void()> & poll) const override {
        poll();
        ++prepared;
        if (refuse_preparation) throw std::runtime_error("preparation directory is unavailable");
        auto file = llama_file::create_temp(std::filesystem::path(path).parent_path().string());
        const auto & data = weights.at(tensor->name).data;
        const size_t half = data.size() / 2;
        file->write_raw(data.data(), half * sizeof(float));
        prepared_bytes += half * sizeof(float);
        poll();
        file->write_raw(data.data() + half, (data.size() - half) * sizeof(float));
        file->finish_write();
        return file;
    }
    void load(ggml_tensor * tensor, bool is_mapped) const override {
        ++loaded;
        const auto & data = weights.at(tensor->name).data;
        if (is_mapped) {
            ++mapped;
            require(std::memcmp(tensor->data, data.data(), data.size() * sizeof(float)) == 0,
                    "mapped source bytes changed");
        } else {
            ggml_backend_tensor_set(tensor, data.data(), 0, data.size() * sizeof(float));
        }
    }
    void validate_complete() const override { require(loaded == bound, "incomplete source"); }
};

// Exercise the ordinary allocation fallback without adding a production flag.
struct without_host_pointer_buffers {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    static inline void (*get_props)(ggml_backend_dev_t, ggml_backend_dev_props *);
    without_host_pointer_buffers() {
        get_props = dev->iface.get_props;
        dev->iface.get_props = [](ggml_backend_dev_t device, ggml_backend_dev_props * props) {
            get_props(device, props);
            props->caps.buffer_from_host_ptr = false;
        };
    }
    ~without_host_pointer_buffers() { dev->iface.get_props = get_props; }
};

static std::vector<float> logits(llama_model * model) {
    auto params = llama_context_default_params();
    params.n_ctx = 128;
    params.n_batch = params.n_ubatch = 32;
    params.n_threads = params.n_threads_batch = 2;
    params.type_k = params.type_v = GGML_TYPE_F16;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context_ptr ctx(llama_init_from_model(model, params));
    require(bool(ctx), "creating test context failed");
    llama_token tokens[] = {2, 3, 4};
    require(llama_decode(ctx.get(), llama_batch_get_one(tokens, 3)) == 0, "test decode failed");
    const float * values = llama_get_logits_ith(ctx.get(), -1);
    require(values != nullptr, "test logits missing");
    return {values, values + 32};
}

int main() try {
    ggml_backend_load_all();
    if (!llama_mmap::SUPPORTED) return 77;
    const auto path = std::filesystem::temp_directory_path() /
        ("llama-source-mmap-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct cleanup {
        std::filesystem::path path;
        ~cleanup() { std::error_code ec; std::filesystem::remove(path, ec); }
    } cleanup {path}, cleanup_gguf {path.string() + ".gguf"};

    if (llama_file::TEMP_SUPPORTED) {
        auto file = llama_file::create_temp(path.parent_path().string());
#if !defined(_WIN32)
        struct stat statbuf {};
        require(fstat(file->file_id(), &statbuf) == 0 && statbuf.st_nlink == 0,
                "temporary weights must be unlinked immediately");
#endif
        const uint32_t data[] = {7, 11, 17, 23};
        file->write_raw(data, sizeof(data) / 2);
        file->sync_write();
        require(file->tell() == sizeof(data) / 2, "writeback moved file position");
        file->write_raw(data + 2, sizeof(data) / 2);
        file->finish_write();
        require(file->size() == sizeof(data), "prepared file size was not refreshed");
        llama_mmap mapping(file.get(), 0);
        file.reset();
        require(std::memcmp(mapping.addr(), data, sizeof(data)) == 0, "mapping lost its backing after file close");
        bool rejected = false;
        try { (void) llama_file::create_temp((path / "missing-directory").string()); }
        catch (const std::runtime_error &) { rejected = true; }
        require(rejected, "invalid backing directory accepted");
#if defined(__linux__)
        // Exercise the buffered write failure without filling any real disk.
        rejected = false;
        try {
            llama_file full("/dev/full", "wb");
            full.write_raw(data, sizeof(data));
            full.finish_write();
        } catch (const std::runtime_error &) { rejected = true; }
        require(rejected, "write failure was not reported");
#endif
    }

    gguf_context_ptr meta(gguf_init_empty());
    gguf_set_val_str(meta.get(), "general.architecture", "llama");
    gguf_set_val_str(meta.get(), "tokenizer.ggml.model", "none");
    for (const auto & kv : std::map<std::string, uint32_t> {
        {"block_count", 1}, {"context_length", 128}, {"embedding_length", 32},
        {"feed_forward_length", 64}, {"attention.head_count", 1},
        {"attention.head_count_kv", 1}, {"vocab_size", 32}}) {
        gguf_set_val_u32(meta.get(), ("llama." + kv.first).c_str(), kv.second);
    }
    gguf_set_val_f32(meta.get(), "llama.attention.layer_norm_rms_epsilon", 1e-5f);

    // Same canonical weights through the existing GGUF path, including tied output.
    {
        source input(path.string());
        gguf_context_ptr control(gguf_init_empty());
        gguf_set_kv(control.get(), meta.get());
        ggml_context_ptr ctx(ggml_init({input.weights.size() * ggml_tensor_overhead(), nullptr, true}));
        for (auto & [name, w] : input.weights) {
            auto * tensor = ggml_new_tensor(ctx.get(), GGML_TYPE_F32, 4, w.ne.data());
            ggml_set_name(tensor, name.c_str());
            tensor->data = w.data.data();
            gguf_add_tensor(control.get(), tensor);
        }
        require(gguf_write_to_file(control.get(), cleanup_gguf.path.string().c_str(), false), "writing GGUF control failed");
    }

    auto control_params = llama_model_default_params();
    control_params.n_gpu_layers = 0;
    control_params.load_mode = LLAMA_LOAD_MODE_MMAP;
    control_params.mmap_prefetch = LLAMA_MMAP_PREFETCH_MODE_OFF;
    llama_model_ptr control(llama_model_load_from_file(cleanup_gguf.path.string().c_str(), control_params));
    require(bool(control), "GGUF control load failed");
    const auto reference = logits(control.get());

    {
        without_host_pointer_buffers fallback;
        source input(path.string());
        input.prepare = input.refuse_preparation = true;
        llama_model_ptr model(llama_model_init_from_source(meta.get(), &input, control_params));
        require(bool(model), "capability-off fallback attempted unavailable preparation");
        require(input.prepared == 0 && input.mapped == 0, "capability-off fallback prepared backing");
        require(input.loaded == input.bound, "capability-off fallback lost source accounting");
        require(logits(model.get()) == reference, "capability-off fallback changed logits");
    }

    if (llama_file::TEMP_SUPPORTED) {
        source input(path.string());
        input.prepare = true;
        auto params = control_params;
        params.progress_callback = [](float, void *) { return false; };
        llama_model_ptr cancelled(llama_model_init_from_source(meta.get(), &input, params));
        require(!cancelled && input.prepared == 0, "preparation ignored cancellation");
        params.progress_callback_user_data = &input;
        params.progress_callback = [](float, void * context) {
            return static_cast<source *>(context)->prepared_bytes == 0;
        };
        cancelled.reset(llama_model_init_from_source(meta.get(), &input, params));
        require(!cancelled && input.prepared == 1 && input.prepared_bytes > 0 && input.loaded == 0,
                "partial preparation ignored cancellation or exposed incomplete weights");
    }

    for (llama_load_mode mode : {LLAMA_LOAD_MODE_MMAP, LLAMA_LOAD_MODE_NONE}) {
        for (bool no_alloc : {false, true}) {
          for (bool prepare : {false, true}) {
            if (prepare && !llama_file::TEMP_SUPPORTED) continue;
            llama_model_ptr model;
            std::map<std::string, weight> expected;
            {
                source input(path.string());
                input.prepare = prepare;
                expected = input.weights;
                auto params = llama_model_default_params();
                params.n_gpu_layers = 0;
                params.load_mode = mode;
                params.mmap_prefetch = LLAMA_MMAP_PREFETCH_MODE_OFF;
                params.no_alloc = no_alloc;
                model.reset(llama_model_init_from_source(meta.get(), &input, params));
                require(bool(model), "source model load failed");
                require(no_alloc ? input.loaded == 0 : input.loaded == input.bound, "wrong load count");
                require((input.mapped != 0) == (mode == LLAMA_LOAD_MODE_MMAP && !no_alloc), "wrong mapping path");
                require((input.prepared != 0) == (prepare && mode == LLAMA_LOAD_MODE_MMAP && !no_alloc),
                        "unexpected preparation during fit or mmap off");
            } // mapped bytes must remain valid after the source is destroyed
            if (!no_alloc) {
                for (const auto & [name, w] : expected) {
                    const auto * tensor = model->get_tensor(name.c_str());
                    require(tensor != nullptr, "missing runtime tensor");
                    std::vector<float> actual(w.data.size());
                    ggml_backend_tensor_get(tensor, actual.data(), 0, actual.size() * sizeof(float));
                    require(actual == w.data, "runtime weights changed after source destruction");
                }
                require(logits(model.get()) == reference, "source logits differ from GGUF control");
            }
          }
        }
    }
    std::cout << "PASS: shared mmap path, mixed allocated weights, tied output, source lifetime, no_alloc, mmap off, exact GGUF logits\n";
    return 0;
} catch (const std::exception & e) {
    std::cerr << e.what() << '\n';
    return 1;
}
