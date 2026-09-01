#include "ggml-backend.h"
#include "gguf.h"
#include "llama-safetensors-names.h"
#include "llama-safetensors-quant.h"
#include "llama-safetensors-qwen3.h"
#include "llama-safetensors-qwen35.h"
#include "llama-safetensors-qwen4exp.h"
#include "llama-safetensors-tensor.h"
#include "llama-safetensors.h"
#include "llama.h"
#include "nlohmann/json.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

using json = nlohmann::json;

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void write_shard(const std::filesystem::path & path, const std::string & header, const std::vector<uint8_t> & data) {
    std::ofstream out(path, std::ios::binary);
    require(static_cast<bool>(out), "failed to create safetensors test shard");
    const uint64_t         n = header.size();
    std::array<uint8_t, 8> prefix{};
    for (size_t i = 0; i < prefix.size(); ++i) {
        prefix[i] = uint8_t(n >> (8 * i));
    }
    out.write(reinterpret_cast<const char *>(prefix.data()), prefix.size());
    out.write(header.data(), header.size());
    out.write(reinterpret_cast<const char *>(data.data()), data.size());
    require(static_cast<bool>(out), "failed to write safetensors test shard");
}

void write_text(const std::filesystem::path & path, const std::string & text) {
    std::ofstream out(path);
    require(static_cast<bool>(out), "failed to create safetensors test metadata");
    out << text;
    require(static_cast<bool>(out), "failed to write safetensors test metadata");
}

void write_bytes(const std::filesystem::path & path, const std::vector<uint8_t> & bytes) {
    std::ofstream out(path, std::ios::binary);
    require(static_cast<bool>(out), "failed to create binary test metadata");
    out.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    require(static_cast<bool>(out), "failed to write binary test metadata");
}

struct tensor_fixture {
    std::string           name;
    std::string           dtype;
    std::vector<uint64_t> shape;
    std::vector<uint8_t>  data;
};

void write_single_shard_model(
        const std::filesystem::path & path,
        const std::vector<tensor_fixture> & tensors,
        const json & metadata = json::object()) {
    std::filesystem::create_directories(path);
    json                 header = json::object();
    if (!metadata.empty()) {
        header["__metadata__"] = metadata;
    }
    std::vector<uint8_t> data;
    for (const tensor_fixture & tensor : tensors) {
        const size_t begin = data.size();
        data.insert(data.end(), tensor.data.begin(), tensor.data.end());
        header[tensor.name] = {
            { "dtype",        tensor.dtype           },
            { "shape",        tensor.shape           },
            { "data_offsets", { begin, data.size() } },
        };
    }
    write_shard(path / "model.safetensors", header.dump(), data);
}

template <typename Fn> void require_rejected(Fn && fn, const char * message) {
    bool rejected = false;
    try {
        fn();
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    require(rejected, message);
}

const char * fp8_channel_config =
    R"({"quantization_config":{"quant_method":"compressed-tensors","format":"mixed-precision","config_groups":{"fp8":{"format":"float-quantized","targets":["module"],"input_activations":{"actorder":null,"block_structure":null,"dynamic":true,"group_size":null,"num_bits":8,"scale_dtype":null,"strategy":"token","symmetric":true,"type":"float","zp_dtype":null},"weights":{"actorder":null,"block_structure":null,"dynamic":false,"group_size":null,"num_bits":8,"scale_dtype":null,"strategy":"channel","symmetric":true,"type":"float","zp_dtype":null}}}}})";

const char * fp8_static_tensor_config =
    R"({"quantization_config":{"quant_method":"compressed-tensors","format":"mixed-precision","config_groups":{"fp8":{"format":"float-quantized","targets":["module"],"input_activations":{"actorder":null,"block_structure":null,"dynamic":false,"group_size":null,"num_bits":8,"scale_dtype":null,"strategy":"tensor","symmetric":true,"type":"float","zp_dtype":null},"weights":{"actorder":null,"block_structure":null,"dynamic":false,"group_size":null,"num_bits":8,"scale_dtype":null,"strategy":"tensor","symmetric":true,"type":"float","zp_dtype":null}}}}})";

const char * fp8_block_config =
    R"({"quantization_config":{"quant_method":"fp8","fmt":"e4m3","activation_scheme":"dynamic","weight_block_size":[128,128]}})";

const char * fp8_block_inferred_format_config =
    R"({"quantization_config":{"quant_method":"fp8","activation_scheme":"dynamic","weight_block_size":[128,128]}})";

const char * fp8_block_embedding_config =
    R"({"quantization_config":{"quant_method":"fp8","activation_scheme":"dynamic","weight_block_size":[128,128],"modules_to_convert":["module.embedding"]}})";

const char * deepseek4_mixed_fp8_config =
    R"({"expert_dtype":"fp4","quantization_config":{"quant_method":"fp8","fmt":"e4m3","scale_fmt":"ue8m0","activation_scheme":"dynamic","weight_block_size":[128,128]}})";

const char * legacy_fp8_static_config =
    R"({"quantization_config":{"quant_method":"fp8","activation_scheme":"static"}})";

const char * modelopt_fp8_pb_wo_config =
    R"({"quantization_config":{"quant_method":"modelopt","quant_algo":"fp8_pb_wo","ignore":["lm_head"],"quantization":{"quant_algo":"fp8_pb_wo","kv_cache_quant_algo":null,"exclude_modules":["output"]}}})";

const char * modelopt_fp8_channel_config =
    R"({"quantization_config":{"quant_method":"modelopt","quant_algo":"FP8_PER_CHANNEL_PER_TOKEN","ignore":["lm_head"]}})";

const char * fbgemm_fp8_config =
    R"({"quantization_config":{"activation_scale_ub":1200.0,"modules_to_not_convert":["ignored"],"quant_method":"fbgemm_fp8"}})";

const char * minimax_mxfp8_config =
    R"({"quantization_config":{"quant_method":"mxfp8","ignored_layers":["lm_head"]}})";

const char * nvfp4_config =
    R"({"quantization_config":{"quant_method":"compressed-tensors","format":"mixed-precision","config_groups":{"nvfp4":{"format":"nvfp4-pack-quantized","targets":["module"],"input_activations":{"actorder":null,"block_structure":null,"dynamic":"local","group_size":16,"num_bits":4,"scale_dtype":"torch.float8_e4m3fn","strategy":"tensor_group","symmetric":true,"type":"float","zp_dtype":null},"weights":{"actorder":"static","block_structure":null,"dynamic":false,"group_size":16,"num_bits":4,"scale_dtype":"torch.float8_e4m3fn","strategy":"tensor_group","symmetric":true,"type":"float","zp_dtype":null}}}}})";

const char * modelopt_w4a16_nvfp4_config =
    R"({"quantization_config":{"quant_method":"modelopt","quant_algo":"W4A16_NVFP4","group_size":16,"ignore":["*ignored*"],"config_groups":{"group_0":{"weights":{"dynamic":false,"num_bits":4,"type":"float","group_size":16},"targets":["Linear"]}}}})";

const char * modelopt_mixed_config =
    R"({"quantization_config":{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","config_groups":{"fp8":{"weights":{"dynamic":false,"num_bits":8,"type":"float"},"input_activations":{"dynamic":false,"num_bits":8,"type":"float"},"targets":["fp8_module"]},"nvfp4":{"weights":{"dynamic":false,"num_bits":4,"type":"float","group_size":16},"input_activations":null,"targets":["nv_module"]}}}})";

const char * mxfp4_config =
    R"({"quantization_config":{"quant_method":"compressed-tensors","format":"mxfp4-pack-quantized","config_groups":{"mxfp4":{"format":"mxfp4-pack-quantized","targets":["module"],"input_activations":{"actorder":null,"block_structure":null,"dynamic":true,"group_size":32,"num_bits":4,"scale_dtype":"torch.uint8","strategy":"group","symmetric":true,"type":"float","zp_dtype":null},"weights":{"actorder":null,"block_structure":null,"dynamic":false,"group_size":32,"num_bits":4,"scale_dtype":"torch.uint8","strategy":"group","symmetric":true,"type":"float","zp_dtype":null}}}}})";

const char * mxfp8_config =
    R"({"quantization_config":{"quant_method":"compressed-tensors","format":"mxfp8-quantized","config_groups":{"mxfp8":{"format":"mxfp8-quantized","targets":["module"],"input_activations":{"actorder":null,"block_structure":null,"dynamic":true,"group_size":32,"num_bits":8,"scale_dtype":"torch.uint8","strategy":"group","symmetric":true,"type":"float","zp_dtype":null},"weights":{"actorder":null,"block_structure":null,"dynamic":false,"group_size":32,"num_bits":8,"scale_dtype":"torch.uint8","strategy":"group","symmetric":true,"type":"float","zp_dtype":null}}}}})";

const char * fp8_group_config =
    R"({"quantization_config":{"quant_method":"compressed-tensors","format":"float-quantized","config_groups":{"group_0":{"targets":["module"],"input_activations":{"actorder":null,"block_structure":null,"dynamic":true,"group_size":32,"num_bits":8,"scale_dtype":"torch.bfloat16","strategy":"group","symmetric":true,"type":"float","zp_dtype":null},"weights":{"actorder":null,"block_structure":null,"dynamic":false,"group_size":32,"num_bits":8,"scale_dtype":"torch.bfloat16","strategy":"group","symmetric":true,"type":"float","zp_dtype":null}}}}})";

const char * w4a8_fp8_config =
    R"({"quantization_config":{"quant_method":"compressed-tensors","format":"pack-quantized","config_groups":{"w4a8":{"format":"pack-quantized","targets":["module"],"input_activations":{"actorder":null,"block_structure":null,"dynamic":true,"group_size":null,"num_bits":8,"observer":null,"strategy":"token","symmetric":true,"type":"float"},"output_activations":null,"weights":{"actorder":null,"block_structure":null,"dynamic":false,"group_size":128,"num_bits":4,"observer":"minmax","strategy":"group","symmetric":true,"type":"int"}}}}})";

const char * w8a8_config =
    R"({"quantization_config":{"quant_method":"compressed-tensors","format":"int-quantized","config_groups":{"int8":{"targets":["module"],"input_activations":{"actorder":null,"block_structure":null,"dynamic":true,"group_size":null,"num_bits":8,"strategy":"token","symmetric":true,"type":"int"},"weights":{"actorder":"static","block_structure":null,"dynamic":false,"group_size":null,"num_bits":8,"strategy":"channel","symmetric":true,"type":"int"}}}}})";

const char * w8a8_static_config =
    R"({"quantization_config":{"quant_method":"compressed-tensors","format":"int-quantized","config_groups":{"int8":{"targets":["module"],"input_activations":{"actorder":null,"block_structure":null,"dynamic":false,"group_size":null,"num_bits":8,"strategy":"tensor","symmetric":true,"type":"int"},"weights":{"actorder":"static","block_structure":null,"dynamic":false,"group_size":null,"num_bits":8,"strategy":"channel","symmetric":true,"type":"int"}}}}})";

const char * w8a8_static_asym_config =
    R"({"quantization_config":{"quant_method":"compressed-tensors","format":"int-quantized","config_groups":{"int8":{"targets":["module"],"input_activations":{"actorder":null,"block_structure":null,"dynamic":false,"group_size":null,"num_bits":8,"strategy":"tensor","symmetric":false,"type":"int","zp_dtype":"torch.int8"},"weights":{"actorder":"static","block_structure":null,"dynamic":false,"group_size":null,"num_bits":8,"strategy":"channel","symmetric":true,"type":"int"}}}}})";

const char * w4a8_config =
    R"({"quantization_config":{"quant_method":"compressed-tensors","format":"int-quantized","config_groups":{"int4":{"targets":["module"],"input_activations":{"actorder":null,"block_structure":null,"dynamic":true,"group_size":null,"num_bits":8,"strategy":"token","symmetric":true,"type":"int"},"weights":{"actorder":"static","block_structure":null,"dynamic":false,"group_size":128,"num_bits":4,"strategy":"group","symmetric":true,"type":"int"}}}}})";

const char * awq_config =
    R"({"quantization_config":{"bits":4,"group_size":128,"modules_to_not_convert":null,"quant_method":"awq","version":"gemm","zero_point":true}})";

const char * awq_g64_config =
    R"({"quantization_config":{"w_bit":4,"q_group_size":64,"modules_to_not_convert":null,"quant_method":"awq","version":"GEMM","zero_point":true}})";

const char * quark_uint4_config =
    R"({"quantization_config":{"quant_method":"quark","global_quant_config":{"bias":null,"input_tensors":null,"output_tensors":null,"weight":{"dtype":"uint4","group_size":128,"is_dynamic":false,"is_scale_quant":false,"qscheme":"per_group","scale_type":"float","symmetric":false}},"layer_quant_config":{},"layer_type_quant_config":{},"exclude":["lm_head","*ignored*"],"export":{"pack_method":"reorder","weight_format":"real_quantized"}}})";

const char * quark_int4_config =
    R"({"quantization_config":{"quant_method":"quark","global_quant_config":{"bias":null,"input_tensors":null,"output_tensors":null,"weight":{"dtype":"int4","group_size":128,"is_dynamic":false,"is_scale_quant":false,"qscheme":"per_group","scale_type":"float","symmetric":true}},"layer_quant_config":{},"layer_type_quant_config":{},"exclude":[],"export":{"pack_method":"reorder","weight_format":"real_quantized"}}})";

const char * quark_int4_g4_config =
    R"({"quantization_config":{"quant_method":"quark","global_quant_config":{"bias":null,"input_tensors":null,"output_tensors":null,"weight":{"dtype":"int4","group_size":4,"is_dynamic":false,"is_scale_quant":false,"qscheme":"per_group","scale_type":"float","symmetric":true}},"layer_quant_config":{},"layer_type_quant_config":{},"exclude":[],"export":{"pack_method":"reorder","weight_format":"real_quantized"}}})";

const char * quark_fp8_config =
    R"({"quantization_config":{"quant_method":"quark","global_quant_config":{"bias":null,"input_tensors":{"dtype":"fp8_e4m3","is_dynamic":false,"qscheme":"per_tensor"},"output_tensors":null,"weight":{"dtype":"fp8_e4m3","is_dynamic":false,"qscheme":"per_tensor"}},"layer_quant_config":{},"layer_type_quant_config":{},"exclude":["lm_head"],"export":{"pack_method":"reorder","weight_format":"real_quantized"}}})";

const char * quark_int8_config =
    R"({"quantization_config":{"quant_method":"quark","global_quant_config":{"bias":null,"input_tensors":{"dtype":"int8","is_dynamic":true,"qscheme":"per_channel","symmetric":true},"output_tensors":null,"weight":{"dtype":"int8","is_dynamic":false,"qscheme":"per_channel","symmetric":true}},"layer_quant_config":{},"layer_type_quant_config":{},"exclude":["lm_head"],"export":{"pack_method":"reorder","weight_format":"real_quantized"}}})";

const char * quark_nvfp4_config =
    R"({"quantization_config":{"quant_method":"quark","global_quant_config":{"bias":null,"input_tensors":[{"ch_axis":-1,"dtype":"fp4","group_size":16,"is_dynamic":true,"is_scale_quant":false,"qscheme":"per_group","scale_type":"float32"},{"dtype":"fp8_e4m3","is_dynamic":false,"is_scale_quant":true,"qscheme":"per_tensor","scale_type":"float32"}],"output_tensors":null,"weight":[{"ch_axis":-1,"dtype":"fp4","group_size":16,"is_dynamic":false,"is_scale_quant":false,"qscheme":"per_group","scale_type":"float32"},{"dtype":"fp8_e4m3","is_dynamic":false,"is_scale_quant":true,"qscheme":"per_tensor","scale_type":"float32"}]},"layer_quant_config":{},"layer_type_quant_config":{},"exclude":["ignored"],"export":{"pack_method":"reorder","weight_format":"real_quantized"}}})";

const char * quark_mxfp4_config =
    R"({"quantization_config":{"quant_method":"quark","global_quant_config":{"bias":null,"input_tensors":{"ch_axis":-1,"dtype":"fp4","group_size":32,"is_dynamic":true,"is_scale_quant":false,"qscheme":"per_group","scale_format":"e8m0","scale_type":"float"},"output_tensors":null,"weight":{"ch_axis":-1,"dtype":"fp4","group_size":32,"is_dynamic":false,"is_scale_quant":false,"qscheme":"per_group","scale_format":"e8m0","scale_type":"float"}},"layer_quant_config":{},"layer_type_quant_config":{},"exclude":["ignored"],"export":{"pack_method":"reorder","weight_format":"real_quantized"}}})";

const char * quark_fp8_ptpc_config =
    R"({"quantization_config":{"quant_method":"quark","global_quant_config":{"bias":null,"input_tensors":{"ch_axis":1,"dtype":"fp8_e4m3","is_dynamic":true,"is_scale_quant":false,"qscheme":"per_channel"},"output_tensors":null,"weight":{"ch_axis":0,"dtype":"fp8_e4m3","is_dynamic":false,"is_scale_quant":false,"qscheme":"per_channel"}},"layer_quant_config":{},"layer_type_quant_config":{},"exclude":["ignored"],"export":{"pack_method":"reorder","weight_format":"real_quantized"}}})";

const char * gptq_config =
    R"({"quantization_config":{"bits":4,"checkpoint_format":"gptq","desc_act":false,"group_size":128,"pack_dtype":"int32","quant_method":"gptq","sym":true}})";

const char * gptq_g32_config =
    R"({"quantization_config":{"bits":4,"desc_act":false,"group_size":32,"quant_method":"gptq","sym":true}})";

const char * gptq_per_channel_config =
    R"({"quantization_config":{"bits":4,"desc_act":true,"group_size":-1,"quant_method":"gptq","sym":true}})";

const char * gptq_act_order_config =
    R"({"quantization_config":{"bits":4,"checkpoint_format":"gptq","desc_act":true,"group_size":128,"pack_dtype":"int32","quant_method":"gptq","sym":true}})";

const char * gptq_int8_config =
    R"({"quantization_config":{"bits":8,"checkpoint_format":"gptq","desc_act":false,"group_size":128,"pack_dtype":"int32","quant_method":"gptq","sym":true}})";

const char * autoround_config =
    R"({"quantization_config":{"autoround_version":"0.12.3","bits":4,"data_type":"int","group_size":128,"iters":0,"packing_format":"auto_round:auto_gptq","quant_method":"auto-round","sym":true}})";

const char * bnb_nf4_config =
    R"({"quantization_config":{"_load_in_4bit":true,"_load_in_8bit":false,"bnb_4bit_compute_dtype":"bfloat16","bnb_4bit_quant_storage":"uint8","bnb_4bit_quant_type":"nf4","bnb_4bit_use_double_quant":true,"load_in_4bit":true,"load_in_8bit":false,"llm_int8_skip_modules":["lm_head"],"quant_method":"bitsandbytes"}})";

const char * bnb_int8_config =
    R"({"quantization_config":{"_load_in_4bit":false,"_load_in_8bit":true,"load_in_4bit":false,"load_in_8bit":true,"llm_int8_skip_modules":["lm_head"],"llm_int8_threshold":6.0,"quant_method":"bitsandbytes"}})";

const char * eetq_config =
    R"({"quantization_config":{"bits":8,"quant_method":"eetq","weights":"int8","zero_point":false}})";

const char * bnb_fp4_config =
    R"({"quantization_config":{"bnb_4bit_compute_dtype":"bfloat16","bnb_4bit_quant_storage":"uint8","bnb_4bit_quant_type":"fp4","bnb_4bit_use_double_quant":false,"load_in_4bit":true,"load_in_8bit":false,"quant_method":"bitsandbytes"}})";

const char * packed_int_config =
    R"({"quantization_config":{"quant_method":"compressed-tensors","format":"pack-quantized","config_groups":{"int4":{"format":"pack-quantized","targets":["int4"],"input_activations":null,"output_activations":null,"weights":{"actorder":"static","block_structure":null,"dynamic":false,"group_size":32,"num_bits":4,"scale_dtype":null,"strategy":"group","symmetric":false,"type":"int","zp_dtype":"torch.int8"}},"int8":{"format":"pack-quantized","targets":["int8"],"input_activations":null,"output_activations":null,"weights":{"actorder":"static","block_structure":null,"dynamic":false,"group_size":128,"num_bits":8,"scale_dtype":null,"strategy":"group","symmetric":true,"type":"int","zp_dtype":null}}}}})";

const char * packed_int4_symmetric_config =
    R"({"quantization_config":{"quant_method":"compressed-tensors","format":"pack-quantized","config_groups":{"int4":{"format":"pack-quantized","targets":["int4"],"input_activations":null,"output_activations":null,"weights":{"actorder":null,"block_structure":null,"dynamic":false,"group_size":128,"num_bits":4,"scale_dtype":null,"strategy":"group","symmetric":true,"type":"int","zp_dtype":null}}}}})";

std::vector<uint8_t> f32_bytes(float value) {
    std::vector<uint8_t> result(sizeof(value));
    std::memcpy(result.data(), &value, sizeof(value));
    return result;
}

std::vector<uint8_t> i32_bytes(const std::vector<uint32_t> & values) {
    std::vector<uint8_t> result(values.size() * sizeof(uint32_t));
    std::memcpy(result.data(), values.data(), result.size());
    return result;
}

std::vector<uint8_t> i64_bytes(const std::vector<int64_t> & values) {
    std::vector<uint8_t> result(values.size() * sizeof(int64_t));
    std::memcpy(result.data(), values.data(), result.size());
    return result;
}

void store_f16(std::vector<uint8_t> & data, size_t index, float value) {
    const ggml_fp16_t bits = ggml_fp32_to_fp16(value);
    std::memcpy(data.data() + index * sizeof(bits), &bits, sizeof(bits));
}

void store_bf16(std::vector<uint8_t> & data, size_t index, float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint16_t bf16 = bits >> 16;
    std::memcpy(data.data() + index * sizeof(bf16), &bf16, sizeof(bf16));
}

uint16_t load_u16(const uint8_t * data) {
    uint16_t result;
    std::memcpy(&result, data, sizeof(result));
    return result;
}

void require_q4_1_block(
        const uint8_t * block,
        float scale,
        uint8_t zero,
        const std::array<uint8_t, 32> & codes,
        const char * message) {
    require(load_u16(block) == ggml_fp32_to_fp16(scale), message);
    require(load_u16(block + sizeof(ggml_fp16_t)) == ggml_fp32_to_fp16(-scale * zero), message);
    for (size_t i = 0; i < codes.size() / 2; ++i) {
        require(block[2 * sizeof(ggml_fp16_t) + i] == (codes[i] | (codes[i + 16] << 4)), message);
    }
}

void require_q4_a32_block(
        const uint8_t * block,
        const std::array<float, 4> & scales,
        const std::array<uint8_t, 4> & zeros,
        const std::array<uint8_t, 128> & codes,
        const char * message) {
    for (size_t group = 0; group < scales.size(); ++group) {
        uint32_t bits;
        std::memcpy(&bits, &scales[group], sizeof(bits));
        require(load_u16(block + group * sizeof(uint16_t)) == uint16_t(bits >> 16), message);
        require(((block[8 + group / 2] >> (4 * (group % 2))) & 0x0f) == zeros[group], message);
    }
    for (size_t i = 0; i < codes.size(); i += 2) {
        require(block[10 + i / 2] == (codes[i] | (codes[i + 1] << 4)), message);
    }
}

std::string read_text(const std::filesystem::path & path) {
    std::ifstream in(path);
    require(static_cast<bool>(in), "failed to open safetensors test metadata");
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

struct temp_dir {
    std::filesystem::path path;

    temp_dir() {
        path = std::filesystem::temp_directory_path() / "llama-safetensors-registry-test";
        std::filesystem::remove_all(path);
        std::filesystem::create_directories(path);
    }

    ~temp_dir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

std::unique_ptr<llama_safetensors_importer> make_importer(const std::filesystem::path & model_dir) {
    llama_safetensors_json config = llama_safetensors_read_json(model_dir / "config.json");
    if (llama_safetensors_qwen3_importer::probe(config)) {
        return std::make_unique<llama_safetensors_qwen3_importer>(model_dir, std::move(config));
    }
    if (llama_safetensors_qwen35_importer::probe(config)) {
        return std::make_unique<llama_safetensors_qwen35_importer>(model_dir, std::move(config));
    }
    throw std::runtime_error("unsupported test importer architecture");
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc == 5 && std::string(argv[1]) == "compare") {
        ggml_backend_load_all();
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers       = 99;
        llama_model * native            = llama_model_load_from_safetensors_dir(argv[2], model_params);
        llama_model * control           = llama_model_load_from_file(argv[3], model_params);
        if (native == nullptr || control == nullptr) {
            throw std::runtime_error("failed to load a live-comparison model");
        }

        const std::string   text          = read_text(argv[4]);
        const llama_vocab * native_vocab  = llama_model_get_vocab(native);
        const llama_vocab * control_vocab = llama_model_get_vocab(control);
        int                 n_tokens = -llama_tokenize(native_vocab, text.data(), text.size(), nullptr, 0, true, true);
        std::vector<llama_token> tokens(n_tokens);
        std::vector<llama_token> control_tokens(n_tokens);
        require(llama_tokenize(native_vocab, text.data(), text.size(), tokens.data(), tokens.size(), true, true) ==
                        n_tokens &&
                    llama_tokenize(control_vocab, text.data(), text.size(), control_tokens.data(),
                                   control_tokens.size(), true, true) == n_tokens &&
                    tokens == control_tokens,
                "native and GGUF tokenization differ");
        constexpr int n_compare = 16384;
        require(n_tokens >= n_compare, "live-comparison corpus is too short");
        tokens.resize(n_compare);

        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx                = n_compare;
        context_params.n_batch              = 512;
        context_params.n_ubatch             = 512;
        context_params.flash_attn_type      = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        llama_context * native_ctx          = llama_init_from_model(native, context_params);
        llama_context * control_ctx         = llama_init_from_model(control, context_params);
        if (native_ctx == nullptr || control_ctx == nullptr) {
            throw std::runtime_error("failed to create live-comparison contexts");
        }

        const int n_vocab  = llama_vocab_n_tokens(native_vocab);
        size_t    compared = 0;
        for (int begin = 0; begin < n_compare; begin += 512) {
            const int   count         = std::min(512, n_compare - begin);
            llama_batch native_batch  = llama_batch_init(count, 0, 1);
            llama_batch control_batch = llama_batch_init(count, 0, 1);
            for (int i = 0; i < count; ++i) {
                for (llama_batch * batch : { &native_batch, &control_batch }) {
                    batch->token[i]     = tokens[begin + i];
                    batch->pos[i]       = begin + i;
                    batch->n_seq_id[i]  = 1;
                    batch->seq_id[i][0] = 0;
                    batch->logits[i]    = true;
                    batch->n_tokens     = count;
                }
            }
            if (llama_decode(native_ctx, native_batch) != 0 || llama_decode(control_ctx, control_batch) != 0) {
                throw std::runtime_error("live-comparison decode failed");
            }
            for (int i = 0; i < count; ++i) {
                const float * actual   = llama_get_logits_ith(native_ctx, i);
                const float * expected = llama_get_logits_ith(control_ctx, i);
                if (std::memcmp(actual, expected, size_t(n_vocab) * sizeof(float)) != 0) {
                    throw std::runtime_error("native and GGUF logits differ at token " + std::to_string(begin + i));
                }
                ++compared;
            }
            llama_batch_free(native_batch);
            llama_batch_free(control_batch);
        }
        std::cout << "exact_logit_rows=" << compared << " vocab=" << n_vocab << '\n';
        llama_free(native_ctx);
        llama_free(control_ctx);
        llama_model_free(native);
        llama_model_free(control);
        return 0;
    }
    if (argc == 4 && std::string(argv[1]) == "metadata") {
        std::unique_ptr<llama_safetensors_importer> importer            = make_importer(argv[2]);
        gguf_context *                              actual              = importer->build_metadata();
        ggml_context *                              expected_tensor_ctx = nullptr;
        gguf_init_params                            params              = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &expected_tensor_ctx,
        };
        gguf_context * expected = gguf_init_from_file(argv[3], params);
        if (actual == nullptr || expected == nullptr) {
            throw std::runtime_error("failed to build or read metadata");
        }
        size_t mismatches = gguf_get_n_kv(actual) == gguf_get_n_kv(expected) ? 0 : 1;
        for (int64_t i = 0; i < gguf_get_n_kv(actual); ++i) {
            const char *  key      = gguf_get_key(actual, i);
            const int64_t j        = gguf_find_key(expected, key);
            bool          mismatch = j < 0 || gguf_get_kv_type(actual, i) != gguf_get_kv_type(expected, j);
            if (!mismatch) {
                switch (gguf_get_kv_type(actual, i)) {
                    case GGUF_TYPE_STRING:
                        mismatch = std::string(gguf_get_val_str(actual, i)) != gguf_get_val_str(expected, j);
                        break;
                    case GGUF_TYPE_UINT32:
                        mismatch = gguf_get_val_u32(actual, i) != gguf_get_val_u32(expected, j);
                        break;
                    case GGUF_TYPE_INT32:
                        mismatch = gguf_get_val_i32(actual, i) != gguf_get_val_i32(expected, j);
                        break;
                    case GGUF_TYPE_FLOAT32:
                        mismatch = gguf_get_val_f32(actual, i) != gguf_get_val_f32(expected, j);
                        break;
                    case GGUF_TYPE_BOOL:
                        mismatch = gguf_get_val_bool(actual, i) != gguf_get_val_bool(expected, j);
                        break;
                    case GGUF_TYPE_ARRAY:
                        {
                            const gguf_type at = gguf_get_arr_type(actual, i);
                            mismatch           = at != gguf_get_arr_type(expected, j) ||
                                       gguf_get_arr_n(actual, i) != gguf_get_arr_n(expected, j);
                            if (!mismatch && at == GGUF_TYPE_STRING) {
                                for (size_t k = 0; k < gguf_get_arr_n(actual, i); ++k) {
                                    if (std::string(gguf_get_arr_str(actual, i, k)) !=
                                        gguf_get_arr_str(expected, j, k)) {
                                        mismatch = true;
                                        break;
                                    }
                                }
                            } else if (!mismatch && at == GGUF_TYPE_INT32) {
                                mismatch = std::memcmp(gguf_get_arr_data(actual, i), gguf_get_arr_data(expected, j),
                                                       gguf_get_arr_n(actual, i) * sizeof(int32_t)) != 0;
                            }
                        }
                        break;
                    default:
                        std::cerr << "unchecked metadata type for " << key << '\n';
                        mismatch = true;
                        break;
                }
            }
            if (mismatch) {
                std::cerr << "metadata mismatch " << key << '\n';
                ++mismatches;
            }
        }
        for (int64_t i = 0; i < gguf_get_n_kv(expected); ++i) {
            if (gguf_find_key(actual, gguf_get_key(expected, i)) < 0) {
                std::cerr << "missing metadata key " << gguf_get_key(expected, i) << '\n';
                ++mismatches;
            }
        }
        if (gguf_get_n_tensors(actual) != 0) {
            std::cerr << "native metadata unexpectedly contains a tensor manifest\n";
            ++mismatches;
        }
        for (int64_t i = 0; i < gguf_get_n_tensors(expected); ++i) {
            const char *                       name = gguf_get_tensor_name(expected, i);
            ggml_type                          type;
            std::array<int64_t, GGML_MAX_DIMS> ne;
            if (!importer->describe(name, type, ne)) {
                std::cerr << "source cannot describe tensor " << name << '\n';
                ++mismatches;
                continue;
            }
            bool            mismatch = type != gguf_get_tensor_type(expected, i);
            const int64_t * ene      = gguf_get_tensor_ne(expected, i);
            for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
                mismatch |= ne[dim] != ene[dim];
            }
            if (mismatch) {
                std::cerr << "tensor contract mismatch " << name << " actual_type=" << ggml_type_name(type)
                          << " expected_type=" << ggml_type_name(gguf_get_tensor_type(expected, i))
                          << " actual_ne=" << ne[0] << ',' << ne[1] << ',' << ne[2] << ',' << ne[3]
                          << " expected_ne=" << ene[0] << ',' << ene[1] << ',' << ene[2] << ',' << ene[3] << '\n';
                ++mismatches;
            }
        }
        std::cout << "metadata_tensors=" << gguf_get_n_tensors(actual)
                  << " source_contracts=" << gguf_get_n_tensors(expected) << " mismatches=" << mismatches << '\n';
        gguf_free(actual);
        gguf_free(expected);
        ggml_free(expected_tensor_ctx);
        return mismatches == 0 ? 0 : 1;
    }
    if (argc == 4 && std::string(argv[1]) == "load") {
        std::unique_ptr<llama_safetensors_importer> importer   = make_importer(argv[2]);
        ggml_context *                              tensor_ctx = nullptr;
        gguf_context *                              metadata   = nullptr;
        const bool                                  native     = std::string(argv[3]) == "native";
        if (!native) {
            gguf_init_params init_params = {
                /*.no_alloc = */ true,
                /*.ctx      = */ &tensor_ctx,
            };
            metadata = gguf_init_from_file(argv[3], init_params);
            if (metadata == nullptr || tensor_ctx == nullptr) {
                throw std::runtime_error("failed to open GGUF metadata control");
            }
        }

        struct callback_state {
            llama_safetensors_importer * importer;
            size_t                       tensors = 0;
            size_t                       bytes   = 0;
        } state{ importer.get() };

        const auto set_tensor_data = [](ggml_tensor * tensor, void * userdata) {
            auto *               state = static_cast<callback_state *>(userdata);
            std::vector<uint8_t> data  = state->importer->materialize(tensor->name, tensor->type, ggml_nbytes(tensor));
            ggml_backend_tensor_set(tensor, data.data(), 0, data.size());
            ++state->tensors;
            state->bytes += data.size();
        };

        ggml_backend_load_all();
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers       = 99;
        llama_model * model             = native ? llama_model_load_from_safetensors_dir(argv[2], model_params) :
                                                   llama_model_init_from_user(metadata, set_tensor_data, &state, model_params);
        if (model == nullptr) {
            throw std::runtime_error("direct safetensors model load failed");
        }
        if (!native) {
            std::cout << "loaded_tensors=" << state.tensors << " loaded_bytes=" << state.bytes << '\n';
        }

        const llama_vocab * vocab    = llama_model_get_vocab(model);
        const std::string   prompt   = "The capital of France is";
        const int           n_prompt = -llama_tokenize(vocab, prompt.data(), prompt.size(), nullptr, 0, true, true);
        std::vector<llama_token> tokens(n_prompt);
        if (llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), tokens.size(), true, true) < 0) {
            throw std::runtime_error("failed to tokenize direct-load smoke prompt");
        }
        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx                = 512;
        context_params.n_batch              = 512;
        context_params.n_ubatch             = 512;
        context_params.flash_attn_type      = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        llama_context * context             = llama_init_from_model(model, context_params);
        if (context == nullptr) {
            throw std::runtime_error("failed to create direct safetensors context");
        }
        llama_sampler * sampler = llama_sampler_init_greedy();
        llama_batch     batch   = llama_batch_get_one(tokens.data(), tokens.size());
        std::string     generated;
        for (int i = 0; i < 8; ++i) {
            if (llama_decode(context, batch) != 0) {
                throw std::runtime_error("direct safetensors decode failed");
            }
            llama_token token = llama_sampler_sample(sampler, context, -1);
            if (llama_vocab_is_eog(vocab, token)) {
                break;
            }
            std::array<char, 256> piece{};
            const int             n_piece = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, true);
            if (n_piece < 0) {
                throw std::runtime_error("failed to render direct-load token");
            }
            generated.append(piece.data(), static_cast<size_t>(n_piece));
            batch = llama_batch_get_one(&token, 1);
        }
        std::cout << "generated=" << generated << '\n';
        llama_sampler_free(sampler);
        llama_free(context);
        llama_model_free(model);
        if (metadata != nullptr) {
            gguf_free(metadata);
        }
        if (tensor_ctx != nullptr) {
            ggml_free(tensor_ctx);
        }
        return 0;
    }
    if (argc == 3) {
        std::unique_ptr<llama_safetensors_importer> importer   = make_importer(argv[1]);
        ggml_context *                              tensor_ctx = nullptr;
        gguf_init_params                            params     = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &tensor_ctx,
        };
        gguf_context * gguf = gguf_init_from_file(argv[2], params);
        if (gguf == nullptr || tensor_ctx == nullptr) {
            throw std::runtime_error("failed to open GGUF control");
        }
        std::ifstream control(argv[2], std::ios::binary);
        if (!control) {
            throw std::runtime_error("failed to open GGUF control data");
        }

        const size_t  data_offset    = gguf_get_data_offset(gguf);
        const int64_t n_tensors      = gguf_get_n_tensors(gguf);
        size_t        compared_bytes = 0;
        for (int64_t i = 0; i < n_tensors; ++i) {
            const char *  name   = gguf_get_tensor_name(gguf, i);
            ggml_tensor * tensor = ggml_get_tensor(tensor_ctx, name);
            require(tensor != nullptr, "GGUF tensor descriptor is missing from its metadata context");
            const size_t         size   = ggml_nbytes(tensor);
            std::vector<uint8_t> actual = importer->materialize(name, tensor->type, size);
            std::vector<uint8_t> expected(size);
            control.seekg(static_cast<std::streamoff>(data_offset + gguf_get_tensor_offset(gguf, i)));
            control.read(reinterpret_cast<char *>(expected.data()), static_cast<std::streamsize>(expected.size()));
            if (!control) {
                throw std::runtime_error(std::string("failed to read GGUF control tensor '") + name + "'");
            }
            bool equal = actual == expected;
            if (!equal) {
                size_t first = 0;
                while (first < size && actual[first] == expected[first]) {
                    ++first;
                }
                throw std::runtime_error(std::string("import mismatch for '") + name + "' at byte " +
                                         std::to_string(first));
            }
            compared_bytes += size;
        }
        std::cout << "matched_tensors=" << n_tensors << " matched_bytes=" << compared_bytes << '\n';
        gguf_free(gguf);
        ggml_free(tensor_ctx);
        return 0;
    }
    if (argc == 2) {
        const auto                             registry = llama_safetensors_registry::load(argv[1]);
        const llama_safetensors_quant_adapters adapters(
            llama_safetensors_read_json(std::filesystem::path(argv[1]) / "config.json"), registry);
        const llama_safetensors_quant_summary & summary = adapters.summary();
        std::cout << "shards=" << registry.shards().size() << " tensors=" << registry.tensors().size()
                  << " nvfp4=" << summary.nvfp4 << " fp8_channel=" << summary.fp8_channel
                  << " fp8_block=" << summary.fp8_block << " w8a8=" << summary.w8a8
                  << " awq=" << summary.awq << " gptq=" << summary.gptq
                  << " gptq_int8=" << summary.gptq_int8
                  << " eetq=" << summary.eetq
                  << " bnb_int8=" << summary.bnb_int8 << '\n';
        return 0;
    }

    const auto q_proj = llama_safetensors_map_decoder_tensor("model.layers.7.", "attn_q.weight");
    require(q_proj.has_value(), "shared decoder mapper missed attention Q");
    require(q_proj->source == "model.layers.7.self_attn.q_proj.weight", "attention Q source name is wrong");
    require(q_proj->module == "model.layers.7.self_attn.q_proj", "attention Q module name is wrong");
    require(q_proj->quant_role == llama_safetensors_quant_role::WEIGHT, "attention Q quant role is wrong");

    const auto ffn_scale = llama_safetensors_map_decoder_tensor("model.layers.3.", "ffn_down.scale");
    require(ffn_scale.has_value(), "shared decoder mapper missed FFN scale");
    require(ffn_scale->source == "model.layers.3.mlp.down_proj.weight_scale", "FFN scale source name is wrong");
    require(ffn_scale->quant_role == llama_safetensors_quant_role::WEIGHT_SCALE, "FFN scale quant role is wrong");

    const auto norm = llama_safetensors_map_decoder_tensor("model.layers.2.", "post_attention_norm.weight");
    require(norm.has_value(), "shared decoder mapper missed post-attention norm");
    require(norm->source == "model.layers.2.post_attention_layernorm.weight" && !norm->quant_role,
            "post-attention norm mapping is wrong");
    require(!llama_safetensors_map_decoder_tensor("model.layers.2.", "ssm_a").has_value(),
            "shared decoder mapper accepted an architecture-specific tensor");
    require(llama_safetensors_qwen3_importer::probe({ { "model_type", "mistral" } }),
            "classic Mistral did not select the shared Llama importer");
    require(!llama_safetensors_qwen3_importer::probe({ { "model_type", "mistral3" } }),
            "Mistral3 incorrectly selected the classic Llama importer");
    require(argc == 1, "usage: test-safetensors-registry [model-directory [control.gguf]]");

    temp_dir dir;

    {
        const auto path = dir.path / "qwen4-permutation-types";
        write_single_shard_model(path, {
            { "model.layers.0.linear_attn.in_proj_qkv.weight", "BF16", { 8, 3 },
              std::vector<uint8_t>(8 * 3 * sizeof(uint16_t), 0) },
            { "model.layers.0.linear_attn.dt_bias", "BF16", { 2 },
              { 0x80, 0x3f, 0x00, 0x40 } },
            { "model.layers.0.linear_attn.A_log", "BF16", { 2 },
              { 0x80, 0x3f, 0x00, 0x40 } },
            { "model.layers.0.linear_attn.conv1d.weight", "BF16", { 8, 3 },
              std::vector<uint8_t>(8 * 3 * sizeof(uint16_t), 0) },
            { "mtp.pre_fc_norm_embedding.weight", "BF16", { 4 },
              std::vector<uint8_t>(4 * sizeof(uint16_t), 0) },
            { "mtp.fc_embedding.weight", "BF16", { 4, 4 },
              std::vector<uint8_t>(4 * 4 * sizeof(uint16_t), 0) },
            { "mtp.fc_hidden.weight", "BF16", { 4, 4 },
              std::vector<uint8_t>(4 * 4 * sizeof(uint16_t), 0) },
            { "mtp.layers.0.attn_hyper_connection.hc_norm.weight", "BF16", { 8 },
              std::vector<uint8_t>(8 * sizeof(uint16_t), 0) },
        });
        write_text(path / "tokenizer.json", "{}");
        const json config = {
            { "model_type", "qwen4_exp" },
            { "num_hidden_layers", 1 },
            { "mtp_num_hidden_layers", 1 },
            { "linear_num_key_heads", 1 },
            { "linear_num_value_heads", 2 },
            { "linear_key_head_dim", 2 },
            { "linear_value_head_dim", 2 },
            { "indexer_n_heads", 1 },
            { "indexer_head_dim", 2 },
            { "full_attention_interval", 2 },
            { "split_ngram_parts", 0 },
            { "ple_layer_ids", json::array() },
            { "num_experts", 2 },
            { "moe_intermediate_size", 128 },
        };
        write_text(path / "config.json", config.dump());
        llama_safetensors_qwen4exp_importer importer(path, config);

        ggml_type type = GGML_TYPE_COUNT;
        std::array<int64_t, GGML_MAX_DIMS> ne{};
        require(importer.describe("blk.0.attn_qkv.weight", type, ne) &&
                    type == GGML_TYPE_BF16 && ne[0] == 3 && ne[1] == 8,
                "Qwen4 permutation-only matrix did not retain BF16");
        require(importer.materialize("blk.0.attn_qkv.weight", type, 48).size() == 48,
                "Qwen4 BF16 matrix permutation changed its byte width");

        require(importer.describe("blk.0.ssm_dt.bias", type, ne) &&
                    type == GGML_TYPE_F32 && ne[0] == 2,
                "Qwen4 transformed runtime scalar did not declare F32");
        require(importer.materialize("blk.0.ssm_dt.bias", type, 8).size() == 8,
                "Qwen4 transformed runtime scalar was not converted to F32");

        require(importer.describe("blk.0.ssm_conv1d.weight", type, ne) &&
                    type == GGML_TYPE_F32 && ne[0] == 3 && ne[1] == 8,
                "Qwen4 SSM convolution kernel did not declare backend-required F32");
        require(importer.materialize("blk.0.ssm_conv1d.weight", type, 96).size() == 96,
                "Qwen4 SSM convolution kernel was not converted to F32");

        require(importer.describe("blk.0.ssm_a", type, ne) &&
                    type == GGML_TYPE_F32 && ne[0] == 2,
                "Qwen4 numerical transform did not declare F32");
        const auto a_log = importer.materialize("blk.0.ssm_a", type, 8);
        float a0;
        float a1;
        std::memcpy(&a0, a_log.data(), sizeof(a0));
        std::memcpy(&a1, a_log.data() + sizeof(a0), sizeof(a1));
        require(std::isfinite(a0) && std::isfinite(a1) && a0 < 0.0f && a1 < 0.0f,
                "Qwen4 A_log permutation/conversion produced invalid values");

        require(importer.describe("blk.1.nextn.enorm.weight", type, ne) &&
                    type == GGML_TYPE_F32 && ne[0] == 4,
                "Qwen4 embedded MTP norm did not map to the draft block");
        require(importer.describe("blk.1.nextn.eh_proj.weight", type, ne) &&
                    type == GGML_TYPE_BF16 && ne[0] == 8 && ne[1] == 4,
                "Qwen4 embedded MTP input projections were not concatenated");
        require(importer.materialize("blk.1.nextn.eh_proj.weight", type, 64).size() == 64,
                "Qwen4 embedded MTP input projection has the wrong byte size");
        require(importer.describe("blk.1.hc_attn_norm.weight", type, ne) &&
                    type == GGML_TYPE_F32 && ne[0] == 8,
                "Qwen4 embedded MTP layer did not use the mtp.layers.0 prefix");
    }

    {
        const auto path = dir.path / "qwen4-split-fp8-experts";
        constexpr size_t dim = 128;
        std::vector<tensor_fixture> tensors;
        for (size_t expert = 0; expert < 2; ++expert) {
            const std::string module = "model.layers.0.mlp.experts." +
                std::to_string(expert) + ".gate_proj";
            tensors.push_back({ module + ".weight", "F8_E4M3", { dim, dim },
                                std::vector<uint8_t>(dim * dim, expert == 0 ? 0x38 : 0x40) });
            tensors.push_back({ module + ".weight_scale_inv", "BF16", { 1, 1 }, { 0x80, 0x3f } });
        }
        write_single_shard_model(path, tensors);
        write_text(path / "tokenizer.json", "{}");
        json config = {
            { "model_type", "qwen4_exp" },
            { "num_hidden_layers", 1 },
            { "mtp_num_hidden_layers", 0 },
            { "linear_num_key_heads", 1 },
            { "linear_num_value_heads", 2 },
            { "linear_key_head_dim", 2 },
            { "linear_value_head_dim", 2 },
            { "indexer_n_heads", 1 },
            { "indexer_head_dim", 2 },
            { "full_attention_interval", 2 },
            { "split_ngram_parts", 0 },
            { "ple_layer_ids", json::array() },
            { "num_experts", 2 },
            { "moe_intermediate_size", dim },
        };
        config["quantization_config"] =
            json::parse(fp8_block_inferred_format_config).at("quantization_config");
        write_text(path / "config.json", config.dump());
        llama_safetensors_qwen4exp_importer importer(path, config);

        ggml_type type = GGML_TYPE_COUNT;
        std::array<int64_t, GGML_MAX_DIMS> ne{};
        require(importer.describe("blk.0.ffn_gate_exps.weight", type, ne) &&
                    type == GGML_TYPE_Q8_0_G128 && ne[0] == dim && ne[1] == dim && ne[2] == 2,
                "Qwen4 split FP8 experts did not select the portable Q8 bridge");
        importer.bind("blk.0.ffn_gate_exps.weight");
        const size_t row_size = ggml_row_size(type, dim);
        const auto packed = importer.materialize(
            "blk.0.ffn_gate_exps.weight", type, 2 * dim * row_size);
        const auto * traits = ggml_get_type_traits(type);
        require(traits != nullptr && traits->to_float != nullptr,
                "Qwen4 expert bridge target cannot be decoded");
        std::vector<float> row(dim);
        traits->to_float(packed.data(), row.data(), dim);
        require(std::all_of(row.begin(), row.end(), [](float value) { return std::fabs(value - 1.0f) < 1e-3f; }),
                "Qwen4 first split FP8 expert was repacked incorrectly");
        traits->to_float(packed.data() + dim * row_size, row.data(), dim);
        require(std::all_of(row.begin(), row.end(), [](float value) { return std::fabs(value - 2.0f) < 1e-3f; }),
                "Qwen4 second split FP8 expert was repacked incorrectly");
        importer.validate_complete();
    }

    {
        const auto path = dir.path / "plain-f16";
        write_single_shard_model(path, {
            { "matrix.weight", "F16", { 2, 2 }, { 0x00, 0x3c, 0x00, 0x40, 0x00, 0x42, 0x00, 0x44 } },
            { "norm.weight",   "F16", { 2 },    { 0x00, 0x3c, 0x00, 0x40 }                         },
        });
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(llama_safetensors_json::object(), registry);
        ggml_type type;
        std::array<int64_t, GGML_MAX_DIMS> ne;
        const auto matrix = llama_safetensors_bind_tensor(
            adapters, { "matrix.weight", {}, std::nullopt });
        require(llama_safetensors_describe_tensor(registry, matrix, type, ne) &&
                    type == GGML_TYPE_F16 && ne[0] == 2 && ne[1] == 2,
                "plain F16 matrix did not preserve its storage type");
        const auto vector = llama_safetensors_bind_tensor(
            adapters, { "norm.weight", {}, std::nullopt });
        std::vector<uint8_t> expected = f32_bytes(1.0f);
        const std::vector<uint8_t> two = f32_bytes(2.0f);
        expected.insert(expected.end(), two.begin(), two.end());
        require(llama_safetensors_describe_tensor(registry, vector, type, ne) && type == GGML_TYPE_F32 &&
                    llama_safetensors_materialize_tensor(
                        registry, adapters, vector, type, expected.size()) == expected,
                "plain F16 vector did not convert to F32");
    }

    // A conventional Qwen3 adapter must reuse the shared block-FP8 contract
    // and decoder naming seam without architecture-specific materialization.
    {
        const std::filesystem::path qwen3_dir = dir.path / "qwen3";
        write_single_shard_model(
            qwen3_dir,
            {
                { "model.layers.0.self_attn.q_proj.weight",           "F8_E4M3", { 128, 128 }, std::vector<uint8_t>(128 * 128) },
                { "model.layers.0.self_attn.q_proj.weight_scale_inv", "BF16",    { 1, 1 },     { 0x80, 0x3f }                  },
        });
        write_text(qwen3_dir / "generation_config.json", R"({"bos_token_id":0,"eos_token_id":1})");
        write_text(qwen3_dir / "tokenizer.json", "{}");
        write_text(qwen3_dir / "tokenizer_config.json", "{}");
        llama_safetensors_json config = {
            { "model_type",          "qwen3" },
            { "num_hidden_layers",   1       },
            { "hidden_size",         128     },
            { "num_attention_heads", 1       },
            { "num_key_value_heads", 1       },
            { "head_dim",            128     },
        };
        config["quantization_config"] = llama_safetensors_json::parse(fp8_block_config).at("quantization_config");
        llama_safetensors_qwen3_importer   importer(qwen3_dir, config);
        ggml_type                          type;
        std::array<int64_t, GGML_MAX_DIMS> ne;
        require(importer.describe("blk.0.attn_q.weight", type, ne), "Qwen3 FP8 weight was not described");
        require(type == GGML_TYPE_F8_E4M3 && ne[0] == 128 && ne[1] == 128, "Qwen3 FP8 weight contract is wrong");
        require(importer.describe("blk.0.attn_q.scale", type, ne), "Qwen3 FP8 scale was not described");
        require(type == GGML_TYPE_F32 && ne[0] == 1 && ne[1] == 1, "Qwen3 FP8 block-scale contract is wrong");
        require(importer.materialize("blk.0.attn_q.scale", GGML_TYPE_F32, sizeof(float)) == f32_bytes(1.0f),
                "Qwen3 FP8 scale materialization changed source bytes");
    }

    // A per-row W8A16 scale bundle must permute only its values. Treating the
    // header as ordinary Q/K rows corrupts the bundle before model loading.
    {
        const std::filesystem::path mistral_dir = dir.path / "mistral-quanto-qint8";
        constexpr size_t rows = 64;
        constexpr size_t cols = 64;
        std::vector<uint8_t> scales(rows * sizeof(uint16_t));
        for (size_t row = 0; row < rows; ++row) {
            store_bf16(scales, row, float(row + 1));
        }
        write_single_shard_model(mistral_dir, {
            { "model.layers.0.self_attn.q_proj.weight._data",  "I8",   { rows, cols }, std::vector<uint8_t>(rows * cols) },
            { "model.layers.0.self_attn.q_proj.weight._scale", "BF16", { rows, 1 },    scales },
        });
        write_text(mistral_dir / "generation_config.json", R"({"bos_token_id":0,"eos_token_id":1})");
        write_text(mistral_dir / "tokenizer.json", "{}");
        write_text(mistral_dir / "tokenizer_config.json", "{}");
        llama_safetensors_json config = {
            { "model_type",          "mistral" },
            { "num_hidden_layers",   1 },
            { "hidden_size",         cols },
            { "intermediate_size",   128 },
            { "num_attention_heads", 1 },
            { "num_key_value_heads", 1 },
            { "head_dim",            rows },
            { "quantization_config", {
                { "quant_method", "quanto" },
                { "quantization_map", {
                    { "model.layers.0.self_attn.q_proj", {
                        { "weights", "qint8" }, { "activations", "none" }
                    } }
                } }
            } }
        };
        llama_safetensors_qwen3_importer importer(mistral_dir, config);
        ggml_type type;
        std::array<int64_t, GGML_MAX_DIMS> ne;
        require(importer.describe("blk.0.attn_q.scale", type, ne) && type == GGML_TYPE_I8,
                "Mistral Quanto qint8 scale bundle was not described");
        const std::vector<uint8_t> bundle = importer.materialize(
            "blk.0.attn_q.scale", type, static_cast<size_t>(ne[0]));
        ggml_w8a16_scale_header header;
        std::memcpy(&header, bundle.data(), sizeof(header));
        require(header.magic == GGML_W8A16_SCALE_MAGIC && header.n_channels == rows,
                "Mistral Quanto qint8 scale bundle header was corrupted by RoPE permutation");
        const auto value_at = [&](size_t row) {
            ggml_bf16_t value;
            std::memcpy(&value, bundle.data() + header.values_offset + row * sizeof(value), sizeof(value));
            return ggml_bf16_to_fp32(value);
        };
        require(value_at(0) == 1.0f && value_at(1) == 33.0f &&
                    value_at(2) == 2.0f && value_at(3) == 34.0f,
                "Mistral Quanto qint8 scale values used the wrong RoPE row permutation");
    }

    // Older compressed-tensors checkpoints use one top-level format and a
    // module-class selector instead of per-group formats and regexes.
    {
        const std::filesystem::path legacy_dir = dir.path / "legacy-channel-fp8";
        const std::string           module     = "model.layers.0.self_attn.q_proj";
        write_single_shard_model(legacy_dir,
                                 {
                                     { module + ".weight",       "F8_E4M3", { 2, 2 }, { 0, 0, 0, 0 }             },
                                     { module + ".weight_scale", "BF16",    { 2 },    { 0x80, 0x3f, 0x80, 0x3f } },
        });
        const llama_safetensors_json           config   = llama_safetensors_json::parse(R"({
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "format": "float-quantized",
                "ignore": ["lm_head"],
                "config_groups": { "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "type": "float", "num_bits": 8, "strategy": "channel",
                        "symmetric": true, "dynamic": false,
                        "group_size": null, "block_structure": null, "actorder": null
                    },
                    "input_activations": {
                        "type": "float", "num_bits": 8, "strategy": "token",
                        "symmetric": true, "dynamic": true,
                        "group_size": null, "block_structure": null, "actorder": null
                    }
                } }
            }
        })");
        const llama_safetensors_registry       registry = llama_safetensors_registry::load(legacy_dir);
        const llama_safetensors_quant_adapters adapters(config, registry);
        require(adapters.summary().fp8_channel == 1, "legacy channel-FP8 group was not recognized");
        require(adapters.bind(module, llama_safetensors_quant_role::WEIGHT).has_value(),
                "legacy Linear selector did not bind a projection");
        require(!adapters.bind("lm_head", llama_safetensors_quant_role::WEIGHT).has_value(),
                "legacy compressed-tensors ignore rule was not honored");

        llama_safetensors_json naive = config;
        naive["quantization_config"]["format"] = "naive-quantized";
        const llama_safetensors_quant_adapters naive_adapters(naive, registry);
        require(naive_adapters.summary().fp8_channel == 1 &&
                    naive_adapters.bind(module, llama_safetensors_quant_role::WEIGHT).has_value(),
                "naive-quantized channel-FP8 group was not recognized");
    }

    // AutoRound/INC publishes the same group-32 MXFP tensors as
    // compressed-tensors under a compact flat config. Both schemas must bind
    // to the same canonical types, including its per-module FP16 exceptions.
    for (const uint32_t bits : { 4U, 8U }) {
        const auto path = dir.path / (bits == 4 ? "autoround-mxfp4" : "autoround-mxfp8");
        std::vector<tensor_fixture> tensors;
        if (bits == 4) {
            tensors = {
                { "module.weight_packed", "U8",      { 2, 16 }, std::vector<uint8_t>(32) },
                { "module.weight_scale",  "U8",      { 2, 1 },  std::vector<uint8_t>(2, 127) },
                { "plain.weight",         "BF16",    { 2, 32 }, std::vector<uint8_t>(128) },
            };
        } else {
            tensors = {
                { "module.weight",       "F8_E4M3", { 2, 32 }, std::vector<uint8_t>(64, 0x38) },
                { "module.weight_scale", "U8",      { 2, 1 },  std::vector<uint8_t>(2, 127) },
                { "plain.weight",        "BF16",    { 2, 32 }, std::vector<uint8_t>(128) },
            };
        }
        write_single_shard_model(path, tensors);
        const json config = {
            { "quantization_config", {
                { "quant_method", "auto-round" },
                { "packing_format", "auto_round:llm_compressor" },
                { "bits", bits },
                { "group_size", 32 },
                { "sym", true },
                { "data_type", "mx_fp" },
                { "act_bits", bits },
                { "act_group_size", 32 },
                { "act_sym", true },
                { "act_dynamic", true },
                { "act_data_type", "mx_fp_rceil" },
                { "extra_config", {
                    { "plain", {
                        { "bits", 16 }, { "group_size", 32 }, { "sym", true },
                        { "data_type", "float" }, { "act_bits", 16 },
                        { "act_group_size", 32 }, { "act_sym", true },
                        { "act_dynamic", true }, { "act_data_type", "float" },
                    } },
                } },
            } },
        };
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(config, registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        const auto input  = adapters.bind("module", llama_safetensors_quant_role::INPUT_SCALE);
        require(weight.has_value() && input.has_value() && !adapters.applies("plain") &&
                    weight->target_type == (bits == 4 ? GGML_TYPE_MXFP4 : GGML_TYPE_F8_E4M3) &&
                    input->materialization == (bits == 4 ?
                        llama_safetensors_quant_materialization::DYNAMIC_MXFP4_MARKER :
                        llama_safetensors_quant_materialization::DYNAMIC_MXFP8_MARKER),
                "AutoRound MXFP did not reuse the canonical microscaling contract");
        adapters.consume(*weight);
        const auto scale = adapters.bind("module", llama_safetensors_quant_role::WEIGHT_SCALE);
        if (scale) {
            adapters.consume(*scale);
        }
        adapters.consume(*input);
        adapters.validate_complete();
    }
    {
        const auto path = dir.path / "autoround-mxfp-sidecar";
        std::filesystem::create_directories(path);
        write_text(path / "config.json", R"({"model_type":"qwen3"})");
        const json quant = {
            { "quant_method", "auto-round" },
            { "packing_format", "auto_round:llm_compressor" },
            { "bits", 4 }, { "group_size", 32 }, { "sym", true },
            { "data_type", "mx_fp" }, { "act_bits", 4 },
            { "act_group_size", 32 }, { "act_sym", true },
            { "act_dynamic", true }, { "act_data_type", "mx_fp_rceil" },
        };
        write_text(path / "quantization_config.json", quant.dump());
        const auto merged = llama_safetensors_read_model_config(path);
        require(merged.contains("quantization_config") &&
                    merged.at("quantization_config").dump() == quant.dump(),
                "AutoRound quantization sidecar was not merged into the model config");
        const auto parsed = llama_safetensors_quant_config::load(path);
        const auto * group = parsed.match("model.layers.0.mlp.gate_proj");
        require(group != nullptr && group->format == llama_safetensors_quant_format::MXFP4_PACK,
                "AutoRound quantization sidecar did not reach the quantization parser");

        json invalid = quant;
        invalid["act_group_size"] = 64;
        write_text(path / "quantization_config.json", invalid.dump());
        require_rejected([&] { (void) llama_safetensors_quant_config::load(path); },
                "AutoRound MXFP accepted an incompatible activation group size");

        invalid = quant;
        invalid["extra_config"] = {
            { "model.layers.0.mlp.gate_proj", {
                { "bits", 8 }, { "data_type", "mx_fp" },
            } },
        };
        write_text(path / "quantization_config.json", invalid.dump());
        require_rejected([&] { (void) llama_safetensors_quant_config::load(path); },
                "AutoRound MXFP accepted a numerically different per-module override");
    }

    // Catch-all producer rules may leave selected modules plain without adding
    // them to an ignore list. Applicability follows the stored representation,
    // while an unexpected quant sidecar remains a malformed contract.
    for (const bool w8a8 : { false, true }) {
        const std::filesystem::path path = dir.path / (w8a8 ? "w8-plain-exception" : "fp8-plain-exception");
        const std::string quant_module = "quant";
        std::vector<tensor_fixture> tensors = {
            { "module.weight", "BF16", { 2, 32 }, std::vector<uint8_t>(128, 0x2a) },
        };
        llama_safetensors_json config = llama_safetensors_json::parse(w8a8 ? w8a8_config : fp8_channel_config);
        auto & group = config["quantization_config"]["config_groups"].begin().value();
        group["targets"][0] = "re:.*";
        if (w8a8) {
            tensors.push_back({ quant_module + ".weight", "I8", { 2, 32 }, std::vector<uint8_t>(64) });
            tensors.push_back({ quant_module + ".weight_scale", "BF16", { 2, 1 },
                                { 0x80, 0x3f, 0x80, 0x3f } });
        } else {
            tensors.push_back({ quant_module + ".weight", "F8_E4M3", { 2, 32 }, std::vector<uint8_t>(64) });
            tensors.push_back({ quant_module + ".weight_scale", "BF16", { 2 },
                                { 0x80, 0x3f, 0x80, 0x3f } });
        }
        write_single_shard_model(path, tensors);
        write_text(path / "config.json", config.dump());
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(config, registry);
        require(!adapters.applies("module") &&
                    !adapters.bind("module", llama_safetensors_quant_role::WEIGHT).has_value() &&
                    adapters.applies(quant_module),
                "catch-all quantization did not preserve a plain module exception");
        const auto binding = llama_safetensors_bind_tensor(
            adapters, { "module.weight", "module", llama_safetensors_quant_role::WEIGHT });
        ggml_type type;
        std::array<int64_t, GGML_MAX_DIMS> ne;
        require(llama_safetensors_describe_tensor(registry, binding, type, ne) && type == GGML_TYPE_BF16 &&
                    ne[0] == 32 && ne[1] == 2 &&
                    llama_safetensors_materialize_tensor(registry, adapters, binding, type, 128) ==
                        std::vector<uint8_t>(128, 0x2a),
                "plain catch-all exception did not follow the ordinary BF16 path");
    }

    // The dense runtime supports the 24-, 32-, and 64-layer Qwen3.5 families.
    // The importer must derive recurrent head geometry from config rather than
    // silently assuming the 27B layout.
    {
        const std::filesystem::path geometry_dir = dir.path / "geometry";
        write_single_shard_model(geometry_dir, {
                                                   { "module.weight",       "F8_E4M3", { 1, 1 }, { 0 }    },
                                                   { "module.weight_scale", "BF16",    { 1 },    { 0, 0 } },
        });
        write_text(geometry_dir / "generation_config.json", "{}");
        write_text(geometry_dir / "tokenizer.json", "{}");

        llama_safetensors_json config = {
            { "model_type",  "qwen3_5" },
            { "text_config",
             {
                  { "model_type", "qwen3_5_text" },
                  { "num_hidden_layers", 24 },
                  { "mtp_num_hidden_layers", 1 },
                  { "linear_num_key_heads", 8 },
                  { "linear_num_value_heads", 16 },
                  { "linear_key_head_dim", 64 },
                  { "linear_value_head_dim", 64 },
              }                        },
        };
        config["quantization_config"] = llama_safetensors_json::parse(fp8_channel_config).at("quantization_config");
        (void) llama_safetensors_qwen35_importer(geometry_dir, config);

        // Text-only checkpoints flatten text_config and use model.layers.*
        // rather than model.language_model.layers.*. They are the same runtime
        // architecture and must select the same importer.
        llama_safetensors_json text_only_config = config.at("text_config");
        text_only_config["quantization_config"] = config.at("quantization_config");
        require(llama_safetensors_qwen35_importer::probe(text_only_config),
                "text-only Qwen3.5 config did not select the Qwen3.5 importer");
        (void) llama_safetensors_qwen35_importer(geometry_dir, text_only_config);

        config["text_config"]["num_hidden_layers"] = 48;
        require_rejected([&] { (void) llama_safetensors_qwen35_importer(geometry_dir, config); },
                         "Qwen importer accepted a layer count unsupported by its model implementation");
        config["text_config"]["num_hidden_layers"]      = 32;
        config["text_config"]["linear_num_value_heads"] = 15;
        require_rejected([&] { (void) llama_safetensors_qwen35_importer(geometry_dir, config); },
                         "Qwen importer accepted non-integral recurrent head groups");
        config["text_config"]["linear_num_value_heads"] = 16;
        config["text_config"]["linear_value_head_dim"]  = 32;
        require_rejected([&] { (void) llama_safetensors_qwen35_importer(geometry_dir, config); },
                         "Qwen importer accepted unequal recurrent key/value dimensions");

        const std::filesystem::path channel_dir    = dir.path / "channel-transform";
        const std::string           channel_module = "model.language_model.layers.0.linear_attn.out_proj";
        write_single_shard_model(channel_dir, {
                                                  { channel_module + ".weight",       "F8_E4M3", { 2, 2 }, { 0, 0, 0, 0 } },
                                                  { channel_module + ".weight_scale", "BF16",    { 2 },    { 1, 2, 3, 4 } },
        });
        write_text(channel_dir / "generation_config.json", "{}");
        write_text(channel_dir / "tokenizer.json", "{}");
        config["text_config"]["num_hidden_layers"]                          = 24;
        config["text_config"]["linear_num_key_heads"]                       = 1;
        config["text_config"]["linear_num_value_heads"]                     = 2;
        config["text_config"]["linear_key_head_dim"]                        = 1;
        config["text_config"]["linear_value_head_dim"]                      = 1;
        config["quantization_config"]["config_groups"]["fp8"]["targets"][0] = channel_module;
        llama_safetensors_qwen35_importer channel_importer(channel_dir, config);
        require(channel_importer.materialize("blk.0.ssm_out.scale", GGML_TYPE_BF16, 4) ==
                    std::vector<uint8_t>({ 1, 2, 3, 4 }),
                "Qwen output channel scale incorrectly followed the weight's column permutation");

        const std::filesystem::path w8_dir = dir.path / "w8-column-transform";
        std::vector<uint8_t> w8_codes(2 * 128);
        for (size_t row = 0; row < 2; ++row) {
            for (size_t block = 0; block < 4; ++block) {
                std::fill_n(w8_codes.begin() + row * 128 + block * 32, 32,
                            static_cast<uint8_t>(10 * row + block + 1));
            }
        }
        write_single_shard_model(w8_dir, {
            { channel_module + ".weight",       "I8",   { 2, 128 }, std::move(w8_codes) },
            { channel_module + ".weight_scale", "BF16", { 2, 1 },  { 0x80, 0x3f, 0x80, 0x3f } },
        });
        write_text(w8_dir / "generation_config.json", "{}");
        write_text(w8_dir / "tokenizer.json", "{}");
        config["quantization_config"] = llama_safetensors_json::parse(w8a8_config).at("quantization_config");
        config["quantization_config"]["config_groups"]["int8"]["targets"][0] = channel_module;
        config["text_config"]["linear_num_key_heads"]   = 2;
        config["text_config"]["linear_num_value_heads"] = 4;
        config["text_config"]["linear_key_head_dim"]    = 32;
        config["text_config"]["linear_value_head_dim"]  = 32;
        llama_safetensors_qwen35_importer w8_importer(w8_dir, config);
        ggml_type w8_type;
        std::array<int64_t, GGML_MAX_DIMS> w8_ne;
        require(w8_importer.describe("blk.0.ssm_out.weight", w8_type, w8_ne) &&
                    w8_type == GGML_TYPE_I8 && w8_ne[0] == 128 && w8_ne[1] == 2,
                "Qwen3.5 W8A8 output projection has the wrong target contract");
        const std::vector<uint8_t> transformed = w8_importer.materialize(
            "blk.0.ssm_out.weight", w8_type, 2 * 128);
        const std::array<size_t, 4> expected_blocks = { 0, 2, 1, 3 };
        for (size_t row = 0; row < 2; ++row) {
            for (size_t block = 0; block < 4; ++block) {
                const uint8_t expected_code = static_cast<uint8_t>(10 * row + expected_blocks[block] + 1);
                const uint8_t * codes = transformed.data() + row * 128 + block * 32;
                require(std::all_of(codes, codes + 32,
                                    [&](uint8_t value) { return value == expected_code; }),
                        "Qwen3.5 W8A8 output projection permuted the wrong quant block");
            }
        }

        const std::filesystem::path awq_dir = dir.path / "awq-row-transform";
        const std::string awq_module = "model.language_model.layers.0.linear_attn.in_proj_z";
        write_single_shard_model(awq_dir, {
            { awq_module + ".qweight", "I32", { 128, 1 }, std::vector<uint8_t>(128 * sizeof(uint32_t)) },
            { awq_module + ".qzeros",  "I32", { 1, 1 },   i32_bytes({ 1 })                              },
            { awq_module + ".scales",  "F16", { 1, 8 },   std::vector<uint8_t>(8 * sizeof(uint16_t), 1) },
        });
        write_text(awq_dir / "generation_config.json", "{}");
        write_text(awq_dir / "tokenizer.json", "{}");
        config["quantization_config"] = llama_safetensors_json::parse(awq_config).at("quantization_config");
        llama_safetensors_qwen35_importer awq_importer(awq_dir, config);
        require_rejected(
            [&] {
                ggml_type type;
                std::array<int64_t, GGML_MAX_DIMS> ne;
                (void) awq_importer.describe("blk.0.attn_gate.weight", type, ne);
            },
            "Qwen3.5 accepted an AWQ row transform after repacking");

        const auto packed_config_json = llama_safetensors_json::parse(packed_int_config);
        const auto make_packed_qwen_config = [&](const std::string & module) {
            llama_safetensors_json result = {
                { "model_type", "qwen3_5" },
                { "text_config",
                 {
                      { "model_type", "qwen3_5_text" },
                      { "num_hidden_layers", 24 },
                      { "mtp_num_hidden_layers", 1 },
                      { "linear_num_key_heads", 2 },
                      { "linear_num_value_heads", 4 },
                      { "linear_key_head_dim", 128 },
                      { "linear_value_head_dim", 128 },
                  } },
            };
            result["quantization_config"] = packed_config_json.at("quantization_config");
            result["quantization_config"]["config_groups"]["int4"]["targets"][0] = module;
            return result;
        };
        const auto write_packed_int4 = [&](const std::filesystem::path & path,
                                           const std::string & module,
                                           size_t rows,
                                           size_t cols,
                                           const std::vector<uint8_t> & row_group_codes) {
            const size_t groups = cols / 32;
            std::vector<uint32_t> weights(rows * cols / 8);
            std::vector<uint32_t> zeros(((rows + 7) / 8) * groups);
            std::vector<uint8_t> scales(rows * groups * sizeof(uint16_t));
            for (size_t row = 0; row < rows; ++row) {
                for (size_t group = 0; group < groups; ++group) {
                    const uint8_t code = row_group_codes[row * groups + group];
                    for (size_t lane = 0; lane < 32; ++lane) {
                        const size_t col = group * 32 + lane;
                        weights[row * (cols / 8) + col / 8] |= uint32_t(code) << (4 * (col % 8));
                    }
                    zeros[(row / 8) * groups + group] |= uint32_t(8) << (4 * (row % 8));
                    store_bf16(scales, row * groups + group, 1.0f);
                }
            }
            write_single_shard_model(path, {
                { module + ".weight_packed",     "I32",  { rows, cols / 8 },         i32_bytes(weights) },
                { module + ".weight_scale",      "BF16", { rows, groups },           scales             },
                { module + ".weight_zero_point", "I32",  { (rows + 7) / 8, groups }, i32_bytes(zeros)   },
                { module + ".weight_shape",      "I64",  { 2 }, i64_bytes({ int64_t(rows), int64_t(cols) }) },
            });
            write_text(path / "generation_config.json", "{}");
            write_text(path / "tokenizer.json", "{}");
        };

        const std::filesystem::path packed_rows_dir = dir.path / "packed-int4-row-transform";
        std::vector<uint8_t> row_codes(512 * 4);
        for (size_t row = 0; row < 512; ++row) {
            for (size_t group = 0; group < 4; ++group) {
                row_codes[row * 4 + group] = 3 * (row / 128);
            }
        }
        write_packed_int4(packed_rows_dir, awq_module, 512, 128, row_codes);
        llama_safetensors_qwen35_importer packed_rows_importer(
            packed_rows_dir, make_packed_qwen_config(awq_module));
        const std::vector<uint8_t> packed_rows =
            packed_rows_importer.materialize("blk.0.attn_gate.weight", GGML_TYPE_Q4_A32, 512 * 74);
        constexpr std::array<size_t, 4> head_order = { 0, 2, 1, 3 };
        for (size_t dst_head = 0; dst_head < head_order.size(); ++dst_head) {
            std::array<uint8_t, 128> codes;
            codes.fill(3 * head_order[dst_head]);
            require_q4_a32_block(
                packed_rows.data() + dst_head * 128 * 74,
                { 1.0f, 1.0f, 1.0f, 1.0f }, { 8, 8, 8, 8 }, codes,
                "Qwen3.5 packed INT4 row transform used packed-source geometry");
        }

        const std::filesystem::path packed_cols_dir = dir.path / "packed-int4-column-transform";
        std::vector<uint8_t> column_codes(2 * 16);
        for (size_t row = 0; row < 2; ++row) {
            for (size_t group = 0; group < 16; ++group) {
                column_codes[row * 16 + group] = 3 * (group / 4);
            }
        }
        write_packed_int4(packed_cols_dir, channel_module, 2, 512, column_codes);
        llama_safetensors_qwen35_importer packed_cols_importer(
            packed_cols_dir, make_packed_qwen_config(channel_module));
        const std::vector<uint8_t> packed_cols =
            packed_cols_importer.materialize("blk.0.ssm_out.weight", GGML_TYPE_Q4_A32, 2 * 4 * 74);
        for (size_t dst_group = 0; dst_group < head_order.size(); ++dst_group) {
            std::array<uint8_t, 128> codes;
            codes.fill(3 * head_order[dst_group]);
            require_q4_a32_block(
                packed_cols.data() + dst_group * 74,
                { 1.0f, 1.0f, 1.0f, 1.0f }, { 8, 8, 8, 8 }, codes,
                "Qwen3.5 packed INT4 column transform split or reordered a quant block incorrectly");
        }
    }

    // Metadata and tokenizer conversion are shared by architecture importers.
    // Keep their strict parsing and GGUF representation covered independently
    // of the full Qwen model fixture.
    {
        const llama_safetensors_json generation = {
            { "top_k",       7     },
            { "top_p",       0.75f },
            { "temperature", 0.5f  },
        };
        const llama_safetensors_json tokenizer = {
            { "model",
             {
                  { "vocab", { { "a", 0 }, { "b", 1 } } },
                  { "merges", { llama_safetensors_json::array({ "a", "b" }), "a b" } },
              } },
            { "added_tokens",
             {
                  { { "id", 2 }, { "content", "<|pad|>" }, { "special", true } },
                  { { "id", 3 }, { "content", "plain-added" }, { "special", false } },
              } },
        };

        llama_safetensors_metadata_sink sink;
        llama_safetensors_emit_sampling_defaults(sink, generation);
        const auto rope = llama_safetensors_parse_rope(
            llama_safetensors_json{
                { "rope_theta",            1000000.0f  },
                { "partial_rotary_factor", 0.5f        },
                { "mrope_section",         { 8, 8, 0 } },
        },
            { 1, 2, 3, 4 }, 0.25f);
        require(rope.theta == 1000000.0f && rope.partial_rotary_factor == 0.5f &&
                    rope.mrope_sections == std::array<int32_t, 4>{ 8, 8, 0, 4 },
                "generic RoPE parsing is wrong");
        llama_safetensors_emit_bpe_tokenizer(sink, tokenizer, { "fixture", 4, 0, 1, std::string("<|pad|>"), true, {} },
                                             std::string("{{ messages }}"));

        gguf_context * metadata = sink.release();
        const int64_t  top_k    = gguf_find_key(metadata, "general.sampling.top_k");
        const int64_t  top_p    = gguf_find_key(metadata, "general.sampling.top_p");
        const int64_t  temp     = gguf_find_key(metadata, "general.sampling.temp");
        const int64_t  tokens   = gguf_find_key(metadata, "tokenizer.ggml.tokens");
        const int64_t  types    = gguf_find_key(metadata, "tokenizer.ggml.token_type");
        const int64_t  pad      = gguf_find_key(metadata, "tokenizer.ggml.padding_token_id");
        const int64_t  chat     = gguf_find_key(metadata, "tokenizer.chat_template");
        require(top_k >= 0 && gguf_get_val_i32(metadata, top_k) == 7 && top_p >= 0 &&
                    gguf_get_val_f32(metadata, top_p) == 0.75f && temp >= 0 && gguf_get_val_f32(metadata, temp) == 0.5f,
                "generic sampling metadata is wrong");
        require(tokens >= 0 && gguf_get_arr_n(metadata, tokens) == 4 &&
                    std::string(gguf_get_arr_str(metadata, tokens, 2)) == "<|pad|>" && types >= 0 &&
                    static_cast<const int32_t *>(gguf_get_arr_data(metadata, types))[2] == 3 &&
                    static_cast<const int32_t *>(gguf_get_arr_data(metadata, types))[3] == 4 && pad >= 0 &&
                    gguf_get_val_u32(metadata, pad) == 2 && chat >= 0 &&
                    std::string(gguf_get_val_str(metadata, chat)) == "{{ messages }}",
                "generic BPE tokenizer metadata is wrong");
        gguf_free(metadata);

        std::vector<uint8_t> spm;
        const auto append_piece = [&](std::string_view token, float score, uint8_t type) {
            std::vector<uint8_t> piece = { 0x0a, static_cast<uint8_t>(token.size()) };
            piece.insert(piece.end(), token.begin(), token.end());
            piece.push_back(0x15);
            const auto * score_bytes = reinterpret_cast<const uint8_t *>(&score);
            piece.insert(piece.end(), score_bytes, score_bytes + sizeof(score));
            piece.push_back(0x18);
            piece.push_back(type);
            spm.push_back(0x0a);
            spm.push_back(static_cast<uint8_t>(piece.size()));
            spm.insert(spm.end(), piece.begin(), piece.end());
        };
        append_piece("<unk>", 0.0f, LLAMA_TOKEN_TYPE_UNKNOWN);
        append_piece("<s>", 0.0f, LLAMA_TOKEN_TYPE_CONTROL);
        append_piece("▁hello", -1.25f, LLAMA_TOKEN_TYPE_NORMAL);
        const auto spm_path = dir.path / "tokenizer.model";
        write_bytes(spm_path, spm);
        llama_safetensors_metadata_sink spm_sink;
        llama_safetensors_emit_spm_tokenizer(
            spm_sink, spm_path, 3, 1, 2, 2, true, std::string("{{ spm }}"));
        gguf_context * spm_metadata = spm_sink.release();
        const int64_t spm_model  = gguf_find_key(spm_metadata, "tokenizer.ggml.model");
        const int64_t spm_tokens = gguf_find_key(spm_metadata, "tokenizer.ggml.tokens");
        const int64_t spm_scores = gguf_find_key(spm_metadata, "tokenizer.ggml.scores");
        const int64_t spm_types  = gguf_find_key(spm_metadata, "tokenizer.ggml.token_type");
        require(spm_model >= 0 && std::string(gguf_get_val_str(spm_metadata, spm_model)) == "llama" &&
                    spm_tokens >= 0 && std::string(gguf_get_arr_str(spm_metadata, spm_tokens, 2)) == "▁hello" &&
                    spm_scores >= 0 && static_cast<const float *>(gguf_get_arr_data(spm_metadata, spm_scores))[2] == -1.25f &&
                    spm_types >= 0 && static_cast<const int32_t *>(gguf_get_arr_data(spm_metadata, spm_types))[0] ==
                        LLAMA_TOKEN_TYPE_UNKNOWN,
                "generic SentencePiece tokenizer metadata is wrong");
        gguf_free(spm_metadata);

        require_rejected(
            [&] {
                (void) llama_safetensors_parse_rope(
                    llama_safetensors_json{
                        { "rope_theta",    1.0f              },
                        { "mrope_section", { 1, 2, 3, 4, 5 } },
                },
                    { 0, 0, 0, 0 }, 1.0f);
            },
            "oversized mrope_section was accepted");
        require_rejected(
            [&] { (void) llama_safetensors_first_token_id(llama_safetensors_json::array(), "eos_token_id"); },
            "empty token-id array was accepted");
    }

    // Quant adapters are architecture-independent contracts. Exercise them
    // directly so an importer cannot accidentally make a correctness-critical
    // scale optional while rearranging canonical names.
    {
        const auto path = dir.path / "fp8-channel";
        write_single_shard_model(path, {
                                           { "module.weight",       "F8_E4M3", { 2, 32 }, std::vector<uint8_t>(64) },
                                           { "module.weight_scale", "BF16",    { 2 },     std::vector<uint8_t>(4)  },
        });
        write_text(path / "config.json", fp8_channel_config);
        const auto registry = llama_safetensors_registry::load(path);

        llama_safetensors_quant_adapters incomplete(llama_safetensors_read_json(path / "config.json"), registry);
        const auto                       weight = incomplete.bind("module", llama_safetensors_quant_role::WEIGHT);
        require(weight.has_value() && weight->target_type == GGML_TYPE_F8_E4M3, "FP8 channel weight binding is wrong");
        incomplete.consume(*weight);
        require_rejected([&] { incomplete.validate_complete(); }, "FP8 channel weight was accepted without its scale");

        llama_safetensors_quant_adapters complete(llama_safetensors_read_json(path / "config.json"), registry);
        const auto complete_weight = complete.bind("module", llama_safetensors_quant_role::WEIGHT);
        const auto scale           = complete.bind("module", llama_safetensors_quant_role::WEIGHT_SCALE);
        const auto input           = complete.bind("module", llama_safetensors_quant_role::INPUT_SCALE);
        require(complete_weight.has_value() && scale.has_value() && scale->target_type == GGML_TYPE_BF16 &&
                    scale->target_shape == std::vector<int64_t>({ 2 }) &&
                    input.has_value() && input->target_type == GGML_TYPE_I32 &&
                    input->materialization == llama_safetensors_quant_materialization::DYNAMIC_FP8_MARKER,
                "FP8 channel scale binding is wrong");
        complete.consume(*complete_weight);
        complete.consume(*scale);
        complete.consume(*input);
        complete.validate_complete();
    }
    {
        const auto path = dir.path / "fp8-channel-missing-scale";
        write_single_shard_model(path, {
                                           { "module.weight", "F8_E4M3", { 2, 32 }, std::vector<uint8_t>(64) },
        });
        write_text(path / "config.json", fp8_channel_config);
        const auto registry = llama_safetensors_registry::load(path);
        require_rejected([&] { (void) llama_safetensors_quant_adapters(llama_safetensors_read_json(path / "config.json"), registry); },
                         "FP8 channel contract accepted a missing scale");
    }
    {
        const auto path = dir.path / "fp8-w8a16";
        write_single_shard_model(path, {
            { "module.weight",       "F8_E4M3", { 2, 32 }, std::vector<uint8_t>(64) },
            { "module.weight_scale", "BF16",    { 2 },     std::vector<uint8_t>(4) },
        });
        llama_safetensors_json config = llama_safetensors_json::parse(fp8_channel_config);
        config["quantization_config"]["config_groups"]["fp8"]["input_activations"] = nullptr;
        write_text(path / "config.json", config.dump());
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(config, registry);
        require(adapters.bind("module", llama_safetensors_quant_role::WEIGHT).has_value() &&
                    adapters.bind("module", llama_safetensors_quant_role::WEIGHT_SCALE).has_value() &&
                    !adapters.bind("module", llama_safetensors_quant_role::INPUT_SCALE).has_value(),
                "FP8 W8A16 binding unexpectedly quantized its activations");
    }
    {
        const auto path = dir.path / "legacy-fp8-static";
        std::vector<uint8_t> codes(64);
        for (size_t i = 0; i < codes.size(); ++i) {
            codes[i] = uint8_t(i);
        }
        write_single_shard_model(path, {
            { "module.weight",       "I8",  { 2, 32 }, codes            },
            { "module.weight_scale", "F32", {},        f32_bytes(0.5f)  },
            { "module.in_scale",     "F32", {},        f32_bytes(0.25f) },
        });
        write_text(path / "config.json", legacy_fp8_static_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(
            llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        const auto scale  = adapters.bind("module", llama_safetensors_quant_role::WEIGHT_SCALE);
        const auto input  = adapters.bind("module", llama_safetensors_quant_role::INPUT_SCALE);
        require(weight.has_value() && weight->target_type == GGML_TYPE_F8_E4M3 &&
                    adapters.read(*weight) == codes && scale.has_value() &&
                    scale->materialization == llama_safetensors_quant_materialization::BROADCAST_BF16_SCALAR &&
                    input.has_value() && input->primary == "module.in_scale" &&
                    input->target_type == GGML_TYPE_F32,
                "legacy static FP8 binding is wrong");
        adapters.consume(*weight);
        adapters.consume(*scale);
        adapters.consume(*input);
        adapters.validate_complete();
    }
    {
        const auto path = dir.path / "fp8-channel-wrong-scale-type";
        write_single_shard_model(path, {
                                           { "module.weight",       "F8_E4M3", { 2, 32 }, std::vector<uint8_t>(64) },
                                           { "module.weight_scale", "F16",     { 2 },     std::vector<uint8_t>(4)  },
        });
        write_text(path / "config.json", fp8_channel_config);
        const auto registry = llama_safetensors_registry::load(path);
        require_rejected([&] { (void) llama_safetensors_quant_adapters(llama_safetensors_read_json(path / "config.json"), registry); },
                         "FP8 channel contract accepted the wrong scale dtype");
    }
    {
        const auto path = dir.path / "fp8-channel-wrong-weight-type";
        write_single_shard_model(path, {
                                           { "module.weight",       "BF16", { 2, 32 }, std::vector<uint8_t>(128) },
                                           { "module.weight_scale", "BF16", { 2 },     std::vector<uint8_t>(4)   },
        });
        write_text(path / "config.json", fp8_channel_config);
        const auto registry = llama_safetensors_registry::load(path);
        require_rejected([&] { (void) llama_safetensors_quant_adapters(llama_safetensors_read_json(path / "config.json"), registry); },
                         "FP8 channel contract accepted a BF16 weight");
    }
    {
        const auto path = dir.path / "fp8-static-tensor";
        write_single_shard_model(path, {
            { "module.weight",       "F8_E4M3", { 3, 32 }, std::vector<uint8_t>(96, 0x38) },
            { "module.weight_scale", "BF16",    { 1 },     { 0xc0, 0x3f }                 },
            { "module.input_scale",  "BF16",    { 1 },     { 0x00, 0x3f }                 },
        });
        write_text(path / "config.json", fp8_static_tensor_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(
            llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        const auto scale  = adapters.bind("module", llama_safetensors_quant_role::WEIGHT_SCALE);
        const auto input  = adapters.bind("module", llama_safetensors_quant_role::INPUT_SCALE);
        require(weight.has_value() && weight->target_type == GGML_TYPE_F8_E4M3 &&
                    scale.has_value() && scale->target_type == GGML_TYPE_BF16 &&
                    scale->target_shape == std::vector<int64_t>({ 3 }) &&
                    scale->materialization == llama_safetensors_quant_materialization::BROADCAST_BF16_SCALAR &&
                    input.has_value() && input->target_type == GGML_TYPE_F32 &&
                    input->target_shape == std::vector<int64_t>({ 1 }) &&
                    adapters.summary().fp8_tensor == 1,
                "static tensor-scale FP8 binding is wrong");
        require(adapters.read(*scale) == std::vector<uint8_t>({ 0xc0, 0x3f, 0xc0, 0x3f, 0xc0, 0x3f }),
                "static FP8 weight scale was not broadcast exactly");
        const std::vector<uint8_t> input_bytes = adapters.finalize(*input, adapters.read(*input));
        float input_value;
        std::memcpy(&input_value, input_bytes.data(), sizeof(input_value));
        require(input_bytes.size() == sizeof(float) && input_value == 0.5f,
                "static FP8 input scale was not converted exactly");
        adapters.consume(*weight);
        adapters.consume(*scale);
        adapters.consume(*input);
        adapters.validate_complete();
    }
    {
        const auto path = dir.path / "fp8-static-tensor-missing-input";
        write_single_shard_model(path, {
            { "module.weight",       "F8_E4M3", { 1, 32 }, std::vector<uint8_t>(32) },
            { "module.weight_scale", "BF16",    { 1 },     { 0x80, 0x3f }           },
        });
        write_text(path / "config.json", fp8_static_tensor_config);
        const auto registry = llama_safetensors_registry::load(path);
        require_rejected([&] {
            (void) llama_safetensors_quant_adapters(
                llama_safetensors_read_json(path / "config.json"), registry);
        }, "static FP8 contract accepted a missing input scale");
    }
    {
        const auto path = dir.path / "fp8-block";
        write_single_shard_model(path,
                                 {
                                     { "module.weight",           "F8_E4M3", { 128, 128 }, std::vector<uint8_t>(128 * 128) },
                                     { "module.weight_scale_inv", "BF16",    { 1, 1 },     std::vector<uint8_t>(2)         },
        });
        write_text(path / "config.json", fp8_block_config);
        const auto                       registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(llama_safetensors_read_json(path / "config.json"), registry);
        const auto                       weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        const auto                       scale  = adapters.bind("module", llama_safetensors_quant_role::WEIGHT_SCALE);
        require(weight.has_value() && scale.has_value() && scale->target_type == GGML_TYPE_F32 &&
                    scale->materialization == llama_safetensors_quant_materialization::FP8_BLOCK_SCALE,
                "FP8 block binding is wrong");
        adapters.consume(*weight);
        adapters.consume(*scale);
        adapters.validate_complete();
        require(adapters.finalize(*scale, adapters.read(*scale)).size() == sizeof(float),
                "FP8 block scale conversion has the wrong size");

        llama_safetensors_quant_adapters inferred(
            llama_safetensors_json::parse(fp8_block_inferred_format_config), registry);
        const auto inferred_weight = inferred.bind("module", llama_safetensors_quant_role::WEIGHT);
        const auto inferred_scale  = inferred.bind("module", llama_safetensors_quant_role::WEIGHT_SCALE);
        require(inferred_weight.has_value() && inferred_scale.has_value() &&
                    inferred_scale->materialization == llama_safetensors_quant_materialization::FP8_BLOCK_SCALE,
                "FP8 block binding did not infer E4M3 from the stored weight dtype");
        inferred.consume(*inferred_weight);
        inferred.consume(*inferred_scale);
        inferred.validate_complete();
    }
    {
        const auto path = dir.path / "fp8-block-wrong-grid";
        write_single_shard_model(path,
                                 {
                                     { "module.weight",           "F8_E4M3", { 128, 128 }, std::vector<uint8_t>(128 * 128) },
                                     { "module.weight_scale_inv", "BF16",    { 1, 2 },     std::vector<uint8_t>(4)         },
        });
        write_text(path / "config.json", fp8_block_config);
        const auto registry = llama_safetensors_registry::load(path);
        require_rejected([&] { (void) llama_safetensors_quant_adapters(llama_safetensors_read_json(path / "config.json"), registry); },
                         "FP8 block contract accepted the wrong scale grid");
    }
    {
        const auto path = dir.path / "fp8-embedding";
        write_single_shard_model(path, {
            { "model.language_model.module.embedding.shard_0.weight", "F8_E4M3", { 2, 32 }, std::vector<uint8_t>(64) },
            { "model.language_model.module.embedding.weight_scale",  "BF16",    { 1 },     { 0x80, 0x3f }           },
            { "linear.weight",                                    "F8_E4M3", { 128, 128 }, std::vector<uint8_t>(16384) },
            { "linear.weight_scale_inv",                          "BF16",     { 1, 1 },     std::vector<uint8_t>(2)     },
        });
        write_text(path / "config.json", fp8_block_embedding_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(
            llama_safetensors_read_json(path / "config.json"), registry);
        require(!adapters.applies("model.language_model.module.embedding.shard_0"),
                "raw FP8 embedding shard was misclassified as a block-scaled linear");
        const auto weight = adapters.bind("linear", llama_safetensors_quant_role::WEIGHT);
        const auto scale  = adapters.bind("linear", llama_safetensors_quant_role::WEIGHT_SCALE);
        require(weight.has_value() && scale.has_value(), "FP8 embedding exception disabled ordinary block FP8");
        adapters.consume(*weight);
        adapters.consume(*scale);
        adapters.validate_complete();
    }
    {
        const auto path = dir.path / "deepseek4-mixed-fp8";
        const std::string expert = "layers.0.ffn.experts.0.w1";
        write_single_shard_model(path, {
            { "module.weight",       "F8_E4M3", { 256, 256 }, std::vector<uint8_t>(256 * 256) },
            { "module.scale",        "F8_E8M0", { 2, 2 },     { 127, 128, 129, 130 }             },
            { expert + ".weight",    "I8",      { 2, 16 },    std::vector<uint8_t>(32)           },
            { expert + ".scale",     "F8_E8M0", { 2, 1 },     { 127, 128 }                       },
        });
        write_text(path / "config.json", deepseek4_mixed_fp8_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(
            llama_safetensors_read_json(path / "config.json"), registry);

        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        const auto scale = adapters.bind("module", llama_safetensors_quant_role::WEIGHT_SCALE);
        require(weight.has_value() && scale.has_value() &&
                    scale->materialization == llama_safetensors_quant_materialization::FP8_BLOCK_SCALE_E8M0,
                "DeepSeek block-FP8 E8M0 binding is wrong");
        const auto converted = adapters.finalize(*scale, adapters.read(*scale));
        std::array<float, 4> values{};
        std::memcpy(values.data(), converted.data(), converted.size());
        require(values == std::array<float, 4>({ 1.0f, 4.0f, 2.0f, 8.0f }),
                "DeepSeek block-FP8 E8M0 grid was not transposed and decoded exactly");

        const auto expert_weight = adapters.bind(expert, llama_safetensors_quant_role::WEIGHT);
        const auto expert_input = adapters.bind(expert, llama_safetensors_quant_role::INPUT_SCALE);
        require(expert_weight.has_value() && expert_weight->target_type == GGML_TYPE_MXFP4 &&
                    expert_weight->target_shape == std::vector<int64_t>({ 32, 2 }) &&
                    adapters.read(*expert_weight).size() == 34 && expert_input.has_value() &&
                    expert_input->materialization == llama_safetensors_quant_materialization::DYNAMIC_MXFP4_MARKER,
                "DeepSeek routed-expert MXFP4 binding is wrong");
        adapters.consume(*weight);
        adapters.consume(*scale);
        adapters.consume(*expert_weight);
        adapters.consume(*expert_input);
        adapters.validate_complete();
    }
    {
        const auto path = dir.path / "modelopt-fp8-pb-wo";
        std::vector<uint8_t> scales(4 * sizeof(float));
        const std::array<float, 4> source_scales = { 1.0f, 2.0f, 3.0f, 4.0f };
        std::memcpy(scales.data(), source_scales.data(), scales.size());
        write_single_shard_model(path, {
            { "module.weight",       "F8_E4M3", { 256, 256 },       std::vector<uint8_t>(256 * 256) },
            { "module.weight_scale", "F32",     { 2, 1, 2, 1 },    scales                         },
        });
        write_text(path / "config.json", modelopt_fp8_pb_wo_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        const auto scale = adapters.bind("module", llama_safetensors_quant_role::WEIGHT_SCALE);
        require(weight.has_value() && weight->target_type == GGML_TYPE_F8_E4M3 &&
                    scale.has_value() && scale->target_type == GGML_TYPE_F32 &&
                    scale->target_shape == std::vector<int64_t>({ 2, 2 }) &&
                    scale->materialization == llama_safetensors_quant_materialization::FP8_BLOCK_SCALE_MODELOPT,
                "ModelOpt FP8_PB_WO binding is wrong");
        const std::vector<uint8_t> target = adapters.finalize(*scale, adapters.read(*scale));
        std::array<float, 4> target_scales;
        std::memcpy(target_scales.data(), target.data(), target.size());
        require(target_scales == std::array<float, 4>({ 1.0f, 3.0f, 2.0f, 4.0f }),
                "ModelOpt FP8_PB_WO block grid was not transposed to GGML layout");
        adapters.consume(*weight);
        adapters.consume(*scale);
        adapters.validate_complete();

    }
    {
        const auto path = dir.path / "modelopt-fp8-per-channel";
        const std::array<float, 2> source_scales = { 0.5f, 2.0f };
        std::vector<uint8_t> scales(sizeof(source_scales));
        std::memcpy(scales.data(), source_scales.data(), scales.size());
        write_single_shard_model(path, {
            { "module.weight",       "F8_E4M3", { 2, 128 }, std::vector<uint8_t>(256) },
            { "module.weight_scale", "F32",     { 2, 1 },   scales                    },
        });
        write_text(path / "config.json", modelopt_fp8_channel_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        const auto scale = adapters.bind("module", llama_safetensors_quant_role::WEIGHT_SCALE);
        const auto input = adapters.bind("module", llama_safetensors_quant_role::INPUT_SCALE);
        require(weight.has_value() && weight->target_type == GGML_TYPE_F8_E4M3 &&
                    scale.has_value() && scale->target_type == GGML_TYPE_BF16 &&
                    scale->target_shape == std::vector<int64_t>({ 2 }) &&
                    scale->materialization == llama_safetensors_quant_materialization::POSITIVE_F32_TO_BF16 &&
                    input.has_value() &&
                    input->materialization == llama_safetensors_quant_materialization::DYNAMIC_FP8_MARKER,
                "ModelOpt per-channel FP8 binding is wrong");
        const std::vector<uint8_t> target = adapters.read(*scale);
        std::array<ggml_bf16_t, 2> converted;
        std::memcpy(converted.data(), target.data(), target.size());
        require(ggml_bf16_to_fp32(converted[0]) == 0.5f && ggml_bf16_to_fp32(converted[1]) == 2.0f,
                "ModelOpt per-channel FP8 scales were not converted to BF16");
        adapters.consume(*weight);
        adapters.consume(*scale);
        adapters.consume(*input);
        adapters.validate_complete();
    }
    {
        const auto path = dir.path / "fbgemm-fp8-per-channel";
        const std::array<float, 2> source_scales = { 0.5f, 2.0f };
        std::vector<uint8_t> scales(sizeof(source_scales));
        std::memcpy(scales.data(), source_scales.data(), scales.size());
        write_single_shard_model(path, {
            { "module.weight",       "F8_E4M3", { 2, 128 }, std::vector<uint8_t>(256) },
            { "module.weight_scale", "F32",     { 2, 1 },   scales                    },
            { "ignored.weight",      "BF16",    { 2, 128 }, std::vector<uint8_t>(512) },
        });
        write_text(path / "config.json", fbgemm_fp8_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        const auto scale = adapters.bind("module", llama_safetensors_quant_role::WEIGHT_SCALE);
        const auto input = adapters.bind("module", llama_safetensors_quant_role::INPUT_SCALE);
        require(weight.has_value() && weight->target_type == GGML_TYPE_F8_E4M3 &&
                    scale.has_value() && scale->target_type == GGML_TYPE_BF16 &&
                    scale->materialization == llama_safetensors_quant_materialization::POSITIVE_F32_TO_BF16 &&
                    input.has_value() && input->target_type == GGML_TYPE_I32 &&
                    input->materialization == llama_safetensors_quant_materialization::DYNAMIC_FP8_MARKER &&
                    !adapters.applies("ignored"),
                "FBGEMM FP8 binding is wrong");
        const std::vector<uint8_t> marker = adapters.read(*input);
        float upper_bound;
        std::memcpy(&upper_bound, marker.data(), sizeof(upper_bound));
        require(upper_bound == 1200.0f, "FBGEMM FP8 activation upper bound was not preserved");
        adapters.consume(*weight);
        adapters.consume(*scale);
        adapters.consume(*input);
        adapters.validate_complete();

        llama_safetensors_json invalid = llama_safetensors_json::parse(fbgemm_fp8_config);
        invalid["quantization_config"]["activation_scale_ub"] = 0.0f;
        require_rejected(
            [&] { (void) llama_safetensors_quant_config::from_json(invalid); },
            "FBGEMM FP8 accepted a non-positive activation upper bound");
    }
    {
        const auto path = dir.path / "modelopt-fp8-per-channel-invalid";
        const float source_scale = std::numeric_limits<float>::infinity();
        std::vector<uint8_t> scale_bytes(sizeof(source_scale));
        std::memcpy(scale_bytes.data(), &source_scale, sizeof(source_scale));
        write_single_shard_model(path, {
            { "module.weight",       "F8_E4M3", { 1, 128 }, std::vector<uint8_t>(128) },
            { "module.weight_scale", "F32",     { 1, 1 },   scale_bytes                },
        });
        write_text(path / "config.json", modelopt_fp8_channel_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(llama_safetensors_read_json(path / "config.json"), registry);
        const auto scale = adapters.bind("module", llama_safetensors_quant_role::WEIGHT_SCALE);
        require(scale.has_value(), "invalid ModelOpt per-channel scale did not bind");
        require_rejected([&] { (void) adapters.read(*scale); },
                         "ModelOpt per-channel FP8 accepted an infinite scale");
    }
    {
        const auto path = dir.path / "minimax-modelopt-mxfp8";
        write_single_shard_model(path, {
            { "module.weight",       "F8_E4M3", { 2, 32 }, std::vector<uint8_t>(64, 0x38) },
            { "module.weight_scale", "U8",      { 2, 1 },  { 127, 128 }                     },
        });
        write_text(path / "config.json", minimax_mxfp8_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        const auto scale = adapters.bind("module", llama_safetensors_quant_role::WEIGHT_SCALE);
        const auto input = adapters.bind("module", llama_safetensors_quant_role::INPUT_SCALE);
        require(weight.has_value() && weight->target_type == GGML_TYPE_F8_E4M3 &&
                    scale.has_value() && scale->target_type == GGML_TYPE_I8 &&
                    input.has_value() &&
                    input->materialization == llama_safetensors_quant_materialization::DYNAMIC_MXFP8_MARKER,
                "MiniMax-style ModelOpt MXFP8 binding is wrong");
        adapters.consume(*weight);
        adapters.consume(*scale);
        adapters.consume(*input);
        adapters.validate_complete();
    }
    {
        const auto path = dir.path / "w8a8";
        std::vector<uint8_t> weights(128);
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] = static_cast<uint8_t>(static_cast<int8_t>(static_cast<int>(i % 64) - 32));
        }
        write_single_shard_model(path, {
                                           { "module.weight",       "I8",   { 2, 64 }, weights                    },
                                           { "module.weight_scale", "F16", { 2, 1 },  { 0x00, 0x3c, 0x00, 0x40 } },
        });
        write_text(path / "config.json", w8a8_config);
        const auto                       registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(llama_safetensors_read_json(path / "config.json"), registry);
        const auto                       weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        const auto scale = adapters.bind("module", llama_safetensors_quant_role::WEIGHT_SCALE);
        require(weight.has_value() && weight->target_type == GGML_TYPE_I8 &&
                    weight->materialization == llama_safetensors_quant_materialization::RAW &&
                    scale.has_value() && scale->target_type == GGML_TYPE_F32 &&
                    adapters.file_type() == LLAMA_FTYPE_MOSTLY_Q8_0,
                "W8A8 binding is wrong");
        require(adapters.read(*weight) == weights,
                "W8A8 raw INT8 materialization changed source codes");
        const std::vector<uint8_t> scale_bytes = adapters.finalize(*scale, adapters.read(*scale));
        require(scale_bytes.size() == 2 * sizeof(float),
                "W8A8 scale conversion has the wrong size");
        float converted_scales[2];
        std::memcpy(converted_scales, scale_bytes.data(), sizeof(converted_scales));
        require(converted_scales[0] == 1.0f && converted_scales[1] == 2.0f,
                "W8A8 scale conversion changed source values");
        adapters.consume(*weight);
        adapters.consume(*scale);
        adapters.validate_complete();

        llama_safetensors_json compact = llama_safetensors_json::parse(w8a8_config);
        auto & compact_group = compact["quantization_config"]["config_groups"]["int8"];
        compact_group["weights"].erase("actorder");
        compact_group["input_activations"].erase("actorder");
        llama_safetensors_quant_adapters compact_adapters(compact, registry);
        require(compact_adapters.bind("module", llama_safetensors_quant_role::WEIGHT).has_value(),
                "current compact W8A8 schema was not recognized");
    }
    {
        const auto path = dir.path / "w8a8-wrong-scale-type";
        write_single_shard_model(path, {
                                           { "module.weight",       "I8",  { 2, 32 }, std::vector<uint8_t>(64) },
                                           { "module.weight_scale", "F32", { 2, 1 },  std::vector<uint8_t>(8)  },
        });
        write_text(path / "config.json", w8a8_config);
        const auto registry = llama_safetensors_registry::load(path);
        require_rejected([&] { (void) llama_safetensors_quant_adapters(llama_safetensors_read_json(path / "config.json"), registry); },
                         "W8A8 contract accepted the wrong scale dtype");
    }
    {
        const auto path = dir.path / "w8a8-static";
        write_single_shard_model(path, {
            { "module.weight",       "I8",   { 2, 32 }, std::vector<uint8_t>(64) },
            { "module.weight_scale", "BF16", { 2, 1 },  { 0x80, 0x3f, 0x00, 0x40 } },
            { "module.input_scale",  "F16",  { 1 },     { 0x00, 0x38 } },
        });
        write_text(path / "config.json", w8a8_static_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(
            llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        const auto scale  = adapters.bind("module", llama_safetensors_quant_role::WEIGHT_SCALE);
        const auto input  = adapters.bind("module", llama_safetensors_quant_role::INPUT_SCALE);
        require(weight.has_value() && scale.has_value() && input.has_value() &&
                    input->target_type == GGML_TYPE_F32 && input->target_shape == std::vector<int64_t>({ 1 }),
                "static W8A8 binding is incomplete");
        const std::vector<uint8_t> input_bytes = adapters.finalize(*input, adapters.read(*input));
        float input_value;
        std::memcpy(&input_value, input_bytes.data(), sizeof(input_value));
        require(input_value == 0.5f, "static W8A8 input scale was not converted exactly");
        adapters.consume(*weight);
        adapters.consume(*scale);
        adapters.consume(*input);
        adapters.validate_complete();
    }
    {
        const auto path = dir.path / "w8a8-static-missing-input";
        write_single_shard_model(path, {
            { "module.weight",       "I8",   { 2, 32 }, std::vector<uint8_t>(64) },
            { "module.weight_scale", "BF16", { 2, 1 }, std::vector<uint8_t>(4, 0x3f) },
        });
        write_text(path / "config.json", w8a8_static_config);
        const auto registry = llama_safetensors_registry::load(path);
        require_rejected([&] {
            (void) llama_safetensors_quant_adapters(
                llama_safetensors_read_json(path / "config.json"), registry);
        }, "static W8A8 contract accepted a missing input scale");
    }
    {
        const auto path = dir.path / "w8a8-static-asymmetric";
        write_single_shard_model(path, {
            { "module.weight",           "I8",   { 2, 32 }, std::vector<uint8_t>(64) },
            { "module.weight_scale",     "BF16", { 2, 1 },  { 0x80, 0x3f, 0x00, 0x40 } },
            { "module.input_scale",      "F16",  { 1 },     { 0x00, 0x38 } },
            { "module.input_zero_point", "I8",   { 1 },     { uint8_t(int8_t(-17)) } },
        });
        write_text(path / "config.json", w8a8_static_asym_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(
            llama_safetensors_read_json(path / "config.json"), registry);
        const auto input = adapters.bind("module", llama_safetensors_quant_role::INPUT_SCALE);
        require(input.has_value() && input->target_type == GGML_TYPE_I64 &&
                    input->materialization ==
                        llama_safetensors_quant_materialization::STATIC_INT8_ASYM_PARAMS &&
                    input->auxiliaries == std::vector<std::string>({ "module.input_zero_point" }),
                "asymmetric static W8A8 binding is wrong");
        const std::vector<uint8_t> params = adapters.read(*input);
        float scale;
        int8_t zero_point;
        std::memcpy(&scale, params.data(), sizeof(scale));
        std::memcpy(&zero_point, params.data() + sizeof(scale), sizeof(zero_point));
        require(params.size() == sizeof(int64_t) && scale == 0.5f && zero_point == -17,
                "asymmetric static W8A8 parameters changed during materialization");
    }
    {
        const auto path = dir.path / "w8a8-static-asymmetric-missing-zero";
        write_single_shard_model(path, {
            { "module.weight",       "I8",   { 2, 32 }, std::vector<uint8_t>(64) },
            { "module.weight_scale", "BF16", { 2, 1 },  std::vector<uint8_t>(4) },
            { "module.input_scale",  "F16",  { 1 },     { 0x00, 0x38 } },
        });
        write_text(path / "config.json", w8a8_static_asym_config);
        const auto registry = llama_safetensors_registry::load(path);
        require_rejected([&] {
            (void) llama_safetensors_quant_adapters(
                llama_safetensors_read_json(path / "config.json"), registry);
        }, "asymmetric static W8A8 contract accepted a missing zero point");
    }
    for (const auto & invalid : std::vector<std::tuple<const char *, const char *, uint16_t>> {
             { "f16-negative", "F16",  0xbc00 },
             { "f16-inf",      "F16",  0x7c00 },
             { "f16-nan",      "F16",  0x7e00 },
             { "bf16-negative", "BF16", 0xbf80 },
             { "bf16-inf",      "BF16", 0x7f80 },
             { "bf16-nan",      "BF16", 0x7fc0 },
         }) {
        const auto path = dir.path / (std::string("w8a8-") + std::get<0>(invalid));
        const uint16_t bits = std::get<2>(invalid);
        write_single_shard_model(path, {
            { "module.weight", "I8", { 1, 32 }, std::vector<uint8_t>(32) },
            { "module.weight_scale", std::get<1>(invalid), { 1, 1 },
                { uint8_t(bits), uint8_t(bits >> 8) } },
        });
        write_text(path / "config.json", w8a8_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(
            llama_safetensors_read_json(path / "config.json"), registry);
        const auto scale = adapters.bind("module", llama_safetensors_quant_role::WEIGHT_SCALE);
        require(scale.has_value(), "W8A8 invalid-scale fixture did not bind");
        require_rejected([&] { (void) adapters.finalize(*scale, adapters.read(*scale)); },
                         "W8A8 accepted a non-finite or negative channel scale");
    }
    {
        const auto path = dir.path / "w8a8-wrong-weight-type";
        write_single_shard_model(path, {
                                           { "module.weight",       "BF16", { 2, 32 }, std::vector<uint8_t>(128) },
                                           { "module.weight_scale", "BF16", { 2, 1 },  std::vector<uint8_t>(4)   },
        });
        write_text(path / "config.json", w8a8_config);
        const auto registry = llama_safetensors_registry::load(path);
        require_rejected([&] { (void) llama_safetensors_quant_adapters(llama_safetensors_read_json(path / "config.json"), registry); },
                         "W8A8 contract accepted a BF16 weight with an INT8 scale sidecar");
    }
    {
        const auto path = dir.path / "w4a8-group128";
        constexpr size_t rows = 2;
        constexpr size_t cols = 128;
        std::vector<uint8_t> weights(rows * cols);
        std::vector<uint8_t> scales(rows * sizeof(uint16_t));
        for (size_t row = 0; row < rows; ++row) {
            for (size_t col = 0; col < cols; ++col) {
                weights[row * cols + col] = uint8_t(int8_t((3 * row + col) % 16 - 8));
            }
            store_bf16(scales, row, 0.5f + 0.5f * row);
        }
        write_single_shard_model(path, {
            { "module.weight",       "I8",   { rows, cols }, weights },
            { "module.weight_scale", "BF16", { rows, 1 },   scales  },
        });
        write_text(path / "config.json", w4a8_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(
            llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        const auto input  = adapters.bind("module", llama_safetensors_quant_role::INPUT_SCALE);
        require(weight.has_value() && weight->target_type == GGML_TYPE_Q4_A32 &&
                    weight->target_shape == std::vector<int64_t>({ int64_t(cols), int64_t(rows) }) &&
                    weight->materialization == llama_safetensors_quant_materialization::INT4_GROUP_REPACK &&
                    input.has_value() && input->target_type == GGML_TYPE_I32 &&
                    input->materialization == llama_safetensors_quant_materialization::DYNAMIC_INT8_MARKER &&
                    adapters.file_type() == LLAMA_FTYPE_MOSTLY_Q4_1,
                "W4A8 group-128 binding is wrong");
        const std::vector<uint8_t> repacked = adapters.finalize(*weight, adapters.read(*weight));
        constexpr size_t block_size = 74;
        require(repacked.size() == rows * block_size, "W4A8 Q4_A32 repack has the wrong size");
        for (size_t row = 0; row < rows; ++row) {
            const uint8_t * block = repacked.data() + row * block_size;
            uint16_t expected_scale;
            std::memcpy(&expected_scale, scales.data() + row * sizeof(uint16_t), sizeof(expected_scale));
            for (size_t group = 0; group < 4; ++group) {
                uint16_t actual_scale;
                std::memcpy(&actual_scale, block + group * sizeof(uint16_t), sizeof(actual_scale));
                require(actual_scale == expected_scale, "W4A8 did not replicate its group-128 scale");
            }
            require(block[8] == 0x88 && block[9] == 0x88, "W4A8 zero point is wrong");
            for (size_t col = 0; col < cols; ++col) {
                const uint8_t packed = block[10 + col / 2];
                const uint8_t code = (packed >> (4 * (col % 2))) & 0x0f;
                require(code == uint8_t(int8_t(weights[row * cols + col]) + 8),
                        "W4A8 repack changed a signed INT4 code");
            }
        }
        adapters.consume(*weight);
        adapters.consume(*input);
        adapters.validate_complete();
    }
    {
        const auto path = dir.path / "w4a8-out-of-range";
        std::vector<uint8_t> weights(128);
        weights[37] = uint8_t(int8_t(8));
        write_single_shard_model(path, {
            { "module.weight",       "I8",   { 1, 128 }, weights },
            { "module.weight_scale", "BF16", { 1, 1 },   { 0x80, 0x3f } },
        });
        write_text(path / "config.json", w4a8_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(
            llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        require(weight.has_value(), "W4A8 invalid-code fixture did not bind");
        require_rejected([&] { (void) adapters.finalize(*weight, adapters.read(*weight)); },
                         "W4A8 accepted a weight value outside signed INT4");
    }
    {
        const auto path = dir.path / "packed-int4-g32";
        constexpr size_t rows = 9;
        constexpr size_t cols = 128;
        constexpr size_t groups = cols / 32;
        std::vector<uint32_t> packed_weight(rows * cols / 8);
        std::vector<uint32_t> packed_zero(((rows + 7) / 8) * groups);
        std::vector<uint8_t> scales(rows * groups * sizeof(uint16_t));
        for (size_t row = 0; row < rows; ++row) {
            for (size_t col = 0; col < cols; ++col) {
                packed_weight[row * (cols / 8) + col / 8] |=
                    uint32_t((3 * row + col) % 16) << (4 * (col % 8));
            }
            for (size_t group = 0; group < groups; ++group) {
                const uint8_t zero = 1 + (row + 2 * group) % 8;
                packed_zero[(row / 8) * groups + group] |= uint32_t(zero) << (4 * (row % 8));
                store_bf16(scales, row * groups + group, 0.5f + 0.25f * ((row + group) % 4));
            }
        }
        write_single_shard_model(path, {
            { "int4.weight_packed",     "I32",  { rows, cols / 8 },       i32_bytes(packed_weight)       },
            { "int4.weight_scale",      "BF16", { rows, groups },         scales                         },
            { "int4.weight_zero_point", "I32",  { (rows + 7) / 8, groups }, i32_bytes(packed_zero)       },
            { "int4.weight_shape",      "I64",  { 2 },                    i64_bytes({ int64_t(rows), int64_t(cols) }) },
        });
        write_text(path / "config.json", packed_int_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("int4", llama_safetensors_quant_role::WEIGHT);
        require(weight.has_value() && weight->target_type == GGML_TYPE_Q4_A32 &&
                    weight->materialization == llama_safetensors_quant_materialization::PACKED_INT4_REPACK &&
                    weight->target_shape == std::vector<int64_t>({ cols, rows }) &&
                    adapters.file_type() == LLAMA_FTYPE_MOSTLY_Q4_1,
                "compressed-tensors INT4 binding is wrong");
        const std::vector<uint8_t> repacked = adapters.read(*weight);
        constexpr size_t block_size = 74;
        require(repacked.size() == rows * (cols / 128) * block_size,
                "compressed-tensors INT4 exact repack has the wrong size");
        for (size_t row = 0; row < rows; ++row) {
            std::array<uint8_t, 128> codes;
            std::array<float, 4> expected_scales;
            std::array<uint8_t, 4> expected_zeros;
            for (size_t group = 0; group < groups; ++group) {
                expected_scales[group] = 0.5f + 0.25f * ((row + group) % 4);
                expected_zeros[group] = 1 + (row + 2 * group) % 8;
                for (size_t i = 0; i < 32; ++i) {
                    codes[group * 32 + i] = (3 * row + group * 32 + i) % 16;
                }
            }
            require_q4_a32_block(repacked.data() + row * block_size,
                expected_scales, expected_zeros, codes,
                "compressed-tensors INT4 repack changed a row/lane/group value");
        }
        adapters.consume(*weight);
        adapters.validate_complete();

    }
    {
        const auto path = dir.path / "packed-int4-symmetric-g128";
        constexpr size_t rows = 2;
        constexpr size_t cols = 128;
        std::vector<uint32_t> packed_weight(rows * cols / 8);
        std::vector<uint8_t> scales(rows * sizeof(uint16_t));
        for (size_t row = 0; row < rows; ++row) {
            for (size_t col = 0; col < cols; ++col) {
                packed_weight[row * (cols / 8) + col / 8] |=
                    uint32_t((5 * row + 3 * col) % 16) << (4 * (col % 8));
            }
            store_bf16(scales, row, 0.5f + 0.75f * row);
        }
        write_single_shard_model(path, {
            { "int4.weight_packed", "I32",  { rows, cols / 8 }, i32_bytes(packed_weight) },
            { "int4.weight_scale",  "BF16", { rows, 1 },        scales                   },
            { "int4.weight_shape",  "I64",  { 2 },              i64_bytes({ int64_t(rows), int64_t(cols) }) },
        });
        write_text(path / "config.json", packed_int4_symmetric_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("int4", llama_safetensors_quant_role::WEIGHT);
        require(weight.has_value() && weight->target_type == GGML_TYPE_Q4_A32 &&
                    weight->materialization == llama_safetensors_quant_materialization::PACKED_INT4_REPACK &&
                    weight->target_shape == std::vector<int64_t>({ cols, rows }),
                "symmetric compressed-tensors W4A16 binding is wrong");
        const std::vector<uint8_t> repacked = adapters.read(*weight);
        constexpr size_t block_size = 74;
        require(repacked.size() == rows * block_size,
                "symmetric compressed-tensors W4A16 repack has the wrong size");
        for (size_t row = 0; row < rows; ++row) {
            std::array<uint8_t, 128> codes;
            for (size_t col = 0; col < cols; ++col) {
                codes[col] = (5 * row + 3 * col) % 16;
            }
            const float scale = 0.5f + 0.75f * row;
            require_q4_a32_block(repacked.data() + row * block_size,
                { scale, scale, scale, scale }, { 8, 8, 8, 8 }, codes,
                "symmetric compressed-tensors W4A16 repack changed a row/lane value");
        }
        adapters.consume(*weight);
        adapters.validate_complete();
    }
    {
        const auto path = dir.path / "packed-int4-symmetric-unexpected-zero";
        write_single_shard_model(path, {
            { "int4.weight_packed",     "I32",  { 1, 16 }, std::vector<uint8_t>(64) },
            { "int4.weight_scale",      "BF16", { 1, 1 },  { 0x00, 0x3f }           },
            { "int4.weight_zero_point", "I32",  { 1, 1 },  i32_bytes({ 8 })         },
            { "int4.weight_shape",      "I64",  { 2 },     i64_bytes({ 1, 128 })    },
        });
        write_text(path / "config.json", packed_int4_symmetric_config);
        require_rejected([&] {
            const auto registry = llama_safetensors_registry::load(path);
            (void) llama_safetensors_quant_adapters(
                llama_safetensors_read_json(path / "config.json"), registry);
        }, "symmetric compressed-tensors W4A16 accepted an unexpected zero-point sidecar");
    }
    {
        const auto path = dir.path / "packed-int8-g128";
        constexpr size_t rows = 3;
        constexpr size_t cols = 256;
        constexpr size_t groups = cols / 128;
        std::vector<uint32_t> packed_weight(rows * cols / 4);
        std::vector<uint8_t> scales(rows * groups * sizeof(uint16_t));
        for (size_t row = 0; row < rows; ++row) {
            for (size_t col = 0; col < cols; ++col) {
                packed_weight[row * (cols / 4) + col / 4] |=
                    uint32_t((17 * row + col) % 256) << (8 * (col % 4));
            }
            for (size_t group = 0; group < groups; ++group) {
                store_bf16(scales, row * groups + group, 0.5f + 0.5f * ((row + group) % 3));
            }
        }
        write_single_shard_model(path, {
            { "int8.weight_packed", "I32",  { rows, cols / 4 }, i32_bytes(packed_weight) },
            { "int8.weight_scale",  "BF16", { rows, groups },   scales                   },
            { "int8.weight_shape",  "I64",  { 2 },              i64_bytes({ int64_t(rows), int64_t(cols) }) },
        });
        write_text(path / "config.json", packed_int_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("int8", llama_safetensors_quant_role::WEIGHT);
        require(weight.has_value() && weight->target_type == GGML_TYPE_Q8_0_G128 &&
                    weight->materialization == llama_safetensors_quant_materialization::PACKED_INT8_REPACK &&
                    weight->target_shape == std::vector<int64_t>({ cols, rows }),
                "compressed-tensors INT8 binding is wrong");
        const std::vector<uint8_t> repacked = adapters.read(*weight);
        constexpr size_t block_size = sizeof(uint16_t) + 128;
        require(repacked.size() == rows * (cols / 128) * block_size,
                "compressed-tensors INT8 exact repack has the wrong size");
        for (size_t row = 0; row < rows; ++row) {
            for (size_t block = 0; block < cols / 128; ++block) {
                const float expected_scale = 0.5f + 0.5f * ((row + block) % 3);
                uint32_t expected_bits;
                std::memcpy(&expected_bits, &expected_scale, sizeof(expected_bits));
                const uint8_t * actual = repacked.data() + (row * (cols / 128) + block) * block_size;
                require(load_u16(actual) == uint16_t(expected_bits >> 16),
                        "compressed-tensors INT8 repack changed a group scale");
                for (size_t i = 0; i < 128; ++i) {
                    const uint8_t code = (17 * row + block * 128 + i) % 256;
                    require(actual[sizeof(uint16_t) + i] == static_cast<uint8_t>(code - 128),
                            "compressed-tensors INT8 repack changed a packed byte lane");
                }
            }
        }
        adapters.consume(*weight);
        adapters.validate_complete();
    }
    {
        const auto path = dir.path / "packed-int4-missing-zero";
        write_single_shard_model(path, {
            { "int4.weight_packed", "I32",  { 1, 4 }, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
            { "int4.weight_scale",  "BF16", { 1, 1 }, { 0, 0 }                                           },
            { "int4.weight_shape",  "I64",  { 2 },    i64_bytes({ 1, 32 })                                },
        });
        write_text(path / "config.json", packed_int_config);
        const auto registry = llama_safetensors_registry::load(path);
        require_rejected(
            [&] { (void) llama_safetensors_quant_adapters(llama_safetensors_read_json(path / "config.json"), registry); },
            "compressed-tensors INT4 accepted a missing zero-point sidecar");
    }
    {
        const auto path = dir.path / "awq";
        constexpr size_t cols = 256;
        constexpr size_t rows = 8;
        constexpr size_t group_size = 64;
        constexpr size_t groups = cols / group_size;
        constexpr uint32_t shifts[8] = { 0, 16, 4, 20, 8, 24, 12, 28 };
        std::vector<uint32_t> qweight(cols);
        std::vector<uint32_t> qzeros(groups);
        std::vector<uint8_t> scales(groups * rows * sizeof(uint16_t));
        for (size_t col = 0; col < cols; ++col) {
            for (size_t row = 0; row < rows; ++row) {
                qweight[col] |= ((col + 3 * row) % 16) << shifts[row];
            }
        }
        for (size_t group = 0; group < groups; ++group) {
            for (size_t row = 0; row < rows; ++row) {
                qzeros[group] |= (1 + (group + row) % 8) << shifts[row];
                store_bf16(scales, group * rows + row, 0.5f + 0.25f * ((group + row) % 8));
            }
        }
        write_single_shard_model(path, {
            { "module.qweight", "I32", { cols, 1 }, i32_bytes(qweight) },
            { "module.qzeros",  "I32", { groups, 1 },    i32_bytes(qzeros)  },
            { "module.scales",  "BF16", { groups, rows }, scales            },
        });
        write_text(path / "config.json", awq_g64_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        require(weight.has_value() && weight->target_type == GGML_TYPE_Q4_1 &&
                    weight->materialization == llama_safetensors_quant_materialization::AWQ_REPACK &&
                    weight->target_shape == std::vector<int64_t>({ cols, rows }) &&
                    adapters.file_type() == LLAMA_FTYPE_MOSTLY_Q4_1,
                "AWQ binding is wrong");
        const std::vector<uint8_t> repacked = adapters.read(*weight);
        constexpr size_t block_size = 2 * sizeof(ggml_fp16_t) + 16;
        require(repacked.size() == rows * (cols / 32) * block_size, "AWQ Q4_1 repack has the wrong size");
        for (size_t row = 0; row < rows; ++row) {
            for (size_t block = 0; block < cols / 32; ++block) {
                const size_t group = block / (group_size / 32);
                std::array<uint8_t, 32> codes{};
                for (size_t col = 0; col < codes.size(); ++col) {
                    codes[col] = (block * 32 + col + 3 * row) % 16;
                }
                require_q4_1_block(
                    repacked.data() + (row * (cols / 32) + block) * block_size,
                    0.5f + 0.25f * ((group + row) % 8), 1 + (group + row) % 8, codes,
                    "AWQ Q4_1 repack changed a row/group code, scale, or zero point");
            }
        }
        adapters.consume(*weight);
        adapters.validate_complete();
    }
    {
        const auto path = dir.path / "awq-missing-zero";
        write_single_shard_model(path, {
            { "module.qweight", "I32", { 128, 1 }, std::vector<uint8_t>(128 * 4) },
            { "module.scales",  "F16", { 1, 8 },   std::vector<uint8_t>(16)      },
        });
        write_text(path / "config.json", awq_config);
        const auto registry = llama_safetensors_registry::load(path);
        require_rejected([&] { (void) llama_safetensors_quant_adapters(llama_safetensors_read_json(path / "config.json"), registry); },
                         "AWQ contract accepted a missing zero-point sidecar");
    }
    {
        const auto path = dir.path / "quark-uint4-w4a16";
        constexpr size_t cols = 256;
        constexpr size_t rows = 16;
        constexpr size_t group_size = 128;
        constexpr size_t groups = cols / group_size;
        constexpr uint32_t shifts[8] = { 0, 16, 4, 20, 8, 24, 12, 28 };
        std::vector<uint32_t> weight(cols * (rows / 8));
        std::vector<uint32_t> zero(groups * (rows / 8));
        std::vector<uint8_t> scales(groups * rows * sizeof(float));
        for (size_t col = 0; col < cols; ++col) {
            for (size_t row = 0; row < rows; ++row) {
                weight[col * (rows / 8) + row / 8] |=
                    uint32_t((5 * col + 3 * row) % 16) << shifts[row % 8];
            }
        }
        for (size_t group = 0; group < groups; ++group) {
            for (size_t row = 0; row < rows; ++row) {
                zero[group * (rows / 8) + row / 8] |=
                    uint32_t(1 + (2 * group + row) % 8) << shifts[row % 8];
                const float value = 0.25f + 0.125f * ((group + row) % 8);
                std::memcpy(scales.data() + (group * rows + row) * sizeof(value), &value, sizeof(value));
            }
        }
        write_single_shard_model(path, {
            { "module.weight",            "I32", { cols, rows / 8 }, i32_bytes(weight) },
            { "module.weight_scale",      "F32", { groups, rows },   scales            },
            { "module.weight_zero_point", "I32", { groups, rows / 8 }, i32_bytes(zero) },
        });
        write_text(path / "config.json", quark_uint4_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(
            llama_safetensors_read_json(path / "config.json"), registry);
        const auto binding = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        require(binding.has_value() && binding->target_type == GGML_TYPE_Q4_1 &&
                    binding->target_shape == std::vector<int64_t>({ cols, rows }) &&
                    binding->materialization ==
                        llama_safetensors_quant_materialization::QUARK_W4A16_REPACK &&
                    adapters.file_type() == LLAMA_FTYPE_MOSTLY_Q4_1,
                "Quark UINT4 W4A16 binding is wrong");
        const std::vector<uint8_t> repacked = adapters.read(*binding);
        constexpr size_t block_size = 2 * sizeof(ggml_fp16_t) + 16;
        require(repacked.size() == rows * (cols / 32) * block_size,
                "Quark UINT4 W4A16 repack has the wrong size");
        for (size_t row = 0; row < rows; ++row) {
            for (size_t block = 0; block < cols / 32; ++block) {
                const size_t group = block / (group_size / 32);
                std::array<uint8_t, 32> codes{};
                for (size_t col = 0; col < codes.size(); ++col) {
                    codes[col] = (5 * (block * 32 + col) + 3 * row) % 16;
                }
                require_q4_1_block(
                    repacked.data() + (row * (cols / 32) + block) * block_size,
                    0.25f + 0.125f * ((group + row) % 8),
                    1 + (2 * group + row) % 8, codes,
                    "Quark UINT4 repack changed a row/group code, scale, or zero point");
            }
        }
        adapters.consume(*binding);
        adapters.validate_complete();
    }
    {
        const auto path = dir.path / "quark-int4-w4a16";
        constexpr size_t cols = 128;
        constexpr size_t rows = 8;
        constexpr uint32_t shifts[8] = { 0, 16, 4, 20, 8, 24, 12, 28 };
        std::vector<uint32_t> weight(cols);
        std::vector<uint8_t> scales(rows * sizeof(uint16_t));
        for (size_t col = 0; col < cols; ++col) {
            for (size_t row = 0; row < rows; ++row) {
                const int8_t value = int8_t((7 * col + 5 * row) % 16) - 8;
                weight[col] |= uint32_t(uint8_t(value) & 0x0f) << shifts[row];
            }
        }
        for (size_t row = 0; row < rows; ++row) {
            store_bf16(scales, row, 0.5f + 0.125f * row);
        }
        write_single_shard_model(path, {
            { "module.weight",            "I32", { cols, 1 }, i32_bytes(weight) },
            { "module.weight_scale",      "BF16", { 1, rows }, scales           },
            { "module.weight_zero_point", "I32", { 1, 1 },    i32_bytes({ 0 })  },
        });
        write_text(path / "config.json", quark_int4_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(
            llama_safetensors_read_json(path / "config.json"), registry);
        const auto binding = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        require(binding.has_value() && binding->auxiliaries == std::vector<std::string>({ "module.weight_scale" }),
                "symmetric Quark W4A16 unnecessarily retained its zero-point sidecar");
        const std::vector<uint8_t> repacked = adapters.read(*binding);
        constexpr size_t block_size = 2 * sizeof(ggml_fp16_t) + 16;
        for (size_t row = 0; row < rows; ++row) {
            for (size_t block = 0; block < cols / 32; ++block) {
                std::array<uint8_t, 32> codes{};
                for (size_t col = 0; col < codes.size(); ++col) {
                    const int8_t value = int8_t((7 * (block * 32 + col) + 5 * row) % 16) - 8;
                    codes[col] = uint8_t(value + 8);
                }
                require_q4_1_block(
                    repacked.data() + (row * (cols / 32) + block) * block_size,
                    0.5f + 0.125f * row, 8, codes,
                    "Quark INT4 repack changed a signed code or scale");
            }
        }
        adapters.consume(*binding);
        adapters.validate_complete();
    }
    require_rejected(
        [&] { (void) llama_safetensors_quant_config::from_json(
            llama_safetensors_json::parse(quark_int4_g4_config)); },
        "Quark group-4 W4A16 was accepted despite lacking an exact GGML representation");
    {
        const auto path = dir.path / "quark-fp8-w8a8";
        write_single_shard_model(path, {
            { "module.weight",       "F8_E4M3", { 2, 32 }, std::vector<uint8_t>(64) },
            { "module.weight_scale", "F32",     {},        f32_bytes(0.5f)           },
            { "module.input_scale",  "F32",     {},        f32_bytes(0.25f)          },
            { "lm_head.weight",      "BF16",    { 2, 32 }, std::vector<uint8_t>(128) },
        });
        write_text(path / "config.json", quark_fp8_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(
            llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        const auto scale  = adapters.bind("module", llama_safetensors_quant_role::WEIGHT_SCALE);
        const auto input  = adapters.bind("module", llama_safetensors_quant_role::INPUT_SCALE);
        require(weight.has_value() && weight->target_type == GGML_TYPE_F8_E4M3 &&
                    scale.has_value() && scale->target_type == GGML_TYPE_BF16 &&
                    scale->target_shape == std::vector<int64_t>({ 2 }) &&
                    input.has_value() && input->target_type == GGML_TYPE_F32 &&
                    !adapters.applies("lm_head"),
                "Quark static FP8 W8A8 binding is wrong");
        const std::vector<uint8_t> broadcast = adapters.read(*scale);
        require(broadcast.size() == 2 * sizeof(uint16_t) &&
                    load_u16(broadcast.data()) == ggml_fp32_to_bf16(0.5f).bits &&
                    load_u16(broadcast.data() + sizeof(uint16_t)) == ggml_fp32_to_bf16(0.5f).bits,
                "Quark FP8 tensor scale was not broadcast across output channels");
        adapters.consume(*weight);
        adapters.consume(*scale);
        adapters.consume(*input);
        adapters.validate_complete();
    }
    {
        const auto path = dir.path / "quark-int8-w8a8";
        std::vector<uint8_t> scales;
        const std::vector<uint8_t> first = f32_bytes(0.5f);
        const std::vector<uint8_t> second = f32_bytes(0.25f);
        scales.insert(scales.end(), first.begin(), first.end());
        scales.insert(scales.end(), second.begin(), second.end());
        write_single_shard_model(path, {
            { "module.weight",       "I8",   { 2, 32 }, std::vector<uint8_t>(64) },
            { "module.weight_scale", "F32",  { 2 },     scales                   },
            { "lm_head.weight",      "BF16", { 2, 32 }, std::vector<uint8_t>(128) },
        });
        write_text(path / "config.json", quark_int8_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(
            llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        const auto scale  = adapters.bind("module", llama_safetensors_quant_role::WEIGHT_SCALE);
        const auto input  = adapters.bind("module", llama_safetensors_quant_role::INPUT_SCALE);
        require(weight.has_value() && weight->target_type == GGML_TYPE_I8 &&
                    scale.has_value() && scale->target_type == GGML_TYPE_F32 &&
                    !input.has_value() && adapters.file_type() == LLAMA_FTYPE_MOSTLY_Q8_0 &&
                    !adapters.applies("lm_head"),
                "Quark dynamic INT8 W8A8 binding is wrong");
        require(adapters.finalize(*scale, adapters.read(*scale)) == scales,
                "Quark INT8 channel scales changed during materialization");
        adapters.consume(*weight);
        adapters.consume(*scale);
        adapters.validate_complete();
    }
    {
        const auto path = dir.path / "quark-qwen35-qkv-transform";
        constexpr size_t cols = 128;
        constexpr size_t rows = 256;
        constexpr size_t packed_rows = rows / 8;
        constexpr uint32_t shifts[8] = { 0, 16, 4, 20, 8, 24, 12, 28 };
        std::vector<uint32_t> weight(cols * packed_rows);
        std::vector<uint32_t> zero(packed_rows);
        std::vector<uint8_t> scales(rows * sizeof(uint16_t));
        for (size_t col = 0; col < cols; ++col) {
            for (size_t row = 0; row < rows; ++row) {
                weight[col * packed_rows + row / 8] |=
                    uint32_t((3 * col + row) % 16) << shifts[row % 8];
            }
        }
        for (size_t row = 0; row < rows; ++row) {
            zero[row / 8] |= uint32_t(1 + row % 8) << shifts[row % 8];
            store_bf16(scales, row, 0.5f + 0.125f * (row % 8));
        }
        const std::string module = "model.layers.0.linear_attn.in_proj_qkv";
        write_single_shard_model(path, {
            { module + ".weight",            "I32",  { cols, packed_rows }, i32_bytes(weight) },
            { module + ".weight_scale",      "BF16", { 1, rows },           scales            },
            { module + ".weight_zero_point", "I32",  { 1, packed_rows },    i32_bytes(zero)   },
        });
        write_text(path / "generation_config.json", "{}");
        write_text(path / "tokenizer.json", "{}");
        llama_safetensors_json config = {
            { "model_type", "qwen3_5_text" },
            { "num_hidden_layers", 24 },
            { "mtp_num_hidden_layers", 0 },
            { "linear_num_key_heads", 2 },
            { "linear_num_value_heads", 4 },
            { "linear_key_head_dim", 32 },
            { "linear_value_head_dim", 32 },
        };
        config["quantization_config"] =
            llama_safetensors_json::parse(quark_uint4_config).at("quantization_config");
        llama_safetensors_qwen35_importer importer(path, config);
        ggml_type target_type;
        std::array<int64_t, GGML_MAX_DIMS> ne;
        require(importer.describe("blk.0.attn_qkv.weight", target_type, ne) &&
                    target_type == GGML_TYPE_Q4_1 && ne[0] == int64_t(cols) && ne[1] == int64_t(rows),
                "Quark Qwen3.5 QKV transform has the wrong target contract");
        constexpr size_t block_size = 2 * sizeof(ggml_fp16_t) + 16;
        const std::vector<uint8_t> transformed = importer.materialize(
            "blk.0.attn_qkv.weight", target_type, rows * (cols / 32) * block_size);
        for (size_t dst_row = 0; dst_row < rows; ++dst_row) {
            size_t src_row = dst_row;
            if (dst_row >= 128) {
                constexpr std::array<size_t, 4> value_head_order = { 0, 2, 1, 3 };
                const size_t value_row = dst_row - 128;
                src_row = 128 + value_head_order[value_row / 32] * 32 + value_row % 32;
            }
            for (size_t block = 0; block < cols / 32; ++block) {
                std::array<uint8_t, 32> codes{};
                for (size_t col = 0; col < codes.size(); ++col) {
                    codes[col] = (3 * (block * 32 + col) + src_row) % 16;
                }
                require_q4_1_block(
                    transformed.data() + (dst_row * (cols / 32) + block) * block_size,
                    0.5f + 0.125f * (src_row % 8), 1 + src_row % 8, codes,
                    "Quark Qwen3.5 QKV transform split a quant block or moved the wrong row");
            }
        }
    }
    {
        const auto path = dir.path / "quark-qwen35-recurrent-scale-transform";
        const std::string module = "model.layers.0.linear_attn.in_proj_a";
        const std::array<float, 4> source_scales = { 0.5f, 1.0f, 1.5f, 2.0f };
        std::vector<uint8_t> scales(sizeof(source_scales));
        std::memcpy(scales.data(), source_scales.data(), scales.size());
        write_single_shard_model(path, {
            { module + ".weight",       "I8",  { 4, 32 }, std::vector<uint8_t>(128) },
            { module + ".weight_scale", "F32", { 4 },     scales                    },
        });
        write_text(path / "generation_config.json", "{}");
        write_text(path / "tokenizer.json", "{}");
        llama_safetensors_json config = {
            { "model_type", "qwen3_5_text" },
            { "num_hidden_layers", 24 },
            { "mtp_num_hidden_layers", 0 },
            { "linear_num_key_heads", 2 },
            { "linear_num_value_heads", 4 },
            { "linear_key_head_dim", 32 },
            { "linear_value_head_dim", 32 },
        };
        config["quantization_config"] =
            llama_safetensors_json::parse(quark_int8_config).at("quantization_config");
        llama_safetensors_qwen35_importer importer(path, config);
        ggml_type target_type;
        std::array<int64_t, GGML_MAX_DIMS> ne;
        require(importer.describe("blk.0.ssm_alpha.scale", target_type, ne) &&
                    target_type == GGML_TYPE_F32 && ne[0] == 4,
                "Quark Qwen3.5 recurrent channel scale has the wrong target contract");
        const std::vector<uint8_t> transformed = importer.materialize(
            "blk.0.ssm_alpha.scale", target_type, sizeof(source_scales));
        std::array<float, 4> actual;
        std::memcpy(actual.data(), transformed.data(), transformed.size());
        require(actual == std::array<float, 4>({ 0.5f, 1.5f, 1.0f, 2.0f }),
                "Qwen3.5 recurrent alpha/beta scale mapping did not preserve head order");
    }
    {
        const auto path = dir.path / "awq-minimum-overflow";
        std::vector<uint8_t> scales(8 * sizeof(uint16_t));
        for (size_t row = 0; row < 8; ++row) {
            const uint16_t maximum = 0x7bff;
            std::memcpy(scales.data() + row * sizeof(maximum), &maximum, sizeof(maximum));
        }
        write_single_shard_model(path, {
            { "module.qweight", "I32", { 128, 1 }, std::vector<uint8_t>(128 * sizeof(uint32_t)) },
            { "module.qzeros",  "I32", { 1, 1 },   i32_bytes({ 15 })                              },
            { "module.scales",  "F16", { 1, 8 },   scales                                         },
        });
        write_text(path / "config.json", awq_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        require(weight.has_value(), "AWQ overflow fixture did not bind");
        require_rejected([&] { (void) adapters.read(*weight); },
                         "AWQ accepted a minimum that overflows Q4_1");
    }
    {
        const auto path = dir.path / "gptq";
        constexpr size_t cols = 256;
        constexpr size_t rows = 8;
        constexpr size_t group_size = 32;
        constexpr size_t groups_count = cols / group_size;
        std::vector<uint32_t> qweight((cols / 8) * rows);
        std::vector<uint32_t> qzeros(groups_count);
        std::vector<uint8_t> scales(groups_count * rows * sizeof(uint16_t));
        for (size_t packed_col = 0; packed_col < cols / 8; ++packed_col) {
            for (size_t row = 0; row < rows; ++row) {
                uint32_t word = 0;
                for (size_t lane = 0; lane < 8; ++lane) {
                    word |= ((packed_col * 8 + lane + 5 * row) % 16) << (4 * lane);
                }
                qweight[packed_col * rows + row] = word;
            }
        }
        for (size_t group = 0; group < groups_count; ++group) {
            for (size_t row = 0; row < rows; ++row) {
                const uint8_t zero = 1 + (2 * group + row) % 8;
                qzeros[group] |= uint32_t(zero - 1) << (4 * row);
                store_f16(scales, group * rows + row, 0.5f + 0.25f * ((2 * group + row) % 8));
            }
        }
        std::vector<uint32_t> groups(cols);
        for (size_t col = 0; col < cols; ++col) {
            groups[col] = col / group_size;
        }
        write_single_shard_model(path, {
            { "module.qweight", "I32", { cols / 8, rows }, i32_bytes(qweight) },
            { "module.qzeros",  "I32", { groups_count, 1 },    i32_bytes(qzeros)  },
            { "module.scales",  "F16", { groups_count, rows }, scales             },
            { "module.g_idx",   "I32", { cols },           i32_bytes(groups)  },
        });
        write_text(path / "config.json", gptq_g32_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        require(weight.has_value() && weight->target_type == GGML_TYPE_Q4_1 &&
                    weight->materialization == llama_safetensors_quant_materialization::GPTQ_REPACK &&
                    weight->target_shape == std::vector<int64_t>({ cols, rows }) &&
                    adapters.file_type() == LLAMA_FTYPE_MOSTLY_Q4_1,
                "GPTQ binding is wrong");
        const std::vector<uint8_t> repacked = adapters.read(*weight);
        constexpr size_t block_size = 2 * sizeof(ggml_fp16_t) + 16;
        require(repacked.size() == rows * (cols / 32) * block_size, "GPTQ Q4_1 repack has the wrong size");
        for (size_t row = 0; row < rows; ++row) {
            for (size_t block = 0; block < cols / 32; ++block) {
                const size_t group = block / (group_size / 32);
                std::array<uint8_t, 32> codes{};
                for (size_t col = 0; col < codes.size(); ++col) {
                    codes[col] = (block * 32 + col + 5 * row) % 16;
                }
                require_q4_1_block(
                    repacked.data() + (row * (cols / 32) + block) * block_size,
                    0.5f + 0.25f * ((2 * group + row) % 8), 1 + (2 * group + row) % 8, codes,
                    "GPTQ Q4_1 repack changed a row/group code, scale, or zero point");
            }
        }
        adapters.consume(*weight);
        adapters.validate_complete();
    }
    {
        const auto path = dir.path / "autoround";
        constexpr size_t cols = 128;
        constexpr size_t rows = 8;
        std::vector<uint8_t> scales(rows * sizeof(uint16_t));
        for (size_t row = 0; row < rows; ++row) {
            store_f16(scales, row, (row & 1 ? 1.0f : -1.0f) * 0.125f * (row + 1));
        }
        write_single_shard_model(path, {
            { "module.qweight", "I32", { cols / 8, rows }, std::vector<uint8_t>(cols * rows / 2) },
            { "module.qzeros",  "I32", { 1, 1 },           i32_bytes({ 0x77777777u })             },
            { "module.scales",  "F16", { 1, rows },        scales                                },
        });
        write_text(path / "config.json", autoround_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(
            llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        require(weight.has_value() && weight->target_type == GGML_TYPE_Q4_1 &&
                    weight->materialization == llama_safetensors_quant_materialization::GPTQ_REPACK &&
                    weight->target_shape == std::vector<int64_t>({ cols, rows }),
                "AutoRound auto_gptq binding is wrong");
        const std::vector<uint8_t> repacked = adapters.read(*weight);
        constexpr size_t block_size = sizeof(uint16_t) * 2 + 16;
        require(repacked.size() == rows * (cols / 32) * block_size,
                "AutoRound auto_gptq repack has the wrong size");
        require(ggml_fp16_to_fp32(load_u16(repacked.data())) == -0.125f &&
                    ggml_fp16_to_fp32(load_u16(repacked.data() + sizeof(uint16_t))) == 1.0f &&
                    std::all_of(repacked.begin() + 2 * sizeof(uint16_t),
                                repacked.begin() + block_size,
                                [](uint8_t code) { return code == 0x00; }),
                "AutoRound signed scale changed during GPTQ repacking");
        adapters.consume(*weight);
        adapters.validate_complete();

        json mixed = json::parse(autoround_config);
        mixed["quantization_config"]["extra_config"] = {
            { "lm_head", { { "bits", 16 }, { "data_type", "float" } } },
            { "module", {
                { "bits", 4 }, { "group_size", 128 }, { "sym", true },
                { "data_type", "int" },
            } },
        };
        llama_safetensors_quant_adapters mixed_adapters(mixed, registry);
        require(!mixed_adapters.applies("lm_head") &&
                    mixed_adapters.bind("module", llama_safetensors_quant_role::WEIGHT).has_value(),
                "AutoRound GPTQ per-module precision exceptions were not honored");
        mixed["quantization_config"]["extra_config"]["module"]["bits"] = 3;
        require_rejected([&] { (void) llama_safetensors_quant_adapters(mixed, registry); },
                "AutoRound GPTQ accepted an unsupported per-module bit width");
    }
    {
        const auto path = dir.path / "gptq-int8";
        constexpr size_t cols = 256;
        constexpr size_t rows = 8;
        constexpr size_t group_size = 128;
        constexpr size_t groups_count = cols / group_size;
        std::vector<uint32_t> qweight((cols / 4) * rows);
        std::vector<uint32_t> qzeros(groups_count * rows / 4);
        std::vector<uint8_t> scales(groups_count * rows * sizeof(uint16_t));
        for (size_t packed_col = 0; packed_col < cols / 4; ++packed_col) {
            for (size_t row = 0; row < rows; ++row) {
                uint32_t word = 0;
                for (size_t lane = 0; lane < 4; ++lane) {
                    word |= uint32_t((packed_col * 4 + lane + 17 * row) % 256) << (8 * lane);
                }
                qweight[packed_col * rows + row] = word;
            }
        }
        for (size_t group = 0; group < groups_count; ++group) {
            for (size_t row = 0; row < rows; ++row) {
                qzeros[group * (rows / 4) + row / 4] |= uint32_t(127) << (8 * (row % 4));
                store_f16(scales, group * rows + row, 0.25f + 0.125f * ((group + row) % 8));
            }
        }
        std::vector<uint32_t> groups(cols);
        for (size_t col = 0; col < cols; ++col) {
            groups[col] = col / group_size;
        }
        write_single_shard_model(path, {
            { "module.qweight", "I32", { cols / 4, rows }, i32_bytes(qweight) },
            { "module.qzeros",  "I32", { groups_count, rows / 4 }, i32_bytes(qzeros) },
            { "module.scales",  "F16", { groups_count, rows }, scales },
            { "module.g_idx",   "I32", { cols }, i32_bytes(groups) },
        });
        write_text(path / "config.json", gptq_int8_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        require(weight.has_value() && weight->target_type == GGML_TYPE_Q8_0 &&
                    weight->materialization == llama_safetensors_quant_materialization::GPTQ8_REPACK &&
                    weight->target_shape == std::vector<int64_t>({ cols, rows }) &&
                    adapters.file_type() == LLAMA_FTYPE_MOSTLY_Q8_0,
                "GPTQ INT8 binding is wrong");
        const std::vector<uint8_t> repacked = adapters.read(*weight);
        const size_t block_size = ggml_type_size(GGML_TYPE_Q8_0);
        require(repacked.size() == rows * (cols / 32) * block_size,
                "GPTQ INT8 repack has the wrong size");
        for (size_t row = 0; row < rows; ++row) {
            for (size_t block = 0; block < cols / 32; ++block) {
                const size_t group = block / 4;
                const uint8_t * actual = repacked.data() + (row * (cols / 32) + block) * block_size;
                require(load_u16(actual) == ggml_fp32_to_fp16(0.25f + 0.125f * ((group + row) % 8)),
                        "GPTQ INT8 repack changed a group scale");
                for (size_t i = 0; i < 32; ++i) {
                    const uint8_t code = (block * 32 + i + 17 * row) % 256;
                    require(actual[sizeof(uint16_t) + i] == uint8_t(code - 128),
                            "GPTQ INT8 repack changed a packed byte lane");
                }
            }
        }
        adapters.consume(*weight);
        adapters.validate_complete();
    }
    {
        const auto path = dir.path / "gptq-act-order";
        constexpr size_t cols = 256;
        constexpr size_t rows = 8;
        constexpr size_t groups_count = 2;
        std::vector<uint32_t> groups(cols);
        for (size_t col = 0; col < cols; ++col) {
            groups[col] = 1 - col / 128;
        }
        std::vector<uint8_t> scales(groups_count * rows * sizeof(uint16_t));
        for (size_t i = 0; i < groups_count * rows; ++i) {
            store_f16(scales, i, 0.25f + 0.125f * i);
        }
        write_single_shard_model(path, {
            { "module.qweight", "I32", { cols / 8, rows }, std::vector<uint8_t>(cols * rows / 2, 0x31) },
            { "module.qzeros",  "I32", { groups_count, rows / 8 }, std::vector<uint8_t>(groups_count * rows / 2, 0x21) },
            { "module.scales",  "F16", { groups_count, rows }, scales },
            { "module.g_idx",   "I32", { cols }, i32_bytes(groups) },
        });
        write_text(path / "config.json", gptq_act_order_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        const auto scale = adapters.bind("module", llama_safetensors_quant_role::WEIGHT_SCALE);
        const auto input = adapters.bind("module", llama_safetensors_quant_role::INPUT_SCALE);
        require(weight.has_value() && weight->target_type == GGML_TYPE_GPTQ_AO &&
                    weight->materialization == llama_safetensors_quant_materialization::RAW &&
                    weight->target_shape == std::vector<int64_t>({ cols, rows }),
                "GPTQ act-order weight binding is wrong");
        require(scale.has_value() && scale->target_type == GGML_TYPE_I8 &&
                    scale->materialization == llama_safetensors_quant_materialization::GPTQ_SCALE_BUNDLE,
                "GPTQ act-order auxiliary binding is wrong");
        const std::vector<uint8_t> bundle = adapters.read(*scale);
        ggml_gptq_ao_header header;
        std::memcpy(&header, bundle.data(), sizeof(header));
        require(header.magic == GGML_GPTQ_AO_MAGIC && header.cols == cols && header.rows == rows &&
                    header.groups == groups_count && header.group_size == 128 &&
                    header.total_size == bundle.size(),
                "GPTQ act-order auxiliary bundle is malformed");
        const uint16_t * compact_groups = reinterpret_cast<const uint16_t *>(
            bundle.data() + header.g_idx_offset);
        for (size_t col = 0; col < cols; ++col) {
            require(compact_groups[col] == groups[col], "GPTQ act-order group map changed during binding");
        }
        adapters.consume(*weight);
        adapters.consume(*scale);
        adapters.validate_complete();
    }
    {
        const auto path = dir.path / "gptq-per-channel";
        constexpr size_t cols = 32;
        constexpr size_t rows = 8;
        std::vector<uint32_t> qweight((cols / 8) * rows);
        for (size_t packed_col = 0; packed_col < cols / 8; ++packed_col) {
            for (size_t row = 0; row < rows; ++row) {
                uint32_t word = 0;
                for (size_t lane = 0; lane < 8; ++lane) {
                    word |= ((packed_col * 8 + lane + row) % 16) << (4 * lane);
                }
                qweight[packed_col * rows + row] = word;
            }
        }
        std::vector<uint32_t> groups(cols, 0);
        std::vector<uint8_t> scales(rows * sizeof(uint16_t));
        for (size_t row = 0; row < rows; ++row) {
            store_bf16(scales, row, 0.5f + 0.25f * row);
        }
        write_single_shard_model(path, {
            { "module.qweight", "I32",  { cols / 8, rows }, i32_bytes(qweight) },
            { "module.qzeros",  "I32",  { 1, 1 },           i32_bytes({ 0 })    },
            { "module.scales",  "BF16", { 1, rows },        scales              },
            { "module.g_idx",   "I32",  { cols },           i32_bytes(groups)   },
        });
        write_text(path / "config.json", gptq_per_channel_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        require(weight.has_value() && weight->target_shape == std::vector<int64_t>({ cols, rows }),
                "per-channel GPTQ binding is wrong");
        const std::vector<uint8_t> repacked = adapters.read(*weight);
        constexpr size_t block_size = 2 * sizeof(ggml_fp16_t) + 16;
        for (size_t row = 0; row < rows; ++row) {
            std::array<uint8_t, 32> codes{};
            for (size_t col = 0; col < cols; ++col) {
                codes[col] = (col + row) % 16;
            }
            require_q4_1_block(
                repacked.data() + row * block_size,
                0.5f + 0.25f * row, 1, codes,
                "per-channel GPTQ repack changed a code, scale, or zero point");
        }
        adapters.consume(*weight);
        adapters.validate_complete();
    }
    {
        const auto path = dir.path / "gptq-partial-packed-row";
        std::vector<uint32_t> groups(128);
        write_single_shard_model(path, {
            { "module.qweight", "I32", { 16, 9 },  std::vector<uint8_t>(16 * 9 * sizeof(uint32_t)) },
            { "module.qzeros",  "I32", { 1, 1 },   std::vector<uint8_t>(sizeof(uint32_t))          },
            { "module.scales",  "F16", { 1, 9 },   std::vector<uint8_t>(9 * sizeof(uint16_t))      },
            { "module.g_idx",   "I32", { 128 },    i32_bytes(groups)                               },
        });
        write_text(path / "config.json", gptq_config);
        const auto registry = llama_safetensors_registry::load(path);
        require_rejected([&] { (void) llama_safetensors_quant_adapters(llama_safetensors_read_json(path / "config.json"), registry); },
                         "GPTQ accepted output rows not divisible by its eight-row packing");
    }
    {
        const auto path = dir.path / "gptq-minimum-overflow";
        std::vector<uint8_t> scales(8 * sizeof(uint16_t));
        for (size_t row = 0; row < 8; ++row) {
            const uint16_t maximum = 0x7bff;
            std::memcpy(scales.data() + row * sizeof(maximum), &maximum, sizeof(maximum));
        }
        std::vector<uint32_t> groups(128);
        write_single_shard_model(path, {
            { "module.qweight", "I32", { 16, 8 }, std::vector<uint8_t>(16 * 8 * sizeof(uint32_t)) },
            { "module.qzeros",  "I32", { 1, 1 },  i32_bytes({ 14 })                               },
            { "module.scales",  "F16", { 1, 8 },  scales                                          },
            { "module.g_idx",   "I32", { 128 },   i32_bytes(groups)                               },
        });
        write_text(path / "config.json", gptq_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        require(weight.has_value(), "GPTQ overflow fixture did not bind");
        require_rejected([&] { (void) adapters.read(*weight); },
                         "GPTQ accepted a minimum that overflows Q4_1");
    }
    {
        const auto path = dir.path / "gptq-nonidentity-groups";
        std::vector<uint32_t> groups(128);
        groups.back() = 1;
        write_single_shard_model(path, {
            { "module.qweight", "I32", { 16, 8 }, std::vector<uint8_t>(16 * 8 * 4) },
            { "module.qzeros",  "I32", { 1, 1 },  std::vector<uint8_t>(4)          },
            { "module.scales",  "F16", { 1, 8 },  std::vector<uint8_t>(16)         },
            { "module.g_idx",   "I32", { 128 },   i32_bytes(groups)                 },
        });
        write_text(path / "config.json", gptq_config);
        const auto registry = llama_safetensors_registry::load(path);
        require_rejected([&] { (void) llama_safetensors_quant_adapters(llama_safetensors_read_json(path / "config.json"), registry); },
                         "GPTQ contract accepted a non-identity group map");
    }
    {
        const auto path = dir.path / "mxfp8";
        write_single_shard_model(path, {
            { "module.weight",       "F8_E4M3", { 2, 32 }, std::vector<uint8_t>(64, 0x38) },
            { "module.weight_scale", "U8",      { 2, 1 },  { 127, 128 }                     },
        });
        write_text(path / "config.json", mxfp8_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        const auto scale  = adapters.bind("module", llama_safetensors_quant_role::WEIGHT_SCALE);
        const auto input  = adapters.bind("module", llama_safetensors_quant_role::INPUT_SCALE);
        require(weight.has_value() && weight->target_type == GGML_TYPE_F8_E4M3 &&
                    weight->target_shape == std::vector<int64_t>({ 32, 2 }),
                "MXFP8 weight binding is wrong");
        require(scale.has_value() && scale->target_type == GGML_TYPE_I8 &&
                    scale->target_shape == std::vector<int64_t>({ 1, 2 }),
                "MXFP8 E8M0 scale binding is wrong");
        require(input.has_value() && input->target_type == GGML_TYPE_I32 &&
                    input->materialization == llama_safetensors_quant_materialization::DYNAMIC_MXFP8_MARKER,
                "MXFP8 dynamic-input marker binding is wrong");
        require(adapters.read(*weight).size() == 64 && adapters.read(*scale) == std::vector<uint8_t>({ 127, 128 }),
                "MXFP8 source bytes changed during binding");
        adapters.consume(*weight);
        adapters.consume(*scale);
        adapters.consume(*input);
        adapters.validate_complete();
    }
    {
        const auto path = dir.path / "mxfp8-reserved-scale";
        write_single_shard_model(path, {
            { "module.weight",       "F8_E4M3", { 1, 32 }, std::vector<uint8_t>(32, 0x38) },
            { "module.weight_scale", "U8",      { 1, 1 },  { 0xff }                         },
        });
        write_text(path / "config.json", mxfp8_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(llama_safetensors_read_json(path / "config.json"), registry);
        const auto scale = adapters.bind("module", llama_safetensors_quant_role::WEIGHT_SCALE);
        require(scale.has_value(), "MXFP8 reserved-scale fixture did not bind");
        require_rejected([&] { (void) adapters.finalize(*scale, adapters.read(*scale)); },
                         "MXFP8 accepted the reserved E8M0 NaN scale encoding");
    }
    {
        const auto path = dir.path / "fp8-group";
        std::vector<uint8_t> scales(4 * sizeof(uint16_t));
        for (size_t i = 0; i < 4; ++i) {
            store_bf16(scales, i, 0.25f * float(i + 1));
        }
        write_single_shard_model(path, {
            { "module.weight",       "F8_E4M3", { 2, 64 }, std::vector<uint8_t>(128, 0x38) },
            { "module.weight_scale", "BF16",    { 2, 2 },  scales                         },
        });
        write_text(path / "config.json", fp8_group_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        const auto scale  = adapters.bind("module", llama_safetensors_quant_role::WEIGHT_SCALE);
        const auto input  = adapters.bind("module", llama_safetensors_quant_role::INPUT_SCALE);
        require(weight.has_value() && weight->target_type == GGML_TYPE_F8_E4M3 &&
                    weight->target_shape == std::vector<int64_t>({ 64, 2 }),
                "grouped FP8 weight binding is wrong");
        require(scale.has_value() && scale->target_type == GGML_TYPE_BF16 &&
                    scale->target_shape == std::vector<int64_t>({ 2, 2 }),
                "grouped FP8 scale binding is wrong");
        require(input.has_value() && input->target_type == GGML_TYPE_I16 &&
                    input->materialization == llama_safetensors_quant_materialization::DYNAMIC_FP8_GROUP_MARKER,
                "grouped FP8 dynamic-input marker binding is wrong");
        require(adapters.finalize(*scale, adapters.read(*scale)) == scales,
                "grouped FP8 BF16 scales changed during binding");
        adapters.consume(*weight);
        adapters.consume(*scale);
        adapters.consume(*input);
        adapters.validate_complete();
    }
    {
        const auto path = dir.path / "eetq-int8";
        constexpr size_t cols = 32;
        constexpr size_t rows = 2;
        std::vector<uint8_t> weights(cols * rows);
        for (size_t col = 0; col < cols; ++col) {
            for (size_t row = 0; row < rows; ++row) {
                weights[col * rows + row] = uint8_t(int8_t(3 * int(row) + int(col) - 16));
            }
        }
        std::vector<uint8_t> scales(rows * sizeof(ggml_fp16_t));
        store_f16(scales, 0, 0.25f);
        store_f16(scales, 1, 0.5f);
        write_single_shard_model(path, {
            { "module.qweight",       "I8",  { cols, rows }, weights },
            { "module.weight_scales", "F16", { rows },       scales  },
        });
        write_text(path / "config.json", eetq_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(
            llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        require(weight.has_value() && weight->target_type == GGML_TYPE_Q8_0 &&
                    weight->target_shape == std::vector<int64_t>({ cols, rows }) &&
                    weight->materialization == llama_safetensors_quant_materialization::EETQ_REPACK &&
                    !adapters.bind("module", llama_safetensors_quant_role::WEIGHT_SCALE).has_value() &&
                    !adapters.bind("module", llama_safetensors_quant_role::INPUT_SCALE).has_value() &&
                    adapters.file_type() == LLAMA_FTYPE_MOSTLY_Q8_0,
                "EETQ W8A16 binding is wrong");
        const std::vector<uint8_t> packed = adapters.read(*weight);
        require(packed.size() == rows * (sizeof(ggml_fp16_t) + cols),
                "EETQ Q8_0 repack size is wrong");
        for (size_t row = 0; row < rows; ++row) {
            const uint8_t * block = packed.data() + row * (sizeof(ggml_fp16_t) + cols);
            require(load_u16(block) == ggml_fp32_to_fp16(row == 0 ? 0.25f : 0.5f),
                    "EETQ Q8_0 scale changed during repack");
            for (size_t col = 0; col < cols; ++col) {
                require(block[sizeof(ggml_fp16_t) + col] == weights[col * rows + row],
                        "EETQ Q8_0 transpose changed an INT8 code");
            }
        }
        adapters.consume(*weight);
        adapters.validate_complete();
    }
    {
        const auto path = dir.path / "quanto-int4";
        constexpr size_t cols = 256;
        constexpr size_t rows = 2;
        constexpr size_t group_size = 128;
        constexpr size_t groups = rows * cols / group_size;
        std::vector<uint8_t> weights((groups / 2) * group_size);
        const auto code = [](size_t group, size_t local) {
            return uint8_t((3 * group + local) & 0x0f);
        };
        for (size_t packed_group = 0; packed_group < groups / 2; ++packed_group) {
            for (size_t local = 0; local < group_size; ++local) {
                weights[packed_group * group_size + local] =
                    code(packed_group, local) | (code(packed_group + groups / 2, local) << 4);
            }
        }
        std::vector<uint8_t> scales(groups * sizeof(uint16_t));
        std::vector<uint8_t> shifts(groups * sizeof(uint16_t));
        for (size_t group = 0; group < groups; ++group) {
            store_bf16(scales, group, 0.25f * float(group + 1));
            store_bf16(shifts, group, 0.125f * float(group + 1));
        }
        write_single_shard_model(path, {
            { "model.layers.0.mlp.down_proj.weight._data._data", "U8",  { groups / 2, group_size }, weights },
            { "model.layers.0.mlp.down_proj.weight._scale",     "BF16", { groups, 1 },            scales  },
            { "model.layers.0.mlp.down_proj.weight._shift",     "BF16", { groups, 1 },            shifts  },
        });
        write_text(path / "config.json", R"({
            "model_type":"mistral","hidden_size":2,"intermediate_size":256,
            "num_attention_heads":1,"num_key_value_heads":1,"head_dim":2,"vocab_size":32,
            "quantization_config":{"quant_method":"quanto","quantization_map":{
                "model.layers.0.mlp.down_proj":{"weights":"qint4","activations":"none"}
            }}})" );
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(
            llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind(
            "model.layers.0.mlp.down_proj", llama_safetensors_quant_role::WEIGHT);
        const auto marker = adapters.bind(
            "model.layers.0.mlp.down_proj", llama_safetensors_quant_role::WEIGHT_SCALE);
        require(weight.has_value() && weight->target_type == GGML_TYPE_Q4_1 &&
                    weight->target_shape == std::vector<int64_t>({ cols, rows }) &&
                    weight->materialization == llama_safetensors_quant_materialization::QUANTO_INT4_REPACK &&
                    adapters.file_type() == LLAMA_FTYPE_MOSTLY_Q4_1,
                "Quanto qint4 binding is wrong");
        require(marker.has_value() && marker->target_type == GGML_TYPE_I8 &&
                    marker->target_shape == std::vector<int64_t>({ 1 }) &&
                    marker->materialization == llama_safetensors_quant_materialization::QUANTO_W4A16_MARKER &&
                    adapters.read(*marker) == std::vector<uint8_t>({ 0 }),
                "Quanto W4A16 execution marker is wrong");
        const std::vector<uint8_t> packed = adapters.read(*weight);
        constexpr size_t block_size = 2 * sizeof(ggml_fp16_t) + 16;
        require(packed.size() == rows * (cols / 32) * block_size,
                "Quanto Q4_1 repack size is wrong");
        for (size_t row = 0; row < rows; ++row) {
            for (size_t block = 0; block < cols / 32; ++block) {
                const size_t group = row * (cols / group_size) + block / (group_size / 32);
                const uint8_t * out = packed.data() + (row * (cols / 32) + block) * block_size;
                require(load_u16(out) == ggml_fp32_to_fp16(0.25f * float(group + 1)) &&
                            load_u16(out + sizeof(ggml_fp16_t)) ==
                                ggml_fp32_to_fp16(-0.125f * float(group + 1)),
                        "Quanto Q4_1 scale or shift changed during repack");
                for (size_t local = 0; local < 16; ++local) {
                    const size_t group_col = (block * 32) % group_size;
                    require(out[2 * sizeof(ggml_fp16_t) + local] ==
                                uint8_t(code(group, group_col + local) |
                                        (code(group, group_col + local + 16) << 4)),
                            "Quanto Q4_1 code changed during repack");
                }
            }
        }
        adapters.consume(*weight);
        adapters.consume(*marker);
        adapters.validate_complete();
    }
    {
        const auto path = dir.path / "quanto-int8";
        constexpr size_t cols = 128;
        constexpr size_t rows = 64;
        std::vector<uint8_t> weights(rows * cols);
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] = uint8_t(int(i % 255) - 127);
        }
        std::vector<uint8_t> scales(rows * sizeof(uint16_t));
        for (size_t row = 0; row < rows; ++row) {
            store_bf16(scales, row, 0.00390625f * float(row + 1));
        }
        write_single_shard_model(path, {
            { "model.layers.0.mlp.down_proj.weight._data",  "I8",   { rows, cols }, weights },
            { "model.layers.0.mlp.down_proj.weight._scale", "BF16", { rows, 1 },    scales  },
        });
        write_text(path / "config.json", R"({
            "model_type":"mistral","hidden_size":64,"intermediate_size":128,
            "num_attention_heads":1,"num_key_value_heads":1,"head_dim":64,"vocab_size":32,
            "quantization_config":{"quant_method":"quanto","quantization_map":{
                "model.layers.0.mlp.down_proj":{"weights":"qint8","activations":"none"}
            }}})" );
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(
            llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind(
            "model.layers.0.mlp.down_proj", llama_safetensors_quant_role::WEIGHT);
        const auto scale = adapters.bind(
            "model.layers.0.mlp.down_proj", llama_safetensors_quant_role::WEIGHT_SCALE);
        const auto marker = adapters.bind(
            "model.layers.0.mlp.down_proj", llama_safetensors_quant_role::INPUT_SCALE);
        require(weight.has_value() && weight->target_type == GGML_TYPE_I8 &&
                    weight->target_shape == std::vector<int64_t>({ cols, rows }) &&
                    weight->materialization == llama_safetensors_quant_materialization::RAW &&
                    adapters.read(*weight) == weights &&
                    adapters.file_type() == LLAMA_FTYPE_MOSTLY_Q8_0,
                "Quanto qint8 weight binding is wrong");
        require(scale.has_value() && scale->target_type == GGML_TYPE_I8 &&
                    scale->target_shape == std::vector<int64_t>({
                        static_cast<int64_t>(sizeof(ggml_w8a16_scale_header) + scales.size()) }) &&
                    scale->materialization == llama_safetensors_quant_materialization::QUANTO_W8A16_SCALE,
                "Quanto qint8 scale binding is wrong");
        const std::vector<uint8_t> bundle = adapters.read(*scale);
        ggml_w8a16_scale_header header;
        std::memcpy(&header, bundle.data(), sizeof(header));
        require(header.magic == GGML_W8A16_SCALE_MAGIC && header.version == 1 &&
                    header.n_channels == rows && header.values_offset == sizeof(header) &&
                    header.total_size == bundle.size() &&
                    std::memcmp(bundle.data() + header.values_offset, scales.data(), scales.size()) == 0,
                "Quanto W8A16 scale bundle is wrong");
        require(!marker.has_value(), "Quanto W8A16 unexpectedly created an activation scale");
        adapters.consume(*weight);
        adapters.consume(*scale);
        adapters.validate_complete();
    }
    {
        const auto path = dir.path / "quanto-fp8";
        constexpr size_t cols = 128;
        constexpr size_t rows = 64;
        std::vector<uint8_t> weights(rows * cols);
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] = uint8_t(i & 0x7f);
        }
        std::vector<uint8_t> scales(rows * sizeof(uint16_t));
        for (size_t row = 0; row < rows; ++row) {
            store_bf16(scales, row, 0.00390625f * float(row + 1));
        }
        write_single_shard_model(path, {
            { "model.layers.0.mlp.down_proj.weight._data",  "F8_E4M3", { rows, cols }, weights },
            { "model.layers.0.mlp.down_proj.weight._scale", "BF16",    { rows, 1 },    scales  },
        });
        write_text(path / "config.json", R"({
            "model_type":"mistral","hidden_size":64,"intermediate_size":128,
            "num_attention_heads":1,"num_key_value_heads":1,"head_dim":64,"vocab_size":32,
            "quantization_config":{"quant_method":"quanto","quantization_map":{
                "model.layers.0.mlp.down_proj":{"weights":"qfloat8_e4m3fn","activations":"none"}
            }}})" );
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(
            llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind(
            "model.layers.0.mlp.down_proj", llama_safetensors_quant_role::WEIGHT);
        const auto scale = adapters.bind(
            "model.layers.0.mlp.down_proj", llama_safetensors_quant_role::WEIGHT_SCALE);
        require(weight.has_value() && weight->target_type == GGML_TYPE_F8_E4M3 &&
                    weight->target_shape == std::vector<int64_t>({ cols, rows }) &&
                    adapters.read(*weight) == weights &&
                    adapters.file_type() == LLAMA_FTYPE_MOSTLY_F8_E4M3,
                "Quanto qfloat8 weight binding is wrong");
        require(scale.has_value() &&
                    scale->materialization == llama_safetensors_quant_materialization::QUANTO_W8A16_SCALE,
                "Quanto qfloat8 scale binding is wrong");
        adapters.consume(*weight);
        adapters.consume(*scale);
        adapters.validate_complete();
    }
    {
        const auto path = dir.path / "torchao-tiled-int4";
        constexpr size_t rows = 8;
        constexpr size_t cols = 128;
        std::vector<uint8_t> logical(rows * cols);
        std::vector<uint8_t> packed_rows(rows * cols / 2);
        for (size_t row = 0; row < rows; ++row) {
            for (size_t col = 0; col < cols; ++col) {
                logical[row * cols + col] = uint8_t((3 * row + col) & 0x0f);
            }
            for (size_t col = 0; col < cols; col += 2) {
                packed_rows[row * (cols / 2) + col / 2] =
                    uint8_t((logical[row * cols + col] << 4) | logical[row * cols + col + 1]);
            }
        }
        std::vector<uint8_t> qdata(32 * 4 * sizeof(uint32_t));
        for (size_t lane = 0; lane < 32; ++lane) {
            const size_t row = lane / 4;
            for (size_t inner = 0; inner < 8; inner += 2) {
                const size_t base = inner * 8;
                const size_t offset = lane % 4;
                const uint8_t v0 = packed_rows[row * (cols / 2) + base + offset];
                const uint8_t v1 = packed_rows[row * (cols / 2) + base + offset + 4];
                const uint8_t v2 = packed_rows[row * (cols / 2) + base + offset + 8];
                const uint8_t v3 = packed_rows[row * (cols / 2) + base + offset + 12];
                const uint32_t word = (uint32_t(v3 & 0x0f) << 28) | (uint32_t(v2 & 0x0f) << 24) |
                    (uint32_t(v1 & 0x0f) << 20) | (uint32_t(v0 & 0x0f) << 16) |
                    (uint32_t(v3 & 0xf0) << 8) | (uint32_t(v2 & 0xf0) << 4) |
                    uint32_t(v1 & 0xf0) | (uint32_t(v0 & 0xf0) >> 4);
                const size_t index = lane * 4 + inner / 2;
                std::memcpy(qdata.data() + index * sizeof(word), &word, sizeof(word));
            }
        }
        std::vector<uint8_t> scale_zero(rows * 2 * sizeof(uint16_t));
        for (size_t row = 0; row < rows; ++row) {
            store_bf16(scale_zero, row * 2, 0.5f);
            store_bf16(scale_zero, row * 2 + 1, 4.0f);
        }
        const json qmap = {
            { "model.layers.0.mlp.down_proj.weight", {
                { "_type", "Int4TilePackedTo4dTensor" },
                { "block_size", { 1, cols } },
                { "shape", { rows, cols } },
            } },
        };
        write_single_shard_model(path, {
            { "model.layers.0.mlp.down_proj.weight.__qdata", "I32", { 1, 1, 32, 4 }, qdata },
            { "model.layers.0.mlp.down_proj.weight.__scale_and_zero", "BF16", { 1, rows, 2 }, scale_zero },
        }, { { "quantization", qmap.dump() } });
        const auto registry = llama_safetensors_registry::load(path);
        require(registry.metadata("quantization") != nullptr,
                "TorchAO safetensors quantization metadata was discarded");
        llama_safetensors_quant_adapters adapters(llama_safetensors_json::object(), registry);
        const auto weight = adapters.bind(
            "model.layers.0.mlp.down_proj", llama_safetensors_quant_role::WEIGHT);
        require(weight.has_value() && weight->target_type == GGML_TYPE_Q4_1 &&
                    weight->target_shape == std::vector<int64_t>({ cols, rows }) &&
                    weight->materialization == llama_safetensors_quant_materialization::TORCHAO_INT4_REPACK,
                "TorchAO tiled-int4 binding is wrong");
        const std::vector<uint8_t> repacked = adapters.read(*weight);
        constexpr size_t q4_block_size = 2 * sizeof(ggml_fp16_t) + 16;
        for (size_t row = 0; row < rows; ++row) {
            for (size_t block = 0; block < cols / 32; ++block) {
                const uint8_t * bytes = repacked.data() + (row * (cols / 32) + block) * q4_block_size;
                ggml_fp16_t d, m;
                std::memcpy(&d, bytes, sizeof(d));
                std::memcpy(&m, bytes + sizeof(d), sizeof(m));
                require(ggml_fp16_to_fp32(d) == 0.5f && ggml_fp16_to_fp32(m) == 0.0f,
                        "TorchAO tiled-int4 scale/zero conversion is wrong");
                for (size_t i = 0; i < 16; ++i) {
                    const size_t col = block * 32;
                    require((bytes[2 * sizeof(ggml_fp16_t) + i] & 0x0f) == logical[row * cols + col + i] &&
                                (bytes[2 * sizeof(ggml_fp16_t) + i] >> 4) == logical[row * cols + col + i + 16],
                            "TorchAO tiled-int4 lane unpacking is wrong");
                }
            }
        }
        adapters.consume(*weight);
        adapters.validate_complete();
    }
    {
        const auto path = dir.path / "torchao-unpacked-int8";
        constexpr size_t rows = 2;
        constexpr size_t cols = 64;
        std::vector<uint8_t> qdata(rows * cols);
        for (size_t i = 0; i < qdata.size(); ++i) {
            qdata[i] = static_cast<uint8_t>(static_cast<int8_t>(int(i % 127) - 63));
        }
        std::vector<uint8_t> scales(rows * sizeof(uint16_t));
        store_bf16(scales, 0, 0.25f);
        store_bf16(scales, 1, 0.5f);
        std::vector<uint8_t> zeros(rows, 0);
        const json qmap = {
            { "model.embed_tokens.weight", {
                { "_type", "IntxUnpackedToInt8Tensor" },
                { "target_dtype", "torch.int8" },
                { "block_size", { 1, cols } },
                { "dtype", "torch.bfloat16" },
                { "activation_quantization", nullptr },
            } },
        };
        write_single_shard_model(path, {
            { "model.embed_tokens.weight.__qdata", "I8", { rows, cols }, qdata },
            { "model.embed_tokens.weight.__scale", "BF16", { rows, 1 }, scales },
            { "model.embed_tokens.weight.__zero_point", "I8", { rows, 1 }, zeros },
        }, { { "quantization", qmap.dump() } });
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(llama_safetensors_json::object(), registry);
        const auto weight = adapters.bind("model.embed_tokens", llama_safetensors_quant_role::WEIGHT);
        require(weight.has_value() && weight->target_type == GGML_TYPE_Q8_0 &&
                    weight->target_shape == std::vector<int64_t>({ cols, rows }) &&
                    weight->materialization == llama_safetensors_quant_materialization::TORCHAO_INTX_REPACK,
                "TorchAO unpacked-int8 binding is wrong");
        const std::vector<uint8_t> repacked = adapters.read(*weight);
        constexpr size_t q8_block_size = sizeof(ggml_fp16_t) + 32;
        for (size_t row = 0; row < rows; ++row) {
            for (size_t block = 0; block < cols / 32; ++block) {
                const uint8_t * bytes = repacked.data() + (row * 2 + block) * q8_block_size;
                ggml_fp16_t d;
                std::memcpy(&d, bytes, sizeof(d));
                require(ggml_fp16_to_fp32(d) == (row == 0 ? 0.25f : 0.5f) &&
                            std::memcmp(bytes + sizeof(d), qdata.data() + row * cols + block * 32, 32) == 0,
                        "TorchAO unpacked-int8 Q8_0 repack is wrong");
            }
        }
        adapters.consume(*weight);
        adapters.validate_complete();
    }
    {
        // ExecuTorch Qwen3.5-MoE stores routed HQQ weights as adjacent
        // low/high nibbles with one BF16 scale per 128 input values. Exercise
        // every row, expert, scale group, and Q4_0 lane rather than relying on
        // the full public-model smoke test alone.
        const auto path = dir.path / "executorch-hqq-experts";
        constexpr size_t experts = 2;
        constexpr size_t rows    = 256;
        constexpr size_t cols    = 128;
        std::vector<uint8_t> packed(experts * rows * cols / 2);
        std::vector<uint8_t> scales(experts * rows * sizeof(uint16_t));
        const auto code = [](size_t expert, size_t row, size_t col) {
            return uint8_t((3 * expert + 5 * row + col) & 0x0f);
        };
        for (size_t expert = 0; expert < experts; ++expert) {
            for (size_t row = 0; row < rows; ++row) {
                const size_t source_row = (expert * rows + row) * (cols / 2);
                for (size_t col = 0; col < cols; col += 2) {
                    packed[source_row + col / 2] =
                        code(expert, row, col) | (code(expert, row, col + 1) << 4);
                }
                store_bf16(scales, expert * rows + row, 0.25f * float(1 + (expert + row) % 3));
            }
        }
        write_single_shard_model(path, {
            { "embed_tokens.weight",             "BF16", { 1 },                    { 0, 0 } },
            { "layers.0.mlp.experts.w1",         "I8",   { experts, rows, cols/2 }, packed },
            { "layers.0.mlp.experts.w1_scale",   "BF16", { experts, rows, 1 },      scales },
        });
        write_text(path / "tokenizer.json", "{}");
        const json config = {
            { "model_type", "qwen3_5_moe_text" },
            { "num_hidden_layers", 40 },
            { "mtp_num_hidden_layers", 0 },
            { "linear_num_key_heads", 2 },
            { "linear_num_value_heads", 4 },
            { "linear_key_head_dim", 32 },
            { "linear_value_head_dim", 32 },
            { "full_attention_interval", 4 },
        };
        llama_safetensors_qwen35_importer importer(path, config);
        ggml_type type;
        std::array<int64_t, GGML_MAX_DIMS> ne;
        require(importer.describe("blk.0.ffn_gate_up_exps.weight", type, ne) &&
                    type == GGML_TYPE_Q4_0 && ne[0] == cols && ne[1] == rows && ne[2] == experts,
                "ExecuTorch HQQ expert binding has the wrong target contract");
        const size_t row_size = ggml_row_size(GGML_TYPE_Q4_0, cols);
        const std::vector<uint8_t> result = importer.materialize(
            "blk.0.ffn_gate_up_exps.weight", type, experts * rows * row_size);
        constexpr size_t block_size = 32;
        constexpr size_t block_bytes = sizeof(ggml_fp16_t) + block_size / 2;
        for (size_t expert = 0; expert < experts; ++expert) {
            for (size_t row = 0; row < rows; ++row) {
                for (size_t block = 0; block < cols / block_size; ++block) {
                    const uint8_t * dst = result.data() +
                        (expert * rows + row) * row_size + block * block_bytes;
                    ggml_fp16_t d;
                    std::memcpy(&d, dst, sizeof(d));
                    require(ggml_fp16_to_fp32(d) == 0.25f * float(1 + (expert + row) % 3),
                            "ExecuTorch HQQ expert scale changed during repack");
                    for (size_t lane = 0; lane < block_size / 2; ++lane) {
                        const size_t col = block * block_size + lane;
                        require(dst[sizeof(d) + lane] ==
                                    (code(expert, row, col) | (code(expert, row, col + 16) << 4)),
                                "ExecuTorch HQQ expert nibble layout changed during repack");
                    }
                }
            }
        }
    }
    {
        // Generic Transformers HQQ serializes 4-bit codes by stacking the
        // high-nibble half of the flattened group matrix over its low-nibble
        // half. Preserve every group/lane and its learned affine parameters.
        const auto path = dir.path / "hqq-int4-axis1";
        constexpr size_t rows = 4;
        constexpr size_t cols = 64;
        constexpr size_t group_size = 64;
        constexpr size_t groups = rows * cols / group_size;
        std::vector<uint8_t> packed((groups / 2) * group_size);
        std::vector<uint8_t> scales(groups * sizeof(uint16_t));
        std::vector<uint8_t> zeros(groups * sizeof(uint16_t));
        const auto code = [](size_t group, size_t col) {
            return uint8_t((5 * group + 3 * col) & 0x0f);
        };
        for (size_t group = 0; group < groups; ++group) {
            store_f16(scales, group, 0.25f * float(group + 1));
            store_f16(zeros, group, 2.0f + float(group));
            for (size_t col = 0; col < group_size; ++col) {
                uint8_t & dst = packed[(group % (groups / 2)) * group_size + col];
                dst |= uint8_t(code(group, col) << (group < groups / 2 ? 4 : 0));
            }
        }
        write_single_shard_model(path, {
            { "module.W_q",           "U8",  { groups/2, group_size }, packed },
            { "module.scale",         "F16", { groups, 1 }, scales },
            { "module.zero",          "F16", { groups, 1 }, zeros },
            { "module.shape",         "I64", { 2 }, i64_bytes({ rows, cols }) },
            { "module.axis",          "I32", {}, i32_bytes({ 1 }) },
            { "module.group_size",    "I32", {}, i32_bytes({ group_size }) },
            { "module.nbits",         "I32", {}, i32_bytes({ 4 }) },
            { "module.packing",       "U8",  { 7 }, { '4','b','i','t','_','u','8' } },
            { "module.channel_wise",  "U8",  {}, { 1 } },
            { "module.quant_scale",   "U8",  {}, { 0 } },
            { "module.quant_zero",    "U8",  {}, { 0 } },
            { "module.round_zero",    "U8",  {}, { 1 } },
            { "module.view_as_float", "U8",  {}, { 0 } },
        });
        const json config = {
            { "quantization_config", {
                { "quant_method", "hqq" }, { "skip_modules", json::array({ "lm_head" }) },
                { "quant_config", {
                    { "offload_meta", false }, { "scale_quant_params", nullptr },
                    { "zero_quant_params", nullptr },
                    { "weight_quant_params", {
                        { "axis", 1 }, { "channel_wise", true }, { "group_size", group_size },
                        { "nbits", 4 }, { "round_zero", true }, { "view_as_float", false },
                    } },
                } },
            } },
        };
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(config, registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        require(weight.has_value() && weight->target_type == GGML_TYPE_Q4_1 &&
                    weight->target_shape == std::vector<int64_t>({ cols, rows }) &&
                    weight->materialization == llama_safetensors_quant_materialization::HQQ_INT4_REPACK &&
                    adapters.summary().hqq_int4 == 1 && !adapters.applies("lm_head"),
                "generic HQQ INT4 binding is wrong");
        const std::vector<uint8_t> repacked = adapters.read(*weight);
        constexpr size_t block_bytes = 2 * sizeof(ggml_fp16_t) + 16;
        for (size_t row = 0; row < rows; ++row) {
            for (size_t block = 0; block < cols / 32; ++block) {
                std::array<uint8_t, 32> codes;
                for (size_t col = 0; col < 32; ++col) {
                    codes[col] = code(row, block * 32 + col);
                }
                require_q4_1_block(
                    repacked.data() + (row * 2 + block) * block_bytes,
                    0.25f * float(row + 1), uint8_t(2 + row), codes,
                    "generic HQQ INT4 repack changed a scale, zero, or code lane");
            }
        }
        adapters.consume(*weight);
        adapters.validate_complete();

        json invalid = config;
        invalid["quantization_config"]["quant_config"]["weight_quant_params"]["axis"] = 0;
        require_rejected([&] { (void) llama_safetensors_quant_adapters(invalid, registry); },
                "generic HQQ accepted an unsupported quantization axis");
    }
    {
        const auto path = dir.path / "bnb-int8";
        constexpr size_t rows = 3;
        constexpr size_t cols = 64;
        std::vector<uint8_t> weights(rows * cols);
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] = uint8_t(int(i % 255) - 127);
        }
        const std::array<float, rows> scb_values = { 12.7f, 25.4f, 6.35f };
        std::vector<uint8_t> scb(sizeof(scb_values));
        std::memcpy(scb.data(), scb_values.data(), scb.size());
        write_single_shard_model(path, {
            { "module.weight", "I8",  { rows, cols }, weights },
            { "module.SCB",    "F32", { rows },       scb     },
        });
        write_text(path / "config.json", bnb_int8_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        const auto scale = adapters.bind("module", llama_safetensors_quant_role::WEIGHT_SCALE);
        const auto input = adapters.bind("module", llama_safetensors_quant_role::INPUT_SCALE);
        require(weight.has_value() && weight->target_type == GGML_TYPE_I8 &&
                    weight->target_shape == std::vector<int64_t>({ cols, rows }) &&
                    adapters.read(*weight) == weights && adapters.file_type() == LLAMA_FTYPE_MOSTLY_Q8_0,
                "BitsAndBytes INT8 weight binding is wrong");
        require(scale.has_value() && scale->target_type == GGML_TYPE_F32 &&
                    scale->target_shape == std::vector<int64_t>({ rows }) &&
                    scale->materialization == llama_safetensors_quant_materialization::BNB_INT8_SCALE,
                "BitsAndBytes INT8 scale binding is wrong");
        require(input.has_value() && input->target_type == GGML_TYPE_I32 &&
                    input->target_shape == std::vector<int64_t>({ 1 }) &&
                    input->materialization == llama_safetensors_quant_materialization::DYNAMIC_INT8_MARKER,
                "BitsAndBytes INT8 outlier-threshold binding is wrong");
        const std::vector<uint8_t> threshold_data = adapters.read(*input);
        float threshold;
        std::memcpy(&threshold, threshold_data.data(), sizeof(threshold));
        require(threshold == 6.0f, "BitsAndBytes INT8 outlier threshold changed during binding");
        const std::vector<uint8_t> scaled = adapters.finalize(*scale, adapters.read(*scale));
        for (size_t row = 0; row < rows; ++row) {
            float value;
            std::memcpy(&value, scaled.data() + row * sizeof(value), sizeof(value));
            require(std::abs(value - scb_values[row] / 127.0f) < 1e-7f,
                    "BitsAndBytes INT8 SCB did not become a channel multiplier");
        }
        adapters.consume(*weight);
        adapters.consume(*scale);
        adapters.consume(*input);
        adapters.validate_complete();
    }
    {
        const auto path = dir.path / "bnb-nf4";
        constexpr size_t rows = 2;
        constexpr size_t cols = 64;
        std::vector<uint8_t> quant_map(16 * sizeof(float));
        std::vector<uint8_t> nested_map(256 * sizeof(float));
        std::vector<uint8_t> nested_absmax(sizeof(float));
        for (size_t i = 0; i < 16; ++i) {
            const float value = float(int(i) - 8) / 8.0f;
            std::memcpy(quant_map.data() + i * sizeof(float), &value, sizeof(value));
        }
        for (size_t i = 0; i < 256; ++i) {
            const float value = float(i) / 255.0f;
            std::memcpy(nested_map.data() + i * sizeof(float), &value, sizeof(value));
        }
        const float nested_scale = 2.0f;
        std::memcpy(nested_absmax.data(), &nested_scale, sizeof(nested_scale));
        const std::string state =
            R"({"quant_type":"nf4","blocksize":64,"dtype":"bfloat16","shape":[2,64],"nested_blocksize":256,"nested_dtype":"float32","nested_offset":0.25})";
        write_single_shard_model(path, {
            { "module.weight", "U8", { rows * cols / 2, 1 }, std::vector<uint8_t>(rows * cols / 2, 0x18) },
            { "module.weight.absmax", "U8", { 2 }, { 0, 255 } },
            { "module.weight.quant_map", "F32", { 16 }, quant_map },
            { "module.weight.quant_state.bitsandbytes__nf4", "U8", { state.size() },
              std::vector<uint8_t>(state.begin(), state.end()) },
            { "module.weight.nested_absmax", "F32", { 1 }, nested_absmax },
            { "module.weight.nested_quant_map", "F32", { 256 }, nested_map },
        });
        write_text(path / "config.json", bnb_nf4_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        const auto scale = adapters.bind("module", llama_safetensors_quant_role::WEIGHT_SCALE);
        require(weight.has_value() && weight->target_type == GGML_TYPE_BNB_NF4 &&
                    weight->target_shape == std::vector<int64_t>({ cols, rows }) &&
                    adapters.read(*weight).size() == rows * cols / 2,
                "BitsAndBytes NF4 weight binding is wrong");
        require(scale.has_value() && scale->target_type == GGML_TYPE_I8 &&
                    scale->materialization == llama_safetensors_quant_materialization::BNB_SCALE_BUNDLE,
                "BitsAndBytes NF4 scale binding is wrong");
        const std::vector<uint8_t> bundle = adapters.read(*scale);
        ggml_bnb_scale_header header;
        std::memcpy(&header, bundle.data(), sizeof(header));
        require(header.magic == GGML_BNB_SCALE_MAGIC && header.n_blocks == 2 &&
                    header.block_size == 64 && header.nested_block_size == 256 &&
                    header.total_size == bundle.size(),
                "BitsAndBytes NF4 scale bundle is malformed");
        adapters.consume(*weight);
        adapters.consume(*scale);
        adapters.validate_complete();
    }
    {
        llama_safetensors_json config = llama_safetensors_json::parse(bnb_int8_config);
        config["quantization_config"]["llm_int8_threshold"] = -1.0;
        require_rejected([&] { (void) llama_safetensors_quant_config::from_json(config); },
                         "BitsAndBytes INT8 accepted a negative outlier threshold");
    }
    {
        const auto config = llama_safetensors_json::parse(
            R"({"quantization_config":{"bits":2,"checkpoint_format":"gptq","desc_act":false,"group_size":128,"pack_dtype":"int32","quant_method":"gptq","sym":true}})");
        require_rejected([&] { (void) llama_safetensors_quant_config::from_json(config); },
                         "GPTQ INT2 was accepted without an exact native affine INT2 type");
    }
    {
        const auto config = llama_safetensors_json::parse(
            R"({"quantization_config":{"quant_method":"compressed-tensors","format":"pack-quantized","config_groups":{"int2":{"format":"pack-quantized","targets":["int2"],"input_activations":null,"output_activations":null,"weights":{"actorder":null,"block_structure":null,"dynamic":false,"group_size":128,"num_bits":2,"scale_dtype":null,"strategy":"group","symmetric":true,"type":"int","zp_dtype":null}}}}})");
        require_rejected([&] { (void) llama_safetensors_quant_config::from_json(config); },
                         "compressed-tensors W2A16 was accepted without an exact native affine INT2 type");
    }
    {
        const auto path = dir.path / "bnb-fp4";
        constexpr size_t rows = 2;
        constexpr size_t cols = 64;
        std::vector<uint8_t> quant_map(16 * sizeof(float));
        std::vector<uint8_t> absmax(rows * sizeof(float));
        for (size_t i = 0; i < 16; ++i) {
            const float value = (i & 8) ? -float(i & 7) / 7.0f : float(i) / 7.0f;
            std::memcpy(quant_map.data() + i * sizeof(float), &value, sizeof(value));
        }
        const std::array<float, rows> block_scales = { 0.5f, 1.25f };
        std::memcpy(absmax.data(), block_scales.data(), absmax.size());
        const std::string state =
            R"({"quant_type":"fp4","blocksize":64,"dtype":"bfloat16","shape":[2,64]})";
        write_single_shard_model(path, {
            { "module.weight", "U8", { rows * cols / 2, 1 }, std::vector<uint8_t>(rows * cols / 2, 0x3c) },
            { "module.weight.absmax", "F32", { rows }, absmax },
            { "module.weight.quant_map", "F32", { 16 }, quant_map },
            { "module.weight.quant_state.bitsandbytes__fp4", "U8", { state.size() },
              std::vector<uint8_t>(state.begin(), state.end()) },
        });
        write_text(path / "config.json", bnb_fp4_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        const auto scale = adapters.bind("module", llama_safetensors_quant_role::WEIGHT_SCALE);
        require(weight.has_value() && weight->target_type == GGML_TYPE_BNB_FP4 &&
                    weight->target_shape == std::vector<int64_t>({ cols, rows }),
                "BitsAndBytes FP4 weight binding is wrong");
        require(scale.has_value() && scale->target_type == GGML_TYPE_I8 &&
                    scale->materialization == llama_safetensors_quant_materialization::BNB_SCALE_BUNDLE,
                "BitsAndBytes FP4 scale binding is wrong");
        const std::vector<uint8_t> bundle = adapters.read(*scale);
        ggml_bnb_scale_header header;
        std::memcpy(&header, bundle.data(), sizeof(header));
        require(header.magic == GGML_BNB_SCALE_MAGIC && header.n_blocks == rows &&
                    header.block_size == cols && header.nested_block_size == 0 &&
                    header.total_size == bundle.size(),
                "BitsAndBytes FP4 scale bundle is malformed");
        adapters.consume(*weight);
        adapters.consume(*scale);
        adapters.validate_complete();
    }
    {
        const auto path = dir.path / "w4a8-fp8";
        constexpr size_t rows = 2;
        constexpr size_t cols = 128;
        std::vector<uint8_t> packed(rows * cols / 2);
        for (size_t i = 0; i < packed.size(); ++i) {
            packed[i] = uint8_t(i);
        }
        std::vector<uint8_t> channel_scales;
        for (float value : { 2.0f, 4.0f }) {
            const auto bytes = f32_bytes(value);
            channel_scales.insert(channel_scales.end(), bytes.begin(), bytes.end());
        }
        write_single_shard_model(path, {
            { "module.weight_packed",     "I32",     { rows, cols / 8 }, packed                },
            { "module.weight_scale",      "F8_E4M3", { rows, 1 },        { 0x38, 0x38 }        },
            { "module.weight_chan_scale", "F32",     { rows, 1 },        channel_scales         },
            { "module.weight_shape",      "I64",     { 2 },              i64_bytes({ rows, cols }) },
        });
        write_text(path / "config.json", w4a8_fp8_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        const auto input  = adapters.bind("module", llama_safetensors_quant_role::INPUT_SCALE);
        require(weight.has_value() && weight->target_type == GGML_TYPE_Q4_A32 &&
                    weight->target_shape == std::vector<int64_t>({ 128, 2 }),
                "W4A8-FP8 weight binding is wrong");
        require(input.has_value() && input->target_type == GGML_TYPE_I16 &&
                    input->materialization == llama_safetensors_quant_materialization::DYNAMIC_W4A8_FP8_MARKER,
                "W4A8-FP8 dynamic-input marker is wrong");
        const auto canonical = adapters.read(*weight);
        require(canonical.size() == rows * 74 && canonical[8] == 0x88 && canonical[9] == 0x88,
                "W4A8-FP8 canonical zero point is wrong");
        ggml_bf16_t first_scale;
        std::memcpy(&first_scale.bits, canonical.data(), sizeof(first_scale.bits));
        std::vector<uint8_t> offset_codes(packed.begin(), packed.begin() + 64);
        for (uint8_t & byte : offset_codes) {
            byte ^= 0x88;
        }
        require(ggml_bf16_to_fp32(first_scale) == 2.0f &&
                    std::memcmp(canonical.data() + 10, offset_codes.data(), offset_codes.size()) == 0,
                "W4A8-FP8 canonical scale or codes are wrong");
        adapters.consume(*weight);
        adapters.consume(*input);
        adapters.validate_complete();
    }
    {
        const auto path = dir.path / "mxfp4";
        constexpr size_t rows = 2;
        constexpr size_t cols = 32;
        std::vector<uint8_t> packed(rows * cols / 2);
        for (size_t row = 0; row < rows; ++row) {
            for (size_t col = 0; col < cols; col += 2) {
                const uint8_t lo = (col + 3 * row) % 16;
                const uint8_t hi = (col + 1 + 3 * row) % 16;
                packed[row * cols / 2 + col / 2] = lo | (hi << 4);
            }
        }
        write_single_shard_model(path, {
            { "module.weight_packed", "U8", { rows, cols / 2 }, packed        },
            { "module.weight_scale",  "U8", { rows, 1 },        { 127, 128 } },
        });
        write_text(path / "config.json", mxfp4_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        const auto input  = adapters.bind("module", llama_safetensors_quant_role::INPUT_SCALE);
        require(weight.has_value() && weight->target_type == GGML_TYPE_MXFP4 &&
                    weight->materialization == llama_safetensors_quant_materialization::MXFP4_REPACK &&
                    weight->target_shape == std::vector<int64_t>({ cols, rows }) &&
                    adapters.file_type() == LLAMA_FTYPE_MOSTLY_MXFP4,
                "MXFP4 packed-weight binding is wrong");
        require(input.has_value() && input->target_type == GGML_TYPE_I32 &&
                    input->materialization == llama_safetensors_quant_materialization::DYNAMIC_MXFP4_MARKER,
                "MXFP4 dynamic-input marker binding is wrong");
        const std::vector<uint8_t> repacked = adapters.read(*weight);
        require(repacked.size() == rows * 17, "MXFP4 repack has the wrong size");
        for (size_t row = 0; row < rows; ++row) {
            const uint8_t * block = repacked.data() + row * 17;
            require(block[0] == 127 + row, "MXFP4 repack changed an E8M0 scale");
            for (size_t col = 0; col < cols; ++col) {
                const uint8_t code = col < 16 ? block[1 + col] & 0x0f : block[1 + col - 16] >> 4;
                require(code == (col + 3 * row) % 16, "MXFP4 repack changed a source code");
            }
        }
        adapters.consume(*weight);
        adapters.consume(*input);
        adapters.validate_complete();
    }
    {
        const auto path = dir.path / "mxfp4-reserved-scale";
        write_single_shard_model(path, {
            { "module.weight_packed", "U8", { 1, 16 }, std::vector<uint8_t>(16) },
            { "module.weight_scale",  "U8", { 1, 1 },  { 0xff }                  },
        });
        write_text(path / "config.json", mxfp4_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        require(weight.has_value(), "MXFP4 reserved-scale fixture did not bind");
        require_rejected([&] { (void) adapters.read(*weight); },
                         "MXFP4 accepted the reserved E8M0 NaN scale encoding");
    }
    {
        const auto path = dir.path / "nvfp4";
        write_single_shard_model(path,
                                 {
                                     { "module.weight_packed", "U8", { 2, 32 }, std::vector<uint8_t>(64) },
                                     { "module.weight_scale", "F8_E4M3", { 2, 4 }, std::vector<uint8_t>(8, 0x38) },
                                     { "module.weight_global_scale", "F32", {}, f32_bytes(1.0f) },
                                     { "module.input_global_scale", "F32", {}, f32_bytes(1.0f) },
        });
        write_text(path / "config.json", nvfp4_config);
        const auto                       registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(llama_safetensors_read_json(path / "config.json"), registry);
        const auto                       weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        const auto                       scale  = adapters.bind("module", llama_safetensors_quant_role::WEIGHT_SCALE);
        const auto                       input  = adapters.bind("module", llama_safetensors_quant_role::INPUT_SCALE);
        require(weight.has_value() && weight->target_type == GGML_TYPE_NVFP4 &&
                    weight->target_shape == std::vector<int64_t>({ 64, 2 }) && adapters.read(*weight).size() == 72,
                "NVFP4 packed-weight binding or repack is wrong");
        require(scale.has_value() && input.has_value(), "NVFP4 global scale binding is incomplete");
        adapters.consume(*weight);
        adapters.consume(*scale);
        adapters.consume(*input);
        adapters.validate_complete();
    }
    {
        const auto path = dir.path / "nvfp4-zero-point";
        write_single_shard_model(path,
                                 {
                                     { "module.weight_packed", "U8", { 2, 32 }, std::vector<uint8_t>(64) },
                                     { "module.weight_scale", "F8_E4M3", { 2, 4 }, std::vector<uint8_t>(8, 0x38) },
                                     { "module.weight_global_scale", "F32", {}, f32_bytes(1.0f) },
                                     { "module.input_global_scale", "F32", {}, f32_bytes(1.0f) },
                                     { "module.weight_zero_point", "U8", { 1 }, std::vector<uint8_t>(1) },
        });
        write_text(path / "config.json", nvfp4_config);
        const auto registry = llama_safetensors_registry::load(path);
        require_rejected([&] { (void) llama_safetensors_quant_adapters(llama_safetensors_read_json(path / "config.json"), registry); },
                         "NVFP4 contract accepted a zero point");
    }
    {
        const auto path = dir.path / "nvfp4-invalid-global-scale";
        write_single_shard_model(path,
                                 {
                                     { "module.weight_packed", "U8", { 2, 32 }, std::vector<uint8_t>(64) },
                                     { "module.weight_scale", "F8_E4M3", { 2, 4 }, std::vector<uint8_t>(8, 0x38) },
                                     { "module.weight_global_scale", "F32", {}, f32_bytes(0.0f) },
                                     { "module.input_global_scale", "F32", {}, f32_bytes(1.0f) },
        });
        write_text(path / "config.json", nvfp4_config);
        const auto registry = llama_safetensors_registry::load(path);
        require_rejected([&] { (void) llama_safetensors_quant_adapters(llama_safetensors_read_json(path / "config.json"), registry); },
                         "NVFP4 contract accepted a non-positive global scale");
    }
    {
        const auto path = dir.path / "modelopt-w4a16-nvfp4";
        write_single_shard_model(path, {
            { "module.weight",         "U8",       { 2, 32 }, std::vector<uint8_t>(64) },
            { "module.weight_scale",   "F8_E4M3", { 2, 4 },  std::vector<uint8_t>(8, 0x38) },
            { "module.weight_scale_2", "F32",      {},        f32_bytes(0.125f) },
            { "ignored.weight",        "BF16",     { 2, 32 }, std::vector<uint8_t>(128) },
        });
        write_text(path / "config.json", modelopt_w4a16_nvfp4_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(
            llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        const auto scale  = adapters.bind("module", llama_safetensors_quant_role::WEIGHT_SCALE);
        const auto input  = adapters.bind("module", llama_safetensors_quant_role::INPUT_SCALE);
        require(weight.has_value() && weight->primary == "module.weight" &&
                    weight->target_type == GGML_TYPE_NVFP4 &&
                    weight->target_shape == std::vector<int64_t>({ 64, 2 }) &&
                    adapters.read(*weight).size() == 72,
                "ModelOpt W4A16 NVFP4 packed-weight binding is wrong");
        require(scale.has_value() && scale->primary == "module.weight_scale_2" &&
                    scale->materialization == llama_safetensors_quant_materialization::POSITIVE_F32 &&
                    adapters.read(*scale) == f32_bytes(0.125f) && !input.has_value(),
                "ModelOpt W4A16 NVFP4 scale contract is wrong");
        require(!adapters.applies("ignored"), "ModelOpt glob ignore did not exclude the module");
        adapters.consume(*weight);
        adapters.consume(*scale);
        adapters.validate_complete();
    }
    {
        const auto path = dir.path / "modelopt-w4a16-nvfp4-experts";
        write_single_shard_model(path, {
            { "module.weight",         "U8",       { 2, 2, 32 }, std::vector<uint8_t>(128) },
            { "module.weight_scale",   "F8_E4M3", { 2, 2, 4 },  std::vector<uint8_t>(16, 0x38) },
            { "module.weight_scale_2", "F32",      {},           f32_bytes(0.125f) },
        });
        write_text(path / "config.json", modelopt_w4a16_nvfp4_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(
            llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        require(weight.has_value() && weight->target_type == GGML_TYPE_NVFP4 &&
                    weight->target_shape == std::vector<int64_t>({ 64, 2, 2 }) &&
                    adapters.read(*weight).size() == 144,
                "ModelOpt W4A16 NVFP4 expert aggregate binding is wrong");
        const auto scale = adapters.bind("module", llama_safetensors_quant_role::WEIGHT_SCALE);
        require(scale.has_value(), "ModelOpt W4A16 NVFP4 expert aggregate scale is missing");
        adapters.consume(*weight);
        adapters.consume(*scale);
        adapters.validate_complete();
    }
    {
        const auto path = dir.path / "quark-nvfp4";
        write_single_shard_model(path, {
            { "module.weight",         "U8",       { 2, 32 }, std::vector<uint8_t>(64) },
            { "module.weight_scale",   "F8_E4M3", { 2, 4 },  std::vector<uint8_t>(8, 0x38) },
            { "module.weight_scale_2", "F32",      {},        f32_bytes(0.125f) },
            { "module.input_scale",    "F32",      {},        f32_bytes(0.25f) },
            { "ignored.weight",        "BF16",     { 2, 32 }, std::vector<uint8_t>(128) },
        });
        write_text(path / "config.json", quark_nvfp4_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(
            llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        const auto scale  = adapters.bind("module", llama_safetensors_quant_role::WEIGHT_SCALE);
        const auto input  = adapters.bind("module", llama_safetensors_quant_role::INPUT_SCALE);
        require(weight.has_value() && weight->primary == "module.weight" &&
                    weight->target_type == GGML_TYPE_NVFP4 && adapters.read(*weight).size() == 72 &&
                    scale.has_value() && scale->primary == "module.weight_scale_2" &&
                    input.has_value() && input->primary == "module.input_scale" &&
                    scale->materialization == llama_safetensors_quant_materialization::POSITIVE_F32 &&
                    input->materialization == llama_safetensors_quant_materialization::POSITIVE_F32 &&
                    !adapters.applies("ignored"),
                "Quark NVFP4 binding is wrong");
        adapters.consume(*weight);
        adapters.consume(*scale);
        adapters.consume(*input);
        adapters.validate_complete();
    }
    {
        const auto path = dir.path / "quark-mxfp4";
        write_single_shard_model(path, {
            { "module.weight",       "U8",   { 2, 16 }, std::vector<uint8_t>(32) },
            { "module.weight_scale", "U8",   { 2, 1 },  std::vector<uint8_t>(2, 127) },
            { "ignored.weight",      "BF16", { 2, 32 }, std::vector<uint8_t>(128) },
        });
        write_text(path / "config.json", quark_mxfp4_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(
            llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        const auto input  = adapters.bind("module", llama_safetensors_quant_role::INPUT_SCALE);
        require(weight.has_value() && weight->primary == "module.weight" &&
                    weight->target_type == GGML_TYPE_MXFP4 &&
                    weight->target_shape == std::vector<int64_t>({ 32, 2 }) &&
                    adapters.read(*weight).size() == 34 && input.has_value() &&
                    input->materialization == llama_safetensors_quant_materialization::DYNAMIC_MXFP4_MARKER &&
                    !adapters.applies("ignored"),
                "Quark MXFP4 binding is wrong");
        adapters.consume(*weight);
        adapters.consume(*input);
        adapters.validate_complete();
    }
    {
        const auto path = dir.path / "quark-fp8-ptpc";
        write_single_shard_model(path, {
            { "module.weight",       "F8_E4M3", { 2, 32 }, std::vector<uint8_t>(64, 0x38) },
            { "module.weight_scale", "BF16",    { 2 },     { 0x00, 0x3f, 0x80, 0x3f }       },
            { "ignored.weight",      "BF16",    { 2, 32 }, std::vector<uint8_t>(128)       },
        });
        write_text(path / "config.json", quark_fp8_ptpc_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(
            llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        const auto scale  = adapters.bind("module", llama_safetensors_quant_role::WEIGHT_SCALE);
        const auto input  = adapters.bind("module", llama_safetensors_quant_role::INPUT_SCALE);
        require(weight.has_value() && weight->target_type == GGML_TYPE_F8_E4M3 &&
                    scale.has_value() && scale->target_type == GGML_TYPE_BF16 &&
                    input.has_value() &&
                    input->materialization == llama_safetensors_quant_materialization::DYNAMIC_FP8_MARKER &&
                    !adapters.applies("ignored"),
                "Quark FP8 PTPC binding is wrong");
        adapters.consume(*weight);
        adapters.consume(*scale);
        adapters.consume(*input);
        adapters.validate_complete();
    }
    {
        llama_safetensors_json config = llama_safetensors_json::parse(modelopt_w4a16_nvfp4_config);
        config["quantization_config"]["quant_algo"] = "NVFP4";
        require_rejected([&] {
            (void) llama_safetensors_quant_config::from_json(config);
        }, "ModelOpt W4A4 NVFP4 was accepted without an FP4 activation contract");
    }
    {
        const auto path = dir.path / "modelopt-mixed";
        write_single_shard_model(path, {
            { "fp8_module.weight",         "F8_E4M3", { 2, 32 }, std::vector<uint8_t>(64, 0x38) },
            { "fp8_module.weight_scale",   "F32",      {},        f32_bytes(0.25f) },
            { "fp8_module.input_scale",    "F32",      {},        f32_bytes(0.5f) },
            { "nv_module.weight",          "U8",       { 2, 32 }, std::vector<uint8_t>(64) },
            { "nv_module.weight_scale",    "F8_E4M3", { 2, 4 },  std::vector<uint8_t>(8, 0x38) },
            { "nv_module.weight_scale_2",  "F32",      {},        f32_bytes(0.125f) },
        });
        write_text(path / "config.json", modelopt_mixed_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(
            llama_safetensors_read_json(path / "config.json"), registry);
        const auto fp8_weight = adapters.bind("fp8_module", llama_safetensors_quant_role::WEIGHT);
        const auto fp8_scale  = adapters.bind("fp8_module", llama_safetensors_quant_role::WEIGHT_SCALE);
        const auto fp8_input  = adapters.bind("fp8_module", llama_safetensors_quant_role::INPUT_SCALE);
        const auto nv_weight  = adapters.bind("nv_module", llama_safetensors_quant_role::WEIGHT);
        require(fp8_weight.has_value() && fp8_weight->target_type == GGML_TYPE_F8_E4M3 &&
                    fp8_scale.has_value() && fp8_scale->target_type == GGML_TYPE_BF16 &&
                    fp8_scale->target_shape == std::vector<int64_t>({ 2 }) &&
                    fp8_input.has_value() && fp8_input->target_type == GGML_TYPE_F32 &&
                    nv_weight.has_value() && nv_weight->target_type == GGML_TYPE_NVFP4,
                "ModelOpt mixed-precision bindings are wrong");
        const auto scale_bytes = adapters.read(*fp8_scale);
        require(scale_bytes.size() == 2 * sizeof(ggml_bf16_t),
                "ModelOpt FP8 weight scale broadcast has the wrong size");
        const ggml_bf16_t expected_scale = ggml_fp32_to_bf16(0.25f);
        require(std::memcmp(scale_bytes.data(), &expected_scale, sizeof(expected_scale)) == 0 &&
                    std::memcmp(scale_bytes.data() + sizeof(expected_scale), &expected_scale,
                                sizeof(expected_scale)) == 0,
                "ModelOpt FP8 weight scale broadcast changed the scale");
        adapters.consume(*fp8_weight);
        adapters.consume(*fp8_scale);
        adapters.consume(*fp8_input);
        adapters.consume(*nv_weight);
        const auto nv_scale = adapters.bind("nv_module", llama_safetensors_quant_role::WEIGHT_SCALE);
        require(nv_scale.has_value(), "ModelOpt mixed NVFP4 scale binding is missing");
        adapters.consume(*nv_scale);
        adapters.validate_complete();
    }

    const std::string shard_a_header =
        R"({"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"packed":{"dtype":"U8","shape":[2],"data_offsets":[4,6]}})";
    const std::string shard_b_header =
        R"({"fp8":{"dtype":"F8_E4M3","shape":[2,2],"data_offsets":[0,4]},"a":{"dtype":"BF16","shape":[2],"data_offsets":[4,8]},"__metadata__":{"format":"pt"}})";
    write_shard(dir.path / "model-00001-of-00002.safetensors", shard_a_header, { 1, 2, 3, 4, 5, 6 });
    write_shard(dir.path / "model-00002-of-00002.safetensors", shard_b_header, { 7, 8, 9, 10, 11, 12, 13, 14 });
    const auto plain_quant = llama_safetensors_quant_config::from_json(llama_safetensors_json::parse(R"({})"));
    require(plain_quant.match("model.layers.0.self_attn.q_proj") == nullptr,
            "plain safetensors config unexpectedly selected a quantization adapter");

    bool malformed_quant_rejected = false;
    try {
        (void) llama_safetensors_quant_config::from_json(
            llama_safetensors_json::parse(R"({"quantization_config":null})"));
    } catch (const std::runtime_error &) {
        malformed_quant_rejected = true;
    }
    require(malformed_quant_rejected, "malformed quantization config was accepted as a plain model");

    write_text(
        dir.path / "model.safetensors.index.json",
        R"({"weight_map":{"a":"model-00001-of-00002.safetensors","packed":"model-00001-of-00002.safetensors","fp8":"model-00002-of-00002.safetensors"}})");

    const auto registry = llama_safetensors_registry::load(dir.path);
    require(registry.shards().size() == 2, "unexpected safetensors shard count");
    require(registry.tensors().size() == 3, "unexpected safetensors tensor count");

    const auto * packed = registry.find("packed");
    require(packed != nullptr, "packed tensor is missing");
    require(packed->dtype == llama_safetensors_dtype::U8, "packed tensor has the wrong dtype");
    require(packed->shape == std::vector<uint64_t>({ 2 }), "packed tensor has the wrong shape");
    require(packed->size == 2, "packed tensor has the wrong size");
    require(packed->offset == 8 + shard_a_header.size() + 4, "packed tensor has the wrong file offset");

    const auto * fp8 = registry.find("fp8");
    require(fp8 != nullptr, "FP8 tensor is missing");
    require(fp8->dtype == llama_safetensors_dtype::F8_E4M3, "FP8 tensor has the wrong dtype");
    require(fp8->shape == std::vector<uint64_t>({ 2, 2 }), "FP8 tensor has the wrong shape");
    require(registry.find("missing") == nullptr, "missing tensor lookup unexpectedly succeeded");

    {
        ggml_init_params params {
            /*.mem_size   =*/ 2 * ggml_tensor_overhead() + 1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(params);
        require(ctx != nullptr, "failed to create direct-upload test context");
        ggml_tensor * destination = ggml_new_tensor_2d(ctx, GGML_TYPE_F8_E4M3, 2, 2);
        ggml_backend_buffer_t buffer =
            ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
        require(buffer != nullptr, "failed to allocate direct-upload test buffer");

        const llama_safetensors_tensor_binding binding { "fp8", std::nullopt };
        require(llama_safetensors_load_tensor_direct(registry, binding, destination, false),
                "mapped canonical tensor did not take the direct-upload path");
        std::array<uint8_t, 4> uploaded{};
        ggml_backend_tensor_get(destination, uploaded.data(), 0, uploaded.size());
        require(uploaded == std::array<uint8_t, 4>({ 7, 8, 9, 10 }),
                "direct upload changed canonical tensor bytes");
        const auto buffered = llama_safetensors_registry::load(
            dir.path, llama_safetensors_io_mode::BUFFERED);
        require(!llama_safetensors_load_tensor_direct(buffered, binding, destination, false),
                "buffered tensor incorrectly took the mapped direct-upload path");

        llama_safetensors_quant_binding transformed {
            llama_safetensors_quant_materialization::NVFP4_REPACK,
            "fp8",
            {},
            GGML_TYPE_F8_E4M3,
            { 2, 2 },
        };
        require(!llama_safetensors_load_tensor_direct(
                    registry, { "fp8", transformed }, destination, false),
                "transformed tensor incorrectly took the direct-upload path");
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }

    const auto no_mmap = llama_safetensors_registry::load(dir.path, llama_safetensors_io_mode::BUFFERED);
    const auto * no_mmap_packed = no_mmap.find("packed");
    require(no_mmap_packed != nullptr && no_mmap.data(*no_mmap_packed) == nullptr &&
                no_mmap.read(*no_mmap_packed) == std::vector<uint8_t>({ 5, 6 }),
            "non-mmap safetensors fallback did not perform an exact bounded read");
    const auto * no_mmap_a = no_mmap.find("a");
    require(no_mmap_a != nullptr && no_mmap.data(*no_mmap_a) == nullptr &&
                no_mmap.read(*no_mmap_a) == std::vector<uint8_t>({ 1, 2, 3, 4 }),
            "non-mmap canonical tensor did not preserve exact bytes");

    write_text(
        dir.path / "config.json",
        R"({"quantization_config":{"quant_method":"compressed-tensors","format":"mixed-precision","ignore":["re:^ignored\\."],"config_groups":{"z_specific":{"format":"float-quantized","targets":["re:.*layers\\.(56|57)\\.mlp\\.(gate|up|down)_proj$"],"input_activations":{"actorder":null,"block_structure":null,"dynamic":true,"group_size":null,"num_bits":8,"scale_dtype":null,"strategy":"token","symmetric":true,"type":"float","zp_dtype":null},"weights":{"actorder":null,"block_structure":null,"dynamic":false,"group_size":null,"num_bits":8,"scale_dtype":null,"strategy":"channel","symmetric":true,"type":"float","zp_dtype":null}},"a_generic":{"format":"nvfp4-pack-quantized","targets":["re:.*mlp\\.(gate|up|down)_proj$"],"input_activations":{"actorder":null,"block_structure":null,"dynamic":"local","group_size":16,"num_bits":4,"scale_dtype":"torch.float8_e4m3fn","strategy":"tensor_group","symmetric":true,"type":"float","zp_dtype":null},"weights":{"actorder":"static","block_structure":null,"dynamic":false,"group_size":16,"num_bits":4,"scale_dtype":"torch.float8_e4m3fn","strategy":"tensor_group","symmetric":true,"type":"float","zp_dtype":null}}}}})");
    const auto   quant = llama_safetensors_quant_config::load(dir.path);
    const auto * early = quant.match("model.layers.12.mlp.gate_proj");
    require(early != nullptr, "early-layer quantization group is missing");
    require(early->format == llama_safetensors_quant_format::NVFP4_PACK, "early layer has the wrong format");
    const auto * late = quant.match("model.layers.56.mlp.gate_proj");
    require(late != nullptr, "late-layer quantization group is missing");
    require(late->format == llama_safetensors_quant_format::FP8_CHANNEL, "late layer has the wrong format");
    require(quant.ignored("ignored.mlp.gate_proj"), "ignored tensor was not recognized");
    require(quant.match("ignored.mlp.gate_proj") == nullptr, "ignored tensor matched a quantization group");

    // The producer selects the first matching target in document order. The
    // group names intentionally sort in the opposite order.
    require(late->name == "z_specific", "overlapping targets were not resolved in declaration order");

    const std::string valid_config = read_text(dir.path / "config.json");
    for (const auto & [needle, replacement] : std::array<std::pair<const char *, const char *>, 3>{
             {
              { "\"symmetric\":true", "\"symmetric\":false" },
              { "\"dynamic\":false", "\"dynamic\":true" },
              { "\"group_size\":16", "\"group_size\":32" },
              }
    }) {
        std::string invalid = valid_config;
        const auto  pos     = invalid.rfind(needle);
        require(pos != std::string::npos, "invalid quantization fixture mutation");
        invalid.replace(pos, std::strlen(needle), replacement);
        write_text(dir.path / "config.json", invalid);
        bool invalid_rejected = false;
        try {
            (void) llama_safetensors_quant_config::load(dir.path);
        } catch (const std::runtime_error &) {
            invalid_rejected = true;
        }
        require(invalid_rejected, "unsupported quantization contract was accepted");
    }
    write_text(dir.path / "config.json", valid_config);

    // Sidecars may duplicate base-model tensors. The index selects the live
    // copy and the registry must ignore the unassigned duplicate.
    write_text(
        dir.path / "model.safetensors.index.json",
        R"({"weight_map":{"a":"model-00002-of-00002.safetensors","packed":"model-00001-of-00002.safetensors","fp8":"model-00002-of-00002.safetensors"}})");
    const auto sidecar_registry = llama_safetensors_registry::load(
        dir.path, llama_safetensors_io_mode::BUFFERED);
    const auto * sidecar_a = sidecar_registry.find("a");
    require(sidecar_a != nullptr &&
                sidecar_registry.read(*sidecar_a) == std::vector<uint8_t>({ 11, 12, 13, 14 }),
            "registry did not select the weight_map copy of a duplicated sidecar tensor");

    // The index is authoritative; a tensor missing from its assigned shard
    // must still be rejected before any model allocation begins.
    write_text(
        dir.path / "model.safetensors.index.json",
        R"({"weight_map":{"a":"model-00001-of-00002.safetensors","packed":"model-00002-of-00002.safetensors","fp8":"model-00002-of-00002.safetensors"}})");
    bool rejected = false;
    try {
        (void) llama_safetensors_registry::load(dir.path);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    require(rejected, "registry accepted a tensor missing from its indexed shard");

    // A valid safetensors container is not sufficient: the architecture and
    // quantization contract must be one the direct importer actually supports.
    write_text(
        dir.path / "model.safetensors.index.json",
        R"({"weight_map":{"a":"model-00001-of-00002.safetensors","packed":"model-00001-of-00002.safetensors","fp8":"model-00002-of-00002.safetensors"}})");
    bool unsupported_rejected = false;
    try {
        (void) llama_safetensors_qwen35_importer(dir.path, llama_safetensors_read_json(dir.path / "config.json"));
    } catch (const std::exception &) {
        unsupported_rejected = true;
    }
    require(unsupported_rejected, "Qwen3.5 importer accepted an unsupported model contract");

    return 0;
}
