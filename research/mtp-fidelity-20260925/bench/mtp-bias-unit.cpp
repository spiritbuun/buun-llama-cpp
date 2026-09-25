// Private exact before/after GPU oracle for batched MMVQ + ADD.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 2) { return 1; }
    ggml_backend_load_all();
    auto backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!backend) { return 2; }
    std::ofstream dump(argv[1], std::ios::binary);
    for (auto type : {GGML_TYPE_IQ2_XS, GGML_TYPE_IQ3_S}) {
        ggml_quantize_init(type);
        for (int n : {64, 65}) for (int m : {1, 2, 3, 4}) for (bool swap : {false, true}) {
            constexpr int k = 512;
            auto ctx = ggml_init({size_t(1)<<20, nullptr, true});
            auto w = ggml_new_tensor_2d(ctx, type, k, n);
            auto x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
            auto b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, m);
            auto mm = ggml_mul_mat(ctx, w, x);
            auto out = swap ? ggml_add(ctx, b, mm) : ggml_add(ctx, mm, b);
            ggml_set_name(w,"test.weight"); ggml_set_name(mm,"test.mm"); ggml_set_name(out,"test.add");
            ggml_set_input(w); ggml_set_input(x); ggml_set_input(b); ggml_set_output(out);
            auto graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph,out);
            auto buffer = ggml_backend_alloc_ctx_tensors(ctx,backend);
            if (!buffer) { return 3; }
            std::vector<float> wf(k*n), xf(k*m), bf(n*m), im(k,1.0f), result(n*m);
            for (int i=0;i<k*n;++i) { wf[i]=std::sin(float(i)*0.071f); }
            for (int i=0;i<k*m;++i) { xf[i]=std::cos(float(i)*0.117f); }
            for (int i=0;i<n*m;++i) { bf[i]=0.3f*std::sin(float(i)*0.013f); }
            std::vector<char> quant(ggml_nbytes(w));
            ggml_quantize_chunk(type,wf.data(),quant.data(),0,n,k,im.data());
            ggml_backend_tensor_set(w,quant.data(),0,quant.size());
            ggml_backend_tensor_set(x,xf.data(),0,xf.size()*sizeof(float));
            ggml_backend_tensor_set(b,bf.data(),0,bf.size()*sizeof(float));
            if (ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS) { return 4; }
            ggml_backend_tensor_get(out,result.data(),0,result.size()*sizeof(float));
            for (float f: result) { if (!std::isfinite(f)) { return 5; } }
            dump.write(reinterpret_cast<const char *>(result.data()),result.size()*sizeof(float));
            std::printf("PASS type=%s n=%d m=%d swapped=%d\n",ggml_type_name(type),n,m,int(swap));
            ggml_backend_buffer_free(buffer); ggml_free(ctx);
        }
    }
    ggml_backend_free(backend);
    return dump.good() ? 0 : 6;
}
