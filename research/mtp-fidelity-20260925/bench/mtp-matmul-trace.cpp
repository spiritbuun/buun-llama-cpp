// Diagnostic preload: observe real matmul inputs/outputs without splitting graphs.
#include "ggml.h"
#include "ggml-backend.h"
#include <dlfcn.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>
struct ggml_backend_cuda_context;
struct ggml_cuda_mm_fusion_args_host;
static int phase;
static int attention_index;
static std::map<std::string,std::vector<float>> rows;
extern "C" void mtp_trace_mark(int value) { phase=value; attention_index=0; }
static void observe(const ggml_tensor * t, const ggml_tensor * weight, const char * path, const char * part) {
    if (!phase || t->type!=GGML_TYPE_F32 || t->nb[0]!=sizeof(float)) { return; }
    auto sync=reinterpret_cast<int(*)()>(dlsym(RTLD_DEFAULT,"cudaDeviceSynchronize"));
    if (!sync || sync()) { std::abort(); }
    std::vector<float> data(t->ne[0]);
    ggml_backend_tensor_get(t,data.data(),0,data.size()*sizeof(float));
    const std::string key=std::string(weight->name)+":"+part;
    if (phase==1) { rows[key]=data; return; }
    auto old=rows.find(key);
    if (old==rows.end() || old->second.size()!=data.size()) { return; }
    double diff=0;
    int count=0;
    for(size_t i=0;i<data.size();++i) {
        diff=std::max(diff,std::abs(double(data[i])-old->second[i]));
        count+=data[i]!=old->second[i];
    }
    std::printf("LIVE_MM phase=%d key=%s path=%s type=%s tensor=%s changed=%d/%zu max_abs=%.9g\n",
        phase,key.c_str(),path,ggml_type_name(weight->type),t->name,count,data.size(),diff);
}
void ggml_cuda_mul_mat_q(ggml_backend_cuda_context & ctx,const ggml_tensor * w,const ggml_tensor * x,
                       const ggml_tensor * ids,ggml_tensor * dst) {
    using fn=void(*)(ggml_backend_cuda_context &,const ggml_tensor *,const ggml_tensor *,const ggml_tensor *,ggml_tensor *);
    static auto original=reinterpret_cast<fn>(dlsym(RTLD_NEXT,"_Z19ggml_cuda_mul_mat_qR25ggml_backend_cuda_contextPK11ggml_tensorS3_S3_PS1_"));
    if (!original) { std::abort(); }
    observe(x,w,"mmq","input");
    original(ctx,w,x,ids,dst);
    observe(dst,w,"mmq","output");
}
void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,const ggml_tensor * w,const ggml_tensor * x,
        const ggml_tensor * ids,ggml_tensor * dst,const ggml_cuda_mm_fusion_args_host * fusion,
        float scale,bool silu,const ggml_tensor * marker) {
    using fn=void(*)(ggml_backend_cuda_context &,const ggml_tensor *,const ggml_tensor *,const ggml_tensor *,ggml_tensor *,
                    const ggml_cuda_mm_fusion_args_host *,float,bool,const ggml_tensor *);
    static auto original=reinterpret_cast<fn>(dlsym(RTLD_NEXT,"_Z23ggml_cuda_mul_mat_vec_qR25ggml_backend_cuda_contextPK11ggml_tensorS3_S3_PS1_PK29ggml_cuda_mm_fusion_args_hostfbS3_"));
    if (!original) { std::abort(); }
    observe(x,w,"mmvq","input");
    original(ctx,w,x,ids,dst,fusion,scale,silu,marker);
    observe(dst,w,"mmvq","output");
}
void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx,ggml_tensor * dst) {
    using fn=void(*)(ggml_backend_cuda_context &,ggml_tensor *);
    static auto original=reinterpret_cast<fn>(dlsym(RTLD_NEXT,"_Z24ggml_cuda_flash_attn_extR25ggml_backend_cuda_contextP11ggml_tensor"));
    if (!original) { std::abort(); }
    if (phase) { std::printf("FLASH_CALL phase=%d name=%s type=%s shape=%lld,%lld,%lld\n",phase,dst->name,ggml_type_name(dst->type),(long long)dst->ne[0],(long long)dst->ne[1],(long long)dst->ne[2]); }
    auto identity=*dst;
    std::snprintf(identity.name,sizeof(identity.name),"flash-%d",attention_index++);
    if (phase) {
        auto * q=dst->src[0];
        for (int h=0;h<q->ne[2];++h) {
            auto row=*q;
            row.data=static_cast<char *>(q->data)+h*q->nb[2];
            observe(&row,&identity,"flash",("q-head"+std::to_string(h)).c_str());
        }
    }
    static const bool fixed_tile=std::getenv("MTP_TRACE_FIXED_FA_TILE") != nullptr;
    if (fixed_tile && dst->src[0]->ne[0]==256 && dst->src[0]->ne[1]<=4 &&
            dst->src[1]->type==GGML_TYPE_F16 && dst->src[2]->type==GGML_TYPE_F16) {
        static auto fixed=reinterpret_cast<fn>(dlsym(RTLD_NEXT,
            "_Z37ggml_cuda_flash_attn_ext_mma_f16_caseILi256ELi256ELi4ELi8ELb0EEvR25ggml_backend_cuda_contextP11ggml_tensor"));
        if (!fixed) { std::fprintf(stderr,"missing fixed FA tile symbol\n"); std::abort(); }
        fixed(ctx,dst);
    } else {
        original(ctx,dst);
    }
    if (phase) {
        auto row=*dst;
        row.ne[0]*=row.ne[1];
        observe(&row,&identity,"flash","output-first-token");
    }
}
void ggml_cuda_op_unary_mul(ggml_backend_cuda_context & ctx,ggml_tensor * unary,ggml_tensor * mul,const ggml_tensor * bf16) {
    using fn=void(*)(ggml_backend_cuda_context &,ggml_tensor *,ggml_tensor *,const ggml_tensor *);
    static auto original=reinterpret_cast<fn>(dlsym(RTLD_NEXT,"_Z22ggml_cuda_op_unary_mulR25ggml_backend_cuda_contextP11ggml_tensorS2_PKS1_"));
    if (!original) { std::abort(); }
    observe(unary->src[0],mul,"unary-mul","unary-input");
    observe(mul->src[mul->src[0]==unary ? 1 : 0],mul,"unary-mul","other-input");
    original(ctx,unary,mul,bf16);
    observe(mul,mul,"unary-mul","output");
}
