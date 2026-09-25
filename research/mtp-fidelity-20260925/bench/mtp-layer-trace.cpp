// Private first-token trace. Always measure observer-vs-unobserved logits too:
// requesting intermediate values can split graphs and inhibit fusion.
#include "arg.h"
#include "common.h"
#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
extern "C" void mtp_trace_mark(int) __attribute__((weak));

struct trace {
    bool active = false;
    bool compare = false;
    std::map<std::string, std::vector<float>> * reference;
    std::map<std::string, int> occurrences;
    static bool callback(ggml_tensor * t, bool ask, void * data) {
        auto & self = *static_cast<trace *>(data);
        const char * dash = std::strrchr(t->name, '-');
        const bool selected = self.active && t->type == GGML_TYPE_F32 &&
            dash && dash[1] >= '0' && dash[1] <= '7' && dash[2] == '\0' &&
            t->op != GGML_OP_NONE && t->op != GGML_OP_VIEW &&
            t->op != GGML_OP_RESHAPE && t->op != GGML_OP_PERMUTE;
        if (ask) { return selected; }
        if (!selected) { return true; }
        // First row only: same first token/head for differently shaped batches.
        if (t->nb[0] != sizeof(float)) { return true; }
        std::vector<float> row(t->ne[0]);
        ggml_backend_tensor_get(t, row.data(), 0, row.size()*sizeof(float));
        const std::string name = std::string(t->name) + ":" + std::to_string(self.occurrences[t->name]++);
        if (!self.compare) { (*self.reference)[name] = row; }
        else {
            const auto found = self.reference->find(name);
            if (found != self.reference->end() && found->second.size() == row.size()) {
                double max_abs = 0, sum = 0, norm = 0;
                int changed = 0;
                for (size_t i = 0; i < row.size(); ++i) {
                    const double delta = double(row[i])-found->second[i];
                    max_abs = std::max(max_abs, std::abs(delta));
                    sum += delta*delta;
                    norm += double(found->second[i])*found->second[i];
                    changed += row[i] != found->second[i];
                }
                std::printf("LAYER_TRACE name=%s op=%s ne=%lld,%lld,%lld,%lld changed=%d/%zu max_abs=%.9g rel_l2=%.9g\n",
                    name.c_str(), ggml_op_name(t->op), (long long)t->ne[0], (long long)t->ne[1],
                    (long long)t->ne[2], (long long)t->ne[3], changed, row.size(), max_abs,
                    std::sqrt(sum/std::max(norm,1e-30)));
            }
        }
        if (t->op == GGML_OP_MUL_MAT && t->src[1]->type == GGML_TYPE_F32) {
            auto * src = t->src[1];
            std::vector<float> input(src->ne[0]);
            ggml_backend_tensor_get(src, input.data(), 0, input.size()*sizeof(float));
            const std::string key = name + ":input";
            if (!self.compare) { (*self.reference)[key] = input; }
            else if (self.reference->count(key) && (*self.reference)[key].size() == input.size()) {
                double diff = 0;
                for (size_t i=0;i<input.size();++i) { diff=std::max(diff,std::abs(double(input[i])-(*self.reference)[key][i])); }
                std::printf("TRACE_MATMUL %s weight=%s type=%s input_max_abs=%.9g\n",name.c_str(),t->src[0]->name,ggml_type_name(t->src[0]->type),diff);
            }
        }
        return true;
    }
};

static void compare(const char * label, const std::vector<float> & p, const std::vector<float> & q) {
    double pm = *std::max_element(p.begin(), p.end()), qm = *std::max_element(q.begin(), q.end());
    double ps = 0, qs = 0, kl = 0, diff = 0;
    for (size_t i = 0; i < p.size(); ++i) { ps += std::exp(p[i]-pm); qs += std::exp(q[i]-qm); }
    for (size_t i = 0; i < p.size(); ++i) {
        kl += std::exp(p[i]-pm)/ps * ((p[i]-pm)-(q[i]-qm)+std::log(qs)-std::log(ps));
        diff = std::max(diff,std::abs(double(p[i])-q[i]));
    }
    std::printf("TRACE_CONTROL %s exact=%d max_abs=%.9g kld=%.12g\n", label,
            std::memcmp(p.data(), q.data(), p.size()*sizeof(float)) == 0, diff, kl);
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PERPLEXITY)) { return 1; }
    common_init(); llama_backend_init();
    auto * model = llama_model_load_from_file(params.model.path.c_str(), common_model_params_to_llama(params));
    if (!model) { return 2; }
    std::ifstream input(std::getenv("MTP_PROBE_TOKENS"));
    std::vector<llama_token> tokens;
    llama_token id;
    while (input >> id) { tokens.push_back(id); }
    constexpr int prefix = 31;
    if (tokens.size() < prefix+4) { return 3; }
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    auto batch = llama_batch_init(512,0,1);
    std::vector<float> plain1, plain4, trace1;
    std::map<std::string,std::vector<float>> reference;
    for (bool observed : {false,true}) {
        if (observed && mtp_trace_mark) { break; }
        for (int width : {1,4}) {
            trace tr{false,width==4,&reference};
            auto cp = common_context_params_to_llama(params);
            cp.n_rs_seq=3; cp.n_seq_max=2;
            if (observed) { cp.cb_eval=trace::callback; cp.cb_eval_user_data=&tr; }
            auto * ctx = llama_init_from_model(model,cp);
            if (!ctx) { return 4; }
            common_batch_clear(batch);
            for (int i=0;i<prefix;++i) { common_batch_add(batch,tokens[i],i,{0},false); }
            if (llama_decode(ctx,batch)) { return 5; }
            llama_synchronize(ctx);
            common_batch_clear(batch);
            for (int i=0;i<width;++i) { common_batch_add(batch,tokens[prefix+i],prefix+i,{0},true); }
            tr.active=true;
            if (mtp_trace_mark) { mtp_trace_mark(width); }
            if (llama_decode(ctx,batch)) { return 6; }
            const float * raw=llama_get_logits_ith(ctx,0);
            std::vector<float> logits(raw,raw+nv);
            if (mtp_trace_mark) { mtp_trace_mark(0); }
            if (!observed && width==1) { plain1=logits; }
            if (!observed && width==4) { plain4=logits; compare("unobserved-width1-vs4",plain1,plain4); }
            if (observed && width==1) { trace1=logits; compare("observer-width1",plain1,logits); }
            if (observed && width==4) {
                compare("observer-width4",plain4,logits);
                compare("observed-width1-vs4",trace1,logits);
            }
            llama_free(ctx);
        }
    }
    llama_batch_free(batch); llama_model_free(model); llama_backend_free();
}
