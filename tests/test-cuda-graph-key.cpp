#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstring>

#include "../ggml/src/ggml-cuda/cuda-graph-key.h"

static ggml_tensor tensor_with_shape(int64_t ne0, int64_t ne1) {
    ggml_tensor tensor = {};
    tensor.ne[0] = ne0;
    tensor.ne[1] = ne1;
    tensor.ne[2] = 1;
    tensor.ne[3] = 1;
    return tensor;
}

int main() {
    ggml_tensor first  = tensor_with_shape(2560, 1);
    ggml_tensor middle = tensor_with_shape(64, 1);
    ggml_tensor last_2 = tensor_with_shape(152064, 2);
    ggml_tensor last_3 = tensor_with_shape(152064, 3);

    ggml_tensor * nodes_2[] = { &first, &middle, &last_2 };
    ggml_tensor * nodes_3[] = { &first, &middle, &last_3 };
    ggml_cgraph graph_2 = {};
    graph_2.n_nodes = 3;
    graph_2.nodes = nodes_2;
    ggml_cgraph graph_3 = graph_2;
    graph_3.nodes = nodes_3;

    const uint64_t key_2 = ggml_cuda_graph_shape_key(&graph_2);
    const uint64_t key_3 = ggml_cuda_graph_shape_key(&graph_3);
    assert(key_2 != key_3);
    assert(key_2 == ggml_cuda_graph_shape_key(&graph_2));

    ggml_cgraph shorter = graph_2;
    shorter.n_nodes = 2;
    assert(key_2 != ggml_cuda_graph_shape_key(&shorter));

    ggml_tensor other_first = first;
    ggml_tensor * other_nodes[] = { &other_first, &middle, &last_2 };
    ggml_cgraph other_graph = graph_2;
    other_graph.nodes = other_nodes;
    assert(key_2 != ggml_cuda_graph_shape_key(&other_graph));

    return 0;
}
