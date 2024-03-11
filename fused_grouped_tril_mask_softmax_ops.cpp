#include <iostream>
#include <torch/extension.h>
#include "fused_grouped_tril_mask_softmax.h"

torch::Tensor& torch_launch_fused_grouped_tril_mask_softmax(torch::Tensor &attn,
                       torch::Tensor &result,
                       const torch::Tensor &seq_lens) {
    const int64_t batch_size = attn.sizes()[0];
    const int64_t num_heads = attn.sizes()[1];
    const int64_t seq_length = attn.sizes()[2];
    fused_grouped_tril_mask_softmax((float *)attn.data_ptr(),
                                    (float *)result.data_ptr(),
                                    (const int64_t *)seq_lens.data_ptr(),
                                    batch_size, num_heads, seq_length);
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_fused_grouped_tril_mask_softmax",
          &torch_launch_fused_grouped_tril_mask_softmax,
          "fused_grouped_tril_mask_softmax kernel warpper");
}

TORCH_LIBRARY(fused_grouped_tril_mask_softmax, m) {
    m.def("torch_launch_fused_grouped_tril_mask_softmax", torch_launch_fused_grouped_tril_mask_softmax);
}