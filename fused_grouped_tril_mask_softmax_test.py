import torch
from torch.utils.cpp_extension import load

device = torch.device("cuda:0")
# dtype = torch.float16
dtype = torch.float32

import warnings

warnings.filterwarnings("ignore")

cuda_module = load(
    name="fused_grouped_tril_mask_softmax",
    extra_include_paths=["include"],
    sources=[
        "fused_grouped_tril_mask_softmax_ops.cpp",
        "kernel/fused_grouped_tril_mask_softmax_kernel.cu",
    ],
    verbose=True,
)

B = 3
S = 5
H = 2
M = 2
SEQ_LENS = [3, 4, 5]


attn = torch.randn(B, M, S, S, device=device, dtype=dtype)
cuda_container = torch.zeros_like(attn, device=device, dtype=dtype)
# print(attn.shape)
# print(attn)
# print(attn.view(-1))


def torch_test(attn):
    for b in range(B):
        i, j = attn.shape[-2:]
        causal_mask = torch.ones((i, j), dtype=torch.bool, device=attn.device)
        seq_len = SEQ_LENS[b]
        for row in range(i):
            for col in range(j):
                invalid = row >= seq_len or col >= seq_len
                need_load = not invalid and row >= col
                if need_load:
                    causal_mask[row][col] = False
        attn[b] = attn[b].masked_fill(causal_mask, -1e10)
        attn[b] = attn[b].softmax(-1)
    attn[attn == 0.2000] = 0
    # print(attn)
    return attn


def cuda_test(attn):
    seq_lens = torch.tensor(SEQ_LENS, dtype=torch.int64, device=attn.device)
    cuda_module.torch_launch_fused_grouped_tril_mask_softmax(
        attn, cuda_container, seq_lens
    )
    # print(cuda_container)
    return cuda_container


torch_result = torch_test(attn.clone())
cuda_result = cuda_test(attn.clone())
print("Maximum error value =", torch.max(torch.abs(torch_result - cuda_result)))
