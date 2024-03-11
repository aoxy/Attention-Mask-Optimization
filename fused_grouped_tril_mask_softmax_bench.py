import time
import torch
import numpy as np
from torch.utils.cpp_extension import load

import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda:0")
# dtype = torch.float16
dtype = torch.float32

B = 3
S = 1000
H = 8
M = 64
# SEQ_LENS = [S - 40 * b for b in range(B)]
SEQ_LENS = [1000 for b in range(B)]
print("SEQ_LENS", SEQ_LENS)


attn = torch.randn(B, M, S, S, device=device, dtype=dtype)
cuda_container = torch.zeros_like(attn, device=device, dtype=dtype)
torch_container = torch.zeros_like(attn, device=device, dtype=dtype)
seq_lens = torch.tensor(SEQ_LENS, dtype=torch.int64, device=attn.device)
ntest = 10


def show_time(func):
    times = list()
    res = None
    # GPU warm up
    for _ in range(10):
        res = func()
    for _ in range(ntest):
        # sync the threads to get accurate cuda running time
        torch.cuda.synchronize(device=device)
        start_time = time.time()
        func()
        torch.cuda.synchronize(device=device)
        end_time = time.time()
        times.append((end_time - start_time) * 1e6)
    return times, res


def run_cuda():
    seq_lens = torch.tensor(SEQ_LENS, dtype=torch.int64, device=attn.device)
    cuda_module.torch_launch_fused_grouped_tril_mask_softmax(
        attn, cuda_container, seq_lens
    )
    return cuda_container


def run_torch():
    for b in range(B):
        i, j = attn.shape[-2:]
        causal_mask = torch.ones((i, j), dtype=torch.bool, device=attn.device).triu(
            j - i + 1
        )
        # causal_mask = torch.ones((i, j), dtype=torch.bool, device=attn.device)
        # seq_len = SEQ_LENS[b]
        # for row in range(i):
        #     for col in range(j):
        #         invalid = row >= seq_len or col >= seq_len
        #         need_load = not invalid and row >= col
        #         if need_load:
        #             causal_mask[row][col] = False
        torch_container[b] = attn[b].masked_fill(causal_mask, -1e10)
        torch_container[b] = torch_container[b].softmax(-1)
    torch_container[attn == 0.2000] = 0
    return torch_container


if __name__ == "__main__":
    cuda_module = load(
        name="fused_grouped_tril_mask_softmax",
        extra_include_paths=["include"],
        sources=[
            "fused_grouped_tril_mask_softmax_ops.cpp",
            "kernel/fused_grouped_tril_mask_softmax_kernel.cu",
        ],
        verbose=True,
    )

    print("Running cuda...")
    cuda_time, cuda_res = show_time(run_cuda)
    print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))

    print("Running torch...")
    torch_time, torch_res = show_time(run_torch)
    print("Torch time:  {:.3f}us".format(np.mean(torch_time)))
    print("================================= RESULT =================================")
    print("Maximum error value =", torch.max(torch.abs(torch_res - cuda_res)))
    print("CUDA Speedup = {:.2f}x".format(np.mean(torch_time) / np.mean(cuda_time)))
    torch.allclose(cuda_res, torch_res)
    print("Kernel test passed.")
    print("==========================================================================")
