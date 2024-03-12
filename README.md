# Fused grouped tril mask softmax

## 性能测试结果（相对于pytorch实现）X.shape = [B, M, S, S]

### DevCloud上（B = 3, S = 1000, H = 8, M = 64）加速 5.67 倍，耗时11.7毫秒

```text
Running cuda...
Cuda time:  11700.296us
Running torch...
Torch time:  66341.925us
================================= RESULT =================================
Maximum error value = tensor(1.1921e-07, device='cuda:0')
CUDA Speedup = 5.67x
Kernel test passed.
==========================================================================
```

### A10物理机上（B = 12, S = 1000, H = 8, M = 64）加速 4.54 倍，耗时16.3毫秒

```text
Running cuda...
Cuda time:  16315.532us
Running torch...
Torch time:  74056.602us
================================= RESULT =================================
Maximum error value = tensor(0.0017, device='cuda:0')
CUDA Speedup = 4.54x
Kernel test passed.
==========================================================================
```

### A10物理机上（B = 3, S = 1000, H = 8, M = 64）加速 4.59 倍，耗时4.0毫秒

```text
Running cuda...
Cuda time:  4012.561us
Running torch...
Torch time:  18420.815us
================================= RESULT =================================
Maximum error value = tensor(0.0023, device='cuda:0')
CUDA Speedup = 4.59x
Kernel test passed.
==========================================================================
```

## 实现思路

主要就是在 [kernel/fused_grouped_tril_mask_softmax_kernel.cu](kernel/fused_grouped_tril_mask_softmax_kernel.cu) 中实现了逻辑Mask，用计算代替访存，具体是在 `GroupedTrilMaskLoad` 和 `GroupedTrilMaskStore` 中。

## 运行代码

```shell
# 在cutlass机器上（jerryao@21.13.75.17）
cd /data1/jerryao/code/Attention-Mask-Optimization
conda activate masked_opt

# 正确性
python fused_grouped_tril_mask_softmax_test.py

# 性能
python fused_grouped_tril_mask_softmax_bench.py
```

## 正确性验证

[fused_grouped_tril_mask_softmax_test.py](fused_grouped_tril_mask_softmax_test.py) 中计算了最大误差，在 1.1921e-07左右

## 算子行为（融合下面的计算）

```python
import torch

device = torch.device("cpu")
dtype = torch.float32


B = 3
S = 5  # 填充长度
H = 2
M = 2
SEQ_LENS = [3, 4, 5]  # 有效长度，每个batch都不同

attn = torch.randn(B, M, S, S, device=device, dtype=dtype)

# fused_grouped_tril_mask_softmax
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
    attn[b] = attn[b].masked_fill(causal_mask, -torch.finfo(attn.dtype).max)
    attn[b] = attn[b].softmax(-1)
    # print(causal_mask)
attn[attn == 1 / S] = 0.0
# print(attn)
# fused_grouped_tril_mask_softmax
```
上面代码运行的打印内容为：

```text
tensor([[False,  True,  True,  True,  True],
        [False, False,  True,  True,  True],
        [False, False, False,  True,  True],
        [ True,  True,  True,  True,  True],
        [ True,  True,  True,  True,  True]])
tensor([[False,  True,  True,  True,  True],
        [False, False,  True,  True,  True],
        [False, False, False,  True,  True],
        [False, False, False, False,  True],
        [ True,  True,  True,  True,  True]])
tensor([[False,  True,  True,  True,  True],
        [False, False,  True,  True,  True],
        [False, False, False,  True,  True],
        [False, False, False, False,  True],
        [False, False, False, False, False]])
tensor([[[[1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.5596, 0.4404, 0.0000, 0.0000, 0.0000],
          [0.0146, 0.9730, 0.0125, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

         [[1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.1611, 0.8389, 0.0000, 0.0000, 0.0000],
          [0.1570, 0.3879, 0.4552, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]],


        [[[1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.2416, 0.7584, 0.0000, 0.0000, 0.0000],
          [0.8261, 0.1336, 0.0404, 0.0000, 0.0000],
          [0.2205, 0.0954, 0.2523, 0.4318, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

         [[1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.4338, 0.5662, 0.0000, 0.0000, 0.0000],
          [0.1283, 0.0794, 0.7923, 0.0000, 0.0000],
          [0.3714, 0.3939, 0.1283, 0.1064, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]],


        [[[1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.5161, 0.4839, 0.0000, 0.0000, 0.0000],
          [0.6362, 0.1044, 0.2594, 0.0000, 0.0000],
          [0.2404, 0.1837, 0.3760, 0.1999, 0.0000],
          [0.0307, 0.0916, 0.5037, 0.1961, 0.1780]],

         [[1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.1792, 0.8208, 0.0000, 0.0000, 0.0000],
          [0.6914, 0.2153, 0.0933, 0.0000, 0.0000],
          [0.3500, 0.2563, 0.2432, 0.1505, 0.0000],
          [0.4088, 0.3471, 0.0822, 0.0416, 0.1203]]]])
```
