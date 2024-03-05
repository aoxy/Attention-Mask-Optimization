import torch

device = torch.device('cuda:0')
# dtype = torch.float16
dtype = torch.float32

B = 1
S = 1000
H = 8
M = 64

attn = torch.randn(B, M, S, S, device=device, dtype=dtype)
i, j = attn.shape[-2:]
causal_mask = torch.ones((i, j), dtype = torch.bool, device = attn.device).triu(j - i + 1)
attn = attn.masked_fill(causal_mask, -torch.finfo(attn.dtype).max)
attn = attn.softmax(-1)
print(attn.shape)
print(attn.sum())
print(attn)

