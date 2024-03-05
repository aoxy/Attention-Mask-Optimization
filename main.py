import torch
from xformers import ops

device = torch.device('cuda:0')
# dtype = torch.float16
dtype = torch.float32

class Model(torch.nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return ops.memory_efficient_attention(query, key, value)

    def forward_attn_bias(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        _attn_bias = ops.LowerTriangularMask()
        return ops.memory_efficient_attention(query, key, value, attn_bias=_attn_bias)

    def inference(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        scale = 1.0 / query.shape[-1] ** 0.5
        query = query * scale
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attn = query @ key.transpose(-2, -1)
        if attn_bias is not None:
            # attn = attn + attn_bias
            i, j = attn.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = attn.device).triu(j - i + 1)
            import pdb; pdb.set_trace()
            attn = attn.masked_fill(causal_mask, -torch.finfo(attn.dtype).max)
        attn = attn.softmax(-1)
        attn = attn @ value
        return attn.transpose(1, 2)


B = 1
S = 1000
H = 8
M = 64

q = torch.randn(B, S, H, M, device=device, dtype=dtype)
k = torch.randn(B, S, H, M, device=device, dtype=dtype)
v = torch.randn(B, S, H, M, device=device, dtype=dtype)

# torch.save(q, "q.pt")
# torch.save(k, "k.pt")
# torch.save(v, "v.pt")

# q = torch.load("q.pt").to(device)
# k = torch.load("k.pt").to(device)
# v = torch.load("v.pt").to(device)

q1 = q.clone()
k1 = k.clone()
v1 = v.clone()

m = Model()
# m.load_state_dict(torch.load("model.pt"))
m.eval()

def attn_bias_test():
    print("==============attn_bias_test================")
    x0 = m.forward_attn_bias(q, k, v)
    attn_bias = tensor = torch.ones([1, 1, S, S])
    # attn_bias = torch.tril(attn_bias).to(device).to(dtype) * -1000000
    # attn_bias = torch.tril(attn_bias).to(device).to(dtype).bool()
    x1 = m.inference(q1, k1, v1, attn_bias)
    d = (x1 - x0).abs().max()

    print(f'ref  : {x0.min()}, {x0.max()}')
    print(f'equal: {x1.min()}, {x1.max()}')
    print(f'diff : {d}')

def test():
    print("==============test================")
    x0 = m.forward(q, k, v)
    x1 = m.inference(q1, k1, v1)
    d = (x1 - x0).abs().max()

    print(f'ref  : {x0.min()}, {x0.max()}')
    print(f'equal: {x1.min()}, {x1.max()}')
    print(f'diff : {d}')



attn_bias_test()
