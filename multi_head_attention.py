# (B, T, n_heads*d_head) -> target shape 

"""
lets see the trace map of the mha 
1. (B, T, C)
2. (B, T, n_heads, d_head)
3. (B, n_heads, T, d_head)
4. (B, n_heads, T, T)
5. (B, n_heads, T, d_head)
6.  (B, T,n_heads, d_head)
7. (B, T, n_heads*d_head)
"""

import torch 
import torch.nn as nn 
import numpy 
import torch.nn.functional as F
import math
from attn_mask import causal_mask

class MHA(nn.Module()):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0, trace_shapes: bool = True, mask : bool = True):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.d_model = d_model 
        self.n_head = n_head 
        self.dropout = dropout 
        self.trace_shapes = trace_shapes
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.trace_shapes = trace_shapes
    
    def forward(self, x):
        B,T,C = x.shape 
        qkv = self.qkv(x)
        qkv = qkv.view(B, T,3, self.n_head, self.d_head)
        if self.trace_shapes:
            print(f"qkv shape rn: {qkv.shape}")
        q, k, v = qkv.unbind(dim=2)
        # the attention model NEEDS ALL TOKENS FOR A FIXED HEAD NOT ALL HEADS FOR A TOKEN
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)
        if self.trace_shapes:
            print(f"q : {q.shape} | v : {v.shape} | k : {k.shape}")
        a = torch.matmul(q, k.transpose(-2,-1))
        scale = 1/math.sqrt(self.d_head)
        attn = a * scale 
        # wherever the mask matrix is true the true elements will be converted into -inf and later after softmax almost 0 
        if self.mask:
            maskout = causal_mask(T, device=x.device)
        attn = attn.masked_fill(maskout, float('-inf'))
        w = F.softmax(attn, dim=-1)
        w = self.dropout(w)
        ctx = torch.matmul(w, v)                 
        if self.trace_shapes:
            print("weights:", w.shape, "ctx:", ctx.shape)
        # contiguous is applied because transpose sometimes breaks memory allocation 
        out = ctx.transpose(1, 2).contiguous().view(B, T, C)  
        out = self.proj(out)
        if self.trace_shapes:
            print("out:", out.shape)
        return out, w
            
            
        
        
        

# more detailed explanation :
"""
Multi-Head Attention: Explicit Shape Trace

1. (B, T, C)
   • Input token embeddings.
   • B = batch size
   • T = sequence length
   • C = d_model (embedding dimension)

2. (B, T, n_heads, d_head)
   • Linear projections (Q, K, V) created from input.
   • d_head = C // n_heads
   • Reshape splits model dimension into multiple heads.

3. (B, n_heads, T, d_head)
   • Transpose step: bring the head dimension forward.
   • Each head now operates independently across the sequence.
   • Enables parallel computation of attention per head.

4. (B, n_heads, T, T)
   • Attention scores = (Q @ Kᵀ) / sqrt(d_head)
   • Represents pairwise similarity between all tokens (query vs key).
   • One (T×T) attention map per head.

5. (B, n_heads, T, d_head)
   • Attention output = softmax(scores) @ V
   • Produces context vectors per token for each head.

6. (B, T, n_heads, d_head)
   • Transpose back to restore token-major layout.
   • Collects all head outputs together per token.

7. (B, T, n_heads * d_head)
   • Flatten / merge all heads.
   • Since n_heads * d_head = C, final output shape = (B, T, C).
   • Often followed by a final linear projection.
"""
