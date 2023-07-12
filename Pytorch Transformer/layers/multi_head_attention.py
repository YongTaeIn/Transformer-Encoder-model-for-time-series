# 23.07.07
# discription: With multi-head attention, 
# you can select and use the desired attention.
# 

# Tip.


import torch
import torch.nn as nn
from layers.Attention_layer_family import ScaleDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,head):
        super().__init__()
        self.d_model = d_model
        self.head = head
        self.head_dim = d_model // head
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        self.w_o = nn.Linear(d_model,d_model)
        self.attention = ScaleDotProductAttention()

    def forward(self, q, k, v, mask=None):
        #  [batch_size, seq_len, d_model]
        batch_size, _, _ = q.size()

        # 1. Q,K,V를  d_k, d_k, d_v 차원으로 projection
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. Q,K,V를 head 수 만큼 분리해주기 
        # (batch_size, seq_len, d_model)  
        # -> (batch_size, head, seq_len, head_dim)
        # 디코더에서는 q와 k,v 의 seq_len 가 다른 경우가 올 수 있음
        q = q.view(batch_size, -1, self.head, self.head_dim).transpose(1,2)
        k = k.view(batch_size, -1, self.head, self.head_dim).transpose(1,2)
        v = v.view(batch_size, -1, self.head, self.head_dim).transpose(1,2)

        # 3. Scaled Dot-Product Attention 을 수행하기
        out, attention_score = self.attention(q,k,v,mask)

        # 4. 분리된 head들을 concat 하기
        # (batch_size, head, seq_len, head_dim)
        # -> (batch_size, seq_len, d_model)
        out = out.transpose(1,2).contiguous().view(batch_size, -1, self.d_model)

        # 5. d_model 차원으로 projection
        out = self.w_o(out)

        return out, attention_score