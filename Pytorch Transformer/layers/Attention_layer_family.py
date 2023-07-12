# 23.07.07
# discription: Various Attention Techniques
# shape 

# Tip. 


import torch
import torch.nn as nn
import math


# ScaleDotProductAttention
class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, q, k, v, mask=None):
        # scaling에 필요한 head_dim 값 얻기
        # (batch_size, head, seq_len, head_dim)
        _, _, _, head_dim = q.size()

        # 1. K를 transpose하기 (seq_len, head_dim의 행렬 전환)
        k_t = k.transpose(-1,-2)

        # 2. Q 와 K^T 의 MatMul
        # (batch_size, head, q_seq_len, k_seq_len)
        attention_score = torch.matmul(q,k_t)

        # 3. Scaling
        attention_score /= math.sqrt(head_dim)

        # 4. Mask가 있다면 마스킹된 부위 -1e10으로 채우기
        # mask는 단어가 있는 곳(True), 마스킹된 곳(False)으로 표시되었기 때문에 False(0)에 해당되는 부분을 -1e10으로 masking out한다.
        # Tensor.masked_fill_(mask_boolean, value) 함수는 True값을 value로 채운다.
        if mask is not None:
            attention_score = attention_score.masked_fill(mask==0,-1e10) 
        
        # 5. Softmax 취하기 
        attention_score = self.softmax(attention_score)

        # 6. Attention 결과와 V의 MatMul 계산하기
        result = torch.matmul(attention_score, v)
        
        return result, attention_score