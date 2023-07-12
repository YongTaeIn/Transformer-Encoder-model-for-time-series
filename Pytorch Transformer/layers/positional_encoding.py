# 23.07.07
# discription: Positional Encoding for sequence order(=word location information+)
# shape [batch , Time_seq, d_model] -> [batch , Time_seq, d_model]

#  Time_seq = seq_len

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch import nn, Tensor



class PositionalEncoding(nn.Module):
 
    def __init__(
        self, 
        dropout: float=0.1, 
        max_len: int=144, 
        d_model: int=512,
        batch_first: bool=False
        ):

        """
        Parameters:
            dropout: the dropout rate
            max_seq_len: the maximum length of the input sequences
            d_model: The dimension of the output of sub-layers in the model 
                     (Vaswani et al, 2017)
        """

        super().__init__()
        self.d_model = d_model      
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        self.x_dim = 1 if batch_first else 0

        # copy pasted from PyTorch tutorial
        position = torch.arange(max_len).unsqueeze(1)       
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))       
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val] or 
               [enc_seq_len, batch_size, dim_val]
        """
        x = x + self.pe[:x.size(self.x_dim)]
        return self.dropout(x)