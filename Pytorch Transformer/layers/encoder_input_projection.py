# 23.07.07
# discription: Embedding with d_model as transformer input.
# shape [batch , Time_seq, feature_dim] -> [batch , Time_seq, d_model]

# Tip. projection mean to pass through a Linear Layer(=fc layer).

import torch
import torch.nn as nn


class input_embedding(nn.Module):
    def __init__(self,input_size,d_model):
        super(input_embedding,self).__init__()
        self.encoder_input_layer = nn.Linear(
            in_features=input_size, 
            out_features=d_model
        )

    def forward(self,x):
        x=self.encoder_input_layer(x)
        return x