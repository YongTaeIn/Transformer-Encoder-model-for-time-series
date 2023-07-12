import torch
import torch.nn as nn

from layers.positional_encoding import PositionalEncoding
from layers.encoder_input_projection import input_embedding
from blocks.encoder_layer import EncoderLayer



class Encoder(nn.Module):
    def __init__(self, input_size, num_predicted_features, d_model, head, d_ff, max_len, dropout, n_layers):
        super().__init__()

        # Embedding
        self.input_emb = input_embedding(input_size=input_size,d_model=d_model)
        
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=max_len,dropout=dropout)
        #self.dropout = nn.Dropout(p=dropout)

        # n개의 encoder layer를 list에 담기
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model=d_model, 
                                                         head=head, 
                                                         d_ff=d_ff, 
                                                         dropout=dropout)
                                             for _ in range(n_layers)])
        self.linear_mapping = nn.Linear(
        in_features=d_model, 
        out_features=num_predicted_features
        )

        self.linear2 = nn.Sequential(
        nn.Linear(144, num_predicted_features)
        )




    def forward(self, x): # ,padding_mask
        # 1. 입력에 대한 input embedding, 
        
        input_emb = self.input_emb(x)
        
        # 2. positional encoding 생성
        x = self.pos_encoding(input_emb)

        # 3. n번 EncoderLayer 반복하기
        for encoder_layer in self.encoder_layers:
            x, attention_score = encoder_layer(x) 

        # 4. 차원축소 수행
        x = self.linear_mapping(x)
        x=x[:,:,0]

        # 5. 원하는 길이 출력
        x = self.linear2(x)
        return x