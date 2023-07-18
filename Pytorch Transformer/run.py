## 실험 내용.23.07.08
## 바닐라 트랜스포머 
## feature: voltage :총 1개

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim


import tensorflow as tf
import numpy as np
import pandas as pd

import math
import random
import glob
import os


from torch import nn, Tensor
from torch.utils.data import Dataset,DataLoader,random_split

# 초기 input d_model로 만듬
from layers.encoder_input_projection import input_embedding
# 위치 인코딩
from layers.positional_encoding import PositionalEncoding
# 멀티헤드 어텐션
from layers.multi_head_attention import MultiHeadAttention
# 마지막단 Position Wise
from layers.positioin_wise import PositionWise

from models.encoder import Encoder
from layers.Earlystopping import EarlyStopping



device = torch.device('cuda:0')
seed=42
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

#############################################################
################   DataLoader  ##############################
#############################################################


class CustomDataset(Dataset):

    # 생성자, 데이터를 전처리 하는 부분
    def __init__(self, folder_path):
        self.data = []
        window_size=144
        # 폴더 안에 있는 모든 CSV 파일을 읽음
        csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            # 3열을 feature, 2열을 label로 사용
            features = df.iloc[:, 3].values.astype('float32')
            labels = df.iloc[:, 2].values.astype('float32')
            # feature를 144개씩 묶음
            for i in range(len(features)-window_size):
                feature_subset = features[i:i+window_size]
                label_subset = labels[i+window_size]
                # 아랫줄 특징 한개일때는 필요함. 
                feature_subset = torch.unsqueeze(torch.FloatTensor(feature_subset), dim=1)
                self.data.append((feature_subset, label_subset))

    # 데이터셋의 총 길이를 반환하는 부분 
    def __len__(self):
        # 데이터셋의 길이를 반환
        return len(self.data)

    def __getitem__(self, idx):

        # 주어진 인덱스에 해당하는 데이터 반환
        features, labels = self.data[idx]
        # 뒤에 차원을 늘려주는것.
        
        inp = torch.FloatTensor(features)
        outp = torch.FloatTensor([labels])
        return inp, outp
    
# instance 생성 
dataset = CustomDataset("../98. IOT_journal/preprocessed_data/data_13/train")


# train,test 분리 
val_ratio = 0.2 
val_size = int(val_ratio * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


# Create dataloaders for train and validation
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)  #시계열 적인 해석이기에 batch가 뒤섞이면 안됨.
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)


model_1= Encoder(input_size=1,
                 num_predicted_features=1,
                d_model=128,
                head=4,
                d_ff=32,
                max_len=144,
                dropout=0.2,
                n_layers=4,
                ).to(device)

optimizer = torch.optim.Adam(model_1.parameters(), lr=0.0001)

early_stopping = EarlyStopping(patience = 100, verbose = True)


# 모델이 학습되는 동안 trainning loss를 track
train_losses = []
# 모델이 학습되는 동안 validation loss를 track
valid_losses = []
# epoch당 average training loss를 track
avg_train_losses = []
# epoch당 average validation loss를 track
avg_valid_losses = []


# 전체 훈련 데이터에 대해 경사 하강법을 20회 반복
best_valid_loss = float('inf') 

nb_epochs = 2000
for epoch in range(nb_epochs+1):
    model_1.train()
    for batch_idx, samples in enumerate(train_dataloader):
        x_train, y_train = samples
        x_train=x_train.to(device)
        y_train=y_train.to(device)
        # H(x) 계산
        prediction = model_1(x_train)
        # print('예측 사이즈: ' ,prediction.size())
        # print('y_train 사이즈',y_train.size())
        # 오차 계산. 
        loss = F.l1_loss(prediction, y_train) # 파이토치 오차함수 mae
        # cost로 H(x) 개선하는 부분
        # gradient를 0으로 초기화
        optimizer.zero_grad()
        # 비용 함수를 미분하여 gradient 계산
        loss.backward() # backward 연산
        # W와 b를 업데이트
        optimizer.step()
        print('Epoch [{}/{}], Batch: [{}/{}], Loss: {:.4f}'.format(epoch, nb_epochs, batch_idx, len(train_dataloader), loss.item()))
        train_losses.append(loss.item())


    model_1.eval()
    for batch_idx, samples in enumerate(val_dataloader): 
        x_val, y_val = samples
        x_val=x_val.to(device)
        y_val=y_val.to(device)

        prediction = model_1(x_val)
        loss = F.l1_loss(prediction, y_val)
        valid_losses.append(loss.item())


    train_loss = np.average(train_losses)
    valid_loss = np.average(valid_losses)
    avg_train_losses.append(train_loss)
    avg_valid_losses.append(valid_loss)

    epoch_len = len(str(nb_epochs))

    print_msg = (f'[{epoch:>{epoch_len}}/{nb_epochs:>{epoch_len}}] ' +
                f'train_loss: {train_loss:.5f} ' +
                f'valid_loss: {valid_loss:.5f}')
    print(print_msg)
    
    train_losses = []
    valid_losses = []

    early_stopping(valid_loss, model_1)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        # Save the model
        model_path = f"{'./pt/'}model_epoch_{epoch}_val_loss_{valid_loss:.4f}.pt"
        torch.save(model_1.state_dict(), model_path)

    if early_stopping.early_stop:
        print("Early stopping")
        break
