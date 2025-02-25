# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import random
from metric.mask_metric import masked_mae,masked_mape,masked_rmse,masked_mse
from block.informer import Informer, InformerStack
from block.gat import GraphAttentionLayer
from block.cross import cross_att
from block.revIN import RevIN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device2 = torch.device("cpu")

random.seed()
torch.manual_seed()
np.random.seed()

batch_size = 32
epoch = 100
IF_mask = 0
DATASET_INPUT_LEN = 14
DATASET_OUTPUT_LEN = 14
num_layer = 2
NUM_NODES = 78
enc_in = NUM_NODES
dec_in = NUM_NODES
c_out  = NUM_NODES
seq_len = DATASET_INPUT_LEN
label_len = DATASET_INPUT_LEN // 2
out_len = DATASET_OUTPUT_LEN
factor = 5
d_model = 512
n_heads = 4

IF_STACK = False
if IF_STACK:
    e_layers = [4, 2, 1]
else:
    e_layers = 2

d_layers = 1
d_ff = 32
dropout = 0.15
attn = 'prob'
embed = "timeF"
activation= "gelu"
output_attention = False
distill = True
mix = True
num_time_features  = 4
time_of_day_size   = 486
day_of_week_size   =  7
day_of_month_size  = 31
day_of_year_size   = 366

IF_cross = True
lr_rate = 0.001
weight_decay = 0.0005
max_norm = 0
num_lr = 5            
gamme = 0.5
milestone = [1,4,10,15,30,50,70,90]

history_seq_len = 14
data_name = 'waterquality'
data_file = "data/" + str(history_seq_len)+ ".npz"
raw_data = np.load(data_file)

def Inverse_normalization(x, max, min):
    return x * (max - min) + min

train_x = raw_data["train_x_raw"]

train_y = raw_data["train_y"]

graph_data = torch.tensor(raw_data["graph"]).to(torch.float32)

input_len = train_x.shape[-1]
output_len = train_y.shape[-1]
num_id = train_x.shape[-2]

train_x = train_x.astype(float)
train_y = train_y.astype(float)
train_x = torch.tensor(train_x)
train_y = torch.tensor(train_y)

train_data = torch.cat([train_x,train_y],dim=2).to(torch.float32)
train_data = DataLoader(train_data,batch_size=batch_size,shuffle=False)

valid_x = raw_data["vail_x_raw"]
valid_y = raw_data["vail_y"]
valid_x = valid_x.astype(float)
valid_y = valid_y.astype(float)
valid_x = torch.tensor(valid_x).to(torch.float32)
valid_y = torch.tensor(valid_y).to(torch.float32)
valid_data = torch.cat([valid_x,valid_y],dim=2).to(torch.float32)
valid_data = DataLoader(valid_data,batch_size=batch_size,shuffle=False)

test_x = raw_data["test_x_raw"]
test_y = raw_data["test_y"]
test_x = test_x.astype(float)
test_y = test_y.astype(float)
test_x = torch.tensor(test_x)
test_y = torch.tensor(test_y)
test_data = torch.cat([test_x, test_y],dim=2).to(torch.float32)
test_data = DataLoader(test_data,batch_size=batch_size,shuffle=False)

max_min = raw_data['max_min']
max_data, min_data = max_min[0],max_min[1]


class HybridAttention(nn.Module):
    def __init__(self, IF_STACK,num_layer, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 time_of_day_size, day_of_week_size, day_of_month_size,day_of_year_size,
                 factor, d_model, n_heads, e_layers, d_layers, d_ff,dropout, attn, embed, freq, activation,
                 output_attention, distill, mix, num_time_features,IF_cross):
        super(HybridAttention, self).__init__()
        self.IF_STACK = IF_STACK
        self.num_layer = num_layer
        self.lay_norm = nn.LayerNorm([out_len])
        self.IF_cross = IF_cross
        self.RevIN = RevIN(enc_in)

        self.GAT1 = GraphAttentionLayer(seq_len, out_len, dropout)
        self.GAT2 = GraphAttentionLayer(out_len, out_len, dropout)

        if self.IF_STACK:
            self.Informer =InformerStack(enc_in, dec_in, c_out, seq_len, label_len, out_len,
                time_of_day_size, day_of_week_size, day_of_month_size,day_of_year_size,
                factor, d_model, n_heads, e_layers, d_layers, d_ff,
                dropout=dropout, attn=attn, embed=embed, freq=freq, activation=activation,
                output_attention=output_attention, distill=distill, mix=mix, num_time_features=num_time_features)
        else:
            self.Informer = Informer(enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 time_of_day_size, day_of_week_size, day_of_month_size=day_of_month_size,
                 day_of_year_size=day_of_year_size,
                 factor=factor, d_model=d_model, n_heads=n_heads, e_layers=e_layers, d_layers=d_layers, d_ff=d_ff,
                 dropout=dropout, attn=attn, embed=embed, freq=freq, activation=activation,
                 output_attention=output_attention, distill=distill, mix=mix, num_time_features=num_time_features)

        self.cross = cross_att(out_len,n_heads,dropout)
        self.decoder = nn.Conv1d(in_channels=out_len,out_channels=out_len,kernel_size=1)

    def forward(self, x, y, graph_data, device):
        graph_data = graph_data.to(device)
        graph_data = HybridAttention.calculate_laplacian_with_self_loop(graph_data)
        x = self.RevIN(x.transpose(-2, -1), 'norm').transpose(-2, -1)

        for i in range(self.num_layer):
            if i == 0:
                prediction_GAT = F.gelu(self.GAT1(x,graph_data))
            else:
                prediction_GAT = F.gelu(self.GAT2(prediction_GAT, graph_data))

        prediction_In = self.Informer(x, y)

        if self.IF_cross:

            x = self.cross(prediction_In, prediction_GAT).transpose(-2, -1)
        else:

            x = prediction_GAT + prediction_In
            x = self.lay_norm(x).transpose(-2, -1)

        x = self.decoder(x)
        x = self.RevIN(x, 'denorm')
        x = x.transpose(-2, -1)
        return x

    @staticmethod
    def calculate_laplacian_with_self_loop(matrix):
        row_sum = matrix.sum(1)
        d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        normalized_laplacian = (
            matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
        )
        return normalized_laplacian

my_net = HybridAttention(IF_STACK, num_layer, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 time_of_day_size, day_of_week_size, day_of_month_size=day_of_month_size,
                 day_of_year_size = day_of_year_size,
                 factor=factor, d_model=d_model, n_heads=n_heads,
                 e_layers=e_layers, d_layers=d_layers, d_ff=d_ff,
                 dropout=dropout, attn=attn, embed=embed, freq='h',
                 activation=activation, output_attention=output_attention,
                 distill=distill, mix=mix, num_time_features=num_time_features, IF_cross=IF_cross)

my_net = my_net.to(device)

optimizer = optim.Adam(params=my_net.parameters(),lr=lr_rate, weight_decay=weight_decay)
num_vail = 0
min_vaild_loss = float("inf")

for i in range(epoch):
    num = 0
    loss_out = 0.0
    my_net.train()
    for data in train_data:
        my_net.zero_grad()
        
        train_feature = data[:,:,:input_len].to(device)
        train_target = data[:,:,input_len:].to(device)
        train_pre = my_net(train_feature,train_target,graph_data, device)

        loss_data = masked_mae(train_pre,train_target,0.0)
        loss_data.backward()

        if max_norm > 0:
            nn.utils.clip_grad_norm_(my_net.parameters(), max_norm = max_norm)
        else:
            pass

        num += 1
        optimizer.step()
        loss_out += loss_data
    loss_out = loss_out/num

    num_va = 0
    loss_vaild = 0.0
    my_net.eval()

    with torch.no_grad():
        for data in valid_data:

            valid_x = data[:, :, :input_len].to(device)
            valid_y = data[:, :, input_len:].to(device)
            valid_pre = my_net(valid_x,valid_y,graph_data, device)
            loss_data = masked_mae(valid_pre, valid_y,0.0)

            num_va += 1
            loss_vaild += loss_data
        loss_vaild = loss_vaild / num_va

    if (i + 1) in milestone:
        for params in optimizer.param_groups:

            params['lr'] *= gamme

    print('loss of epoch {} of the training set: {:02.4f}, loss of valid_data:{:02.4f}:'.format(i+1,loss_out,loss_vaild))

my_net.eval()
my_net = my_net.to(device2)

with torch.no_grad():
    all_pre = 0.0
    all_true = 0.0
    num = 0

    for data in test_data:
        test_feature = data[:,:,:input_len].to(device2)
        test_target = data[:,:,input_len:].to(device2)
        test_pre = my_net(test_feature,test_target,graph_data, device2)
        if num == 0:
            all_pre = test_pre
            all_true = test_target
        else:
            all_pre = torch.cat([all_pre, test_pre], dim=0)
            all_true = torch.cat([all_true, test_target], dim=0)
        num += 1

test_x = Inverse_normalization(test_x, max_data, min_data)
final_pred = Inverse_normalization(all_pre, max_data, min_data)
final_target = Inverse_normalization(all_true, max_data, min_data)
np.savetxt('results/pred.csv', pd.DataFrame(final_pred[:, :, 0]), delimiter =',')
np.savetxt('results/target.csv', pd.DataFrame(final_target[:, :, 0]), delimiter =',')

mae, mape, rmse = (masked_mae(final_pred, final_target, 0.0),\
                        masked_mape(final_pred, final_target, 0.0)*100,\
                        masked_rmse(final_pred, final_target, 0.0))

print('Masked_Overall prediction performance:\nRMSE: {}, MAPE: {}, MAE: {}'.format(rmse, mape, mae))

mae1, mape1, rmse1 = (masked_mae(final_pred[:, :, 0], final_target[:, :, 0], 0.0),\
                    masked_mape(final_pred[:, :, 0], final_target[:, :, 0],0.0)*100,\
                    masked_rmse(final_pred[:, :, 0], final_target[:, :, 0], 0.0))
print('Masked_Prediction performance in the first time step:\nRMSE: {}, MAPE: {}, MAE: {}'.format(rmse1, mape1, mae1))

mae2, mape2, rmse2 = (masked_mae(final_pred[:,:,-1], final_target[:,:,-1],0.0),\
                    masked_mape(final_pred[:,:,-1], final_target[:,:,-1],0.0)*100,\
                    masked_rmse(final_pred[:,:,-1], final_target[:,:,-1],0.0))
print('Masked_Prediction performance in the last time step: \nRMSE: {}, MAPE: {}, MAE: {}'.format(rmse2, mape2, mae2))

# with open('log/run.txt', 'a+', encoding='utf-8') as fw:
#     fw.write("Masked_Overall prediction performance:\nRMSE: {}, MSE: {}, MAPE: {}, MAE: {}\n".format(rmse, mape, mae))
#     fw.write("Masked_Prediction performance in the first time step:\nRMSE: {}, MSE: {}, MAPE: {}, MAE: {}\n".format(rmse1, mape1, mae1))
#     fw.write("Masked_Prediction performance in the last time step:\nRMSE: {}, MSE: {}, MAPE: {}, MAE: {}\n".format(rmse2, mape2, mae2))