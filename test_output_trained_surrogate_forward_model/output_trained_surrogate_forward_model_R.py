
from __future__ import absolute_import, division, print_function
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import pandas as pd
import torch.optim as optim
# from openpyxl import load_workbook

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import numpy as np
import time
from thop import profile

df13 = pd.read_csv("Th_large_testing.txt",sep=',',header=None)
Th_large_testing_df = df13.T

Th_large_testing_df_array_double = np.array(Th_large_testing_df)
Th_large_testing_df_array_float = Th_large_testing_df_array_double.astype(np.float32)



df33 = pd.read_csv("phi_target_3_3_testing.txt",sep=',',header=None)
phi_target_3_3_testing_df = df33.T

phi_target_3_3_testing_df_array_double = np.array(phi_target_3_3_testing_df)
phi_target_3_3_testing_df_array_float = phi_target_3_3_testing_df_array_double.astype(np.float32)
        
Th_phi_target_testing_array_float=np.hstack((Th_large_testing_df_array_float,phi_target_3_3_testing_df_array_float))
Th_phi_target_testing = torch.tensor(Th_phi_target_testing_array_float)

df23 = pd.read_csv("phi_final_testing.txt",sep=',',header=None)
phi_final_testing_df = df23.T

phi_final_testing_df_array_double = np.array(phi_final_testing_df)
phi_final_testing_df_array_float = phi_final_testing_df_array_double.astype(np.float32)
phi_final_testing = torch.tensor(phi_final_testing_df_array_float)
        
        #Moving to GPU
phi_final_testing=phi_final_testing.cuda(0)
        

    
#Moving to GPU
Th_phi_target_testing=Th_phi_target_testing.cuda(0)


# hyper parameter of forward nn
EPOCHS_1= 10000
INPUT_DIM_1=25
OUTPUT_DIM_1=1296
HIDDEN_DIM_1=53
NUM_LAYERS_1=5           #total number of layers
WEIGHT_DECAY_1=0

######### load the pre-trained forward neural network
# def restore_forwardNN():

class NetForward(nn.Module):
    def __init__(self,input_dim_1=INPUT_DIM_1,num_layers_1=NUM_LAYERS_1,hidden_dim_1=HIDDEN_DIM_1,output_dim_1=OUTPUT_DIM_1):
        super(NetForward, self).__init__()
        self.num_layers_1 = num_layers_1
        self.linears = nn.ModuleList([nn.Linear(input_dim_1,hidden_dim_1)])
        for i in range(1, self.num_layers_1-2):
            self.linears.append(nn.Linear(hidden_dim_1, hidden_dim_1))
        self.linears.append(nn.Linear(hidden_dim_1, output_dim_1))

    def forward(self, x_1):
        for i in range(self.num_layers_1-2):
            x_1 = torch.sigmoid(self.linears[i](x_1))
        x_1 = self.linears[-1](x_1)
        return x_1

#construct the forward nn structure
net_1 = NetForward()
net_1=net_1.cuda(0)
# create a loss function
criterion = nn.MSELoss()

#将模型参数加载到新模型中
state_dict = torch.load('net_params1.pkl')
net_1.load_state_dict(state_dict)

# require_gradient=False
for param in net_1.parameters():
    param.requires_grad = False
    


net_out1 = net_1(Th_phi_target_testing)
            # sum up batch loss
test_loss = criterion(net_out1, phi_final_testing)


E_test=test_loss.item()

# phinnout_testing_df_array_double = np.array(phinnout_testing_df)
#phinnout_testing_df_array_float = net_out1.detach().cpu().numpy()
phinnout_testing_df_array_float = net_out1.detach().cpu().numpy()
np.savetxt('phinnout_testing.txt',phinnout_testing_df_array_float,fmt='%f',delimiter=' ')
        



