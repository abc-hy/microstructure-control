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

# import gpustat


# hyper parameter of inverse nn
EPOCHS_2 = 7786
INPUT_DIM_2=1305
OUTPUT_DIM_2=16
HIDDEN_DIM_2=53
NUM_LAYERS_2=5
WEIGHT_DECAY_2=0

# hyper parameter of forward nn
EPOCHS_1= 7786
INPUT_DIM_1=25
OUTPUT_DIM_1=1296
HIDDEN_DIM_1=53
NUM_LAYERS_1=5          #total number of layers
WEIGHT_DECAY_1=0



#training and testing data
# df11 =  pd.read_csv("TH_training.txt",sep=',',header=None)
# TH_training_df = df11.T
# df12 =  pd.read_csv("TH_training_test.txt",sep=',',header=None)
# TH_training_test_df = df12.T
# df13 =  pd.read_csv("TH_testing.txt",sep=',',header=None)
# TH_testing_df = df13.T
# df21 =  pd.read_csv("phi_training.txt",sep=',',header=None)
# phi_training_df = df21.T
# df22 =  pd.read_csv("phi_training_test.txt",sep=',',header=None)
# phi_training_test_df = df22.T
# df23 =  pd.read_csv("phi_testing.txt",sep=',',header=None)
# phi_testing_df = df23.T

#phi_target training and testing data 3 by 3
df31 =  pd.read_csv("phi_target_3_3_training.txt",sep=',',header=None)
phi_target_3_3_training_df = df31.T
df32 =  pd.read_csv("phi_target_3_3_training_test.txt",sep=',',header=None)
phi_target_3_3_training_test_df = df32.T
df33 =  pd.read_csv("phi_target_3_3_testing.txt",sep=',',header=None)
phi_target_3_3_testing_df = df33.T


#phi_target training and testing data 36 by 36
df41 =  pd.read_csv("phi_final_training_inverse.txt",sep=',',header=None)
phi_final_training_inverse_df = df41.T
df42 =  pd.read_csv("phi_final_training_test_inverse.txt",sep=',',header=None)
phi_final_training_test_inverse_df = df42.T
df43 =  pd.read_csv("phi_final_testing_inverse.txt",sep=',',header=None)
phi_final_testing_inverse_df = df43.T



phi_target_3_3_training_df_array_double = np.array(phi_target_3_3_training_df)
phi_target_3_3_training_df_array_float = phi_target_3_3_training_df_array_double.astype(np.float32)
phi_target_3_3_training = torch.tensor(phi_target_3_3_training_df_array_float)
phi_target_3_3_training=phi_target_3_3_training.cuda(0)


phi_target_3_3_training_test_df_array_double = np.array(phi_target_3_3_training_test_df)
phi_target_3_3_training_test_df_array_float = phi_target_3_3_training_test_df_array_double.astype(np.float32)
phi_target_3_3_training_test = torch.tensor(phi_target_3_3_training_test_df_array_float)
phi_target_3_3_training_test=phi_target_3_3_training_test.cuda(0)



phi_target_3_3_testing_df_array_double = np.array(phi_target_3_3_testing_df)
phi_target_3_3_testing_df_array_float = phi_target_3_3_testing_df_array_double.astype(np.float32)
phi_target_3_3_testing = torch.tensor(phi_target_3_3_testing_df_array_float)
phi_target_3_3_testing=phi_target_3_3_testing.cuda(0)




phi_final_training_inverse_df_array_double = np.array(phi_final_training_inverse_df)
phi_final_training_inverse_df_array_float = phi_final_training_inverse_df_array_double.astype(np.float32)
phi_final_training_inverse = torch.tensor(phi_final_training_inverse_df_array_float)
phi_final_training_inverse=phi_final_training_inverse.cuda(0)


phi_final_training_test_inverse_df_array_double = np.array(phi_final_training_test_inverse_df)
phi_final_training_test_inverse_df_array_float = phi_final_training_test_inverse_df_array_double.astype(np.float32)
phi_final_training_test_inverse = torch.tensor(phi_final_training_test_inverse_df_array_float)
phi_final_training_test_inverse=phi_final_training_test_inverse.cuda(0)



phi_final_testing_inverse_df_array_double = np.array(phi_final_testing_inverse_df)
phi_final_testing_inverse_df_array_float = phi_final_testing_inverse_df_array_double.astype(np.float32)
phi_final_testing_inverse = torch.tensor(phi_final_testing_inverse_df_array_float)
phi_final_testing_inverse=phi_final_testing_inverse.cuda(0)








#change dataframe to tensor and changing double type to float32
# TH_training_df_array_double = np.array(TH_training_df)
# TH_training_df_array_float = TH_training_df_array_double.astype(np.float32)
# TH_training = torch.tensor(TH_training_df_array_float)

# #Moving to GPU
# TH_training=TH_training.cuda(0)


# # TH_training_test_df_array = np.array(TH_training_test_df)
# # TH_training_test_double = torch.tensor(TH_training_test_df_array)

# TH_training_test_df_array_double = np.array(TH_training_test_df)
# TH_training_test_df_array_float = TH_training_test_df_array_double.astype(np.float32)
# TH_training_test = torch.tensor(TH_training_test_df_array_float)

# #Moving to GPU
# TH_training_test=TH_training_test.cuda(0)


# TH_testing_df_array_double = np.array(TH_testing_df)
# TH_testing_df_array_float = TH_testing_df_array_double.astype(np.float32)
# TH_testing = torch.tensor(TH_testing_df_array_float)

# #Moving to GPU
# TH_testing=TH_testing.cuda(0)




# phi_training_df_array_double = np.array(phi_training_df)
# phi_training_df_array_float = phi_training_df_array_double.astype(np.float32)
# phi_training = torch.tensor(phi_training_df_array_float)

# #Moving to GPU
# phi_training=phi_training.cuda(0)



# phi_training_test_df_array_double = np.array(phi_training_test_df)
# phi_training_test_df_array_float = phi_training_test_df_array_double.astype(np.float32)
# phi_training_test = torch.tensor(phi_training_test_df_array_float)

# #Moving to GPU
# phi_training_test=phi_training_test.cuda(0)



# phi_testing_df_array_double = np.array(phi_testing_df)
# phi_testing_df_array_float = phi_testing_df_array_double.astype(np.float32)
# phi_testing = torch.tensor(phi_testing_df_array_float)

# #Moving to GPU
# phi_testing=phi_testing.cuda(0)


#Training data number and testing data number
Ntraining=np.size(df31,1)

input_number=np.size(df31,0)

Ntesting=np.size(df33,1)

# output_number=np.size(df21,0)

# target=phi_training

#Moving to GPU
# target=target.cuda(0)



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


#将模型参数加载到新模型中
state_dict = torch.load('net_params1.pkl')
net_1.load_state_dict(state_dict)

# require_gradient=False
for param in net_1.parameters():
    param.requires_grad = False


##########creating and train a inverse nn
training_start_2 = time.time()
# creating a inverse neural network
class NetInverse(nn.Module):
    def __init__(self,input_dim_2=INPUT_DIM_2,num_layers_2=NUM_LAYERS_2,hidden_dim_2=HIDDEN_DIM_2,output_dim_2=OUTPUT_DIM_2):
        super(NetInverse, self).__init__()
        self.num_layers_2 = num_layers_2
        self.linears = nn.ModuleList([nn.Linear(input_dim_2,hidden_dim_2)])
        for i in range(1, self.num_layers_2-2):
            self.linears.append(nn.Linear(hidden_dim_2, hidden_dim_2))
        self.linears.append(nn.Linear(hidden_dim_2, output_dim_2))

    def forward(self, x_2):
        for i in range(self.num_layers_2-2):
            x_2 = torch.sigmoid(self.linears[i](x_2))
        x_2 = self.linears[-1](x_2)
        return x_2


net_2 = NetInverse()


net_2=net_2.cuda(0)

# optimizer of inverse nn    
optimizer_2=optim.Adam(net_2.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=WEIGHT_DECAY_2, amsgrad=False)


# create a loss function
criterion = nn.MSELoss()


loss_2_values=[]
epoch_2_plot=[]



for epoch_2 in range(0,EPOCHS_2):
    # total_batch_training_loss = 0  
    # for batch_idx in range(0,Ntraining):
    optimizer_2.zero_grad()
    
    phi_final_phi_target=torch.cat((phi_final_training_inverse,phi_target_3_3_training),1)
    
    
    TH_net_out_2 = net_2(phi_final_phi_target)
    
    Th_out_phi_target=torch.cat((TH_net_out_2,phi_target_3_3_training),1)
    
    phi_final_net_out_1 = net_1(Th_out_phi_target)

    
    loss_2 = criterion(phi_final_net_out_1, phi_final_training_inverse)
    
    loss_2.backward()

    optimizer_2.step()

        
    if epoch_2 % 5 == 0:           
        loss_2_values.append(loss_2)
        epoch_2_plot.append(epoch_2) 
    ##plotting the loss vs. epochs figure
        plt.plot(epoch_2_plot, loss_2_values,'r-o')
        plt.xlabel('epochs')
        plt.ylabel('Inverse nn training loss') 
        plt.title('Inverse nn training loss vs. epochs')  
        
        plt.savefig("Inverse nn training error vs. epochs.png")
        plt.show()
        plt.close()
        
training_end_2 = time.time()     

torch.save(net_2.state_dict(), "net_params%d.pkl"%(2))

print('Inverse nn 程序运行时间:%s毫秒' % ((training_end_2 - training_start_2)*1000))   

# print(loss)
# print(EPOCHS)
# print(loss.item())
# print(net_out)
# run a test loop using testing data
test_loss = 0
# for i in range(0,Ntesting):

phi_final_phi_target_testing=torch.cat((phi_final_testing_inverse,phi_target_3_3_testing),1)


TH_net_out_2_testing = net_2(phi_final_phi_target_testing)

Th_out_phi_target_testing=torch.cat((TH_net_out_2_testing,phi_target_3_3_testing),1)


Th_out_phi_target_testing_numpy=Th_out_phi_target_testing.detach().cpu().numpy()

phi_final_net_out_1_testing = net_1(Th_out_phi_target_testing)

phi_final_net_out_1_testing_numpy=phi_final_net_out_1_testing.detach().cpu().numpy()  
    
test_loss = criterion(phi_final_net_out_1_testing, phi_final_testing_inverse)

flops, params = profile(net_2, inputs=(phi_final_phi_target_testing, ))

print('Inverse nn的flops:%s' % flops) 
    
    # print(average_test_loss.item())
# print(net_out1)
# print(np.size(net_out1))
    
    ##plotting the testing error vs. training data number figure
plt.plot(Ntraining, test_loss.item(),'r-o')
plt.xlabel('Training data number')
plt.ylabel('Error') 
plt.title('Testing error vs. Training data number of Inverse nn')  
# plt.show()
plt.savefig("Testing_error vs. training_data_number of Inverse nn.png")
plt.show()
plt.close()
    

np.savetxt('Th_out_phi_target_testing.txt',Th_out_phi_target_testing_numpy)    
np.savetxt('phi_final_net_out_1_testing.txt',phi_final_net_out_1_testing_numpy)   



    
# run a test loop using training data
training_test_loss = 0

phi_final_phi_target_training_test=torch.cat((phi_final_training_test_inverse,phi_target_3_3_training_test),1)

TH_net_out_2_training_test = net_2(phi_final_phi_target_training_test)

Th_out_phi_target_training_test=torch.cat((TH_net_out_2_training_test,phi_target_3_3_training_test),1)


phi_final_net_out_1_training_test= net_1(Th_out_phi_target_training_test)

    
training_test_loss = criterion(phi_final_net_out_1_training_test, phi_final_training_test_inverse)

       ##plotting the training error vs. training data number figure
# print(net_out2)  
# print(np.size(net_out2))

plt.plot(Ntraining, training_test_loss.item(),'b-s',label="Training error")
plt.plot(Ntraining, test_loss.item(),'k-o',label="Testing error")
plt.xlabel('Training data number')
plt.ylabel('Error') 
plt.title('Training&testing error vs. Training data number of Inverse nn')  
plt.legend()
# plt.show()
plt.savefig("Training&testing error vs. Training data number of Inverse nn.png")
plt.show()
plt.close() 

    