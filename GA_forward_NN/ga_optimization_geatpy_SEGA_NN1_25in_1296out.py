from __future__ import absolute_import, division, print_function
## Writtrn by Haiying Yang
## Date: August,2021
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
import geatpy as ea  # import geatpy
# from ga_overfitting_nn_optimization_geatpy import ga_overfitting_nn_optimization_geatpy


class MyProblem(ea.Problem): # 继承Problem父类
    def __init__(self):
        name = 'MyProblem' # 初始化name（函数名称，可以随意设置）
        M = 1 # 初始化M（目标维数）
        maxormins = [1] # 初始化目标最小最大化标记列表，1：min；-1：max
        Dim = 3 # 初始化Dim（决策变量维数）
        varTypes = [1,1,1]  # 初始化决策变量类型，0：连续；1：离散
        lb = [2,15,6000] # 决策变量下界
        ub = [3,53,8000] # 决策变量上界
        # lbin = [1,1,1,1] # 决策变量下边界
        # ubin = [1,1,1,1] # 决策变量上边界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
# 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb,ub, lbin, ubin)


#defining function for calculating E_test and overfitting quantity
    def nn_training_overfitting(self,x1,x2,x3,count):
        
    
        
        
        EPOCHS = int(x3)
        INPUT_DIM=25
        OUTPUT_DIM=1296
        HIDDEN_DIM=int(x2)
        NUM_LAYERS=int(x1+2) 
        WEIGHT_DECAY=0

        
        

        #training and testing data
        df11 =  pd.read_csv("Th_large_training.txt",sep=',',header=None)
        Th_large_training_df = df11.T
        df12 =  pd.read_csv("Th_large_training_test.txt",sep=',',header=None)
        Th_large_training_test_df = df12.T
        df13 =  pd.read_csv("Th_large_testing.txt",sep=',',header=None)
        Th_large_testing_df = df13.T
        df21 =  pd.read_csv("phi_final_training.txt",sep=',',header=None)
        phi_final_training_df = df21.T
        df22 =  pd.read_csv("phi_final_training_test.txt",sep=',',header=None)
        phi_final_training_test_df = df22.T
        df23 =  pd.read_csv("phi_final_testing.txt",sep=',',header=None)
        phi_final_testing_df = df23.T
        
        #phi_target training and testing data
        df31 =  pd.read_csv("phi_target_3_3_training.txt",sep=',',header=None)
        phi_target_3_3_training_df = df31.T
        df32 =  pd.read_csv("phi_target_3_3_training_test.txt",sep=',',header=None)
        phi_target_3_3_training_test_df = df32.T
        df33 =  pd.read_csv("phi_target_3_3_testing.txt",sep=',',header=None)
        phi_target_3_3_testing_df = df33.T
        
        
        #change dataframe to tensor and changing double type to float32
        
        #Th_phi_target_training 
        Th_large_training_df_array_double = np.array(Th_large_training_df)
        Th_large_training_df_array_float = Th_large_training_df_array_double.astype(np.float32)
        
        phi_target_3_3_training_df_array_double = np.array(phi_target_3_3_training_df)
        phi_target_3_3_training_df_array_float = phi_target_3_3_training_df_array_double.astype(np.float32)
        
        Th_phi_target_training_array_float=np.hstack((Th_large_training_df_array_float,phi_target_3_3_training_df_array_float))
        Th_phi_target_training = torch.tensor(Th_phi_target_training_array_float)
        
        #Moving to GPU
        Th_phi_target_training=Th_phi_target_training.cuda(0)
        
        
        
        #Th_phi_target_training_test                                   
        Th_large_training_test_df_array_double = np.array(Th_large_training_test_df)
        Th_large_training_test_df_array_float = Th_large_training_test_df_array_double.astype(np.float32)
                                                     
        phi_target_3_3_training_test_df_array_double = np.array(phi_target_3_3_training_test_df)
        phi_target_3_3_training_test_df_array_float = phi_target_3_3_training_test_df_array_double.astype(np.float32)
        
        Th_phi_target_training_test_array_float=np.hstack((Th_large_training_test_df_array_float,phi_target_3_3_training_test_df_array_float))
        Th_phi_target_training_test = torch.tensor(Th_phi_target_training_test_array_float)
        
        #Moving to GPU
        Th_phi_target_training_test=Th_phi_target_training_test.cuda(0)
        
        
        
        #Th_phi_target_testing                                   
        
        Th_large_testing_df_array_double = np.array(Th_large_testing_df)
        Th_large_testing_df_array_float = Th_large_testing_df_array_double.astype(np.float32)
        
        phi_target_3_3_testing_df_array_double = np.array(phi_target_3_3_testing_df)
        phi_target_3_3_testing_df_array_float = phi_target_3_3_testing_df_array_double.astype(np.float32)
        
        Th_phi_target_testing_array_float=np.hstack((Th_large_testing_df_array_float,phi_target_3_3_testing_df_array_float))
        Th_phi_target_testing = torch.tensor(Th_phi_target_testing_array_float)
        
        #Moving to GPU
        Th_phi_target_testing=Th_phi_target_testing.cuda(0)
        
        
        
        # Th_large_training_test = torch.tensor(Th_large_training_test_df_array_float)
        
        # #Moving to GPU
        # Th_large_training_test=Th_large_training_test.cuda(0)
        
        
        
        # Th_large_testing = torch.tensor(Th_large_testing_df_array_float)
        
        # #Moving to GPU
        # Th_large_testing=Th_large_testing.cuda(0)
        
        
        
        
        phi_final_training_df_array_double = np.array(phi_final_training_df)
        phi_final_training_df_array_float = phi_final_training_df_array_double.astype(np.float32)
        phi_final_training = torch.tensor(phi_final_training_df_array_float)
        
        #Moving to GPU
        phi_final_training=phi_final_training.cuda(0)
        
        
        
        phi_final_training_test_df_array_double = np.array(phi_final_training_test_df)
        phi_final_training_test_df_array_float = phi_final_training_test_df_array_double.astype(np.float32)
        phi_final_training_test = torch.tensor(phi_final_training_test_df_array_float)
        
        #Moving to GPU
        phi_final_training_test=phi_final_training_test.cuda(0)
        
        
        
        phi_final_testing_df_array_double = np.array(phi_final_testing_df)
        phi_final_testing_df_array_float = phi_final_testing_df_array_double.astype(np.float32)
        phi_final_testing = torch.tensor(phi_final_testing_df_array_float)
        
        #Moving to GPU
        phi_final_testing=phi_final_testing.cuda(0)
        
        
        #Training data number and testing data number
        Ntraining=np.size(df11,1)
        batch_size=Ntraining
        input_number=np.size(df11,0)
        
        Ntesting=np.size(df13,1)
        
        output_number=np.size(df21,0)
        
        target=phi_final_training
        
        #Moving to GPU
        target=target.cuda(0)
        
        # print(df11(1))
        # print(df11(0))
        
        training_start = time.time()
        
        
        
        
        # creating a neural network
        class Net(nn.Module):
            def __init__(self,input_dim=INPUT_DIM,num_layers=NUM_LAYERS,hidden_dim=HIDDEN_DIM,output_dim=OUTPUT_DIM):
                super(Net, self).__init__()
                self.num_layers = num_layers
                self.linears = nn.ModuleList([nn.Linear(input_dim,hidden_dim)])
                # if self.num_layers > 4:
                for i in range(1, self.num_layers-2):
                    self.linears.append(nn.Linear(hidden_dim, hidden_dim))
                self.linears.append(nn.Linear(hidden_dim, output_dim))
        
            def forward(self, x):
                for i in range(self.num_layers-2):
                    x = torch.sigmoid(self.linears[i](x))
                x = self.linears[-1](x)
                return x
        
        
        net = Net()
        net=net.cuda(0)
        
        
        # print(net)
        
        # print(np.shape (net.linears[1].weight))
        # print(np.shape (net.linears[0].weight))
        
        # print(np.shape (net.linears[1].bias))
        # print(np.shape (net.linears[0].bias))
        
        # print(net.linears[1].bias)
        # print(net.linears[0].bias)
        # print(net.linears[2].bias)
        
        # print(net.linears[1].weight)
        # print(net.linears[0].weight)
        # print(net.linears[2].weight)
        
        
        
        
        # optimizer    
        optimizer=optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=WEIGHT_DECAY, amsgrad=False)
        
        # create a loss function
        criterion = nn.MSELoss()
        
        
        loss_values = []
        epoch_plot=[]
        
        
        for epoch in range(0,EPOCHS):
            # total_batch_training_loss = 0  
            # for batch_idx in range(0,Ntraining):
            optimizer.zero_grad()
            net_out = net(Th_phi_target_training)
            loss = criterion(net_out, target)
            # loss1=loss.detach().cpu().numpy()
            
            loss.backward()
            
            
            
            
            optimizer.step()
                 #sum up total loss in each batch
            # total_batch_training_loss += loss         
            #average training loss for different epoch
            # average_training_loss=total_batch_training_loss/Ntraining 
        
            # print(loss.item())
            # net_out_11=net_out.var.detach().numpy()
            # net_out_11=net_out.detach().cpu().numpy()
                
            # if epoch % 5 == 0:           
            #     loss_values.append(loss)
            #     epoch_plot.append(epoch) 
            #     print(epoch_plot)
            # ##plotting the loss vs. epochs figure
            #     plt.plot(epoch_plot, loss_values,'r-o')
            #     plt.xlabel('epochs')
            #     plt.ylabel('Training loss') 
            #     plt.title('Training loss vs. epochs')  
            #     plt.show()
            #     plt.savefig("Training error vs. epochs.png")
            #     plt.close()
                
        training_end = time.time()     
        
        # torch.save(net.state_dict(), "net_params%d.pkl"%(1))
        
        # print('程序运行时间:%s毫秒' % ((training_end - training_start)*1000))   
        
        
        print('程序运行时间:%s毫秒' % ((training_end - training_start)*1000))   
        torch.save(net.state_dict(), 'net_params%d.pkl'%(count+1))
        
        
        # print(loss)
        # print(EPOCHS)
        # print(loss.item())
        # print(net_out)
        # run a test loop using testing data
        test_loss = 0
        # for i in range(0,Ntesting):
        
        net_out1 = net(Th_phi_target_testing)
            # sum up batch loss
        test_loss = criterion(net_out1, phi_final_testing)
        
        flops, params = profile(net, inputs=(Th_phi_target_testing, ))
        
        
        E_test=test_loss.item()
        
            # print(average_test_loss.item())
        # print(net_out1)
        # print(np.size(net_out1))
            
            ##plotting the testing error vs. training data number figure
        # plt.plot(Ntraining, test_loss.item(),'r-o')
        # plt.xlabel('Training data number')
        # plt.ylabel('Error') 
        # plt.title('Testing error vs. Training data number')  
        
        # plt.savefig("Testing_error vs. training_data_number.png")
        # plt.show()
        # plt.close()
            
            
        # run a test loop using training data
        training_test_loss = 0
        
        
        net_out2 = net(Th_phi_target_training_test)
            # sum up batch loss
        training_test_loss = criterion(net_out2, phi_final_training_test)
        
               ##plotting the training error vs. training data number figure
        # print(net_out2)  
        # print(np.size(net_out2))
        
        # plt.plot(Ntraining, training_test_loss.item(),'b-s',label="Training error")
        # plt.plot(Ntraining, test_loss.item(),'k-o',label="Testing error")
        # plt.xlabel('Training data number')
        # plt.ylabel('Error') 
        # plt.title('Training&testing error vs. Training data number')  
        # plt.legend()
        
        # plt.savefig("E_training_number.png")
        # plt.show()
        # plt.close() 
        
        
        phinnout_training_df_array_float = net_out2.detach().cpu().numpy()
    
        # phinnout_testing_df_array_double = np.array(phinnout_testing_df)
        phinnout_testing_df_array_float = net_out1.detach().cpu().numpy()
 

    #load phi data
        # df21 =  pd.read_csv("phi_training.txt",sep=',',header=None)
        # phi_training_df = df21.T
        # df22 =  pd.read_csv("phi_training_test.txt",sep=',',header=None)
        # phi_training_test_df = df22.T
        # df23 =  pd.read_csv("phi_testing.txt",sep=',',header=None)
        # phi_testing_df = df23.T
        
        
        
        phi_low_tr=phi_final_training_df_array_float.min()
        phi_high_tr=phi_final_training_df_array_float.max()
    
        phi_low_te=phi_final_testing_df_array_float.min()
        phi_high_te=phi_final_testing_df_array_float.max()
        
        # Ntesting=np.size(df23,1)
        
        #Data for plotting nnoutput phi-training 
        row3=np.size(phinnout_training_df_array_float,0)*np.size(phinnout_training_df_array_float,1)
        phinnout_training_plot= phinnout_training_df_array_float.reshape((1,row3))
      
        
      
        #Data for plotting nnoutput phi-testing
        row4=np.size(phinnout_testing_df_array_float,0)*np.size(phinnout_testing_df_array_float,1)
        phinnout_testing_plot= phinnout_testing_df_array_float.reshape((1,row4))
        
        
        
        #Normalized_training_test data
        hist_tr, x_nn_tr = np.histogram(phinnout_training_plot, bins=20,density=True)
        w_nn_tr=x_nn_tr[1]-x_nn_tr[0]
        z_nn_tr=hist_tr*w_nn_tr
        # total_tr=np.sum(z_nn_tr)
        
        
        #Normalized_testing data
        hist_te, x_nn_te = np.histogram(phinnout_testing_plot, bins=20,density=True)
        w_nn_te=x_nn_te[1]-x_nn_te[0]
        z_nn_te=hist_te*w_nn_te
        # total_te=np.sum(z_nn_te)
        iiil=[]
        iiir=[]
        # metric for defining overfitting 
        num_test=np.size(x_nn_te)
        num_tr=np.size(x_nn_tr)
        
        
        for ii in range(num_tr):
            if x_nn_tr[ii] < phi_low_tr:
                if ii==0:
                    iitrl=ii
                else:
                    iitrl=ii-1
        
                iiil.append(iitrl)
            elif x_nn_tr[ii] > phi_high_tr:
                if ii==0:
                    iitrr=ii
                else:
                    iitrr=ii-1
            
                iiir.append(iitrr)
    
        ntraining=np.sum(z_nn_tr[iiil])+np.sum(z_nn_tr[iiir])
        
        jjjr=[]
        jjjl=[]
        
        for jj in range(num_test):
            if x_nn_te[jj] < phi_low_te:
                if jj==0:
                    jjtel=jj
                else:
                    jjtel=jj-1
        
                jjjl.append(jjtel)
            elif x_nn_te[jj] > phi_high_te:
                if jj==0:
                    jjter=jj
                else:
                    jjter=jj-1
                
                jjjr.append(jjter)
        

    
        ntesting=np.sum(z_nn_te[jjjl])+np.sum(z_nn_te[jjjr])
        overfittingM=ntesting-ntraining
        if overfittingM < 0:
            overfittingM=0

        net.parameters=[]
        
        return overfittingM, E_test, flops
        
        


    ###Genetic algorithm
    outputtotal=[]
    outputtotal_scaled=[]
    nn_parameter_total=[]
    parameter_output_totoal=[]
    parameter_output_totoal_scaled=[]
    sum_output=[]
    sum_output2=[]
    # output=[]
    def aimFunc(self, pop):
        nnparameter = pop.Phen # 得到决策变量矩阵
        # np.savez('data',nnparameter)
        x1 = nnparameter[:, [0]] # 取出第一列，得到所有个体的第一个自变量
        x2 = nnparameter[:, [1]] # 取出第二列，得到所有个体的第二个自变量
        x3 = nnparameter[:, [2]] # 取出第一列，得到所有个体的第一个自变量
        # x4 = nnparameter[:, [3]] # 取出第二列，得到所有个体的第二个自变量
        
    
        population_size=int(np.size(x1[:,0]))
        for ii1 in range(0,population_size):
            
            count=len(self.nn_parameter_total)
            if count == []:
                count=0
            

            overfittingM, E_test,flops = self.nn_training_overfitting(x1[ii1],x2[ii1],x3[ii1],count)
            
            #unsclaed output
            output=overfittingM+E_test+flops
            #scaled output
            output_scaled=overfittingM/0.2+E_test/0.05+flops/300000000
        
            
            outputprofile=[overfittingM,E_test,flops,output]
            outputprofile_scaled=[overfittingM/0.2,E_test/0.05,flops/300000000,output_scaled]
            
            self.outputtotal.append(outputprofile)
            self.outputtotal_scaled.append(outputprofile_scaled)
            
            nn_parameter_profile=[x1[ii1],x2[ii1],x3[ii1]]
            
            
            self.nn_parameter_total.append(nn_parameter_profile)
            
            
            parameter_output=[x1[ii1],x2[ii1],x3[ii1],overfittingM,E_test,flops,output,overfittingM/0.2,E_test/0.05,flops/300000000,output_scaled]
            # parameter_output_scaled=[x1[ii1],x2[ii1],x3[ii1],overfittingM/0.02,E_test/0.3,flops/800000000,output_scaled]
            
            self.parameter_output_totoal.append(parameter_output)
            # self.parameter_output_totoal_scaled.append(parameter_output_scaled)
            
            # print(np.shape(self.parameter_output_totoal))
            
            # print(np.size(self.parameter_output_totoal))
            
            # print(np.size(parameter_output))
            
            # print(np.shape(parameter_output))
            
            # print(ii1)
            # print(parameter_output)
             
            np.savetxt('progress_forwardnn.txt',self.parameter_output_totoal,fmt='%d %d %d %f %f %f %f %f %f %f %f',delimiter=' ',header='Hidden_layer_number Hidden_layer_nodes epochs overfitting error flops output overfitting_scaled error_scaled flops_scaled output_scaled')
            
            
            
            
            self.sum_output.append(output_scaled)
        # print(type(self.output))    
        # print(type(self.sum_output))
        # # print(self.sum_output.shape)
        # print(self.sum_output)
        self.sum_output2=np.array(self.sum_output)
        # print(type(self.sum_output2))
        # print(self.sum_output2)
        # # Nchrom=int(np.size(self.sum_output2))
        # column=int(Nchrom/population_size)
        # print(Nchrom)
        # print(column)
        self.sum_output2=self.sum_output2.reshape(population_size,1)
        # print(self.sum_output2)
        pop.ObjV = self.sum_output2  # 计算目标函数值，赋值给pop种群对象的ObjV属性
        self.sum_output=[]
        
        # return output







