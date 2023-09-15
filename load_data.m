clear all
clc
format long;

load 'phi_training.mat'
load 'phi_training_test.mat'
load 'phi_testing.mat'

phi_training=dlmread('phi_training.txt');

phi_plottraining=phi_training;
phi_plottraining1=phi_plottraining(:);

figure(1)
[y6,x6]=hist(phi_plottraining1);      % histogram of h for testing
plot(x6,y6,'-o','lineWidth',1.5,'Markersize',6);
title('phi for training','fontsize',16);
xlabel('value of phi');
ylabel('count');
legend('phi for training');
savefig('phi for training')


T_training=dlmread('T_training.txt');

% Data for plotting T_training
T_trainingplot=T_training(:);

figure(6)
[y1,x1]=hist(T_trainingplot);      % histogram of h for testing
plot(x1,y1,'-o','lineWidth',1.5,'Markersize',6);
title('T with 5 possible values','fontsize',16);
xlabel('value of T');
ylabel('count');
legend('T for training');
savefig('T for training')




h_training=dlmread('h_training.txt');

% Data for plotting h_training
h_trainingplot=h_training(:);

figure(8)
[y3,x3]=hist(h_trainingplot);      % histogram of h for testing
plot(x3,y3,'-o','lineWidth',1.5,'Markersize',6);
title('h with 5 possible values','fontsize',16);
xlabel('value of h');
ylabel('count');
legend('h for training');
savefig('h for training')




