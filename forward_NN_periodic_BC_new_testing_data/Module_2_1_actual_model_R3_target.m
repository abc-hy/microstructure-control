
% Neural network_Haiying Yang
% Using Zirui's script to solve the Allen-Cahn equation 
% T_h_phi_database generation written by Haiying Yang
% 32_input_1296_output
% Revised: Haiying Yang
% Date: August, 2021

clear all
clc
format long;

E=[];
%%%% computational domain specification %%%%
N=36; % number of node per edge of square
ds=1/(N);
L=1; % Length of each edge

%%% Control parameter %%%
gamma=0.059;
T=zeros(N,N);
h=zeros(N,N);
phi_target_index=zeros(N,N);


for i=1:N
    for j=1:N
        x(i,j)=(j-1)*ds;
        y(i,j)=(i-1)*ds;
    end
end


%%%% time integration %%%%
dtime=0.001; % 1e-2
ttime=0.0;
nstep=500;   %5e7;
nprint=100; 

%%%% Parameters for T&h&phi database generation %%%%
load 'TH_training.mat'
load 'TH_training_test.mat'
load 'TH_testing.mat'

Ntraining=size(TH_training,2);           %training data
Ntesting=size(TH_training_testing,2);           %testing data 
Number_T_h=size(TH_training,1);           %Number of T and h

% [nquadrupletT,nquadrupleth]=T_h_generation(Ntraining,Ntesting);
load 'nquadrupletT.mat'
load 'nquadrupleth.mat'

nquadrulet=Ntraining+Ntesting;

%%%% Initialization %%%%
phi0=zeros(N,N);
phi2=phi0;
xx=[0:0.0005:L];
phi_theory=tanh((xx-L/2)/sqrt(2*gamma/4));
phi_t=tanh((x(1,:)-L/2)/sqrt(2*gamma/4));

% [E0,e_local,e_gradient] = energy (phi1,N,ds,M,T,h); 
% E1=E0; E2=E0; 
Ttime=0; 
% Error=sqrt(sum((phi2(1,:)-phi_t).^2));
k=0; 
% [DM, DN, Qm, Qn, M, N] = matrix_para (Nx,Ny);

% initialize the plotting
set(0,'defaultfigurecolor','w')    % set the background as white.
[Dn, Qn] = matrix_para (N);

% lamata=1+dtime*(1-a1)*2*T;
eta=-gamma*dtime;

Qnt=Qn';
% evolve
pic_num1=1;
pic_num2=1;
pic_num3=1;
pic_num4=1;

possibility=zeros(512,9);
index=0;
for s1=1:2
    for s2=1:2
        for s3=1:2
            for s4=1:2
                for s5=1:2
                    for s6=1:2
                        for s7=1:2
                            for s8=1:2
                                for s9=1:2
                                    index=index+1;
                                    possibility(index,:)=[s1 s2 s3 s4 s5 s6 s7 s8 s9];
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end
% phi_target_3_3=zeros(9,512);
           
    
p_num=512;      % Number of possibility  
phi_target_3_3=possibility(1:p_num,1:9).';     
Th_large_training=zeros(Number_T_h,(Ntraining));           %large matrix containing T and h
Th_large_testing=zeros(Number_T_h,Ntesting);           %large matrix containing T and h



for III=1:(nquadrulet/p_num)
    
    phi_target_3_3_512(:,(III-1)*p_num+1:III*p_num)=phi_target_3_3;
    
    
   for index=1:p_num
    
    phi2=phi0;
    
    phi_target_index(1:N/3,1:N/3)=possibility(index,1);
    phi_target_index(1:N/3,N/3+1:2*N/3)=possibility(index,2);
    phi_target_index(1:N/3,2*N/3+1:N)=possibility(index,3);
    phi_target_index(N/3+1:2*N/3,1:N/3)=possibility(index,4);
    phi_target_index(N/3+1:2*N/3,N/3+1:2*N/3)=possibility(index,5);
    phi_target_index(N/3+1:2*N/3,2*N/3+1:N)=possibility(index,6);
    phi_target_index(2*N/3+1:N,1:N/3)=possibility(index,7);
    phi_target_index(2*N/3+1:N,N/3+1:2*N/3)=possibility(index,8);
    phi_target_index(2*N/3+1:N,2*N/3+1:N)=possibility(index,9);
    
    Nt=(III-1)*p_num+index;
    phi_quadruplet_target(Nt,:)={Nt,phi_target_index};
    phi_onecolumn_target(:,Nt)=phi_target_index(:);
    
    
    if Nt<=Ntraining
    Th_large_training(:,Nt)=TH_training(:,Nt);
    else
    Th_large_testing(:,(Nt-Ntraining))=TH_testing(:,(Nt-Ntraining));
    end
%   phi_target_small=
    
    
for n=1:nstep
    
    if n<(nstep/4)
    ii=1;

    if possibility(index,1)==1
        
    h(1:N/3,1:N/3)=nquadrupleth{Nt,2}(ii,1);
    T(1:N/3,1:N/3)=nquadrupletT{Nt,2}(ii,1);
%     phi_target_total(1:N/3,1:N/3)=possibility(index,1);
    else
    h(1:N/3,1:N/3)=nquadrupleth{Nt,2}(ii,2);
    T(1:N/3,1:N/3)=nquadrupletT{Nt,2}(ii,2);
    end
    
    
    if possibility(index,2)==1
    h(1:N/3,N/3+1:2*N/3)=nquadrupleth{Nt,2}(ii,1);
    T(1:N/3,N/3+1:2*N/3)=nquadrupletT{Nt,2}(ii,1);
    else
    h(1:N/3,N/3+1:2*N/3)=nquadrupleth{Nt,2}(ii,2);
    T(1:N/3,N/3+1:2*N/3)=nquadrupletT{Nt,2}(ii,2);
    end
    
    
    if possibility(index,3)==1
    h(1:N/3,2*N/3+1:N)=nquadrupleth{Nt,2}(ii,1);
    T(1:N/3,2*N/3+1:N)=nquadrupletT{Nt,2}(ii,1);
    else
    h(1:N/3,2*N/3+1:N)=nquadrupleth{Nt,2}(ii,2);
    T(1:N/3,2*N/3+1:N)=nquadrupletT{Nt,2}(ii,2);    
    end
    
    
    if possibility(index,4)==1
    h(N/3+1:2*N/3,1:N/3)=nquadrupleth{Nt,2}(ii,1);
    T(N/3+1:2*N/3,1:N/3)=nquadrupletT{Nt,2}(ii,1);
    else
    h(N/3+1:2*N/3,1:N/3)=nquadrupleth{Nt,2}(ii,2);
    T(N/3+1:2*N/3,1:N/3)=nquadrupletT{Nt,2}(ii,2);
    end
    
    
    if possibility(index,5)==1
    h(N/3+1:2*N/3,N/3+1:2*N/3)=nquadrupleth{Nt,2}(ii,1);
    T(N/3+1:2*N/3,N/3+1:2*N/3)=nquadrupletT{Nt,2}(ii,1);
    else
    h(N/3+1:2*N/3,N/3+1:2*N/3)=nquadrupleth{Nt,2}(ii,2);
    T(N/3+1:2*N/3,N/3+1:2*N/3)=nquadrupletT{Nt,2}(ii,2);    
    end
    
    
    if possibility(index,6)==1
    h(N/3+1:2*N/3,2*N/3+1:N)=nquadrupleth{Nt,2}(ii,1);
    T(N/3+1:2*N/3,2*N/3+1:N)=nquadrupletT{Nt,2}(ii,1);
    else
    h(N/3+1:2*N/3,2*N/3+1:N)=nquadrupleth{Nt,2}(ii,2);
    T(N/3+1:2*N/3,2*N/3+1:N)=nquadrupletT{Nt,2}(ii,2);  
    end
    
    if possibility(index,7)==1
    h(2*N/3+1:N,1:N/3)=nquadrupleth{Nt,2}(ii,1);
    T(2*N/3+1:N,1:N/3)=nquadrupletT{Nt,2}(ii,1);
    else
    h(2*N/3+1:N,1:N/3)=nquadrupleth{Nt,2}(ii,2);
    T(2*N/3+1:N,1:N/3)=nquadrupletT{Nt,2}(ii,2); 
    end
       
    if possibility(index,8)==1
    h(2*N/3+1:N,N/3+1:2*N/3)=nquadrupleth{Nt,2}(ii,1);
    T(2*N/3+1:N,N/3+1:2*N/3)=nquadrupletT{Nt,2}(ii,1);
    else
    h(2*N/3+1:N,N/3+1:2*N/3)=nquadrupleth{Nt,2}(ii,2);
    T(2*N/3+1:N,N/3+1:2*N/3)=nquadrupletT{Nt,2}(ii,2); 
    end
    
    if possibility(index,9)==1
    h(2*N/3+1:N,2*N/3+1:N)=nquadrupleth{Nt,2}(ii,1);
    T(2*N/3+1:N,2*N/3+1:N)=nquadrupletT{Nt,2}(ii,1);
    else
    h(2*N/3+1:N,2*N/3+1:N)=nquadrupleth{Nt,2}(ii,2);
    T(2*N/3+1:N,2*N/3+1:N)=nquadrupletT{Nt,2}(ii,2);
    end
    
   T_max=max(max(abs(T)));
     
     
    
    elseif n>=(nstep/4)&&n<(nstep/2)
    ii=2;
   
    if possibility(index,1)==1
        
    h(1:N/3,1:N/3)=nquadrupleth{Nt,2}(ii,1);
    T(1:N/3,1:N/3)=nquadrupletT{Nt,2}(ii,1);
    else
    h(1:N/3,1:N/3)=nquadrupleth{Nt,2}(ii,2);
    T(1:N/3,1:N/3)=nquadrupletT{Nt,2}(ii,2);
    end
    
    
    if possibility(index,2)==1
    h(1:N/3,N/3+1:2*N/3)=nquadrupleth{Nt,2}(ii,1);
    T(1:N/3,N/3+1:2*N/3)=nquadrupletT{Nt,2}(ii,1);
    else
    h(1:N/3,N/3+1:2*N/3)=nquadrupleth{Nt,2}(ii,2);
    T(1:N/3,N/3+1:2*N/3)=nquadrupletT{Nt,2}(ii,2);
    end
    
    
    if possibility(index,3)==1
    h(1:N/3,2*N/3+1:N)=nquadrupleth{Nt,2}(ii,1);
    T(1:N/3,2*N/3+1:N)=nquadrupletT{Nt,2}(ii,1);
    else
    h(1:N/3,2*N/3+1:N)=nquadrupleth{Nt,2}(ii,2);
    T(1:N/3,2*N/3+1:N)=nquadrupletT{Nt,2}(ii,2);    
    end
    
    
    if possibility(index,4)==1
    h(N/3+1:2*N/3,1:N/3)=nquadrupleth{Nt,2}(ii,1);
    T(N/3+1:2*N/3,1:N/3)=nquadrupletT{Nt,2}(ii,1);
    else
    h(N/3+1:2*N/3,1:N/3)=nquadrupleth{Nt,2}(ii,2);
    T(N/3+1:2*N/3,1:N/3)=nquadrupletT{Nt,2}(ii,2);
    end
    
    
    if possibility(index,5)==1
    h(N/3+1:2*N/3,N/3+1:2*N/3)=nquadrupleth{Nt,2}(ii,1);
    T(N/3+1:2*N/3,N/3+1:2*N/3)=nquadrupletT{Nt,2}(ii,1);
    else
    h(N/3+1:2*N/3,N/3+1:2*N/3)=nquadrupleth{Nt,2}(ii,2);
    T(N/3+1:2*N/3,N/3+1:2*N/3)=nquadrupletT{Nt,2}(ii,2);    
    end
    
    
    if possibility(index,6)==1
    h(N/3+1:2*N/3,2*N/3+1:N)=nquadrupleth{Nt,2}(ii,1);
    T(N/3+1:2*N/3,2*N/3+1:N)=nquadrupletT{Nt,2}(ii,1);
    else
    h(N/3+1:2*N/3,2*N/3+1:N)=nquadrupleth{Nt,2}(ii,2);
    T(N/3+1:2*N/3,2*N/3+1:N)=nquadrupletT{Nt,2}(ii,2);  
    end
    
    if possibility(index,7)==1
    h(2*N/3+1:N,1:N/3)=nquadrupleth{Nt,2}(ii,1);
    T(2*N/3+1:N,1:N/3)=nquadrupletT{Nt,2}(ii,1);
    else
    h(2*N/3+1:N,1:N/3)=nquadrupleth{Nt,2}(ii,2);
    T(2*N/3+1:N,1:N/3)=nquadrupletT{Nt,2}(ii,2); 
    end
       
    if possibility(index,8)==1
    h(2*N/3+1:N,N/3+1:2*N/3)=nquadrupleth{Nt,2}(ii,1);
    T(2*N/3+1:N,N/3+1:2*N/3)=nquadrupletT{Nt,2}(ii,1);
    else
    h(2*N/3+1:N,N/3+1:2*N/3)=nquadrupleth{Nt,2}(ii,2);
    T(2*N/3+1:N,N/3+1:2*N/3)=nquadrupletT{Nt,2}(ii,2); 
    end
    
    if possibility(index,9)==1
    h(2*N/3+1:N,2*N/3+1:N)=nquadrupleth{Nt,2}(ii,1);
    T(2*N/3+1:N,2*N/3+1:N)=nquadrupletT{Nt,2}(ii,1);
    else
    h(2*N/3+1:N,2*N/3+1:N)=nquadrupleth{Nt,2}(ii,2);
    T(2*N/3+1:N,2*N/3+1:N)=nquadrupletT{Nt,2}(ii,2);
    end
    
    
    
    T_max=max(max(abs(T)));
    
    elseif n>=(nstep/2)&&n<(3*nstep/4)
    ii=3;
    
    if possibility(index,1)==1
        
    h(1:N/3,1:N/3)=nquadrupleth{Nt,2}(ii,1);
    T(1:N/3,1:N/3)=nquadrupletT{Nt,2}(ii,1);
    else
    h(1:N/3,1:N/3)=nquadrupleth{Nt,2}(ii,2);
    T(1:N/3,1:N/3)=nquadrupletT{Nt,2}(ii,2);
    end
    
    
    if possibility(index,2)==1
    h(1:N/3,N/3+1:2*N/3)=nquadrupleth{Nt,2}(ii,1);
    T(1:N/3,N/3+1:2*N/3)=nquadrupletT{Nt,2}(ii,1);
    else
    h(1:N/3,N/3+1:2*N/3)=nquadrupleth{Nt,2}(ii,2);
    T(1:N/3,N/3+1:2*N/3)=nquadrupletT{Nt,2}(ii,2);
    end
    
    
    if possibility(index,3)==1
    h(1:N/3,2*N/3+1:N)=nquadrupleth{Nt,2}(ii,1);
    T(1:N/3,2*N/3+1:N)=nquadrupletT{Nt,2}(ii,1);
    else
    h(1:N/3,2*N/3+1:N)=nquadrupleth{Nt,2}(ii,2);
    T(1:N/3,2*N/3+1:N)=nquadrupletT{Nt,2}(ii,2);    
    end
    
    
    if possibility(index,4)==1
    h(N/3+1:2*N/3,1:N/3)=nquadrupleth{Nt,2}(ii,1);
    T(N/3+1:2*N/3,1:N/3)=nquadrupletT{Nt,2}(ii,1);
    else
    h(N/3+1:2*N/3,1:N/3)=nquadrupleth{Nt,2}(ii,2);
    T(N/3+1:2*N/3,1:N/3)=nquadrupletT{Nt,2}(ii,2);
    end
    
    
    if possibility(index,5)==1
    h(N/3+1:2*N/3,N/3+1:2*N/3)=nquadrupleth{Nt,2}(ii,1);
    T(N/3+1:2*N/3,N/3+1:2*N/3)=nquadrupletT{Nt,2}(ii,1);
    else
    h(N/3+1:2*N/3,N/3+1:2*N/3)=nquadrupleth{Nt,2}(ii,2);
    T(N/3+1:2*N/3,N/3+1:2*N/3)=nquadrupletT{Nt,2}(ii,2);    
    end
    
    
    if possibility(index,6)==1
    h(N/3+1:2*N/3,2*N/3+1:N)=nquadrupleth{Nt,2}(ii,1);
    T(N/3+1:2*N/3,2*N/3+1:N)=nquadrupletT{Nt,2}(ii,1);
    else
    h(N/3+1:2*N/3,2*N/3+1:N)=nquadrupleth{Nt,2}(ii,2);
    T(N/3+1:2*N/3,2*N/3+1:N)=nquadrupletT{Nt,2}(ii,2);  
    end
    
    if possibility(index,7)==1
    h(2*N/3+1:N,1:N/3)=nquadrupleth{Nt,2}(ii,1);
    T(2*N/3+1:N,1:N/3)=nquadrupletT{Nt,2}(ii,1);
    else
    h(2*N/3+1:N,1:N/3)=nquadrupleth{Nt,2}(ii,2);
    T(2*N/3+1:N,1:N/3)=nquadrupletT{Nt,2}(ii,2); 
    end
       
    if possibility(index,8)==1
    h(2*N/3+1:N,N/3+1:2*N/3)=nquadrupleth{Nt,2}(ii,1);
    T(2*N/3+1:N,N/3+1:2*N/3)=nquadrupletT{Nt,2}(ii,1);
    else
    h(2*N/3+1:N,N/3+1:2*N/3)=nquadrupleth{Nt,2}(ii,2);
    T(2*N/3+1:N,N/3+1:2*N/3)=nquadrupletT{Nt,2}(ii,2); 
    end
    
    if possibility(index,9)==1
    h(2*N/3+1:N,2*N/3+1:N)=nquadrupleth{Nt,2}(ii,1);
    T(2*N/3+1:N,2*N/3+1:N)=nquadrupletT{Nt,2}(ii,1);
    else
    h(2*N/3+1:N,2*N/3+1:N)=nquadrupleth{Nt,2}(ii,2);
    T(2*N/3+1:N,2*N/3+1:N)=nquadrupletT{Nt,2}(ii,2);
    end
   
    
    
    T_max=max(max(abs(T)));
    
    elseif n>=(3*nstep/4)
   
    ii=4;
   
    
    if possibility(index,1)==1
        
    h(1:N/3,1:N/3)=nquadrupleth{Nt,2}(ii,1);
    T(1:N/3,1:N/3)=nquadrupletT{Nt,2}(ii,1);
    else
    h(1:N/3,1:N/3)=nquadrupleth{Nt,2}(ii,2);
    T(1:N/3,1:N/3)=nquadrupletT{Nt,2}(ii,2);
    end
    
    
    if possibility(index,2)==1
    h(1:N/3,N/3+1:2*N/3)=nquadrupleth{Nt,2}(ii,1);
    T(1:N/3,N/3+1:2*N/3)=nquadrupletT{Nt,2}(ii,1);
    else
    h(1:N/3,N/3+1:2*N/3)=nquadrupleth{Nt,2}(ii,2);
    T(1:N/3,N/3+1:2*N/3)=nquadrupletT{Nt,2}(ii,2);
    end
    
    
    if possibility(index,3)==1
    h(1:N/3,2*N/3+1:N)=nquadrupleth{Nt,2}(ii,1);
    T(1:N/3,2*N/3+1:N)=nquadrupletT{Nt,2}(ii,1);
    else
    h(1:N/3,2*N/3+1:N)=nquadrupleth{Nt,2}(ii,2);
    T(1:N/3,2*N/3+1:N)=nquadrupletT{Nt,2}(ii,2);    
    end
    
    
    if possibility(index,4)==1
    h(N/3+1:2*N/3,1:N/3)=nquadrupleth{Nt,2}(ii,1);
    T(N/3+1:2*N/3,1:N/3)=nquadrupletT{Nt,2}(ii,1);
    else
    h(N/3+1:2*N/3,1:N/3)=nquadrupleth{Nt,2}(ii,2);
    T(N/3+1:2*N/3,1:N/3)=nquadrupletT{Nt,2}(ii,2);
    end
    
    
    if possibility(index,5)==1
    h(N/3+1:2*N/3,N/3+1:2*N/3)=nquadrupleth{Nt,2}(ii,1);
    T(N/3+1:2*N/3,N/3+1:2*N/3)=nquadrupletT{Nt,2}(ii,1);
    else
    h(N/3+1:2*N/3,N/3+1:2*N/3)=nquadrupleth{Nt,2}(ii,2);
    T(N/3+1:2*N/3,N/3+1:2*N/3)=nquadrupletT{Nt,2}(ii,2);    
    end
    
    
    if possibility(index,6)==1
    h(N/3+1:2*N/3,2*N/3+1:N)=nquadrupleth{Nt,2}(ii,1);
    T(N/3+1:2*N/3,2*N/3+1:N)=nquadrupletT{Nt,2}(ii,1);
    else
    h(N/3+1:2*N/3,2*N/3+1:N)=nquadrupleth{Nt,2}(ii,2);
    T(N/3+1:2*N/3,2*N/3+1:N)=nquadrupletT{Nt,2}(ii,2);  
    end
    
    if possibility(index,7)==1
    h(2*N/3+1:N,1:N/3)=nquadrupleth{Nt,2}(ii,1);
    T(2*N/3+1:N,1:N/3)=nquadrupletT{Nt,2}(ii,1);
    else
    h(2*N/3+1:N,1:N/3)=nquadrupleth{Nt,2}(ii,2);
    T(2*N/3+1:N,1:N/3)=nquadrupletT{Nt,2}(ii,2); 
    end
       
    if possibility(index,8)==1
    h(2*N/3+1:N,N/3+1:2*N/3)=nquadrupleth{Nt,2}(ii,1);
    T(2*N/3+1:N,N/3+1:2*N/3)=nquadrupletT{Nt,2}(ii,1);
    else
    h(2*N/3+1:N,N/3+1:2*N/3)=nquadrupleth{Nt,2}(ii,2);
    T(2*N/3+1:N,N/3+1:2*N/3)=nquadrupletT{Nt,2}(ii,2); 
    end
    
    if possibility(index,9)==1
    h(2*N/3+1:N,2*N/3+1:N)=nquadrupleth{Nt,2}(ii,1);
    T(2*N/3+1:N,2*N/3+1:N)=nquadrupletT{Nt,2}(ii,1);
    else
    h(2*N/3+1:N,2*N/3+1:N)=nquadrupleth{Nt,2}(ii,2);
    T(2*N/3+1:N,2*N/3+1:N)=nquadrupletT{Nt,2}(ii,2);
    end
   
    
    
    
    T_max=max(max(abs(T)));
end
   
   
   a1=(-6-T_max/2)./T;

    
for i2=1:N
        for jj=1:N
lamata(i2,jj)=1+dtime*(1-a1(i2,jj))*2.*T(i2,jj);           %% corresponding element multiplication

        end
end
D=lamata+eta*(Dn+Dn');

    C=phi2-dtime*2*T.*phi2.*a1-4*dtime*phi2.^3-dtime.*h;          %% corresponding element multiplication
    
    [phi2]= evolution_implicit (C,D,Qn,Qnt);
    phii2(n,:)={ttime,phi2};
        
    ttime=ttime+dtime;
      
%     if  (n/nprint==round(n/nprint))
%         k=k+1;
%         sum_c2=sum(sum(phi2));        % summing all the data in matrix
% %         fprintf('Time = %1.4e;  File# = %i;   sum_c = %.4e\n',ttime,k, sum_c2)
%         Ttime=[Ttime; ttime]; 
%         phi_2=phi2((N)/2,:);
%         x_0=x((N)/2,:);
%         error=sqrt(sum((phi_2-phi_t).^2));
% %         Error=[Error; error];
%         
%         figure(1)
%         image(phi2,'CDataMapping','scaled')
%         title('Image of order parameter(whole model)');
%         axis square
%         axis xy
%         xlabel('x');
%         ylabel('y');
%         set(gca,'TickDir','out');
%         %axis([0 Nx*dx 0 Ny*dy]);
%         %caxis([-1 1]);
%         colorbar;
%         caxis([-4 4])
%         colormap jet;
%         set(gca, 'Fontname', 'Times New Roman','FontSize',12);
%         drawnow;
%         hh=subtitle(sprintf('dtime = %.1e  |  Time = %1.2e sec', dtime, ttime));
%         set(hh, 'fontweight', 'bold','Fontname', 'Times New Roman','FontSize',10);
%         F=getframe(gcf);
%         I=frame2im(F);
%         [I,map]=rgb2ind(I,256);
%         if pic_num1 == 1
%             imwrite(I,map,'image_four_curve_2.gif','gif', 'Loopcount',inf,'DelayTime',0.);
%         else
%             imwrite(I,map,'image_four_curve_2.gif','gif','WriteMode','append','DelayTime',0.);
%         end
%         pic_num1 = pic_num1 + 1;
%         
%  
%         figure(2)
%         surf(phi2,'EdgeColor','None','facecolor','interp');
%         view(2);
%         title('Evolution of order parameter(whole model)');
%         xlabel('x');
%         ylabel('y');
%         axis equal
%         set(gca,'TickDir','out');
%         axis([1 36 1 36]);
%         caxis([-4 4]);
%         colorbar
%         colormap jet
%         set(gca, 'Fontname', 'Times New Roman','FontSize',12);
% 
%         drawnow;
%         hh=subtitle(sprintf('dtime = %.1e  |  Time = %1.2e sec', dtime, ttime));
%         set(hh, 'fontweight', 'bold','Fontname', 'Times New Roman','FontSize',10);
%         F=getframe(gcf);
%         I=frame2im(F);
%         [I,map]=rgb2ind(I,256);
%         if pic_num2 == 1
%             imwrite(I,map,'result_four_curve_2.gif','gif', 'Loopcount',inf,'DelayTime',0.);
%         else
%             imwrite(I,map,'result_four_curve_2.gif','gif','WriteMode','append','DelayTime',0.);
%         end
%         pic_num2 = pic_num2 + 1;
%    
% %         phi2_actual_plot=phi2(123:158,123:158);     
% %         
% %         figure(3)
% %         image(phi2_actual_plot,'CDataMapping','scaled')
% %         title('Image of order parameter(small model)');
% %         axis square
% %         axis xy
% %         xlabel('x');
% %         ylabel('y');
% %         set(gca,'TickDir','out');
% %         %axis([0 Nx*dx 0 Ny*dy]);
% %         %caxis([-1 1]);
% %         colorbar;
% %         caxis([-4 4])
% %         colormap jet;
% %         set(gca, 'Fontname', 'Times New Roman','FontSize',12);
% %         drawnow;
% %         hh=subtitle(sprintf('dtime = %.1e  |  Time = %1.2e sec', dtime, ttime));
% %         set(hh, 'fontweight', 'bold','Fontname', 'Times New Roman','FontSize',10);
% %         F=getframe(gcf);
% %         I=frame2im(F);
% %         [I,map]=rgb2ind(I,256);
% %         if pic_num3 == 1
% %             imwrite(I,map,'image_four_curve_3.gif','gif', 'Loopcount',inf,'DelayTime',0.);
% %         else
% %             imwrite(I,map,'image_four_curve_3.gif','gif','WriteMode','append','DelayTime',0.);
% %         end
% %         pic_num3 = pic_num3 + 1;
% %         
% %  
% %         figure(4)
% %         surf(phi2_actual_plot,'EdgeColor','None','facecolor','interp');
% %         view(2);
% %         title('Evolution of order parameter(small model)');
% %         xlabel('x');
% %         ylabel('y');
% %         axis equal
% %         set(gca,'TickDir','out');
% %         axis([1 36 1 36]);
% %         caxis([-4 4]);
% %         colorbar
% %         colormap jet
% %         set(gca, 'Fontname', 'Times New Roman','FontSize',12);
% % 
% %         drawnow;
% %         hh=subtitle(sprintf('dtime = %.1e  |  Time = %1.2e sec', dtime, ttime));
% %         set(hh, 'fontweight', 'bold','Fontname', 'Times New Roman','FontSize',10);
% %         F=getframe(gcf);
% %         I=frame2im(F);
% %         [I,map]=rgb2ind(I,256);
% %         if pic_num4 == 1
% %             imwrite(I,map,'result_four_curve_3.gif','gif', 'Loopcount',inf,'DelayTime',0.);
% %         else
% %             imwrite(I,map,'result_four_curve_3.gif','gif','WriteMode','append','DelayTime',0.);
% %         end
% %         pic_num4 = pic_num4 + 1;
%         
%         
%         
%         
%         
%         
%         
%     end
    
end
    

% phi2_actual=phi2(123:158,123:158);



        
%         figure(5)
%         surf(phi2,'EdgeColor','None','facecolor','interp');
%         view(2);
%         title('Order parameter of end state');
%         xlabel('x');
%         ylabel('y');
%         axis equal
%         set(gca,'TickDir','out');
%         axis([1 36 1 36]);
%         caxis([-4 4]);
%         colorbar
%         colormap jet
%         set(gca, 'Fontname', 'Times New Roman','FontSize',12);
%        
%         figure(6)
%         surf(phi2,'EdgeColor','None','facecolor','interp');
%         view(2);
%         title('Order parameter of end state(whole model)');
%         xlabel('x');
%         ylabel('y');
%         axis equal
%         set(gca,'TickDir','out');
%         axis([1 280 1 280]);
%         caxis([-4 4]);
%         colorbar
%         colormap jet
%         set(gca, 'Fontname', 'Times New Roman','FontSize',12);
%         
        
        
%actual model
phi_quadruplet(Nt,:)={Nt,phi2};
phi_onecolumn(:,Nt)=phi2(:);


% phi_quadruplet(II,:)={II,phi2};
% phi_onecolumn(:,II)=phi2(:);

end
end
%%% Training data and testing data storage   
% for III1=1:Ntraining
% phi_training{III1}=phi_quadruplet{III1,2};      % training data of phi 
% end
% 
% for III2=1:Ntesting
% phi_testing{III2}=phi_quadruplet{(Ntraining+III2),2};      % testing data of T and h for each quadruplet
% end

% save (['final.mat'],'Ttime','Error');
% save('phi_final.mat','phi_quadruplet')
% save('phi_training.mat','phi_training')
% save('phi_testing.mat','phi_testing')

%Th for training
dlmwrite('Th_large_training.txt',Th_large_training,'precision',10);

%Th for testing
dlmwrite('Th_large_testing.txt',Th_large_testing,'precision',10);

%Th training for testing
Th_large_training_test=Th_large_training(:,(Ntraining-Ntesting)+1:Ntraining);
dlmwrite('Th_large_training_test.txt',Th_large_training_test,'precision',10);


%phi_target_3_3_512_training
phi_target_3_3_training=phi_target_3_3_512(:,1:Ntraining);
dlmwrite('phi_target_3_3_training.txt',phi_target_3_3_training,'precision',10);


%phi_target_3_3_512_training_for_test
phi_target_3_3_training_test=phi_target_3_3_512(:,(Ntraining-Ntesting)+1:Ntraining);
dlmwrite('phi_target_3_3_training_test.txt',phi_target_3_3_training_test,'precision',10);

%phi_target_3_3_512_testing
phi_target_3_3_testing=phi_target_3_3_512(:,(Ntraining+1):(Ntraining+Ntesting));
dlmwrite('phi_target_3_3_testing.txt',phi_target_3_3_testing,'precision',10);



%phi_target_3*3 for corresponding T and h
% dlmwrite('phi_target_3_3_512.txt',phi_target_3_3_512,'precision',10);

% training data of phi_final
phi_final_training=phi_onecolumn(:,1:Ntraining);
dlmwrite('phi_final_training.txt',phi_onecolumn(:,1:Ntraining), 'precision',10);
save('phi_final_training.mat','phi_final_training')

% training data of phi_target
phi_target_training=phi_onecolumn_target(:,1:Ntraining);
dlmwrite('phi_target_training.txt',phi_onecolumn_target(:,1:Ntraining), 'precision',10);
save('phi_target_training.mat','phi_target_training')


% training data of phi_final used for testing
phi_final_training_test=phi_onecolumn(:,(Ntraining-Ntesting)+1:Ntraining);
dlmwrite('phi_final_training_test.txt',phi_onecolumn(:,(Ntraining-Ntesting)+1:Ntraining), 'precision',10);
save('phi_final_training_test.mat','phi_final_training_test')


% training data of phi_target used for testing
phi_target_training_test=phi_onecolumn_target(:,(Ntraining-Ntesting)+1:Ntraining);
dlmwrite('phi_target_training_test.txt',phi_onecolumn_target(:,(Ntraining-Ntesting)+1:Ntraining), 'precision',10);
save('phi_target_training_test.mat','phi_target_training_test')


% testing data of phi_final
phi_final_testing=phi_onecolumn(:,(Ntraining+1):(Ntraining+Ntesting));
dlmwrite('phi_final_testing.txt',phi_onecolumn(:,(Ntraining+1):(Ntraining+Ntesting)), 'precision',10);
save('phi_final_testing.mat','phi_final_testing')

% testing data of phi_target
phi_target_testing=phi_onecolumn_target(:,(Ntraining+1):(Ntraining+Ntesting));
dlmwrite('phi_target_testing.txt',phi_onecolumn_target(:,(Ntraining+1):(Ntraining+Ntesting)), 'precision',10);
save('phi_target_testing.mat','phi_target_testing')




%% Plot

% Data for plotting phi_training
phi_plottraining=phi_final_training;
phi_plottraining1=phi_plottraining(:);

% Data for plotting phi_testing
phi_plottesting=phi_final_testing;
phi_plottesting1=phi_plottesting(:);

figure(1)
[y6,x6]=hist(phi_plottraining1);      % histogram of h for testing
plot(x6,y6,'-o','lineWidth',1.5,'Markersize',6);
title('phi for training','fontsize',16);
xlabel('value of phi');
ylabel('count');
legend('phi for training');
savefig('phi for training')


figure(2)
[y5,x5]=hist(phi_plottesting1);      % histogram of h for testing
plot(x5,y5,'-o','lineWidth',1.5,'Markersize',6);
title('phi for testing','fontsize',16);
xlabel('value of phi');
ylabel('count');
legend('phi for testing');
savefig('phi for testing')


