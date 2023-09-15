%% Name: Haiying Yang
%% Date: August, 2021

clear all
clc
format short
% phi_final_testing_PFM=dlmread('phi_final_testing_PFM.txt');
Th_out_phi_target_testing1=dlmread('Th_out_phi_target_testing.txt');
Th_out_phi_target_testing=transpose(Th_out_phi_target_testing1);
Ntesting=size(Th_out_phi_target_testing,2);

Th_out_testing=Th_out_phi_target_testing(1:16,:);
phi_target_3_3=Th_out_phi_target_testing(17:25,:);

for II=1:Ntesting
    
Thte=reshape(Th_out_testing(:,II),[],2);
Tte=Thte(1:4,1:2);
hte=Thte(5:8,1:2);
nquadrupletT(II,:)={II,Tte};
nquadrupleth(II,:)={II,hte};

end




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
nprint=25; 

%%%% Parameters for T&h&phi database generation %%%%
% load 'TH_training.mat'
% load 'TH_training_test.mat'
% load 'TH_testing.mat'

% Ntraining=size(TH_training,2);           %training data
% % Ntesting=size(TH_training_testing,2);           %testing data 
% Number_T_h=size(TH_training,1);           %Number of T and h

% [nquadrupletT,nquadrupleth]=T_h_generation(Ntraining,Ntesting);
% load 'nquadrupletT.mat'
% load 'nquadrupleth.mat'

nquadrulet=Ntesting;

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


% phi_target_3_3=zeros(9,512);
           
    
% p_num=1;      % Number of possibility  
% N_possibility=158;
% phi_target_3_3_1=possibility(:,1:9).';
% phi_target_3_3=possibility(N_possibility,1:9).';     
% Th_large_training=zeros(Number_T_h,(Ntraining));           %large matrix containing T and h
% Th_large_testing=zeros(Number_T_h,Ntesting);           %large matrix containing T and h
% 


for III=1:Ntesting
phi2=phi0;
for n=1:nstep
    
    if n<(nstep/4)
    ii=1;

    if phi_target_3_3(1,III)==1
        
    h(1:N/3,1:N/3)=nquadrupleth{III,2}(ii,1);
    T(1:N/3,1:N/3)=nquadrupletT{III,2}(ii,1);
%     phi_target_total(1:N/3,1:N/3)=possibility(index,1);
    else
    h(1:N/3,1:N/3)=nquadrupleth{III,2}(ii,2);
    T(1:N/3,1:N/3)=nquadrupletT{III,2}(ii,2);
    end
    
    
    if phi_target_3_3(2,III)==1
    h(1:N/3,N/3+1:2*N/3)=nquadrupleth{III,2}(ii,1);
    T(1:N/3,N/3+1:2*N/3)=nquadrupletT{III,2}(ii,1);
    else
    h(1:N/3,N/3+1:2*N/3)=nquadrupleth{III,2}(ii,2);
    T(1:N/3,N/3+1:2*N/3)=nquadrupletT{III,2}(ii,2);
    end
    
    
    if phi_target_3_3(3,III)==1
    h(1:N/3,2*N/3+1:N)=nquadrupleth{III,2}(ii,1);
    T(1:N/3,2*N/3+1:N)=nquadrupletT{III,2}(ii,1);
    else
    h(1:N/3,2*N/3+1:N)=nquadrupleth{III,2}(ii,2);
    T(1:N/3,2*N/3+1:N)=nquadrupletT{III,2}(ii,2);    
    end
    
    
    if phi_target_3_3(4,III)==1
    h(N/3+1:2*N/3,1:N/3)=nquadrupleth{III,2}(ii,1);
    T(N/3+1:2*N/3,1:N/3)=nquadrupletT{III,2}(ii,1);
    else
    h(N/3+1:2*N/3,1:N/3)=nquadrupleth{III,2}(ii,2);
    T(N/3+1:2*N/3,1:N/3)=nquadrupletT{III,2}(ii,2);
    end
    
    
    if phi_target_3_3(5,III)==1
    h(N/3+1:2*N/3,N/3+1:2*N/3)=nquadrupleth{III,2}(ii,1);
    T(N/3+1:2*N/3,N/3+1:2*N/3)=nquadrupletT{III,2}(ii,1);
    else
    h(N/3+1:2*N/3,N/3+1:2*N/3)=nquadrupleth{III,2}(ii,2);
    T(N/3+1:2*N/3,N/3+1:2*N/3)=nquadrupletT{III,2}(ii,2);    
    end
    
    
    if phi_target_3_3(6,III)==1
    h(N/3+1:2*N/3,2*N/3+1:N)=nquadrupleth{III,2}(ii,1);
    T(N/3+1:2*N/3,2*N/3+1:N)=nquadrupletT{III,2}(ii,1);
    else
    h(N/3+1:2*N/3,2*N/3+1:N)=nquadrupleth{III,2}(ii,2);
    T(N/3+1:2*N/3,2*N/3+1:N)=nquadrupletT{III,2}(ii,2);  
    end
    
    if phi_target_3_3(7,III)==1
    h(2*N/3+1:N,1:N/3)=nquadrupleth{III,2}(ii,1);
    T(2*N/3+1:N,1:N/3)=nquadrupletT{III,2}(ii,1);
    else
    h(2*N/3+1:N,1:N/3)=nquadrupleth{III,2}(ii,2);
    T(2*N/3+1:N,1:N/3)=nquadrupletT{III,2}(ii,2); 
    end
       
    if phi_target_3_3(8,III)==1
    h(2*N/3+1:N,N/3+1:2*N/3)=nquadrupleth{III,2}(ii,1);
    T(2*N/3+1:N,N/3+1:2*N/3)=nquadrupletT{III,2}(ii,1);
    else
    h(2*N/3+1:N,N/3+1:2*N/3)=nquadrupleth{III,2}(ii,2);
    T(2*N/3+1:N,N/3+1:2*N/3)=nquadrupletT{III,2}(ii,2); 
    end
    
    if phi_target_3_3(9,III)==1
    h(2*N/3+1:N,2*N/3+1:N)=nquadrupleth{III,2}(ii,1);
    T(2*N/3+1:N,2*N/3+1:N)=nquadrupletT{III,2}(ii,1);
    else
    h(2*N/3+1:N,2*N/3+1:N)=nquadrupleth{III,2}(ii,2);
    T(2*N/3+1:N,2*N/3+1:N)=nquadrupletT{III,2}(ii,2);
    end
    
   T_max=max(max(abs(T)));
     
     
    
    elseif n>=(nstep/4)&&n<(nstep/2)
    ii=2;
   
    if phi_target_3_3(1,III)==1
        
    h(1:N/3,1:N/3)=nquadrupleth{III,2}(ii,1);
    T(1:N/3,1:N/3)=nquadrupletT{III,2}(ii,1);
    else
    h(1:N/3,1:N/3)=nquadrupleth{III,2}(ii,2);
    T(1:N/3,1:N/3)=nquadrupletT{III,2}(ii,2);
    end
    
    
    if phi_target_3_3(2,III)==1
    h(1:N/3,N/3+1:2*N/3)=nquadrupleth{III,2}(ii,1);
    T(1:N/3,N/3+1:2*N/3)=nquadrupletT{III,2}(ii,1);
    else
    h(1:N/3,N/3+1:2*N/3)=nquadrupleth{III,2}(ii,2);
    T(1:N/3,N/3+1:2*N/3)=nquadrupletT{III,2}(ii,2);
    end
    
    
    if phi_target_3_3(3,III)==1
    h(1:N/3,2*N/3+1:N)=nquadrupleth{III,2}(ii,1);
    T(1:N/3,2*N/3+1:N)=nquadrupletT{III,2}(ii,1);
    else
    h(1:N/3,2*N/3+1:N)=nquadrupleth{III,2}(ii,2);
    T(1:N/3,2*N/3+1:N)=nquadrupletT{III,2}(ii,2);    
    end
    
    
    if phi_target_3_3(4,III)==1
    h(N/3+1:2*N/3,1:N/3)=nquadrupleth{III,2}(ii,1);
    T(N/3+1:2*N/3,1:N/3)=nquadrupletT{III,2}(ii,1);
    else
    h(N/3+1:2*N/3,1:N/3)=nquadrupleth{III,2}(ii,2);
    T(N/3+1:2*N/3,1:N/3)=nquadrupletT{III,2}(ii,2);
    end
    
    
    if phi_target_3_3(5,III)==1
    h(N/3+1:2*N/3,N/3+1:2*N/3)=nquadrupleth{III,2}(ii,1);
    T(N/3+1:2*N/3,N/3+1:2*N/3)=nquadrupletT{III,2}(ii,1);
    else
    h(N/3+1:2*N/3,N/3+1:2*N/3)=nquadrupleth{III,2}(ii,2);
    T(N/3+1:2*N/3,N/3+1:2*N/3)=nquadrupletT{III,2}(ii,2);    
    end
    
    
    if phi_target_3_3(6,III)==1
    h(N/3+1:2*N/3,2*N/3+1:N)=nquadrupleth{III,2}(ii,1);
    T(N/3+1:2*N/3,2*N/3+1:N)=nquadrupletT{III,2}(ii,1);
    else
    h(N/3+1:2*N/3,2*N/3+1:N)=nquadrupleth{III,2}(ii,2);
    T(N/3+1:2*N/3,2*N/3+1:N)=nquadrupletT{III,2}(ii,2);  
    end
    
    if phi_target_3_3(7,III)==1
    h(2*N/3+1:N,1:N/3)=nquadrupleth{III,2}(ii,1);
    T(2*N/3+1:N,1:N/3)=nquadrupletT{III,2}(ii,1);
    else
    h(2*N/3+1:N,1:N/3)=nquadrupleth{III,2}(ii,2);
    T(2*N/3+1:N,1:N/3)=nquadrupletT{III,2}(ii,2); 
    end
       
    if phi_target_3_3(8,III)==1
    h(2*N/3+1:N,N/3+1:2*N/3)=nquadrupleth{III,2}(ii,1);
    T(2*N/3+1:N,N/3+1:2*N/3)=nquadrupletT{III,2}(ii,1);
    else
    h(2*N/3+1:N,N/3+1:2*N/3)=nquadrupleth{III,2}(ii,2);
    T(2*N/3+1:N,N/3+1:2*N/3)=nquadrupletT{III,2}(ii,2); 
    end
    
    if phi_target_3_3(9,III)==1
    h(2*N/3+1:N,2*N/3+1:N)=nquadrupleth{III,2}(ii,1);
    T(2*N/3+1:N,2*N/3+1:N)=nquadrupletT{III,2}(ii,1);
    else
    h(2*N/3+1:N,2*N/3+1:N)=nquadrupleth{III,2}(ii,2);
    T(2*N/3+1:N,2*N/3+1:N)=nquadrupletT{III,2}(ii,2);
    end
    
    
    
    T_max=max(max(abs(T)));
    
    elseif n>=(nstep/2)&&n<(3*nstep/4)
    ii=3;
    
    if phi_target_3_3(1,III)==1
        
    h(1:N/3,1:N/3)=nquadrupleth{III,2}(ii,1);
    T(1:N/3,1:N/3)=nquadrupletT{III,2}(ii,1);
    else
    h(1:N/3,1:N/3)=nquadrupleth{III,2}(ii,2);
    T(1:N/3,1:N/3)=nquadrupletT{III,2}(ii,2);
    end
    
    
    if phi_target_3_3(2,III)==1
    h(1:N/3,N/3+1:2*N/3)=nquadrupleth{III,2}(ii,1);
    T(1:N/3,N/3+1:2*N/3)=nquadrupletT{III,2}(ii,1);
    else
    h(1:N/3,N/3+1:2*N/3)=nquadrupleth{III,2}(ii,2);
    T(1:N/3,N/3+1:2*N/3)=nquadrupletT{III,2}(ii,2);
    end
    
    
    if phi_target_3_3(3,III)==1
    h(1:N/3,2*N/3+1:N)=nquadrupleth{III,2}(ii,1);
    T(1:N/3,2*N/3+1:N)=nquadrupletT{III,2}(ii,1);
    else
    h(1:N/3,2*N/3+1:N)=nquadrupleth{III,2}(ii,2);
    T(1:N/3,2*N/3+1:N)=nquadrupletT{III,2}(ii,2);    
    end
    
    
    if phi_target_3_3(4,III)==1
    h(N/3+1:2*N/3,1:N/3)=nquadrupleth{III,2}(ii,1);
    T(N/3+1:2*N/3,1:N/3)=nquadrupletT{III,2}(ii,1);
    else
    h(N/3+1:2*N/3,1:N/3)=nquadrupleth{III,2}(ii,2);
    T(N/3+1:2*N/3,1:N/3)=nquadrupletT{III,2}(ii,2);
    end
    
    
    if phi_target_3_3(5,III)==1
    h(N/3+1:2*N/3,N/3+1:2*N/3)=nquadrupleth{III,2}(ii,1);
    T(N/3+1:2*N/3,N/3+1:2*N/3)=nquadrupletT{III,2}(ii,1);
    else
    h(N/3+1:2*N/3,N/3+1:2*N/3)=nquadrupleth{III,2}(ii,2);
    T(N/3+1:2*N/3,N/3+1:2*N/3)=nquadrupletT{III,2}(ii,2);    
    end
    
    
    if phi_target_3_3(6,III)==1
    h(N/3+1:2*N/3,2*N/3+1:N)=nquadrupleth{III,2}(ii,1);
    T(N/3+1:2*N/3,2*N/3+1:N)=nquadrupletT{III,2}(ii,1);
    else
    h(N/3+1:2*N/3,2*N/3+1:N)=nquadrupleth{III,2}(ii,2);
    T(N/3+1:2*N/3,2*N/3+1:N)=nquadrupletT{III,2}(ii,2);  
    end
    
    if phi_target_3_3(7,III)==1
    h(2*N/3+1:N,1:N/3)=nquadrupleth{III,2}(ii,1);
    T(2*N/3+1:N,1:N/3)=nquadrupletT{III,2}(ii,1);
    else
    h(2*N/3+1:N,1:N/3)=nquadrupleth{III,2}(ii,2);
    T(2*N/3+1:N,1:N/3)=nquadrupletT{III,2}(ii,2); 
    end
       
    if phi_target_3_3(8,III)==1
    h(2*N/3+1:N,N/3+1:2*N/3)=nquadrupleth{III,2}(ii,1);
    T(2*N/3+1:N,N/3+1:2*N/3)=nquadrupletT{III,2}(ii,1);
    else
    h(2*N/3+1:N,N/3+1:2*N/3)=nquadrupleth{III,2}(ii,2);
    T(2*N/3+1:N,N/3+1:2*N/3)=nquadrupletT{III,2}(ii,2); 
    end
    
    if phi_target_3_3(9,III)==1
    h(2*N/3+1:N,2*N/3+1:N)=nquadrupleth{III,2}(ii,1);
    T(2*N/3+1:N,2*N/3+1:N)=nquadrupletT{III,2}(ii,1);
    else
    h(2*N/3+1:N,2*N/3+1:N)=nquadrupleth{III,2}(ii,2);
    T(2*N/3+1:N,2*N/3+1:N)=nquadrupletT{III,2}(ii,2);
    end
   
    
    
    T_max=max(max(abs(T)));
    
    elseif n>=(3*nstep/4)
   
    ii=4;
   
    
    if phi_target_3_3(1,III)==1
        
    h(1:N/3,1:N/3)=nquadrupleth{III,2}(ii,1);
    T(1:N/3,1:N/3)=nquadrupletT{III,2}(ii,1);
    else
    h(1:N/3,1:N/3)=nquadrupleth{III,2}(ii,2);
    T(1:N/3,1:N/3)=nquadrupletT{III,2}(ii,2);
    end
    
    
    if phi_target_3_3(2,III)==1
    h(1:N/3,N/3+1:2*N/3)=nquadrupleth{III,2}(ii,1);
    T(1:N/3,N/3+1:2*N/3)=nquadrupletT{III,2}(ii,1);
    else
    h(1:N/3,N/3+1:2*N/3)=nquadrupleth{III,2}(ii,2);
    T(1:N/3,N/3+1:2*N/3)=nquadrupletT{III,2}(ii,2);
    end
    
    
    if phi_target_3_3(3,III)==1
    h(1:N/3,2*N/3+1:N)=nquadrupleth{III,2}(ii,1);
    T(1:N/3,2*N/3+1:N)=nquadrupletT{III,2}(ii,1);
    else
    h(1:N/3,2*N/3+1:N)=nquadrupleth{III,2}(ii,2);
    T(1:N/3,2*N/3+1:N)=nquadrupletT{III,2}(ii,2);    
    end
    
    
    if phi_target_3_3(4,III)==1
    h(N/3+1:2*N/3,1:N/3)=nquadrupleth{III,2}(ii,1);
    T(N/3+1:2*N/3,1:N/3)=nquadrupletT{III,2}(ii,1);
    else
    h(N/3+1:2*N/3,1:N/3)=nquadrupleth{III,2}(ii,2);
    T(N/3+1:2*N/3,1:N/3)=nquadrupletT{III,2}(ii,2);
    end
    
    
    if phi_target_3_3(5,III)==1
    h(N/3+1:2*N/3,N/3+1:2*N/3)=nquadrupleth{III,2}(ii,1);
    T(N/3+1:2*N/3,N/3+1:2*N/3)=nquadrupletT{III,2}(ii,1);
    else
    h(N/3+1:2*N/3,N/3+1:2*N/3)=nquadrupleth{III,2}(ii,2);
    T(N/3+1:2*N/3,N/3+1:2*N/3)=nquadrupletT{III,2}(ii,2);    
    end
    
    
    if phi_target_3_3(6,III)==1
    h(N/3+1:2*N/3,2*N/3+1:N)=nquadrupleth{III,2}(ii,1);
    T(N/3+1:2*N/3,2*N/3+1:N)=nquadrupletT{III,2}(ii,1);
    else
    h(N/3+1:2*N/3,2*N/3+1:N)=nquadrupleth{III,2}(ii,2);
    T(N/3+1:2*N/3,2*N/3+1:N)=nquadrupletT{III,2}(ii,2);  
    end
    
    if phi_target_3_3(7,III)==1
    h(2*N/3+1:N,1:N/3)=nquadrupleth{III,2}(ii,1);
    T(2*N/3+1:N,1:N/3)=nquadrupletT{III,2}(ii,1);
    else
    h(2*N/3+1:N,1:N/3)=nquadrupleth{III,2}(ii,2);
    T(2*N/3+1:N,1:N/3)=nquadrupletT{III,2}(ii,2); 
    end
       
    if phi_target_3_3(8,III)==1
    h(2*N/3+1:N,N/3+1:2*N/3)=nquadrupleth{III,2}(ii,1);
    T(2*N/3+1:N,N/3+1:2*N/3)=nquadrupletT{III,2}(ii,1);
    else
    h(2*N/3+1:N,N/3+1:2*N/3)=nquadrupleth{III,2}(ii,2);
    T(2*N/3+1:N,N/3+1:2*N/3)=nquadrupletT{III,2}(ii,2); 
    end
    
    if phi_target_3_3(9,III)==1
    h(2*N/3+1:N,2*N/3+1:N)=nquadrupleth{III,2}(ii,1);
    T(2*N/3+1:N,2*N/3+1:N)=nquadrupletT{III,2}(ii,1);
    else
    h(2*N/3+1:N,2*N/3+1:N)=nquadrupleth{III,2}(ii,2);
    T(2*N/3+1:N,2*N/3+1:N)=nquadrupletT{III,2}(ii,2);
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
    phi22=phi2.';    
   
    phi222=flipud(phi2);
    
    ttime=ttime+dtime;

    
end
    


        
%actual model
phi_quadruplet(III,:)={III,phi2};
phi_onecolumn(:,III)=phi2(:);


% phi_quadruplet(II,:)={II,phi2};
% phi_onecolumn(:,II)=phi2(:);

end

% testing data of phi_final_PFM
phi_final_testing_PFM=phi_onecolumn(:,1:Ntesting);
% phi_final_testing_PFM=transpose(phi_final_testing_PFM1);
dlmwrite('phi_final_testing_PFM.txt',phi_final_testing_PFM, 'precision',10);
save('phi_final_testing_PFM.mat','phi_final_testing_PFM')











