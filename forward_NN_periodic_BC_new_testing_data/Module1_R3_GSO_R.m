%% Name: Haiying Yang
%% Date: July, 2021

clear all
clc
format short
Ththeta11=[];
Th22=[];
theta00=[];
ntrajectory=2;
nstep=3;

%% range of T and h
Tmax=1;
Tmin=-3/2*((-20).^2).^(1/3);
hmax=20;
hmin=-20;
%% L theta randomly generated sampling data
 n=60000;  
%% Number of quadrulet
Ntraining=2048;           %training data----must be mutiple of 512
Ntesting=512;           %testing data ----must be mutiple of 512

nquadrulet=Ntraining+Ntesting;

for II=1:nquadrulet

T0=rand(ntrajectory,1)*(-12)+1;
h0=hmax*rands(ntrajectory,1);
T0h0=[T0,h0];
TH=[];

for I=1:ntrajectory
Ththeta11=[];
Th22=[];
Thshort=[];
Thh22=[];
T3h3short33=[];
Thh33=[];
Th33=[];
theta00=[];
% continuours uniform distribution of theta in the range of [0,360]
% rtheta=normrnd(0,20,[n,1]);
rtheta=unifrnd(0,360,n,1);
theta(I,:)={I,(rtheta./180).*pi};   

% Poisson distribution of L
rL=random('poisson',18,n,1);
r1(I,:)={I,rL};

%% First step
ii=1;
for ii1= 1:n
Tnext1=T0(I)+r1{I,2}(ii1,1).*sin(theta{I,2}(ii1,1));
hnext1=h0(I)+r1{I,2}(ii1,1).*cos(theta{I,2}(ii1,1));

if Tnext1>=Tmin && Tnext1<=Tmax && hnext1>=hmin && hnext1<=hmax  % To check whether each sampling points is in the range
    Ththeta1=[Tnext1,hnext1];
    Ththeta11=[Ththeta11;Ththeta1];
break
end
end

Theachh(ii,:)=Ththeta11(1,:);

%% Loop--from ii=2

for ii=2:nstep

% n: number of sampling of L and theta generated from Gaussian distribution
% Gaussian distribution of theta
rtheta=normrnd(0,50,[n,1]);
theta(I,:)={I,(rtheta./180).*pi};   

% Poisson distribution of L
rL=random('poisson',5,n,1);
r1(I,:)={I,rL};

Thh22=[];
for i3=1:n
Th22=[];
Tnextt=[];
hnextt=[];
L2x=r1{I,2}(i3,1);
h1x=Theachh(ii-1,2);
h0x=h0(I);
T1x=Theachh(ii-1,1);
T0x=T0(I);
thetax=theta{I,2}(i3,1);

T2x_sol=(T0x^2*T1x - 2*T0x*T1x^2 + T1x*h0x^2 + T1x*h1x^2 + T1x^3 + L2x*T0x*h0x - L2x*T0x*h1x - L2x*T1x*h0x + L2x*T1x*h1x - 2*T1x*h0x*h1x - L2x*h0x*sin(thetax)*((T0x - T1x)^2 + (h0x - h1x)^2)^(1/2) + L2x*h1x*sin(thetax)*((T0x - T1x)^2 + (h0x - h1x)^2)^(1/2))/(T0x^2 - 2*T0x*T1x + T1x^2 + h0x^2 - 2*h0x*h1x + h1x^2) + ((T0x - T1x)*(h1x - h0x + (T0x^2 - 2*h0x*h1x - 2*T0x*T1x + T1x^2 - sin(thetax)^2*((T0x - T1x)^2 + (h0x - h1x)^2) + h0x^2 + h1x^2)^(1/2))*(L2x*T0x - L2x*T1x + L2x*sin(thetax)*((T0x - T1x)^2 + (h0x - h1x)^2)^(1/2)))/((T0x - T1x + sin(thetax)*((T0x - T1x)^2 + (h0x - h1x)^2)^(1/2))*(T0x^2 - 2*T0x*T1x + T1x^2 + h0x^2 - 2*h0x*h1x + h1x^2));
h2x_sol=(L2x*h0x^2 + L2x*h1x^2 + T0x^2*h1x + T1x^2*h1x - 2*h0x*h1x^2 + h0x^2*h1x + h1x^3 - 2*T0x*T1x*h1x - 2*L2x*h0x*h1x + L2x*T0x*sin(thetax)*((T0x - T1x)^2 + (h0x - h1x)^2)^(1/2) - L2x*T1x*sin(thetax)*((T0x - T1x)^2 + (h0x - h1x)^2)^(1/2))/(T0x^2 - 2*T0x*T1x + T1x^2 + h0x^2 - 2*h0x*h1x + h1x^2) + ((h0x - h1x)*(h1x - h0x + (T0x^2 - 2*h0x*h1x - 2*T0x*T1x + T1x^2 - sin(thetax)^2*((T0x - T1x)^2 + (h0x - h1x)^2) + h0x^2 + h1x^2)^(1/2))*(L2x*T0x - L2x*T1x + L2x*sin(thetax)*((T0x - T1x)^2 + (h0x - h1x)^2)^(1/2)))/((T0x - T1x + sin(thetax)*((T0x - T1x)^2 + (h0x - h1x)^2)^(1/2))*(T0x^2 - 2*T0x*T1x + T1x^2 + h0x^2 - 2*h0x*h1x + h1x^2));

Tnextt=T2x_sol;
hnextt=h2x_sol;

if Tnextt>=Tmin && Tnextt<= Tmax && hnextt>=hmin && hnextt<=hmax   
    Th2=[Tnextt,hnextt];
    Th22=[Th22;Th2];
    Thh22=[Thh22;Th22];
    break  
end

end

Theachh(ii,:)=Thh22(1,:);

end
%% Trajectory
Th=[];
TrajectoryT(:,I)=[T0(I);Theachh(1:nstep,1)];
Trajectoryh(:,I)=[h0(I);Theachh(1:nstep,2)];
TrajectoryTh(:,(2*I-1):2*I)=[T0h0(I,:);Theachh(1:nstep,1:2)];
Th=[T0h0(I,:);Theachh(1:nstep,1:2)];
TH(:,I)=Th(:);               % one trajectory's T and h
% TH1=reshape(TH,[nstep+1,2]);

end

nquadrupletT(II,:)={II,TrajectoryT(:,1:ntrajectory)};

nquadrupleth(II,:)={II,Trajectoryh(:,1:ntrajectory)};

nquadrupletTh(II,:)={II,TH(:,1:ntrajectory)};

nquadrupletTh2(II,:)={II,TrajectoryTh(:,1:2*ntrajectory)};

THtj(:,II)=TH(:);             % reshaping every trajectory's T and h in the IIth quadruplet 

T(:,II)=TrajectoryT(:);       % reshaping every trajectory's T in the IIth quadruplet 

h(:,II)=Trajectoryh(:);       % reshaping every trajectory's h in the IIth quadruplet 


end

save('nquadrupletT.mat','nquadrupletT')
save('nquadrupleth.mat','nquadrupleth')
% save('nquadrupletTh.mat','nquadrupletTh')
% save('nquadrupletTh2.mat','nquadrupletTh2')
% save('TH_testing.mat','TH_testing')
% save('TH_training.mat','TH_training')

% training data of T and h
TH_training=THtj(:,1:Ntraining);
dlmwrite('TH_training.txt',THtj(:,1:Ntraining), 'precision',10);
save('TH_training.mat','TH_training')



% training data of T and h--test
TH_training_testing=THtj(:,(Ntraining-Ntesting+1):Ntraining);
dlmwrite('TH_training_test.txt',THtj(:,(Ntraining-Ntesting+1):Ntraining), 'precision',10);
save('TH_training_test.mat','TH_training_testing')

% testing data of T and h
TH_testing=THtj(:,(Ntraining+1):(Ntraining+Ntesting));
dlmwrite('TH_testing.txt',THtj(:,(Ntraining+1):(Ntraining+Ntesting)), 'precision',10);
save('TH_testing.mat','TH_testing')

% Whole data of T and h

save('THtj.mat','THtj');
dlmwrite('THtj.txt',THtj, 'precision',10);




% training data of T
dlmwrite('T_training.txt',T(:,1:Ntraining), 'precision',10);

% testing data of T 
dlmwrite('T_testing.txt',T(:,(Ntraining+1):(Ntraining+Ntesting)), 'precision',10);

% training data of h
dlmwrite('h_training.txt',h(:,1:Ntraining), 'precision',10);

% testing data of h 
dlmwrite('h_testing.txt',h(:,(Ntraining+1):(Ntraining+Ntesting)), 'precision',10);


T_training=dlmread('T_training.txt');

% Data for plotting T_training
T_trainingplot=T_training(:);

figure(6)
[y1,x1]=hist(T_trainingplot);      % histogram of h for testing
plot(x1,y1,'-o','lineWidth',1.5,'Markersize',6);
title('T for training','fontsize',16);
xlabel('value of T');
ylabel('count');
legend('T for training');
savefig('T for training')



T_testing=dlmread('T_testing.txt');

%Data for plotting T_esting
T_testingplot=T_testing(:);

figure(7)
[y2,x2]=hist(T_testingplot);      % histogram of h for testing
plot(x2,y2,'-o','lineWidth',1.5,'Markersize',6);
title('T for testing','fontsize',16);
xlabel('value of T');
ylabel('count');
legend('T for testing');


h_training=dlmread('h_training.txt');

% Data for plotting h_training
h_trainingplot=h_training(:);

figure(8)
[y3,x3]=hist(h_trainingplot);      % histogram of h for testing
plot(x3,y3,'-o','lineWidth',1.5,'Markersize',6);
title('h for training','fontsize',16);
xlabel('value of h');
ylabel('count');
legend('h for training');
savefig('h for training')


h_testing=dlmread('h_testing.txt');

%Data for plotting h_testing
h_testingplot=h_testing(:);

figure(11)
[y4,x4]=hist(h_testingplot);      % histogram of h for testing
plot(x4,y4,'-o','lineWidth',1.5,'Markersize',6);
title('h for testing','fontsize',16);
xlabel('value of h');
ylabel('count');
legend('h for testing');

%% Boundary of 1 minimum and 2 minima

% x5=-40:1:40;
% y5=-3/2*(x5.^2).^(1/3);
% figure(1)
% plot(Trajectoryh(:,1:ntrajectory),TrajectoryT(:,1:ntrajectory),x5,y5,'b-p','Markersize',6,'Linewidth',1);
% rectangle('Position',[-20 -3/2*((-20).^2).^(1/3) 40 3/2*((-20).^2).^(1/3)+1],'EdgeColor','r','LineWidth',2,'LineStyle','-');
% 
% line([0 0],[-30 52],'linestyle','--','Color','k','LineWidth',1);
% ylim([-20, 10]);
% xlabel('h','Fontname', 'Times New Roman','FontSize',12,'FontWeight','bold')
% ylabel('T','Fontname', 'Times New Roman','FontSize',12,'FontWeight','bold')
% legend('Trajectory 1','Trajectory 2','Trajectory 3','Trajectory 4','Boundary of 1 minimum & 2 minima','Fontname', 'Times New Roman','FontSize',8);
% 
% title('Four T and h trajectories','Fontname', 'Times New Roman','FontSize',14)
% 
