
%% Name: Haiying Yang
%% Date: September, 2023
%% micorstructure comparison
clear all
clc
format short


II=1;               % set this value based on the size of testing database
p_num=512;
Npossibility=120+(II-1)*p_num;



phi_final_net_out_1_testing1=dlmread('phi_final_net_out_1_testing.txt');
phi_final_net_out_1_testing=transpose(phi_final_net_out_1_testing1);
Ntesting=size(phi_final_net_out_1_testing,2);

phi_final_testing_PFM=dlmread('phi_final_testing_PFM.txt');

phi_target_testing=dlmread('phi_final_testing_inverse.txt');





phi_final_net_1_possibility=phi_final_net_out_1_testing(:,Npossibility);
phi_final_net_1_possibility_R=reshape(phi_final_net_1_possibility,36,36);
phi_final_net_1_possibility_R2=flipud(phi_final_net_1_possibility_R);


phi_final_testing_PFM_possibility=phi_final_testing_PFM(:,Npossibility);
phi_final_testing_PFM_possibility_R=reshape(phi_final_testing_PFM_possibility,36,36);
phi_final_testing_PFM_possibility_R2=flipud(phi_final_testing_PFM_possibility_R);

phi_target_testing_possibility=phi_target_testing(:,Npossibility);
phi_target_testing_possibility_R=reshape(phi_target_testing_possibility,36,36);
phi_target_testing_possibility_R2=flipud(phi_target_testing_possibility_R);







       figure(3)
        surf(phi_final_net_1_possibility_R2,'EdgeColor','None','facecolor','interp');
        view(2);
        title('Order parameter of end state by using forward nn');
        xlabel('x');
        ylabel('y');
        axis equal
        set(gca,'TickDir','out');
        axis([1 36 1 36]);
        caxis([-3 3]);
        colorbar
        colormap winter
        set(gca, 'Fontname', 'Times New Roman','FontSize',12);


       figure(4)
        surf(phi_final_testing_PFM_possibility_R2,'EdgeColor','None','facecolor','interp');
        view(2);
        title('Order parameter of end state by using PFM');
        xlabel('x');
        ylabel('y');
        axis equal
        set(gca,'TickDir','out');
        axis([1 36 1 36]);
        caxis([-3 3]);
        colorbar
        colormap winter
        set(gca, 'Fontname', 'Times New Roman','FontSize',12);

       figure(5)
        surf(phi_target_testing_possibility_R2,'EdgeColor','None','facecolor','interp');
        view(2);
        title('Order parameter of target state');
        xlabel('x');
        ylabel('y');
        axis equal
        set(gca,'TickDir','out');
        axis([1 36 1 36]);
        caxis([-3 3]);
        colorbar
        colormap winter
        set(gca, 'Fontname', 'Times New Roman','FontSize',12);
