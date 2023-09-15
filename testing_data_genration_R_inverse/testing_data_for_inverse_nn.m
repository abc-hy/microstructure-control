%Generating the testing data for inverse nn

clear all
clc
format long;

N=36; 
Ntesting=1;
phi1=zeros(Ntesting,1);
phi2=zeros(Ntesting,1);

%Setting the phi_final that you want to achieve
phi1(1)=1.6;
% phi1(2)=0.7;
% phi1(3)=2.63;
% phi1(4)=1.7;
phi2(1)=-2.2;
% phi2(2)=-0.5;
% phi2(3)=-1.07;
% phi2(4)=-2.09;
% 
possibility=zeros(512,9);
p_num=512;      % Number of possibility  


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
phi_target_3_3=possibility(1:p_num,1:9).';     


for II=1:Ntesting
    phi_target_3_3_512(:,(II-1)*p_num+1:II*p_num)=phi_target_3_3;
 for index=1:p_num
     Nt=(II-1)*p_num+index;
     
    if possibility(index,1)==1
        
    phi_final_testing(1:N/3,1:N/3)=phi1(II);
    
    else
    phi_final_testing(1:N/3,1:N/3)=phi2(II);
    
    end
    
    
    if possibility(index,2)==1
    phi_final_testing(1:N/3,N/3+1:2*N/3)=phi1(II);

    else
    phi_final_testing(1:N/3,N/3+1:2*N/3)=phi2(II);

    end
    
    
    if possibility(index,3)==1
    phi_final_testing(1:N/3,2*N/3+1:N)=phi1(II);

    else
    phi_final_testing(1:N/3,2*N/3+1:N)=phi2(II);

    end
    
    
    if possibility(index,4)==1
    phi_final_testing(N/3+1:2*N/3,1:N/3)=phi1(II);

    else
    phi_final_testing(N/3+1:2*N/3,1:N/3)=phi2(II);

    end
    
    
    if possibility(index,5)==1
    phi_final_testing(N/3+1:2*N/3,N/3+1:2*N/3)=phi1(II);

    else
    phi_final_testing(N/3+1:2*N/3,N/3+1:2*N/3)=phi2(II);
 
    end
    
    
    if possibility(index,6)==1
    phi_final_testing(N/3+1:2*N/3,2*N/3+1:N)=phi1(II);

    else
    phi_final_testing(N/3+1:2*N/3,2*N/3+1:N)=phi2(II);

    end
    
    if possibility(index,7)==1
    phi_final_testing(2*N/3+1:N,1:N/3)=phi1(II);

    else
    phi_final_testing(2*N/3+1:N,1:N/3)=phi2(II);

    end
       
    if possibility(index,8)==1
    phi_final_testing(2*N/3+1:N,N/3+1:2*N/3)=phi1(II);

    else
    phi_final_testing(2*N/3+1:N,N/3+1:2*N/3)=phi2(II);

    end
    
    if possibility(index,9)==1
    phi_final_testing(2*N/3+1:N,2*N/3+1:N)=phi1(II);

    else
    phi_final_testing(2*N/3+1:N,2*N/3+1:N)=phi2(II);

    end
    phi_final_testing_1(Nt,:)={Nt,phi_final_testing};
    
    phi_final_testing_2(:,Nt)=phi_final_testing(:);

 end
 
end

% testing data of phi_final

dlmwrite('phi_final_testing_inverse.txt',phi_final_testing_2, 'precision',10);
save('phi_final_testing_inverse.mat','phi_final_testing_2')


%phi_target_3_3_512_testing
phi_target_3_3_testing=phi_target_3_3_512;
dlmwrite('phi_target_3_3_testing.txt',phi_target_3_3_testing,'precision',10);




