clear all
clc
format long;

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


