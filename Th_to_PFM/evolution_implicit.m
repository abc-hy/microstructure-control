function [phi] = evolution_implicit (C,D,Qn,Qnt)



C=Qnt*C*Qn;
Y=C./D;
phi=Qn*Y*Qnt;
end