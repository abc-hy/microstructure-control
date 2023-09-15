function [Dn, Qn] = matrix_para (N)

h=1/(N-1);
n=linspace(1,1,N);
n1=n(1:end-1); 

A0=-2*diag(n); A0(1,end)=1; A0(end,1)=1;
Ap=diag(n1,1); Am=diag(n1,-1);
A=(A0+Ap+Am)/h/h;
[Qn, Dn]=eig(A);
Dn=diag(Dn);
Dn=Dn*n;

end


