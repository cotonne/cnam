function [Xa,Ya]=GetSamples(N,X,Y)
fprintf('\n Cliquez %d points dans la figure \n',N*2)
[xval, yval]=ginput(2*N);
for i=1:2*N
[val(i) x(i)]=min(abs(xval(i)-X));
end
x(2*N+1)=length(X);
Xa=[];Ya=[];
k=1;
for i=1:N
Xa=[Xa;X((x(k)+1):x(k+1))];
Ya=[Ya;Y((x(k)+1):x(k+1))];
k=k+2;
end
drawnow