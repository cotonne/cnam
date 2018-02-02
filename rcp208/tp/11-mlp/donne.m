n=1000;
X=4*(rand(n,1)-.5);
X=sort(X);
Y=zeros(size(X));
Y=Y+sin(pi*X).*((X>-1) & (X<1));
Yb=Y+.1*randn(size(Y,1),1);
figure
plot(X,Y,'b-',X,Yb,'c.')
legend(': Y',': Yb');
title('data2')
save data2 X Y Yb

