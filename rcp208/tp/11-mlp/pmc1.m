global options
addpath netlab
figure
load base1
NbreNC = 10; % Nombre de neurones dans la couche cachée
Net = mlp(size(Xapp,2), NbreNC, size(Yapp,2), 'linear');
options = foptions;
options(1) = 1; options(14) = 100; options(18) = 0.001;
[Net options errlog] = netopt(Net, options, Xapp, Yapp, 'graddesc');
figure
plot(errlog)
Yc=mlpfwd(Net,X);
figure
plot(X,Y,'b-',X,Yc,'r-')
legend(': Y',': Yc');
Etest=mlperr(Net,Xtest,Ytest);