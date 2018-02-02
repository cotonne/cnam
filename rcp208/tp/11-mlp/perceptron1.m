global options
addpath netlab

load data1
Xapp=X(1:2:end);
Yapp=Yb(1:2:end);
Xtest=X(2:2:end);
Ytest=Yb(2:2:end);
% RESEAU
Net = glm(size(Xapp,2),size(Yapp,2),'linear');
options    = foptions;
options(1) = 1;
options(14) = 100;
options(18) = 0.001;
% CONSTRUCTION DU RESEAU
% Net => réseau utilisé
% graddesc => DESCENTE DE GRADIENT
[Net options errlog] = netopt(Net, options, Xapp, Yapp, 'graddesc');
figure
plot(errlog)
Yc=glmfwd(Net,X);
figure
plot(X,Y,'b-',X,Yc,'r-')
legend(': Y',': Yc');
Etest=glmerr(Net,Xtest,Ytest)

%>> Net
%Net =
%
%  scalar structure containing the fields:
%
%    type = glm
%    nin =  1 ==> neurone en entrée
%    nout =  1 ==> neurone en sortie
%    nwts =  2 
%    outfn = linear
%    w1 =  2.0069
%    b1 =  3.0011