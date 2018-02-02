global options
addpath netlab

data = load('Sunspots');
dataSunspots=data(:,2);
% Normalisation [-1;1]
dataSunspots = 2 * dataSunspots - 1;

% Data input
Entree(:,1)=dataSunspots(1:268);
Entree(:,2)=dataSunspots(2:269);
Entree(:,3)=dataSunspots(3:270);
Entree(:,4)=dataSunspots(4:271);
Entree(:,5)=dataSunspots(5:272);
Entree(:,6)=dataSunspots(6:273);
Entree(:,7)=dataSunspots(7:274);
Entree(:,8)=dataSunspots(8:275);
Entree(:,9)=dataSunspots(9:276);
Entree(:,10)=dataSunspots(10:277);
Entree(:,11)=dataSunspots(11:278);
Entree(:,12)=dataSunspots(12:279);

% Format Entree : 12 colonnes, 268 lignes

% Data output
Sortie=dataSunspots(13:280);
% Bases : Apprentissage, Validation et Test
DappInput=Entree(1:209,:);
% Learning input
DappOutput=Sortie(1:209);
% Learning output
DvalInput=Entree(210:244,:); % Validation input
DvalOutput=Sortie(210:244);
% Validation output
DtestInput=Entree(245:268,:); % Test input
DtestOutput=Sortie(245:268); % Test output
DtestOutput=Sortie(245:268); % Test output

nbre_neur_entree = size(DappInput,2);
nbre_neur_sortie = size(DappOutput,2);
nbre_neur_cache = 10;
Net = mlp(nbre_neur_entree, nbre_neur_cache, nbre_neur_sortie, 'linear');
options = foptions;
options(1) = 1;
options(14) = 1000;
algorithm = 'scg';
[Net options errlog] = netopt(Net, options, DappInput, DappOutput, algorithm);

figure
plot(errlog)
Yc=mlpfwd(Net,DtestInput);
figure
plot(range(279),dataSunspots,'b-',range(279),Yc,'r-')
legend(': Y',': Yc');
%Etest=mlperr(Net,Xtest,Ytest);