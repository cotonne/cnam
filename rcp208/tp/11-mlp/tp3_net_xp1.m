global options
addpath Netlab

data = load('Sunspots');
dataSunspots=data(:,2);
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
% Data output
Sortie=dataSunspots(13:280);
% Bases : Apprentissage, Validation et Test
DappInput=Entree(1:209,:); % Learning input
DappOutput=Sortie(1:209); % Learning output
DvalInput=Entree(210:244,:); % Validation input
DvalOutput=Sortie(210:244); % Validation output
DtestInput=Entree(245:268,:); % Test input
DtestOutput=Sortie(245:268); % Test output

Xapp=DappInput;
Yapp=DappOutput;  %Xapp et Y app : pour l’apprentissage (pour calculer les poids du réseau)
Xval=DvalInput;
Yval=DvalOutput;  %Xval et Y val : pour la validation (pour déterminer quand arrêter l’apprentissage)
Xtest=DtestInput;
Ytest=DtestOutput;  %Xtest et Y test : pour le test (évaluer les performances du réseau obtenu)

% Expérience 1
% Apprentissage avec les 12 neurones
nbre_neur_entree = size(Xapp,2);
nbre_neur_sortie = size(Yapp,2);
nbre_neur_cache = 3;
Net = mlp(nbre_neur_entree, nbre_neur_cache, nbre_neur_sortie, 'linear');

% sauvegarde les poids initiaux
Net_initial=Net;

%
Net_meilleur1=Net;

% A modifier si nécessaire
nbre_iterations=500;
nbre_iterations_avant_test=5;

%On garde les traces des erreurs dans un vecteur pour les visualiser
Err_app=[];
err_a= mlperr(Net,Xapp,Yapp);
Err_app = [Err_app err_a];
Err_val=[];
err_v= mlperr(Net,Xval,Yval);
Err_val = [Err_val err_v];

%On initialise la meilleure erreur
err_val_meilleure=err_v;

% Quelques instructions matlab (faciles à comprendre)
Nb_Iter = 0:nbre_iterations_avant_test:nbre_iterations;
fig_error=figure;
set(gca,'YScale','log')
Cur_iter = [ Nb_Iter(1) ];
plot(Cur_iter,Err_app,'b-',Cur_iter,Err_val,'r-')
hold on

options = foptions;
algorithm = 'scg';
options=foptions;
options(14)=nbre_iterations_avant_test;

for i_iter = 2:length(Nb_Iter)
    [Net, options] = netopt(Net, options, Xapp, Yapp, algorithm);
    err_a= mlperr(Net,Xapp,Yapp);
    Err_app = [Err_app err_a];
    err_v= mlperr(Net,Xval,Yval);
    Err_val = [Err_val err_v];
    
    % Recherche du réseau qui minimise l'erreur
    if err_v < err_val_meilleure
        err_val_meilleure=err_v;
        Net_meilleur1=Net;
    end
    Cur_iter = [Cur_iter, Nb_Iter(i_iter)];
    figure(fig_error);
    plot(Cur_iter,Err_app,'b-',Cur_iter,Err_val,'r-')
end

% ARV
variance_data_sunspots = mean((Sortie - mean(Sortie)).^2);
YLearningCalculee = mlpfwd(Net_meilleur1,Xapp);
arv_app= mean((Yapp-YLearningCalculee).^2)/variance_data_sunspots
YValidationCalculee = mlpfwd(Net_meilleur1,Xval);
arv_val=mean((Yval-YValidationCalculee).^2)/variance_data_sunspots
YTestCalculee = mlpfwd(Net_meilleur1, Xtest);
arv_test=mean((Ytest-YTestCalculee).^2)/variance_data_sunspots

% Visualisation de la série qu’on veut apprendre et de celle apprise par le réseau
figure
plot(dataSunspots)
hold on
SunspotsToPredict=dataSunspots(13:280);
plot(13:280,SunspotsToPredict,'g-');
pause
NetSunspots=mlpfwd(Net_meilleur1,Entree);
plot(13:280,NetSunspots,'r-');




