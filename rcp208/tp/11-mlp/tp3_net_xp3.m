global options
addpath netlab

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
Yapp=DappOutput;  %Xapp et Y app : pour l�apprentissage (pour calculer les poids du r�seau)
Xval=DvalInput;
Yval=DvalOutput;  %Xval et Y val : pour la validation (pour d�terminer quand arr�ter l�apprentissage)
Xtest=DtestInput;
Ytest=DtestOutput;  %Xtest et Y test : pour le test (�valuer les performances du r�seau obtenu)

% Exp�rience 1
% Apprentissage avec les 12 neurones
nbre_neur_entree = size(Xapp,2);
nbre_neur_sortie = size(Yapp,2);
nbre_neur_cache = 10;
Net = mlp(nbre_neur_entree, nbre_neur_cache, nbre_neur_sortie, 'linear');

% sauvegarde les poids initiaux
Net_initial=Net;

%
Net_meilleur3=Net;

% A modifier si necessaire
nbre_iterations_initiale=0;
nbre_iterations=200;
nbre_iterations_avant_test=10;

%On garde les traces des erreurs dans un vecteur pour les visualiser
Err_app=[];
err_a= mlperr(Net,Xapp,Yapp);
%Err_app = [Err_app err_a];
Err_val=[];
err_v= mlperr(Net,Xval,Yval);
%Err_val = [Err_val err_v];

%On initialise la meilleure erreur
err_val_meilleure=err_v;

% Quelques instructions matlab (faciles a comprendre)
Nb_Iter = nbre_iterations_initiale:nbre_iterations_avant_test:nbre_iterations;
fig_error=figure;
set(gca,'YScale','log')
Cur_iter = [ ];
%Cur_iter = [ Nb_Iter(1) ];
%plot(Cur_iter,Err_app,'b-',Cur_iter,Err_val,'r-')
hold on

options = foptions;
algorithm = 'scg';
options=foptions;
options(14)=Nb_Iter(1);

for i_iter = 2:length(Nb_Iter)
    [Net, options] = netopt(Net, options, Xapp, Yapp, algorithm);
    err_a= mlperr(Net,Xapp,Yapp);
    Err_app = [Err_app err_a];
    err_v= mlperr(Net,Xval,Yval);
    Err_val = [Err_val err_v];
    
    % Recherche du r�seau qui minimise l'erreur
    if err_v < err_val_meilleure
        err_val_meilleure=err_v;
        Net_meilleur3=Net;
    end
    Cur_iter = [Cur_iter, Nb_Iter(i_iter)];
    figure(fig_error);
    plot(Cur_iter,Err_app,'g-',Cur_iter,Err_val,'b-')
    options(14)=nbre_iterations_avant_test;
end

variables_conservees = [3 4 6 7 8 9];
Xapp = Xapp(:, variables_conservees);
Xval = Xval(:, variables_conservees);
Xtest = Xtest(:, variables_conservees);
Net_meilleur3.w1 = Net_meilleur3.w1(variables_conservees, :);
Net_meilleur3.nin = size(variables_conservees, 2);
Net_meilleur3.nwts = 81;

Net = Net_meilleur3;

Nb_Iter = nbre_iterations_initiale:nbre_iterations_avant_test:1000;
Err_app = [mlperr(Net,Xapp,Yapp)];
Err_val = [mlperr(Net,Xval,Yval)];
Cur_iter = [Nb_Iter(1) + nbre_iterations ];
plot(Cur_iter,Err_app,'g-',Cur_iter,Err_val,'b-');

for i_iter = 2:length(Nb_Iter)
    [Net, options] = netopt(Net, options, Xapp, Yapp, algorithm);
    err_a= mlperr(Net,Xapp,Yapp);
    Err_app = [Err_app err_a];
    err_v= mlperr(Net,Xval,Yval);
    Err_val = [Err_val err_v];
    
    % Recherche du r�seau qui minimise l'erreur
    if err_v < err_val_meilleure
        err_val_meilleure=err_v;
        Net_meilleur3=Net;
        %disp("Better network at Iteration")
        %disp(Nb_Iter(i_iter))
    end
    Cur_iter = [Cur_iter, Nb_Iter(i_iter)+ nbre_iterations];
    figure(fig_error);
    plot(Cur_iter,Err_app,'g-',Cur_iter,Err_val,'b-')
end
hold on

% ARV
variance_data_sunspots = mean((Sortie - mean(Sortie)).^2);
YLearningCalculee = mlpfwd(Net_meilleur3,Xapp);
arv_app= mean((Yapp-YLearningCalculee).^2)/variance_data_sunspots
YValidationCalculee = mlpfwd(Net_meilleur3,Xval);
arv_val=mean((Yval-YValidationCalculee).^2)/variance_data_sunspots
YTestCalculee = mlpfwd(Net_meilleur3, Xtest);
arv_test=mean((Ytest-YTestCalculee).^2)/variance_data_sunspots

% Visualisation de la s�rie qu�on veut apprendre et de celle apprise par le r�seau
figure
plot(dataSunspots)
hold on
SunspotsToPredict=dataSunspots(13:280);
plot(13:280,SunspotsToPredict,'g-');
NetSunspots=mlpfwd(Net_meilleur3,Entree(:, variables_conservees));
plot(13:280,NetSunspots,'r-');




