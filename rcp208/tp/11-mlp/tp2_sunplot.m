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

Xapp=DappInput;
Yapp=DappOutput; % Xapp et Y app : pour l’apprentissage (pour calculer les poids du réseau)
Xval=DvalInput;
Yval=DvalOutput; % Xval et Y val : pour la validation (pour déterminer quand arrêter l’apprentissage)
Xtest=DtestInput;
Ytest=DtestOutput; % Xtest et Y test : pour le test (évaluer les performances du réseau obtenu)

nbre_neur_entree = size(Xapp,2);
nbre_neur_sortie = size(Yapp,2);
nbre_neur_cache = 7;

Net = mlp(nbre_neur_entree, nbre_neur_cache, nbre_neur_sortie, 'linear');
options = foptions;


steps = [5 10 15 20 25 50 75 100 125 150 175 200 300 400 500];
steps_delta = [5];
for j = 1:length(steps) - 1
  steps_delta = [steps_delta (steps(j+1) - steps(j))];
end

err_apprentissage = [];
err_validation = [];
err_test = [];

arv_err_apprentissage = [];
arv_err_validation = [];
arv_err_test = [];

options(1) = 0;
algorithm = 'scg';

% pour activer le log d'erreur, mettre 1
%option(1) = 1
%option(2) = 0
%option(3) = 0
%option(14) = i
%algorithm = 'graddesc';

for i = steps_delta
  options(14) = i;
  % Apprentissage
  [Net, options, errlog] = netopt(Net, options, Xapp, Yapp, algorithm);
  variance_data_sunspots = mean((Sortie - mean(Sortie)).^2);
  
  ErreurApprentissage= mlperr(Net,Xapp,Yapp); % Erreur sur l’ensemble d’apprentissage
  err_apprentissage = [err_apprentissage; ErreurApprentissage];
  ErreurValidation= mlperr(Net,Xval,Yval); % Erreur sur l’ensemble de validation
  err_validation = [err_validation; ErreurValidation];
  ErreurTest= mlperr(Net,Xtest,Ytest); % Erreur sur l’ensemble de test
  err_test = [err_test; ErreurTest];
  
  % MLPFWD  Forward propagation through 2-layer network.
  YappCalculee = mlpfwd(Net,Xapp);
  arv_apprentissage= mean((Yapp-YappCalculee).^2)/variance_data_sunspots;
  arv_err_apprentissage = [arv_err_apprentissage; arv_apprentissage];
  
  YValCalculee = mlpfwd(Net,Xval);
  arv_validation=mean((Yval-YValCalculee).^2)/variance_data_sunspots;
  arv_err_validation = [arv_err_validation; arv_validation];
  
  YTestCalculee = mlpfwd(Net, Xtest);
  arv_test=mean((Ytest-YTestCalculee).^2)/variance_data_sunspots;
  arv_err_test = [arv_err_test; arv_test];
end

% Trouve le pas qui produit l'erreur la plus petite
[~, ii] = min(err_validation);
step_min_validation = steps(ii);
[~, ii] = min(err_test);
step_min_test = steps(ii);

all_values = [err_validation;err_apprentissage;err_test];
max_all = max(all_values);

[~, ii] = min(arv_err_validation);
step_min_arv_validation = steps(ii);
[~, ii] = min(arv_err_test);
step_min_arv_test = steps(ii);

all_arv_values = [arv_err_apprentissage;arv_err_validation;arv_err_test];
max_all_arv = max(all_arv_values);

disp([steps' arv_err_apprentissage arv_err_validation arv_err_test]);

figure
plot(steps,err_apprentissage,'b-', ...
  steps,err_validation,'r-', ...
  steps, err_test, 'g-', ...
  [step_min_validation step_min_validation], [0 max_all], 'r-', ...
  [step_min_test step_min_test], [0 max_all], 'g-');
legend(': err apprentissage',': err validation', ': err test', 'Min erreur');
title('ERREUR QUADRATIQUE');
figure
plot(steps,arv_err_apprentissage,'b-', ...
  steps,arv_err_validation,'r-', ...
  steps, arv_err_test, 'g-', ...
  [step_min_validation step_min_arv_validation], [0 max_all_arv], 'r-', ...
  [step_min_test step_min_test], [0 max_all_arv], 'g-')
legend(': arv apprentissage',': arv validation', ': arv test');
title('ERREUR ARV')