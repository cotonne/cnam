addpath Data_Classif_Ocean
addpath somtoolbox-octave
load L2Flags_LAND.mat;
lin_incr = 2; lignesApp = 1:lin_incr:size(LAND,1);
MasqueTerreApp = LAND(lignesApp,:);
load Labels_Aer_Map.mat;
AerMapApp = Aer_Map(lignesApp,:);
valid_pixels = find(MasqueTerreApp == 0);
AerMapApp = AerMapApp(valid_pixels);
D=[];
load L1Brad_rho_412; x=rho_412(lignesApp,:); x=x(valid_pixels);D=[D,x]; clear x
load L1Brad_rho_443; x=rho_443(lignesApp,:); x=x(valid_pixels);D=[D,x]; clear x
load L1Brad_rho_490; x=rho_490(lignesApp,:); x=x(valid_pixels);D=[D,x]; clear x
load L1Brad_rho_510; x=rho_510(lignesApp,:); x=x(valid_pixels);D=[D,x]; clear x
load L1Brad_rho_555; x=rho_555(lignesApp,:); x=x(valid_pixels);D=[D,x]; clear x
load L1Brad_rho_670; x=rho_670(lignesApp,:); x=x(valid_pixels);D=[D,x]; clear x
load L1Brad_rho_765; x=rho_765(lignesApp,:); x=x(valid_pixels);D=[D,x]; clear x
load L1Brad_rho_865; x=rho_865(lignesApp,:); x=x(valid_pixels);D=[D,x]; clear x
NomLabels = {'nuage', 'marit', 'cont', 'desert'};
labs = cell(length(valid_pixels),1);
labsNumerique=zeros(length(valid_pixels),1);
for i_flag = 1:4
  j_flag = find(AerMapApp == i_flag);
  if ~isempty(j_flag);
    labsNumerique(j_flag)=i_flag;
    for jj = j_flag'
      labs{jj} = NomLabels{i_flag};
    end
  end
end
cnames={'r412nm','r443nm','r490nm','r510nm','r555nm','r670nm','r765nm','r865nm'}';
sD = som_data_struct(D,'name','CouleurOcean','labels',labs,'comp_names',cnames);
n_data = size(sD.data,1); insize = size(sD.data,2);
msize = [10 10]; 
lattice = 'rect'; 
shape= 'sheet';
sM = som_map_struct(insize,'msize',msize, lattice, shape);
sM = som_lininit(sD, sM);
figure
subplot(1,2,1)
epochs = 40; radius_ini = 2.5; radius_fin = 1; Neigh = 'gaussian';
tr_lev= 2;
[sM,sT] = som_batchtrain(sM, sD,'trainlen',epochs,...
'radius_ini',radius_ini,'radius_fin',radius_fin, ...
'neigh',Neigh,'tracking',tr_lev);
legend('Phase 1 : auto-organisation')
subplot(1,2,2)
epochs= 60; radius_ini = 1; radius_fin = 0.5;
[sM,sT] = som_batchtrain(sM, sD,'trainlen',epochs,...
'radius_ini',radius_ini,'radius_fin',radius_fin, ...
'neigh',Neigh,'tracking',tr_lev);
legend('Phase 2 : convergence')
[qe,te]=som_quality(sM,sD);
disp([ 'qe = ' num2str(qe)]); disp([ 'te = ' num2str(te)]);

% Projection de la carte par ACP : avec la commande pcaproj
PCAneurones=pcaproj(sM.codebook,2);
figure
som_grid(sM,'Coord',PCAneurones); axis on
title('Projection par ACP');
% Projection de la carte par Sammon : avec la commande sammon
figure
SammonNeurones=sammon(sM,2);
som_grid(sM,'coord',SammonNeurones);axis on
title('Projection par Sammon');

[Bmus, Qerrors] = som_bmus(sM, sD);
Hits = som_hits(sM, sD);
NbClasses = length(NomLabels);
MatClass = zeros(size(sM.codebook,1),NbClasses);
for iDon = 1: size(sD.data,1)
MatClass(Bmus(iDon),labsNumerique(iDon))=MatClass(Bmus(iDon),labsNumerique(iDon))+1;
end
sM = som_label(sM, 'clear','all');
for iCell = 1: size(sM.codebook,1)
for iClass = 1: NbClasses
if MatClass(iCell,iClass) ~= 0
curname = NomLabels{iClass};
cellLabels = [ curname(1:2) ' ' num2str(MatClass(iCell,iClass)) ];
sM = som_label(sM,'add',iCell, cellLabels);
end
end
end
figure('Position',[80 80 600 600]);
h=som_show(sM,'empty','Cardinalite par classe');
hlbl=som_show_add('label',sM);


% Vote majoritaire sur chaque neurone
sM = som_label(sM, 'clear','all');
Labl = cell(size(sM.codebook,1),1);
for ii = 1:size(sM.codebook,1), Labl{ii} = num2str(ii); end
sM = som_label(sM,'add', 'all', Labl);
sM = som_autolabel(sM,sD,'vote');
figure('Position',[50 50 650 650]);
som_show(sM,'empty','numero du neurone et son label');
som_show_add('label',sM);

Neuronedechaquepixel= som_bmus(sM,sD);
Classifparcarte=sM.labels(Neuronedechaquepixel,2);
correct= find(strcmp(Classifparcarte,sD.labels)==1);
pourcentage=length(correct)*100/length(sD.labels);
nbreclasses=length(NomLabels);
confuse=zeros(nbreclasses,nbreclasses);
for i=1:nbreclasses
for j=1:nbreclasses
confuse(i,j)=length(find(strcmp(sD.labels,NomLabels(i))==1 ...
& strcmp(Classifparcarte,NomLabels(j))==1));
end
end
disp(' ')
disp(['pourcentage de bonne classification : ' num2str(pourcentage) '%'])
disp('matrice de confusion')
disp(num2str(confuse))
figure
PloterMatrice(confuse)
title(['Pourcentage de classification: ' num2str(pourcentage) '%'], 'FontSize', 14);

load L2Flags_LAND.mat; load Labels_Aer_Map.mat;
load L1Brad_rho_412; load L1Brad_rho_443; load L1Brad_rho_490; load L1Brad_rho_510;
load L1Brad_rho_555; load L1Brad_rho_670; load L1Brad_rho_765; load L1Brad_rho_865;
% Base de test
pixels_test = find(LAND == 0 & ...
isnan(rho_412)==0 & isnan(rho_443)==0 & ...
isnan(rho_490)==0 & isnan(rho_510)==0 & ...
isnan(rho_555)==0 & isnan(rho_670)==0 & ...
isnan(rho_765)==0 & isnan(rho_865)==0 ...
);
AerMap_test = Aer_Map(pixels_test);
Dtest=[rho_412(pixels_test),rho_443(pixels_test),rho_490(pixels_test), ...
rho_510(pixels_test),rho_555(pixels_test),rho_670(pixels_test), ...
rho_765(pixels_test),rho_865(pixels_test)];
NomLabels = {'nuage', 'marit', 'cont', 'desert'};
labs = cell(length(pixels_test),1);
for i_flag = 1:4
j_flag = find(AerMapApp == i_flag);
if ~isempty(j_flag);
labsNumerique(j_flag)=i_flag;
for jj = j_flag'
labs{jj} = NomLabels{i_flag};
end
end
end
Neuronedechaquepixel= som_bmus(sM,Dtest);
Classifparcarte=sM.labels(Neuronedechaquepixel,2);
Classifparexpert=transpose(NomLabels(AerMap_test));
correct= find(strcmp(Classifparcarte,Classifparexpert)==1);
pourcentage=length(correct)*100/length(AerMap_test);
nbreclasses=length(NomLabels);
confuse=zeros(nbreclasses,nbreclasses);
for i=1:nbreclasses
for j=1:nbreclasses
confuse(i,j)=length(find(strcmp(Classifparexpert,NomLabels(i))==1 ...
& strcmp(Classifparcarte,NomLabels(j))==1));
end
end
disp(' ')
disp(['pourcentage de bonne classification : ' num2str(pourcentage) '%'])
disp('matrice de confusion')
disp(num2str(confuse))
figure
PloterMatrice(confuse)
title(['Pourcentage de classification: ' num2str(pourcentage) '%'], 'FontSize', 14);

num_neurone=1;
[Bmus, Qerrors] = som_bmus(sM, Dtest);
pixels_captes_par_num_neurone=find(Bmus==num_neurone);
nbre_pixels_captes_par_num_neurone=length(pixels_captes_par_num_neurone);
% on visualise sur le masque terre
carte=LAND;
carte(pixels_test(pixels_captes_par_num_neurone))=0.5;
Palette = [ 1 1 1; 1 0 0; 0 0 0];
figure
imagesc(carte,[0 1])
colormap(Palette)
axis image
colorbar
title([ 'neurone ' num2str(num_neurone) ' : ' ...
num2str(nbre_pixels_captes_par_num_neurone) 'pixels'])

pixelsmalclasses=find(strcmp(Classifparcarte,Classifparexpert)~=1);
% on visualise sur la carte des labels
carte=Aer_Map;
carte(pixels_test(pixelsmalclasses))=5;
MesCouleurs = [1 1 1 ; 0 0 1; 0 1 0; 1 1 0; 1 0 0];
figure
imagesc(carte)
colormap(MesCouleurs)
axis image
title([num2str(length(pixelsmalclasses)) ' pixels mal classs : en rouge'])