addpath somtoolbox-octave

 % CHARGER RCP208Data4Classes.ZIP

% CREATION DE LA STRUCTURE
sData = som_data_struct(D,'name','Donnees2','labels',labs,'comp names',cnames);

%CREATION DE LA CARTE
msize = [6 6];
insize = size(sData.data,2);
lattice = 'rect'; % lattice = 'rect' ou 'hexa'
shape = 'sheet'; % shape = 'sheet', 'cyl', ou 'toroid'
sMap = som_map_struct(insize,'msize',msize, lattice, shape);
sMap = som_lininit(sData, sMap); %sMap = som_randinit(sData, sMap)

% CHARGEMENT DES DONNEES
load Data2.mat;
i1=find(classes==1);
i2=find(classes==2);
i3=find(classes==3);
i4=find(classes==4);
i5=find(classes==5);

figure;
plot(D(i1,1),D(i1,2),'r.');
hold on;
plot(D(i2,1),D(i2,2),'g.');
plot(D(i3,1),D(i3,2),'b.');
plot(D(i4,1),D(i4,2),'k.');
plot(D(i5,1),D(i5,2),'m.');


% Entraı̂nement de la carte : Phase 1 (Auto organisation)
figure;
epochs = 500;
radius_ini = 5;
radius_fin = 0.05;
Neigh = 'gaussian'; % Neigh = 'gaussian', 'cutgauss', 'bubble' ou 'ep'
tr_lev = 3;
[sMap,sT] = som_batchtrain(sMap, sData,'trainlen',epochs, ...
    'radius_ini',radius_ini,'radius_fin',radius_fin, 'neigh',Neigh,'tracking',tr_lev);

% Affichage de la grille
figure;
plot(D(i1, 1), D(i1, 2), 'r.');
hold on;
plot(D(i2, 1), D(i2, 2), 'g.');
plot(D(i3, 1), D(i3, 2), 'b.');
plot(D(i4, 1), D(i4, 2), 'k.');
plot(D(i5, 1), D(i5, 2), 'm.');
som_grid(sMap,'Coord',sMap.codebook);


[Bmus, Qerrors] = som_bmus(sMap, sData);
neur=4; % neurone choisi
exemples_captes = find(Bmus == neur); % données captées par le neurone choisi
plot(D(exemples_captes,1),D(exemples_captes,2),'r+'); % données captées par neur en rouge
plot(sMap.codebook(neur,1),sMap.codebook(neur,2),'go'); % neurone neur en vert
title(['exemples captés par le neurone ' num2str(neur)]); % titre de la figure


Labl = cell(size(sMap.codebook,1),1); % Création d'une matrice de 36 cellules
for ii = 1:size(sMap.codebook,1), Labl{ii} = num2str(ii); end % donner un nom à chaque neurone
sMap = som_label(sMap,'clear','all'); % plus de labels dans sMap.labels
sMap = som_label(sMap,'add', 'all', Labl); % On assigne les labels de Labl à sMap
sMap = som_autolabel(sMap,sData,'vote'); % On a donné un label à chaque neurone en utilisant le vote majoritaire

figure
som_show(sMap,'empty','numéro de chaque neurone et son étiquette');
som_show_add('label',sMap,'subplot',1);

%figure
%for ii = 1:size(sMap.codebook,1)
%    [Bmus, Qerrors] = som_bmus(sMap, sData);
%    neur=ii; % neurone choisi
%    exemples_captes = find(Bmus == neur); % données captées par le neurone choisi
%    plot(D(exemples_captes,1),D(exemples_captes,2),'g.') % données captées par neur en rouge
%    title(['exemples captés par le neurone ' num2str(neur)]) % titre de la figure
%end % donner un nom à chaque neurone

indice_agregation = 'ward'; %'single','average','complete','centroid','ward'
liens = 'neighbors'; %'neighbors','any'
nbClasses = 5;
LesClasses = maCAH(sMap,nbClasses,'linkage',indice_agregation,'connect',liens);
%LesClasses = som_cllinkage(sMap,'linkage',indice_agregation,'connect',liens);
%LesClasses=cluster(sC.tree,nbClasses);

figure
MesCouleurs={'ro', 'go', 'bo', 'ko', 'mo'};
som_grid(sMap,'Coord',sMap.codebook,'MarkerColor',[0.75 0.75 0.75],'LineColor',[0 0 0])
hold on
for j=1:nbClasses
plot(sMap.codebook(LesClasses{j},1),sMap.codebook(LesClasses{j},2),MesCouleurs{j},...
'LineWidth',3,'MarkerSize',6)
end
title(['Indice : ' indice_agregation 'Liens : ' liens],'FontSize', 18)


