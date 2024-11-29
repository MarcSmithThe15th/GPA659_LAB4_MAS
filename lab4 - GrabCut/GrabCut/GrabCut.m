clear;
close('all');
clc;
Addpath_items;

%% Liste des images pour tester GrabCut
im_number = 2; % Choisissez un nombre entre 1 et 50
image_set = {'38801G','banana1','banana2','banana3','book','bool','bush','ceramic','cross','doll',...
    'elefant','flower','fullmoon','grave','llama','memorial','music','person1','person2', 'person3',...
    'person4', 'person5', 'person6', 'person7','person8','scissors','sheep','stone1','stone2','teddy',...
    'tennis','21077','24077','37073','65019','69020','86016','106024','124080','153077',...
    '153093','181079','189080','208001','209070','227092','271008','304074','326038','376043'};

%% Param�tres
file_image='Images'; % Dossier contenant les images
file_rectangle='Rectangle'; % Dossier contenant les rectangles (segmentation initiale approximative)
file_GT='GT'; % Dossier contenant le ground truth.

lambda=5; % Le lambda pour le crit�re de r�gularisation
nbIterations = 5; % Nombre maximal d'it�rations

%% Chargement de l'image, initialisation et affichage
image_name=image_set{im_number};
img_file = dir(fullfile(file_image, [image_name '.*']));
groundTruth=imread(fullfile(file_GT, [image_name '.bmp']));
image=imread(fullfile(file_image, img_file.name));
Rectangle=imread(fullfile(file_rectangle, [image_name '.bmp']));
masque = (Rectangle==128); % Masque initial
[M,N,~] = size(masque);

%% Pr�-calcul des poids de voisinage (n-links)
optimizationOptions.NEIGHBORHOOD_TYPE = 8;
optimizationOptions.LAMBDA_POTTS = lambda;
optimizationOptions.neighborhoodBeta = computeNeighborhoodBeta(image, optimizationOptions);
[neighborhoodWeights,~,~] = getNeighborhoodWeights_radius(image, optimizationOptions);

%% Initialisation de la librairie de coupe de graphe
% Cr�ation de l'objet de coupe de graphe
BKhandle = BK_Create(numel(masque));
BK_SetNeighbors(BKhandle, neighborhoodWeights);

%% Premi�re s�rie d'it�rations (sans contraintes)
for iter = 1:nbIterations
    fprintf('It�ration %d\n', iter);
    
    % Sauvegarde du masque pr�c�dent pour le test de convergence
    masque_precedent = masque;
    
    %% �tape a: Mise � jour des probabilit�s par pixel
    [objProbabilites, bkgProbabilitees] = calculerProbabilitesParPixel(image, masque);
    probabilitesParPixel = [objProbabilites(:), bkgProbabilitees(:)]';
    
    % (Optionnel) Affichage des probabilit�s
    figure(1)
    imshow(objProbabilites,[]), colormap('jet'), colorbar;
    title(sprintf('-log(probabilit�) que le pixel appartienne � l''avant-plan - It�ration %d', iter))
    figure(2)
    imshow(bkgProbabilitees,[]), colormap('jet'), colorbar;
    title(sprintf('-log(probabilit�) que le pixel appartienne � l''arri�re-plan - It�ration %d', iter))
    
    %% �tape b: Application de la coupe de graphe
    % R�initialisation des capacit�s des t-links avec les nouvelles probabilit�s
    BK_SetUnary(BKhandle, probabilitesParPixel);
    
    % Ex�cution de l'optimisation
    BK_Minimize(BKhandle);
    
    % R�cup�ration des labels
    L = BK_GetLabeling(BKhandle);
    L = reshape(L, M, N);
    
    % Mise � jour du masque
    masque = (L == 1); % 1 pour l'avant-plan, 0 pour l'arri�re-plan
    
    % (Optionnel) Calcul de l'�nergie
    E = computeEnergy(neighborhoodWeights, double(L==1), objProbabilites, bkgProbabilitees);
    fprintf('�nergie = %.2f\n', E);
    
    % Test de convergence
    changements = sum(masque(:) ~= masque_precedent(:));
    fprintf('Nombre de pixels chang�s : %d\n', changements);
    if changements == 0
        fprintf('Convergence atteinte � l''it�ration %d\n', iter);
        break;
    end
    
    % (Optionnel) Affichage de la segmentation actuelle
    figure(3); imagesc(image); axis image; axis off; hold on;
    [c,h] = contour(L, 'LineWidth',3,'Color', 'r');
    title(sprintf('Segmentation � l''it�ration %d - �nergie = %.2f', iter, E))
    hold off;
    drawnow;
end

%% Suppression de l'objet de coupe de graphe (avant de le recr�er pour la prochaine s�rie)
BK_Delete(BKhandle);
clear BKhandle;

%% *** Ajout des contraintes ***

% Initialisation des tableaux de contraintes
foreground_constraints = zeros(M, N); % Contraintes avant-plan
background_constraints = zeros(M, N); % Contraintes arri�re-plan

% Demande des contraintes � l'utilisateur
% Contraintes avant-plan
response_fg = input('Souhaitez-vous imposer une contrainte de type avant-plan? (oui/non): ', 's');

if strcmp(response_fg, 'oui')
    figure; imshow(image);
    title('S�lectionnez une r�gion pour imposer une contrainte avant-plan');
    rect = getrect; % [xmin ymin largeur hauteur]
    x_min = max(1, floor(rect(1)));
    y_min = max(1, floor(rect(2)));
    x_max = min(N, ceil(rect(1) + rect(3)));
    y_max = min(M, ceil(rect(2) + rect(4)));
    foreground_constraints(y_min:y_max, x_min:x_max) = 1;
    close;
end

% Contraintes arri�re-plan
response_bg = input('Souhaitez-vous imposer une contrainte de type arri�re-plan? (oui/non): ', 's');

if strcmp(response_bg, 'oui')
    figure; imshow(image);
    title('S�lectionnez une r�gion pour imposer une contrainte arri�re-plan');
    rect = getrect; % [xmin ymin largeur hauteur]
    x_min = max(1, floor(rect(1)));
    y_min = max(1, floor(rect(2)));
    x_max = min(N, ceil(rect(1) + rect(3)));
    y_max = min(M, ceil(rect(2) + rect(4)));
    background_constraints(y_min:y_max, x_min:x_max) = 1;
    close;
end

%% R�initialisation de la librairie de coupe de graphe pour la nouvelle s�rie
% Cr�ation d'un nouvel objet de coupe de graphe
BKhandle = BK_Create(numel(masque));
BK_SetNeighbors(BKhandle, neighborhoodWeights);

%% Deuxi�me s�rie d'it�rations (avec contraintes)
for iter = 1:nbIterations
    fprintf('It�ration avec contraintes %d\n', iter);
    
    % Sauvegarde du masque pr�c�dent pour le test de convergence
    masque_precedent = masque;
    
    %% Mise � jour des probabilit�s par pixel
    [objProbabilites, bkgProbabilitees] = calculerProbabilitesParPixel(image, masque);
    
    % Calcul du poids total et d�finition d'un poids tr�s grand
    total_weight = sum(objProbabilites(:)) + sum(bkgProbabilitees(:));
    large_weight = total_weight + 1e6; % Ajustez si n�cessaire
    
    % Application des contraintes aux probabilit�s
    % Pour les contraintes avant-plan, on augmente bkgProbabilitees
    bkgProbabilitees(foreground_constraints == 1) = bkgProbabilitees(foreground_constraints == 1) + large_weight;
    
    % Pour les contraintes arri�re-plan, on augmente objProbabilites
    objProbabilites(background_constraints == 1) = objProbabilites(background_constraints == 1) + large_weight;
    
    % Construction de la matrice des probabilit�s pour la coupe de graphe
    probabilitesParPixel = [objProbabilites(:), bkgProbabilitees(:)]';
    
    %% Application de la coupe de graphe
    % R�initialisation des capacit�s des t-links avec les nouvelles probabilit�s
    BK_SetUnary(BKhandle, probabilitesParPixel);
    
    % Ex�cution de l'optimisation
    BK_Minimize(BKhandle);
    
    % R�cup�ration des labels
    L = BK_GetLabeling(BKhandle);
    L = reshape(L, M, N);
    
    % Mise � jour du masque
    masque = (L == 1); % 1 pour l'avant-plan, 0 pour l'arri�re-plan
    
    % (Optionnel) Calcul de l'�nergie
    E = computeEnergy(neighborhoodWeights, double(L==1), objProbabilites, bkgProbabilitees);
    fprintf('�nergie = %.2f\n', E);
    
    % Test de convergence
    changements = sum(masque(:) ~= masque_precedent(:));
    fprintf('Nombre de pixels chang�s : %d\n', changements);
    if changements == 0
        fprintf('Convergence atteinte � l''it�ration %d\n', iter);
        break;
    end
    
    % (Optionnel) Affichage de la segmentation actuelle
    figure(4); imagesc(image); axis image; axis off; hold on;
    [c,h] = contour(masque, 'LineWidth',3,'Color', 'g');
    title(sprintf('Segmentation avec contraintes � l''it�ration %d - �nergie = %.2f', iter, E))
    hold off;
    drawnow;
end

%% Suppression de l'objet de coupe de graphe
BK_Delete(BKhandle);
clear BKhandle;

%% Affichage de la solution finale
figure; imagesc(image); axis image; axis off; hold on;
[c,h] = contour(masque, 'LineWidth',3,'Color', 'r');
title(sprintf('Solution finale du GrabCut avec contraintes - �nergie = %.2f', E))
