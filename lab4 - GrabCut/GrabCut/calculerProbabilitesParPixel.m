function [objProbabilites, bkgProbabilites] = calculerProbabilitesParPixel(image, masque)

% Validation des entrées
if size(image, 1) ~= size(masque, 1) || size(image, 2) ~= size(masque, 2)
    error('L''image et le masque doivent avoir la même taille MxN.');
end
[M, N, C] = size(image);

% Vérification que l'image est en couleur
if C ~= 3
    error('L''image doit être en couleur (RGB).');
end

% Normalisation des valeurs entre 0 et 1.0
if max(image(:)) > 1.0
    image = double(image) ./ 255;
end

% Paramètre
Nbins = 64; % Nombre de classes par dimension (Nbins x Nbins x Nbins en RGB)

% Quantification des valeurs des pixels pour les indices d'histogramme
imageQuant = floor(image * (Nbins - 1)) + 1;

% Initialisation des histogrammes 3D
objHist = zeros(Nbins, Nbins, Nbins);
bkgHist = zeros(Nbins, Nbins, Nbins);

% Construction des histogrammes 3D
for x = 1:M
    for y = 1:N
        idxR = imageQuant(x, y, 1);
        idxG = imageQuant(x, y, 2);
        idxB = imageQuant(x, y, 3);
        if masque(x, y)
            objHist(idxR, idxG, idxB) = objHist(idxR, idxG, idxB) + 1;
        else
            bkgHist(idxR, idxG, idxB) = bkgHist(idxR, idxG, idxB) + 1;
        end
    end
end

assert(isequal(sum(objHist(:)) + sum(bkgHist(:)), M * N), 'Les histogrammes n''ont pas correctement compté tous les pixels');

% Normalisation des histogrammes pour obtenir des PDF
objPDF = objHist / sum(masque(:));
bkgPDF = bkgHist / sum(~masque(:));

% Remplacement des zéros par une petite valeur
histAlpha = 1e-6; % Probabilité minimale dans chaque classe
objPDF = (1 - histAlpha) * objPDF + histAlpha / (Nbins^3);
bkgPDF = (1 - histAlpha) * bkgPDF + histAlpha / (Nbins^3);

% Calcul des indices pour chaque pixel
idxR = imageQuant(:, :, 1);
idxG = imageQuant(:, :, 2);
idxB = imageQuant(:, :, 3);

% Conversion des indices 3D en indices linéaires pour accéder aux PDF
idx = sub2ind(size(objPDF), idxR(:), idxG(:), idxB(:));

% Calcul des probabilités par pixel
objProbabilites = -log(reshape(objPDF(idx), M, N));
bkgProbabilites = -log(reshape(bkgPDF(idx), M, N));

end
