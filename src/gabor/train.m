class = ["boxing","running","walking","handwaving"];
%class = ["basketball","biking","diving","golf_swing"];
meas = [];
species = [];
for k = class
    i = find(class==k);
    path = strcat('/Users/thoithanh/Desktop/2312/feature_extracted/',k,'_feature.txt');
    k_feature = dlmread(path);
    
    
    meas =[meas; k_feature(2:end,:)];
    
    temp_spe = [k];
    %temp_spe = repmat(temp_spe,99,1);
    temp_spe = repmat(temp_spe,64,1);
    species = [species;temp_spe];
end

X = meas;
Y = species;

Mdl = fitcecoc(X,Y);

Mdl.ClassNames
CodingMat = Mdl.CodingMatrix
isLoss = resubLoss(Mdl)

%[label, score] = predict(Mdl, test_feature);








