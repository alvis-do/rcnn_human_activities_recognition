%class = ["boxing","running","walking","handclapping","jogging","handwaving"];
%class = ["basketball","biking","diving","golf_swing"];
class = ["boxing","running","walking","handwaving"];
num_class = size(class,2);
index = [1:1:150];

total_feature = [];
mean_feature = [];
std_feature = [];
for k = class
    try
        i = find(class==k);
        path = strcat('feature/',k,'_feature.txt');
        k_feature = dlmread(path);
        k_feature = [index;k_feature];
        disp(size(k_feature));
        k_feature = k_feature(1:65,:);

        total_feature(:,:,i) = k_feature;
    catch
    end
end

for i = 1:num_class
    try
        temp_mean_feature = mean(total_feature(2:end,:,i));
        temp_std_feature = std(total_feature(2:end,:,i));

        mean_feature = [mean_feature; temp_mean_feature];
        std_feature = [std_feature; temp_std_feature];
    catch
    end
end



a = sum(mean_feature)/num_class;
aa = repmat(a,num_class,1);

tn = (mean_feature - aa);
temp_numerator = sum(tn.*tn)/num_class;
numerator = temp_numerator.^0.5;

denominator = sum(std_feature)/num_class;

rank = numerator./denominator;
rank = [index; rank];
rank = sortrows(rank.',2,'descend').';

index_of_feature_retained = rank(1,1:70);

for k = class
    i = find(class==k);
    temp_feature = total_feature(:,index_of_feature_retained,i);
    path = strcat('feature_extracted/',k,'_feature.txt');
    dlmwrite(path,temp_feature);
end
