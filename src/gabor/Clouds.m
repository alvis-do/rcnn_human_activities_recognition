global gabor_0 gabor_22 gabor_45 gabor_67 gabor_90 gamma psi lambda sigma;
gamma=0.3;
psi=0;
lambda=3.5;
sigma=2.8;
gabor_0 = gabor2_fn(sigma,0,lambda,psi,gamma);
gabor_22 = gabor2_fn(sigma,22,lambda,psi,gamma);
gabor_45 = gabor2_fn(sigma,45,lambda,psi,gamma);
gabor_67 = gabor2_fn(sigma,67,lambda,psi,gamma);
gabor_90 = gabor2_fn(sigma,90,lambda,psi,gamma);
%     vid = '/Users/thoithanh/Desktop/DataSet/running_mp4/person01_running_d1_uncomp.mp4';
%     feature = vid2feature_fn(vid);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 class = ["boxing","running","walking","handwaving"];
 feature = [];
 for k = class
     k = char(k);
     for d = 1:4
         for p = 1:16 %first 16 acc 86%
             if p < 10
                 strp = strcat('0',int2str(p));
             else
                 strp = int2str(p);
             end
             video = strcat('/Users/thoithanh/Desktop/Database_mp4/',k,'/person',strp,'_',k,'_d',int2str(d),'_uncomp.mp4');
             temp_feature = vid2feature_fn(video);
             feature = [feature; temp_feature];
             disp([{k} p d ]);
             %filename = strcat('/Users/thoithanh/Desktop/New_features/',k,'_feature/person',strp,'_',k,'_d',int2str(d),'_uncomp.txt');
             %dlmwrite(filename,temp_feature);
         end
     end
     ffff = strcat('/Users/thoithanh/Desktop/KTH_feature_80bins_all/feature/',k,'_feature.txt');
     dlmwrite(ffff,feature);
     feature = [];
 end


% class = ["basketball","biking","diving","golf_swing"];
% feature = [];
% for k = class
%     k = char(k);
%     fid = fopen('video_train.txt');
%     while ~feof(fid)
%         tline = fgetl(fid);
%         C = strsplit(tline,'/');
%         if strcmp(C(7),strcat(k))
%             disp(C(7))
%             temp_feature = vid2feature_fn(tline);
%             if size(temp_feature,2)==150
%                 feature = [feature; temp_feature];
%             end
%         end
% %         disp(tline);
%     end
%     fclose(fid);
%     
%     ffff = strcat('/Users/thoithanh/Desktop/UCF_feature/feature/',k,'_feature.txt');
%     dlmwrite(ffff,feature);
%     feature = [];
% end



function vid2feature = vid2feature_fn(link)
vid=VideoReader(link);
numFrames = vid.NumberOfFrames;
n=numFrames;

points= [];


frames = vid.read(1);
previous_frame = rgb2gray(frames);
previous_frame = imresize(previous_frame,[128 128]);

feature = [];
object = [];
for i = 1:n
    frames = vid.read(i);
    %
    frames= imresize(frames,[128 128]);
    %
    frames_gray = rgb2gray(frames);
    
    frame_dif = frames_gray - previous_frame;
    previous_frame = frames_gray;
    frame_dif(frame_dif<40)=0;
    frame_dif(frame_dif>0)=255;
    
    temp_extrema = bound_fn(frame_dif);
    if(isempty(temp_extrema))
        temp_extrema = [0 0 0 0];
    end
    
    %frame_dif(frame_dif>0)=255;
    
    
    
    
    new_points = extract_interest_points(frames_gray);
    position_delete_points = find( new_points(:,1)<temp_extrema(2) | new_points(:,1)>(temp_extrema(2)+temp_extrema(4)) | new_points(:,2) < temp_extrema(1) | new_points(:,2)>(temp_extrema(1)+temp_extrema(3)));
    
    position_delete_points = position_delete_points.';
    new_points(position_delete_points,:) = [];
    new_points(:,3) = i;
    
    points  = cat(1,points,new_points);
    temp_extrema(:,5) = i;
    object = cat(1,object,temp_extrema);
    temp_extrema(:,5) = [];

       
    
    
%               imshow(frames);
%               hold on,
%               rectangle('Position', temp_extrema, 'EdgeColor', 'yellow'),
%               plot(points(:,2),points(:,1),'g.'),
%               title('SSS');
    
end
%
feature = extract_feature_fn(object,points,6,5,n);
feature_processed = pre_processing_fn(feature);
feature_q = quantum_fn(feature_processed, 3);
vid2feature = feature_q;
%

% temple_frame = read(vid,30);
% temple_frame = imresize(temple_frame,[128 128]);
% figure, imshow(temple_frame); hold on,
% plot(new_points(:,2),new_points(:,1),'r.'), title('Cloud in frame 30');

end


function feature = extract_feature_fn(object,points,s,Ns,n) %n = num of frame
f = [];
pre_O_center = [0 0];
pre_C_center = zeros(s,2);
for i = 1:n
    temp_extrema = object(i,:);
    temp_extrema(5) = [];
    O_center = [(temp_extrema(1)+temp_extrema(3)/2), (temp_extrema(2)+temp_extrema(3)/2) ]; %w/h
    if temp_extrema(3) == 0
        Ort = 0;
    else
        Ort = temp_extrema(4)/temp_extrema(3);
    end
    Ospt = sqrt((O_center(1) - pre_O_center(1))^2 + (O_center(2) - pre_O_center(2))^2);
    
    feature_in_frame = [Ort, Ospt];
    for j = 1:s
        temp_points =  points(points(:,3)<=i & points(:,3)>=(i-s*Ns),:); %cloud in s scale
        if size(temp_points,1)>1
            temp_points(:,3) = [];
            
            pmax = max(temp_points);
            pmin = min(temp_points);
            row_max = pmax(1);
            col_max = pmax(2);
            row_min = pmin(1);
            col_min = pmin(2);
            width = col_max - col_min;
            heigh = row_max - row_min;
            area = width*heigh;
            C_center = [(row_min + row_max)/2, (col_min + col_max)/2];
            Cr = heigh/width;
            Csp = sqrt((pre_C_center(j,1)-C_center(1))^2 + (pre_C_center(j,2)-C_center(2))^2);
            Csp = Csp/heigh;
            Cd = size(temp_points,1)/area;
            Cvd = abs(O_center(2) - C_center(2));
            Chd = abs(O_center(1) - C_center(1));
            Chr = temp_extrema(4)/heigh;
            Cwr = temp_extrema(3)/width;
            %
            ol_row_max = min(row_max,temp_extrema(1)+temp_extrema(3));
            ol_row_min = max(row_min,temp_extrema(1));
            ol_col_max = min(col_max,temp_extrema(2)+temp_extrema(4));
            ol_col_min = max(col_min,temp_extrema(2));
            if (ol_row_max > ol_row_min) && (ol_col_max > ol_col_min)
                Cor = (ol_row_max - ol_row_min)*(ol_col_max - ol_col_min);
            else
                Cor = 0;
            end
            
            if (width == 0) ||  (heigh == 0)
                feature_in_scale = [0 0 0 0 0 0 0 0];
            else
                feature_in_scale = [Cr, Csp, Cd, Cvd, Chd, Chr, Cwr, Cor];
            end
        else
            feature_in_scale = [0 0 0 0 0 0 0 0];
            C_center = [0 0];
        end
        feature_in_frame = [feature_in_frame feature_in_scale];
        pre_C_center(j,:) = C_center;
    end
    f = [f;feature_in_frame];
    pre_O_center = O_center;
end

feature = f;
end

function pre_processing = pre_processing_fn(feature)
feature(feature(:,2)==0,:)=[];
feature(feature(:,4)==0,:)=[];
pre_processing = feature;
end

function quantum = quantum_fn(feature, Nb)
quantum = [];
nrow = size(feature,1);

interval = nrow/Nb;
for i = 1:interval:(Nb*interval)
    a = i + interval -1;
    if a>nrow
        a = nrow;
    end
    temp_mt = feature(i:a,:);
    
    if size(temp_mt,1)==1
        temp_quantum = temp_mt;
    else
        temp_quantum = sum(temp_mt);
    end
    quantum = [quantum; temp_quantum];
end
quantum = quantum/nrow;
quantum = reshape(quantum,1,[]);
end


function bound = bound_fn(image)
% //Threshold and remove last 10 rows
y=imbinarize(image);
y = y(1:end-10,:);

% //Calculate all bounding boxes
s=regionprops(y, 'BoundingBox');

%// Obtain all of the bounding box co-ordinates
bboxCoords = reshape([s.BoundingBox], 4, []).';

% // Calculate top left corner
topLeftCoords = bboxCoords(:,1:2);

% // Calculate top right corner
topRightCoords = [topLeftCoords(:,1) + bboxCoords(:,3) topLeftCoords(:,2)];

% // Calculate bottom left corner
bottomLeftCoords = [topLeftCoords(:,1) topLeftCoords(:,2) + bboxCoords(:,4)];

% // Calculate bottom right corner
bottomRightCoords = [topLeftCoords(:,1) + bboxCoords(:,3) ...
    topLeftCoords(:,2) + bboxCoords(:,4)];

% // Calculating the minimum and maximum X and Y values
finalCoords = [topLeftCoords; topRightCoords; bottomLeftCoords; bottomRightCoords];
minX = min(finalCoords(:,1));
maxX = max(finalCoords(:,1));
minY = min(finalCoords(:,2));
maxY = max(finalCoords(:,2));

width = (maxX - minX + 1);
height = (maxY - minY + 1);
rect = [minX minY width height];


bound = [minX minY width height];

end


function ip = extract_interest_points(image)
size = 128;
%image_gray = rgb2gray(image);
image_gray = image;
image_resize = imresize(image_gray, [size size]);
im = im2double(image_resize);
gamma=0.3;
psi=0;
lambda=3.5;

sigma=2.8;



qxy = zeros(size);
for th=[0 22 45 67 90]
    qxy = qxy + q2_fn(im,size,sigma,th,lambda,psi,gamma);
end
gb = qxy./5;

radius = 1;
order = (2*radius + 1);
threshold = 600;
threshold = threshold/1000;
mx = ordfilt2(gb,order^2,ones(order));
interestPoints = (gb==mx)&(gb>threshold);
[rows,cols] = find(interestPoints);

% figure, imshow(image_resize), hold on,
% plot(cols,rows,'g.'), title('Interest Points by Gabor filter');


ip = [rows, cols];
end

function gb2=gabor2_fn(sigma,theta,lambda,psi,gamma)
sigma_x = sigma;
sigma_y = sigma/gamma;

% Bounding box
nstds = 3;
xmax = max(abs(nstds*sigma_x*cos(theta)),abs(nstds*sigma_y*sin(theta)));
xmax = ceil(max(1,xmax));
ymax = max(abs(nstds*sigma_x*sin(theta)),abs(nstds*sigma_y*cos(theta)));
ymax = ceil(max(1,ymax));
xmin = -xmax; ymin = -ymax;
[x,y] = meshgrid(xmin:xmax,ymin:ymax);

% Rotation
x_theta=x*cos(theta)+y*sin(theta);
y_theta=-x*sin(theta)+y*cos(theta);

%gb2= exp(-.5*(x_theta.^2/sigma_x^2+y_theta.^2/sigma_y^2)).*cos(2*pi/lambda*x_theta+psi);

mu = 1/22;
po = 11;
gb2 = exp(-.5*(x.^2/po^2 + y.^2/po^2)).*cos(2*pi*(x*mu+y*mu)+theta);
end

function q2 = q2_fn(im,size,sigma,theta,lambda,psi,gamma)
global gabor_0 gabor_22 gabor_45 gabor_67 gabor_90;
if theta == 0
    Wxy = conv2(im,gabor_0,'same');
elseif theta == 22
    Wxy = conv2(im,gabor_22,'same');
elseif theta == 45
    Wxy = conv2(im,gabor_45,'same');
elseif theta == 67
    Wxy = conv2(im,gabor_67,'same');
else
    Wxy = conv2(im,gabor_90,'same');
end

q2 = abs(Wxy);
end




