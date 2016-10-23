%The main function to run is imgstitch which is a wrapper function. It
%handles 2 modes (stitch 2 images or stitch from a video). Both modes make
%use of the panorama() function which is adapted and modified from 
%Mr. Ganesh Kumar code with whom I had worked with during our summer
%internship.

function imgstitch()
close all;
mode_ = input('Enter mode (1 for image 2 for video): ');

if mode_ == 1  %stitch 2 images
    %example
    %C:\Sayantan\acads\enee631\project\matlabcode\balcony
    %C:\Sayantan\acads\enee631\project\matlabcode\building
    %C:\Sayantan\acads\enee631\project\matlabcode\hill
    dp_ = input('Enter directory: ', 's');
    file_ = dir(dp_);
    filenames_ = {file_.name};
    filenames = filenames_(end);
    namesplit = strsplit(filenames{1}, '.');
    
    parts = strsplit(dp_,'\');
    f1 = strcat(dp_, '\', parts(end), num2str(1), '.', namesplit(2));
    f2 = strcat(dp_, '\', parts(end), num2str(2), '.', namesplit(2));
    img1 = im2double(imread(f1{1}));
    img2 = im2double(imread(f2{1}));
    
    octaves = 4;
    levels = 2;
    [keyPoints1 keyPoints2 match inliers im] = panorama(img1, img2, octaves, levels, 8, 200, 1.6);
    figure; imshow(im);
    
else  %video mode
    %C:\Sayantan\acads\enee631\project\matlabcode\vid1.mp4
    octaves = 4;
    levels = 2;
    vidname = input('Enter video: ', 's');
    vidObj = VideoReader(vidname);
    nFrames = vidObj.NumberOfFrames;
    vidHeight = vidObj.Height;
    vidWidth = vidObj.Width;
    for k = 1 : nFrames
        mov(k).cdata = read(vidObj,k);
        mov(k).cdata = im2double(mov(k).cdata);
    end
    im = videoPanorama(mov, octaves, levels, 8, 200, 1.6, 2, nFrames);
    figure; imshow(im);
end
end

function im = videoPanorama(mov, octaves, levels, RANSACth, RANSACiter, sigma, fusenum, nFrames)
img1 = mov(1).cdata;
figure; imshow(img1); pause(0.001);
for i = 1:floor(nFrames/fusenum):nFrames
    if (i + floor(nFrames/fusenum)) < nFrames
        img2 = mov(i + floor(nFrames/fusenum)).cdata;
        figure; imshow(img2); pause(0.001);
        [keyPoints1 keyPoints2 match inliers im] = panorama(img1, img2, octaves, levels, RANSACth, RANSACiter, sigma);
        img1 = im;
        figure; imshow(im); pause(0.001);
    end
end
end

function  [keyPoints1 keyPoints2 match inliers im] = panorama(img1, img2, octaves, levels, RANSACth, RANSACiter, sigma)
if (0)  %enable for histogram equalization
    i1 = rgb2hsv(img1); i2 = rgb2hsv(img2);
    imeq1(:,:,1) = (i1(:,:,1));
    imeq1(:,:,2) = (i1(:,:,2));
    imeq1(:,:,3) = histeq(i1(:,:,3));
    imeq1 = hsv2rgb(imeq1);
%     figure;
%     subplot(2,2,1), subimage(img1);
%     subplot(2,2,2), subimage(i1(:,:,3))
%     subplot(2,2,3), subimage(histeq(i1(:,:,3)))
%     subplot(2,2,4), subimage(imeq1)
    
    imeq2(:,:,1) = (i2(:,:,1));
    imeq2(:,:,2) = (i2(:,:,2));
    imeq2(:,:,3) = histeq(i2(:,:,3));
    imeq2 = hsv2rgb(imeq2);
else
    imeq1 = img1; imeq2 = img2;
end

figure; imshow((imeq1));
figure; imshow((imeq2));

imggray1 = rgb2gray((imeq1));
imggray2 = rgb2gray((imeq2));

[keyPoints1,descriptors1] = sift(imggray1, octaves, levels, sigma);
%display SIFT points
J = imggray1;
for c = 1:size(keyPoints1,1)
    shapeInserter = vision.ShapeInserter;
    rectangle = int32([keyPoints1(c,2) keyPoints1(c,1) 5 5]);
    J = step(shapeInserter, J, rectangle);
end
figure; imshow(J); 
[keyPoints2,descriptors2] = sift(imggray2, octaves, levels, sigma);
J = imggray2;
for c = 1:size(keyPoints2,1)
    shapeInserter = vision.ShapeInserter;
    rectangle = int32([keyPoints2(c,2) keyPoints2(c,1) 5 5]);
    J = step(shapeInserter, J, rectangle);
end
figure; imshow(J); 

match = findBestMatch(descriptors1, descriptors2);

%find transformzation using RANSAC
[best, inliers]  = ransac(keyPoints1,keyPoints2,match,RANSACth,RANSACiter);

%merge
im = merge(best, img1, img2);
end

function [gaussPyramid dogPyramid sigmas] = findPyramids(image, octaves, levels, baseSigma)
%gaussian and DoG pyramids
sigmas = zeros(octaves, levels);
workingImage = image;
for octave = 1:octaves
    sigma = baseSigma * 2^(octave-1);
    gaussFilter = getGaussFilter(sigma);
    gaussPyramid{octave,1} = conv2( gaussFilter, gaussFilter, workingImage, 'same');
    sigmas(octave, 1) = sigma;
    for level=2:levels+3
        sigma = sigma*sqrt(2);
        sigmas(octave,level) = sigma;
        gaussFilter = getGaussFilter(sigma);
        gaussPyramid{octave,level} = conv2(gaussFilter, gaussFilter, workingImage , 'same' );
        dogPyramid{octave} (:,:,level-1)= gaussPyramid{octave,level} - gaussPyramid{octave,level-1};
    end
    workingImage = imresize(workingImage, 0.5);
end
end

function [H, D] = computeDerivatives(dogPyramid, octave, x, y, level)
% building the Hessian matrix
dx = (dogPyramid{octave}(y,x+1,level) - dogPyramid{octave}(y,x-1,level))*0.5;
dy = (dogPyramid{octave}(y+1,x,level) - dogPyramid{octave}(y-1,x,level))*0.5;
ds = (dogPyramid{octave}(y,x,level+1) - dogPyramid{octave}(y,x,level-1))*0.5;
dxx = sum(dogPyramid{octave}(y, x-1:x+1, level).*[1 -2 1]);
dyy = sum(dogPyramid{octave}(y-1:y+1, x, level).*[1 -2 1]');
dss = dogPyramid{octave}(y,x,level+1) + dogPyramid{octave}(y,x,level-1) - dogPyramid{octave}(y,x,level) *2;
dxy = (dogPyramid{octave}(y+1,x+1,level) + dogPyramid{octave}(y-1,x-1,level) - dogPyramid{octave}(y+1,x-1,level) - dogPyramid{octave}(y-1,x+1,level) )*0.25;
dxs = (dogPyramid{octave}(y,x+1,level+1) + dogPyramid{octave}(y,x-1,level-1) - dogPyramid{octave}(y,x-1,level+1) - dogPyramid{octave}(y,x+1,level-1) )*0.25;
dys = (dogPyramid{octave}(y+1,x,level+1) + dogPyramid{octave}(y-1,x,level-1) - dogPyramid{octave}(y-1,x,level+1) - dogPyramid{octave}(y+1,x,level-1) )*0.25;

H = [dxx dxy dxs; dxy dyy dys; dxs dys dss];
D = [dx dy ds]';
end

function [keyPoints, descriptors] = sift(image, octaves, levels, s)
[gaussPyramid dogPyramid sigmas] = findPyramids(image, octaves, levels, s);

% find maxima and minima
keyPointCount = 0;
for octave = 1:octaves
    for level = 2:levels+1
        for y = 5:size(dogPyramid{octave} (:,:,level), 1) - 5
            for x = 5:size(dogPyramid{octave} (:,:,level), 2) - 5
                area = dogPyramid{octave}((y-1):(y+1),(x-1):(x+1),(level-1):(level+1)); %27 point neighbourhood
                point = area(2,2,2);  %centre of neighbourhood
                if((point == max(area(:)) || point == min(area(:))) && abs(point) > 0.03)
                    
                    [H, D] = computeDerivatives(dogPyramid, octave, x, y, level);
        
                    %curvature threshold
                    if ((trace(H)^2) / det(H) < (11^2)/10)
                        delta = (H^(-1))*D;
                        
                        % build histogram
                        region =  gaussPyramid{octave, level}(y-2:y+2,x-2:x+2);
                        bin = zeros(1,36);
                        deltaX = conv2(region,[-1 1],'same');
                        deltaY = conv2(region,[-1 1]','same');
                        mag = sqrt(deltaX.^2 + deltaY.^2);
                        
                        for cc1 = 1:size(deltaX,1)
                            for cc2 = 1:size(deltaX,2)
                                binIndex(cc1,cc2) = ceil((180/pi)*myatan2(deltaY(cc1,cc2), deltaX(cc1,cc2))/10);
                            end
                        end
           
                        for i=2:4
                            for j = 2:4
                                bin(binIndex(i,j)) = bin(binIndex(i,j)) + mag(i,j)*gauss(1.5*sigmas(octave,level),  sqrt((i-3)^2+(j-3)^2));
                            end
                        end

                        maxBin = max(bin);
                        maxList = find(bin>=0.8*maxBin);
                        for ii = 1:size(maxList)
                            modxy = [x;y] + delta(1:2); %correction
                            keyPointCount = keyPointCount + 1;                     
                            keyPoints(keyPointCount, [1:3]) = [modxy(2) modxy(1) octave];
                            
                            [gridX, gridY] = meshgrid(-6:4:6);
                            grid1 = [gridX(:) gridY(:)]';
                            [gridX, gridY] = meshgrid(-8:7);
                            grid2 = [gridX(:) gridY(:)]';
                            rotMtx = [cos(maxList(ii)*pi/18) -sin(maxList(ii)*pi/18); sin(maxList(ii)*pi/18) cos(maxList(ii)*pi/18)];
                            rotGrid1 = rotMtx*grid1 + repmat(modxy, 1, 16);
                            rotGrid2 = rotMtx*grid2 + repmat(modxy, 1, 256);
                            rot1X = rotGrid1(1,:); rot1Y = rotGrid1(2,:);
                            rot2X = rotGrid2(1,:); rot2Y = rotGrid2(2,:);
                            
                            %get neighbours in rotated grid
                            tempX = interp2(gaussPyramid{octave,level}, [rot2X' rot2X'+1], [rot2Y' rot2Y'], 'linear');
                            tempY = interp2(gaussPyramid{octave,level}, [rot2X' rot2X'], [rot2Y' rot2Y'+1], 'linear');
                            tempX(find(isnan(tempX))) = 0;
                            tempY(find(isnan(tempY))) = 0;
                            dx = conv2(tempX,[-1 1], 'same');
                            dy = conv2(tempY,[-1 1], 'same');
                            mag = sqrt(dx(:,1).^2 + dy(:,1).^2);
                            deg = zeros(256,1);
                            for cc = 1:256
                                deg(cc,1) = myatan2(dy(cc,1), dx(cc,1));
                            end
                            
                            % calculate descriptors
                            deg = mod((deg*(180/pi)-maxList(ii)*10)+360,360);
                            descriptor = zeros(1,128);
                            for i=1:256
                                weightX = max(1 - (abs(rot1X- rot2X(i))/4), 0);
                                weightY = max(1 - (abs(rot1Y - rot2Y(i))/4), 0);
                                weight = reshape(repmat(weightX.*weightY,8,1),1,128);
                                diff = mod(deg(i) - [-180:45:135] + 180, 360) - 180;
                                weightDeg = repmat(max(1 - abs(diff)/45,0), 1, 16);
                                weightGauss = gauss(8, sqrt((rot2X(i)-modxy(1))^2+(rot2Y(i)-modxy(2))^2));
                                descriptor = descriptor + weight.*weightDeg*weightGauss*mag(i); 
                            end
                            descriptor = descriptor/norm(descriptor);
                            descriptor(descriptor>0.2) = 0.2; %saturate
                            descriptor = descriptor/norm(descriptor); %normalize
                            descriptors(keyPointCount,1:128) = descriptor;
                        end
                    end
                end
            end
        end
    end
end
% scale keypoint by octave
keyPoints(:,2) = keyPoints(:,2).*(2.^(keyPoints(:,3)-1));
keyPoints(:,1) = keyPoints(:,1).*(2.^(keyPoints(:,3)-1));
end

function im = merge(m, a, b)
%transformation matrix
transform = [m(1) m(2) m(5); m(3) m(4) m(6); m(7) m(8) 1];
transform = transform^(-1);

%source image corners
tl = [1 1 1]';  %topleft
bl = [1 size(b,1) 1]';  %bottomleft
tr = [size(b,2) 1 1]';  %topright
br = [size(b,2) size(b,1) 1]';  %bottomright

xx = transform *([tl bl tr br]);
yy = xx./repmat(xx(3,:), 3, 1);
tl_ = yy(:,1); bl_ = yy(:,2); tr_ = yy(:,3); br_ = yy(:,4); %transformed coordinates of corners

minX = floor(min([tl_(1) bl_(1) tr_(1) br_(1)]));
maxX = ceil(max([tl_(1) bl_(1) tr_(1) br_(1)]));
minY = floor(min([tl_(2) bl_(2) tr_(2) br_(2)]));
maxY = ceil(max([tl_(2) bl_(2) tr_(2) br_(2)]));

%compute source location for all the target points
x = minX:maxX;
y = minY:maxY;
[X1, Y1] = meshgrid(x,y);
XT = (m(1)*X1+m(2)*Y1+m(5))./(m(7)*X1 + m(8)*Y1+ 1);
YT = (m(3)*X1+m(4)*Y1+m(6))./(m(7)*X1 + m(8)*Y1 + 1);

%take care when minimum coordinate is negative
offset = [1,1];
if(minX < 0)
    offset(1) = 1 - minX;
end
if(minY < 0)
    offset(2) = 1 - minY;
end

for cc = 1:3  %stitch the 3 channels separately
    im(:,:,cc) = stitchPyramid(a(:,:,cc), b(:,:,cc), XT, YT, minX, maxX, minY, maxY, offset);
end
im(im < 0) =0;
im(im> 1) = 1;
end


function im = stitchPyramid(a, b, XT, YT, minX, maxX, minY, maxY, offset)    
    im1 = zeros(max([maxY size(a,1)]) + offset(2), max([maxX size(a,2)]) + offset(1));
    im2 = im1;
    im1(offset(2):offset(2) -1 + size(a,1),offset(1):offset(1)  -1 + size(a,2)) = a;
    im2(offset(2) + minY:offset(2) + maxY, offset(1) + minX:offset(1) + maxX) = interp2((b),XT,YT);
    im2(find(isnan(im2))) = 0;
    ROI = (im1 + im2) > 0;
    filter = [1 4 6 4 1]/16;
    filter = filter'*filter;
    mask = im1;
    mask(mask>0) = 1;
    
    %borders
    num = 40;
    szim1 = size(im1); szim2 = size(im2); sza = size(a); szb = size(b);
    if(offset(2) > 1)
        border = min(offset(2)-1, num);
        im1(offset(2) - border:offset(2)-1,:) = repmat(im1(offset(2) ,:), border, 1);
    end
    if(offset(2) -1 + sza(1)<szim1(1))
        border = min(szim1(1) - (offset(2) -1 + sza(1)), num);
        im1(offset(2)  -1 + sza(1) +1:offset(2) -1 + sza(1) + border,:) = repmat(im1(offset(2) -1 + sza(1),: ), border, 1);
    end
    if(offset(1) > 1)
        border = min(offset(1)-1, num);
        im1(:,offset(1) - border:offset(1)-1) = repmat(im1(:,offset(1)), 1, border);
    end
    if(offset(1) -1 + sza(2)<szim1(2))
        border = min(szim1(2) - (offset(1) -1 + sza(2)), num);
        im1(:,offset(1) -1 + sza(2) +1:offset(1) -1 + sza(2) + border) = repmat(im1(:,offset(1) -1 + sza(2) ), 1, border);
    end

    %build pyramids
    pySize = 6;
    py1 = cell(pySize,2); py1(1) = {im1};
    py2 = cell(pySize,2); py2(1) = {im2};
    pyMask = cell(pySize,1); pyMask(1) = {mask};
    pyJoin = cell(pySize,1);

    for i=2:pySize
        py1(i,1) = {downsampleby2(py1{i-1,1}, filter)};
        py1(i-1,2) = {py1{i-1,1} - upsampleby2(py1{i,1}, size(py1{i-1,1}), filter)};
        py2(i,1) = {downsampleby2(py2{i-1,1}, filter)};
        py2(i-1,2) = {py2{i-1,1} - upsampleby2(py2{i,1}, size(py2{i-1,1}), filter)};
        pyMask(i,1) = {downsampleby2(pyMask{i-1,1}, filter)};
        pyJoin(i-1,1) = {(pyMask{i-1,1}).*py1{i-1,2} + (1-pyMask{i-1,1}).*py2{i-1,2}};
    end
    py1(pySize,2) = py1(pySize,1);
    py2(pySize,2) = py2(pySize,1);
    pyJoin(pySize,1) = {pyMask{pySize,1}.*py1{pySize,1} + (1-pyMask{pySize,1}).*py2{pySize,1} };

    im = zeros(size(pyJoin{pySize,1}));
    for i=pySize:-1:2
        im = im + pyJoin{i,1};
        im = upsampleby2(im, size(pyJoin{i-1,1}), filter);
    end
    im = im + pyJoin{1,1};
    im(ROI == 0) = 0;
end

function down = downsampleby2(img, filter)
[X Y] = meshgrid(1:2:size(img,2), 1:2:size(img,1));
down = interp2(filter2(filter,img),X,Y); 
end

function expand = upsampleby2(img, sz, filter)
expand = zeros(sz);
y = 1:2:sz(1); x = 1:2:sz(2);
expand(y, x) = img(1:size(y, 2),1:size(x, 2)); %expansion
expand = filter2(filter,expand);
expand = expand/max(expand(:)); %normalize
end

function [best, inliers] = ransac(keyPoints1, keyPoints2, match, RANSACth, RANSACiter)
best = eye(3);
bestCount=0;
for i=1:RANSACiter
    num = 4;
    rnd =  randperm(size(match,1)); rnd = rnd(1:num);
    maybeinliers = match(rnd,:);

    d = keyPoints2(maybeinliers(:,2),[2 1]);
    s = keyPoints1(maybeinliers(:,1),[2 1]);
    %transformation matrix
    transformation = [ s(:,1) s(:,2) zeros(size(s,1),1) zeros(size(s,1),1) ones(size(s,1),1) zeros(size(s,1),1) -s(:,1).*d(:,1) -s(:,2).*d(:,1) ;
        zeros(size(s,1),1) zeros(size(s,1),1) s(:,1) s(:,2) zeros(size(s,1),1) ones(size(s,1),1) -s(:,1).*d(:,2) -s(:,2).*d(:,2)];
    dv = reshape(d,2*num,1);
    p = pinv(transformation)*dv;   
    
    [count, index] = countMatches(keyPoints1, keyPoints2, match, p, RANSACth);
    if(count > bestCount)
        best = p;
        bestCount = count;
        inliers = index;
    end
end
end

function [count, index] = countMatches(keyPoints1, keyPoints2, match, m, RANSACth)
%source coordinates of the target
x1 = keyPoints1(match(:,1),2); y1 = keyPoints1(match(:,1),1);
x2 = keyPoints2(match(:,2),2); y2 = keyPoints2(match(:,2),1);
x = (m(1)*x1+m(2)*y1+m(5))./(m(7)*x1+m(8)*y1+ 1);
y = (m(3)*x1+m(4)*y1+m(6))./(m(7)*x1+m(8)*y1 + 1);
index = find(sqrt((x-x2).^2+(y-y2).^2) < RANSACth);
count = size(index,1);
end

% Find minimum distance pairs between the two keypoints
function match = findBestMatch(d1, d2)
matchCount = 0;
for i=1:size(d1, 1)
    dist = d2 - repmat(d1(i,:), size(d2,1), 1);
    dist = sqrt(sum(dist.*dist,2));
    mind = min(dist);
    if(sum(dist < (10/8)*mind) == 1)
        matchCount = matchCount+1;
        match(matchCount,1:2) = [i find(dist == mind)];
    end
end
end

function gaussianFilter = getGaussFilter(sigma)
x=1:2*round(3.5 * sigma)+1;
x=x-ceil(length(x)/2);
gaussianFilter = 1/(sigma*sqrt(2*pi)) * exp(-(x.^2)/(2*sigma^2));
end

function gauss = gauss(sigma,dist)
gauss = 1/(sigma*sqrt(2*pi)) * exp(-(dist^2)/(2*sigma^2));
end

%this function is modified from a Mathworks forum answer
%http://www.mathworks.com/matlabcentral/answers/9330-changing-the-atan-function-so-that-it-ranges-from-0-to-2-pi
function v = myatan2(y,x)
if nargin==1 %just in case the user only gives the value of y myatan(y)
    x=1;
end
v = 0.000001;
if x>0
    v=atan(y/x);
end
if y>=0 && x<0
    v=pi+atan(y/x);
end
if y<0 && x<0
    v=-pi+atan(y/x);
end
if y>0 && x==0
    v=pi/2;
end
if y<0 && x==0
    v=-pi/2;
end
if v<0
    v=v+2*pi;
end
if (v == 0)
   v = 0.001; 
end     
end