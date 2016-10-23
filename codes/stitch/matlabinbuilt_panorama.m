%this code is from Mathwork's site
%http://www.mathworks.com/help/vision/examples/feature-based-panoramic-image-stitching.html?refresh=true

function matlabinbuilt_panorama()
close all;
mode = input('Enter mode (1 for image 2 for video): ') ;
% Load images.

if (mode == 1)
    dirpath = input('Enter Directory: ','s');
    scene = imageSet(dirpath);
else
    vidpath = input('Enter video path: ','s');
    num = input('Enter number of frames to fuse: ');
    scene = extractFrames(vidpath, num);
end
makepanorama(scene)
if (mode ~= 1)
rmdir('tempframes','s')
end
end

function scene = extractFrames(vidpath, fusenum)
mkdir('tempframes');
vidObj = VideoReader(vidpath);
nFrames = vidObj.NumberOfFrames;
i = 1;
for k = 1 : floor(nFrames/fusenum): nFrames
    temp = read(vidObj,k);
    temp = im2double(temp);
    imwrite(temp, sprintf('tempframes\\frame%d.jpg',i));
    i = i+1;
end
scene = imageSet('tempframes');
end


function makepanorama(scene)
% Display images to be stitched
montage(scene.ImageLocation)

% Read the first image from the image set.
I = read(scene, 1);

% Initialize features for I(1)
grayImage = rgb2gray(I);
points = detectSURFFeatures(grayImage);
[features, points] = extractFeatures(grayImage, points);

%Show a few of the strongest SURF points
figure; imshow(I); hold on;
strongestPoints = points.selectStrongest(20);
strongestPoints.plot('showOrientation',true);

% Initialize all the transforms to the identity matrix. Note that the
% projective transform is used here because the building images are fairly
% close to the camera. Had the scene been captured from a further distance,
% an affine transform would suffice.
tforms(scene.Count) = projective2d(eye(3));

% Iterate over remaining image pairs
for n = 2:scene.Count

    % Store points and features for I(n-1).
    pointsPrevious = points;
    featuresPrevious = features;

    % Read I(n).
    I = read(scene, n);

    % Detect and extract SURF features for I(n).
    grayImage = rgb2gray(I);
    points = detectSURFFeatures(grayImage);
    figure; imshow(I); hold on;
    strongestPoints = points.selectStrongest(20);
    strongestPoints.plot('showOrientation',true);
    [features, points] = extractFeatures(grayImage, points);

    % Find correspondences between I(n) and I(n-1).
    indexPairs = matchFeatures(features, featuresPrevious, 'Unique', true);

    matchedPoints = points(indexPairs(:,1), :);
    matchedPointsPrev = pointsPrevious(indexPairs(:,2), :);

    % Estimate the transformation between I(n) and I(n-1).
    tforms(n) = estimateGeometricTransform(matchedPoints, matchedPointsPrev,...
        'projective', 'Confidence', 99.9, 'MaxNumTrials', 2000);

    % Compute T(1) * ... * T(n-1) * T(n)
    tforms(n).T = tforms(n-1).T * tforms(n).T;
end

imageSize = size(I);  % all the images are the same size

% Compute the output limits  for each transform
for i = 1:numel(tforms)
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(2)], [1 imageSize(1)]);
end

avgXLim = mean(xlim, 2);

[~, idx] = sort(avgXLim);

centerIdx = floor((numel(tforms)+1)/2);

centerImageIdx = idx(centerIdx);

Tinv = invert(tforms(centerImageIdx));

for i = 1:numel(tforms)
    tforms(i).T = Tinv.T * tforms(i).T;
end

for i = 1:numel(tforms)
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(2)], [1 imageSize(1)]);
end

% Find the minimum and maximum output limits
xMin = min([1; xlim(:)]);
xMax = max([imageSize(2); xlim(:)]);

yMin = min([1; ylim(:)]);
yMax = max([imageSize(1); ylim(:)]);

% Width and height of panorama.
width  = round(xMax - xMin);
height = round(yMax - yMin);

% Initialize the "empty" panorama.
panorama = zeros([height width 3], 'like', I);

blender = vision.AlphaBlender('Operation', 'Binary mask', ...
    'MaskSource', 'Input port');

% Create a 2-D spatial reference object defining the size of the panorama.
xLimits = [xMin xMax];
yLimits = [yMin yMax];
panoramaView = imref2d([height width], xLimits, yLimits);

% Create the panorama.
for i = 1:scene.Count

    I = read(scene, i);

    % Transform I into the panorama.
    warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);

    % Create an mask for the overlay operation.
    warpedMask = imwarp(ones(size(I(:,:,1))), tforms(i), 'OutputView', panoramaView);

    % Clean up edge artifacts in the mask and convert to a binary image.
    warpedMask = warpedMask >= 1;

    % Overlay the warpedImage onto the panorama.
    panorama = step(blender, panorama, warpedImage, warpedMask);
end

figure
imshow(panorama)
end