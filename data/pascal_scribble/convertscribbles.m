% The original scribbles (provided by http://www.jifengdai.org/downloads/scribble_sup/)
% are in XML format storing coordinates of spots along the scribbles.
% Here we dilate with linewidth three and save labeling as png.
clear all
close all
addpath('scribble_annotation/demo');
load('scribble_annotation/demo/classes.mat');
load('scribble_annotation/demo/cmap.mat');
xmldir='scribble_annotation/pascal_2012';
d = dir([xmldir '/*.xml']);
numimages = size(d,1);
mkdir(['pascal_2012_scribble']);

tic
for n=1:numimages
    if n==2
        visualize = 1;
    else
        visualize = 0;
    end
    imgname = d(n).name;
    imgname = imgname(1:end-4);
    
    clsIdx = 1:numel(classes);
    clsMap = containers.Map(classes, clsIdx);

    % read the image and the scribble annotation
    im = imread(['JPEGImages/' imgname '.jpg']);
    img_h = size(im, 1);
    img_w = size(im, 2);
    spots = readspots([xmldir '/' imgname '.xml'], clsMap);

    numScribble = numel(unique(spots(:, 4)));
    clsScribble = unique(spots(:, 3));
    clsScribble = [clsScribble ones(numel(clsScribble), 1)];
    if visualize
        figure;him = image(im);
        set(him, 'AlphaData', 0.5);
        hold on;
        axis off;axis equal;
    end
    for ii = 1:numScribble
        scribble = spots(spots(:, 4) == ii, :);
        if visualize
            plot(scribble(:, 1), scribble(:, 2), 'Color', cmap(scribble(1, 3), :), 'LineWidth', 3);
            %scatter(scribble(:, 1), scribble(:, 2), 46, cmap(scribble(1, 3), :),'x');
            %imline(gca, scribble(:, 1), scribble(:, 2));
        end
        idx = find(clsScribble(:, 1) == scribble(1, 3));
        if clsScribble(idx, 2) == 1
            if visualize
                meanX = round(mean(scribble(:, 1)));
                meanY = round(mean(scribble(:, 2)));
                text(scribble(1, 1)+5, scribble(1, 2)+5, classes{scribble(1, 3)}, 'Color', [0 0 0], 'FontSize', 13);
            end
            clsScribble(idx, 2) = 0;
        end
    end
    
    gt = zeros(img_h, img_w, 1, 'uint8') + 255; % ignore label
    for ii = 1:numScribble
        scribble = spots(spots(:, 4) == ii, :);
        %if visualize
        %    plot(scribble(:, 1), scribble(:, 2), 'Color', cmap(scribble(1, 3), :), 'LineWidth', 3);
        %    scatter(scribble(:, 1), scribble(:, 2),'x');
        %end
        whiteimg = zeros(img_h, img_w, 'uint8')+255;
        pos_x = scribble(:, 1);
        pos_y = scribble(:, 2);
        pos_idx = pos_y - 1 + (pos_x -1 ) * img_h + 1;
        scribblemask=insertShape(whiteimg, 'line',[pos_x(1:end-1) pos_y(1:end-1) pos_x(2:end) pos_y(2:end)],...
            'LineWidth',3,'Color',[1 1 1], 'Opacity', 1);
        scribblemask = rgb2gray(scribblemask);
        scribblemask = (scribblemask~=255);
        gt(scribblemask) = scribble(1, 3) -1 ; % start from zero (background)
        %if visualize; figure,imshow(scribblemask); end
    end
    if visualize; figure,imshow(gt); end
    imwrite(gt,['pascal_2012_scribble/' imgname '.png']);
    fprintf("image %d %.1f percents of pixels labeled\n", n, sum(gt(:)~=255) / numel(gt(:))*100);
end
toc