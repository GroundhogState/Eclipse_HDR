clear all
eclipse_init
%%
% Hotpix correlation
% Let's load up a test CR2 image and pick a hotspot. It should look like an x

im_idx = 6;
img.name = fnames{im_idx};
img.hotpix = hotpix.(img.name(1:end-4));
img.raw = imread(fullfile(raw_dir,img.name));


%%

% find the common hotspots
stfig('view hotspots');
clf

view_half_range = [15,15];
colormap(inferno)
chan='r';
for i=1:length(img.hotpix.(chan))
    view_centre = img.hotpix.(chan)(i,:);
    view_area = [view_centre-view_half_range;view_centre+view_half_range];
    subplot(2,2,1)
    imagesc(img.raw(view_area(1,1):view_area(2,1),view_area(1,2):view_area(2,2),:))
    for j=1:3
       subplot(2,2,j+1)
       imagesc(img.raw(view_area(1,1):view_area(2,1),view_area(1,2):view_area(2,2),j))
    end
    drawnow   
    suptitle(sprintf('hotspot %u',i))
    pause(0.3)
end


%%
% %
% % eg (2709        2454)

view_centre = [2685,2453];
view_half_range = [15,15];
view_area = [view_centre-view_half_range;view_centre+view_half_range];
view_centre_2 = [2837,2365];
view_area_2 = [view_centre_2-view_half_range;view_centre_2+view_half_range];
% view_area = [2833,2361;2841,2369];
test_area = view_area + 30*[-1,-1;1,1];

img.crop = img.raw(view_area(1,1):view_area(2,1),view_area(1,2):view_area(2,2),:);
img.crop_2 = img.raw(view_area_2(1,1):view_area_2(2,1),view_area_2(1,2):view_area_2(2,2),:);

% img.filter = normxcorr2(rgb2gray(img.crop),rgb2gray(img.raw));
% img.filter_crop = img.filter(test_area(1,1):test_area(2,1),test_area(1,2):test_area(2,2),:);

corr_peaks = find(img.filter>0.9);
length(corr_peaks);

cli_header(2,'Calcs done');


% %
stfig('hotpix view');
clf
tiledlayout(2,3,'TileSpacing','Compact')

nexttile
hold on
imagesc(img.raw)
plot([view_area(1,2),view_area(1,2),view_area(2,2),view_area(2,2),view_area(1,2)],...
    [view_area(1,1),view_area(2,1),view_area(2,1),view_area(1,1),view_area(1,1)],'r','LineWidth',3)
daspect([1,1,1])
title('Raw image')

nexttile
hold on
surf(view_area(1,2):view_area(2,2),view_area(1,1):view_area(2,1),rgb2gray(img.crop))
colormap(inferno)
daspect([1,1,1])
title('view area')

nexttile
hold on
surf(view_area_2(1,2):view_area_2(2,2),view_area_2(1,1):view_area_2(2,1),rgb2gray(img.crop_2))
colormap(inferno)
daspect([1,1,1])
title('view area')

nexttile
plot(rescale(rgb2gray(img.crop)))

nexttile
plot(rescale(rgb2gray(img.crop_2)))
% imshow(img.filter) 
% hold on
% plot([view_area(1,2),view_area(1,2),view_area(2,2),view_area(2,2),view_area(1,2)],...
%     [view_area(1,1),view_area(2,1),view_area(2,1),view_area(1,1),view_area(1,1)],'r','LineWidth',3)
% title('Correlation ')


% nexttile
% surf(img.filter_crop) 
% title('Correlation crop')

% nexttile

cli_header(2,'Plots done');

% %
zone_1 = [2685,2687;2451,2453];
zone_2 = [2834,2836;2362,2364];
img.zone1 = img.raw(zone_1(1,1):zone_1(1,2),zone_1(2,1):zone_1(2,2),:);
img.zone2 = img.raw(zone_2(1,1):zone_2(1,2),zone_2(2,1):zone_2(2,2),:);

stfig('Tight');
clf
tiledlayout(2,2)
nexttile
imagesc(img.zone1)
nexttile
imagesc(img.zone2)

z1=(rgb2gray(img.raw(zone_1(1,1):zone_1(1,2),zone_1(2,1):zone_1(2,2),:)));
z2=(rgb2gray(img.raw(zone_2(1,1):zone_2(1,2),zone_2(2,1):zone_2(2,2),:)));
z = .5*(z1+z2);


z1x = normxcorr2(double(z),double(rgb2gray(img.crop)));
z2x = normxcorr2(double(z),double(rgb2gray(img.crop_2)));

nexttile
plot(z1x)

nexttile
plot(z2x)


%% Look in the whole image...
zfull = normxcorr2(rescale(z),double(rgb2gray(img.raw)));

peak_loc = find(zfull>.8);

stfig('Full find');
tiledlayout(2,2)

nexttile
imagesc(zfull>.8)
hold on
for idx=1:length(peak_loc)
   [row,col] = ind2sub(size(img.raw,1:2),peak_loc(idx));
   [row,col]
   plot(row,col,'rx')
end

nexttile
histogram(zfull)
set(gca,'Yscale','log')
ylim([0.5,1e6])
cli_header('Full search done');