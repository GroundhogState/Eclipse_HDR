
%  Try open a CR2?

clear all
outpath = 'C:\Users\jaker\Pictures\Oregon_Eclipse\code\matlab\out\raw';
rawdir = 'C:\Users\jaker\Pictures\Oregon_Eclipse\raw';
pngdir = 'C:\Users\jaker\Pictures\Oregon_Eclipse\16bit\png';
% offsets = load(fullfile('C:\Users\jaker\Pictures\Oregon_Eclipse\code\matlab\out\',offsets.mat'));
% etimes = exposure_times();
% fnames = get_files(rawdir,'CR2');
% file_idxs = cell2mat(cellfun(@(x) str2double(x(7:8)),fnames,'uni',0));

% %
this_img.img = imread(fullfile(rawdir,'IMG_0587.tif'))
% this_img.img = imread(fullfile(pngdir,'IMG_0584.png'));
cli_header('cr2 loaded');
%%
this_img.snip = this_img.img(2000:3400,2000:3000,:);

stfig('Raw edit');
clf
imagesc(this_img.img)
daspect([1,1,1])
stfig('bits');
clf
for i=1:3
    subplot(2,2,i)
    imagesc(this_img.img(:,:,i))
    daspect([1,1,1])
end
subplot(2,2,4)
imagesc(this_img.img)
daspect([1,1,1])
colormap(plasma)
daspect([1,1,1])   
%%

 
% subplot(2,1,2)
% imagesc(snip)
% daspect([1,1,1])
% %
% rows = 2681:2688; % hot pixel
% cols = 2449:2456;
rows = 1681:3388; % disk img 0584
cols = 449:3456;

% rows = 2701:2718; % hot pixel
% cols = 2449:2460;
% rows = 2810:2900; % img_0589
% cols = 1300:1500;
% rows = (1344:1360); % red bit?
% cols = (2750:2772); 
options.unit_bins = false;
options.min_val = 1;
options.max_val = 2^16;
options.num_bins = options.max_val+1;
imcounts = rgb_hist(2^16*im2double(this_img.img(rows,cols,:)),options);
% for row = 2680:2694
%     for col = 2441:2465
stfig('flipbits');
clf
subplot(2,2,1)
imagesc(cols,rows,this_img.img(rows,cols,:))
subplot(2,2,2)
hold on
plot(imcounts.bins,imcounts.red.counts,'r')
plot(imcounts.bins,imcounts.blue.counts,'g')
plot(imcounts.bins,imcounts.green.counts,'b')
subplot(2,2,3)
surf(cols,rows,this_img.img(rows,cols,1),'EdgeAlpha',0)
subplot(2,2,4)
surf(cols,rows,this_img.img(rows,cols,2)-(this_img.img(rows,cols,1)),'EdgeAlpha','0')
colormap(viridis)
%     end
% end
% %
imdiff = this_img.img(rows,cols,3)-this_img.img(rows,cols,1);
% imcounts=rgb_hist(2^16*im2double(imdiff),options);
bin_edges = (0:1:10)-.5;
diff.counts.rg= histcounts(this_img.img(rows,cols,1)-this_img.img(rows,cols,2),bin_edges);
diff.counts.gb = histcounts(this_img.img(rows,cols,2)-this_img.img(rows,cols,3),bin_edges);
diff.counts.br = histcounts(this_img.img(rows,cols,3)-this_img.img(rows,cols,1),bin_edges);
diff.bins = 0.5*(bin_edges(2:end) + bin_edges(1:end-1));

options.label = 'diff';
stfig('diff')
subplot(2,2,1)
imagesc(cols,rows,imdiff)
title('blue-red')
subplot(2,2,2)
imagesc(cols,rows,this_img.img(rows,cols,3)-this_img.img(rows,cols,2))
title('blue-green')
subplot(2,2,3)
imagesc(cols,rows,this_img.img(rows,cols,1)-this_img.img(rows,cols,3))
title('red-green')
subplot(2,2,4)
hold on
plot(this_img.img(rows,cols,1)-this_img.img(rows,cols,3))
set(gca,'Yscale','log')

% subplot(2,2,3)
% imagesc(cols,rows,this_img.img(rows,cols,1)-this_img.img(rows,cols,2))
% subplot(2,2,4)
% imagesc(cols,rows,this_img.img(rows,cols,2)-this_img.img(rows,cols,3))
colormap(viridis)
cli_header('Plots done');
% [this_img.img(rows,cols,1),zeros(length(rows),1),this_img.img(rows,cols,2),zeros(length(rows),1),this_img.img(rows,cols,3)]