cli_header(1,'Running darkfield...');
dirname = 'C:\Users\jaker\Pictures\Oregon_Eclipse\darkfield\png';
fnames = get_files(dirname,'png');
% imgs = cellfun(@(x) imread(fullfile(dirname,x)),fnames,'uni',0);
cli_header('Done.');
num_imgs = length(fnames);
%%
% num_bins = 1e3;
% bin_edges = linspace(0,1e4,num_bins+1);
num_bins = 2^8;
bin_edges = logspace(0,4,num_bins+1);


tally.red = nan(num_imgs,num_bins);
tally.green = nan(num_imgs,num_bins);
tally.blue = nan(num_imgs,num_bins);

cli_header(2,'Loading images...');
n_img = 1;

stfig('Darkfield analysis');
clf;
imcounter = 0;
for i=1:2
    cli_header(3,'%u/%u...',i,num_imgs);
    imcounter = imcounter + 1;
    this_dark=[];
    this_dark_img = imread(fullfile(dirname,fnames{i}));
%     this_dark.im g g 
    this_dark.grey = rgb2gray(this_dark_img);
    if i==1
        mean_img = this_dark_img;
        sqr_img = im2double(this_dark_img.^2);
        fprintf('New img, %u, %.0f\n',max(this_dark_img,[],'all'),max(mean_img,[],'all'));
    else
        mean_img = mean_img+this_dark_img;
        sqr_img = sqr_img + im2double(this_dark_img).^2;
        fprintf('Added img, %u, %.0f\n',max(this_dark_img,[],'all'),max(mean_img,[],'all'));
    end
    
    this_dark.red.img=squeeze(this_dark_img(:,:,1));
    this_dark.green.img=squeeze(this_dark_img(:,:,2));
    this_dark.blue.img=squeeze(this_dark_img(:,:,3));

    [this_dark.red.counts,~] = histcounts(this_dark.red.img,bin_edges);
    [this_dark.green.counts,~] = histcounts(this_dark.green.img,bin_edges);
    [this_dark.blue.counts,~] = histcounts(this_dark.blue.img,bin_edges); 
    
    tally.red(i,:) = this_dark.red.counts;
    tally.green(i,:) = this_dark.green.counts;
    tally.blue(i,:) = this_dark.blue.counts;
    hist_centres = 0.5*(bin_edges(1:end-1)+bin_edges(2:end));
    
    
    
    subplot(3,2,1)
    hold on
    plot(hist_centres,this_dark.red.counts,'r')
    plot(hist_centres,this_dark.green.counts,'g')
    plot(hist_centres,this_dark.blue.counts,'b')
    subplot(3,2,2)
    hold on
    plot(hist_centres,this_dark.red.counts,'r')
    plot(hist_centres,this_dark.green.counts,'g')
    plot(hist_centres,this_dark.blue.counts,'b') 
    set(gca,'Yscale','log')
    set(gca,'Xscale','log')
    drawnow
    
%     mean_all = histcounts(mean_img(:,:,1)/imcounter,bin_edges);
%     subplot(3,2,6)
%     plot(hist_centres,mean_all,'r')
%     ylim([0,1e7])
end
% clear this_dark
%%
% mean_img = mean_img;
cli_header(2,'Plotting.');
[meanstats.r,~] = histcounts(mean_img(:,:,1)/imcounter,bin_edges);
meanstats.g = histcounts(mean_img(:,:,2)/imcounter,bin_edges);
meanstats.b = histcounts(mean_img(:,:,3)/imcounter,bin_edges);

% end
subplot(3,2,3)
cla
hold on
plot(hist_centres,meanstats.r,'r')
plot(hist_centres,meanstats.g,'g')
plot(hist_centres,meanstats.b,'b')
title('Mean')
xlabel('Intensity')
ylabel('Counts')


% colormap(viridis)
subplot(3,2,4)
cla
hold on
plot(hist_centres,meanstats.r,'r')
plot(hist_centres,meanstats.g,'g')
plot(hist_centres,meanstats.b,'b')
xlabel('Intensity')
ylabel('Counts')
title('Mean log')
set(gca,'Xscale','log')
set(gca,'Yscale','log')


cli_header(2,'Done.');
%%
mean_img = mean_img/imcounter;
cli_header(1,'Saving output...');
outpath = 'C:\Users\jaker\Pictures\Oregon_Eclipse\code\matlab\out';
save(fullfile(outpath,'darkfield_out.mat'),'mean_img')
% save(fullfile(outpath,'darkfield_out'),'mean_img','.PNG')
% std_img = 2^32*sqr_img/num_imgs - mean_img.^2;
cli_header(2,'Done.');
%%
% stfig('Darkfield analysis');
% clf;

% % %
stats.red.mean = nanmean(tally.red);
stats.red.std = nanstd(tally.red);
stats.blue.mean= nanmean(tally.blue);
stats.blue.std = nanstd(tally.blue);
stats.green.mean= nanmean(tally.green);
stats.green.std = nanstd(tally.green);
subplot(3,2,5)
hold on
cla
plot(hist_centres,stats.red.mean,'r')
plot(hist_centres,stats.red.mean+stats.red.std,'r:')
plot(hist_centres,stats.red.mean-stats.red.std,'r:')
plot(hist_centres,stats.green.mean,'g')
plot(hist_centres,stats.green.mean+stats.green.std,'g:')
plot(hist_centres,stats.green.mean-stats.green.std,'g:')
plot(hist_centres,stats.blue.mean,'b')
plot(hist_centres,stats.blue.mean+stats.blue.std,'b:')
plot(hist_centres,stats.blue.mean-stats.blue.std,'b:')
xlim([1,1e4])
ylim([1,2e5])
xlabel('Intensity')
ylabel('Counts')

subplot(3,2,6)
hold on
plot(hist_centres,stats.red.mean,'r')
plot(hist_centres,stats.red.mean+stats.red.std,'r:')
plot(hist_centres,stats.red.mean-stats.red.std,'r:')
plot(hist_centres,stats.green.mean,'g')
plot(hist_centres,stats.green.mean+stats.green.std,'g:')
plot(hist_centres,stats.green.mean-stats.green.std,'g:')
plot(hist_centres,stats.blue.mean,'b')
plot(hist_centres,stats.blue.mean+stats.blue.std,'b:')
plot(hist_centres,stats.blue.mean-stats.blue.std,'b:')
xlim([1,1e4])
ylim([1,2e5])
xlabel('Intensity')
ylabel('Counts')
set(gca,'Yscale','log')
set(gca,'Xscale','log')

suptitle('Dark field intensity histograms')

% % Check correlations between images? Thought with that variation prob not much...



% stfig('Correlations and stationarity in noise');
% clf


%%
% To a real extent, the mean image is only useful if there is persistent
% structure in the pixel noise. Otherwise, you'll just have to estimate the
% noise floor and subtract that off, which is a bit less satisfying.
% So you need to know: Is this noise, or structured noise? Given one image,
% can you make predictions about the rest? There are some sophisticated
% tests one could do. A simple one would be to check the correlation
% between two images:
% corrs.normal=[];
% corrs.invert=[];
% corrs.translate=[];
% corrs.flip=[];
% c = 0;
% for i = 1:5
%     i
% 	im1 = imread(fullfile(dirname,fnames{i}));
%     for j=i+1:i+5
%         c = c+1;
%         j
%     im2 = imread(fullfile(dirname,fnames{j}));
%     corrs.normal(c,:)= [corr2(squeeze(im1(:,:,1)),squeeze(im2(:,:,1))),...
%              corr2(squeeze(im1(:,:,2)),squeeze(im2(:,:,2))),...
%              corr2(squeeze(im1(:,:,3)),squeeze(im2(:,:,3)))];
%     corrs.flip(c,:) = [corr2(squeeze(im1(:,:,1)),fliplr(squeeze(im2(:,:,1)))),...
%             corr2(squeeze(im1(:,:,2)),fliplr(squeeze(im2(:,:,2)))),...
%             corr2(squeeze(im1(:,:,3)),fliplr(squeeze(im2(:,:,3))))];
%         corrs.invert(c,:) = [corr2(squeeze(im1(:,:,1)),flipud(squeeze(im2(:,:,1)))),...
%             corr2(squeeze(im1(:,:,2)),flipud(squeeze(im2(:,:,2)))),...
%             corr2(squeeze(im1(:,:,3)),flipud(squeeze(im2(:,:,3))))];
%         corrs.translate(c,:) = [corr2(squeeze(im1(:,1:end-1,1)),flipud(squeeze(im2(:,2:end,1)))),...
%             corr2(squeeze(im1(:,1:end-1,2)),flipud(squeeze(im2(:,2:end,2)))),...
%             corr2(squeeze(im1(:,1:end-1,3)),flipud(squeeze(im2(:,2:end,3))))];
%     end
% end
% % after checking a couple it's pretty clear all the channels are
% % correlated (having used the flipped image as a proxy for a random image
% % with the same noise statistics). If they were both strongly random, then
% % the correlation would be comparable between the two. Instead it's about
% % 100x stronger.
% %%
% stfig('corrs');
% clf;
% subplot(3,1,1)
% hold on
% plot(corrs.normal(:,1),'rx')
% plot(corrs.flip(:,1),'ro')
% plot(corrs.invert(:,1),'rv')
% plot(corrs.translate(:,1),'r+')
% ylim([-.02,.04])
% legend('Inter-image','Intra-image (flipped LR)','intra-image flip UD','translate 1px')
% subplot(3,1,2)
% hold on
% plot(corrs.normal(:,2),'gx')
% plot(corrs.flip(:,2),'go')
% plot(corrs.invert(:,2),'gv')
% plot(corrs.translate(:,2),'g+')
% ylim([-.02,.04])
% legend('Inter-image','Intra-image (flipped LR)','intra-image flip UD','translate 1px')
% subplot(3,1,3)
% hold on
% plot(corrs.normal(:,3),'bx')
% plot(corrs.flip(:,3),'bo')
% plot(corrs.invert(:,3),'bv')
% plot(corrs.translate(:,3),'b+')
% legend('Inter-image','Intra-image (flipped LR)','intra-image flip UD','translate 1px')
% ylim([-.02,.04])
% suptitle('Correlation: Intra-image vs automorphism')
% % The separation between these guys is interesting!
% Seems concordant with column-readout noise, plus individual pixel noise.
% Ergo *yes* there is indeed structure in the images - and this should be
% testable - by translating the images just a couple of pixels, one should
% kill off the correlation function, right?
%%



%%
% function iplot(xdata,ydata,udata,color)
%     plot(xdata,ydata,color)
%     
% 
% end


