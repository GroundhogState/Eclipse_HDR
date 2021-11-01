% % % exposure times
% % % If you haven't already, run:
% clear all
% get_img_translations
% darkfield
%%
% retrieve centre offsets and mean image
clear all
outpath = 'C:\Users\jaker\Pictures\Oregon_Eclipse\code\matlab\out';
cli_header(2, 'Reading images');
dirname = 'C:\Users\jaker\Pictures\Oregon_Eclipse\16bit\png';
fnames = get_files(dirname,'png');
imgs = cellfun(@(x) imread(fullfile(dirname,x)),fnames,'uni',0);
etimes = exposure_times();
dark_mean= load(fullfile(outpath,'darkfield_out.mat'));
offsets = load(fullfile(outpath,'offsets.mat'));
cli_header(2,'Loaded up');

%%
% -----   Noise floor cutoffs for 0.999 percentile:
% -------   Red:   0.0554
% -------   Green: 0.0192
% -------   Blue:  0.0243

num_imgs = length(imgs);
num_pixels = 3457*5194;
mean_dark = im2double(dark_mean.mean_img);

% imagesc(sum_img+imtranslate(imgs{2},offsets.full(1,:)))
net_offset = [0,0];

num_bins = 2^8;
bin_edges = logspace(0,log10(2^16),num_bins+1);
bin_centres = 0.5*(bin_edges(1:end-1) + bin_edges(2:end));

[dark.red.counts,dark_edges] = histcounts(mean_dark(:,:,1),num_bins);
dark.green.counts = histcounts(mean_dark(:,:,2),num_bins);
dark.blue.counts = histcounts(mean_dark(:,:,3),num_bins);
dark_bins = 0.5*(dark_edges(2:end)+dark_edges(1:end-1));

file_idxs = cell2mat(cellfun(@(x) str2double(x(7:8)),fnames,'uni',0));

h=3;
w=3;
cli_header('Init fig');
stfig('Combining images');
clf

subplot(h,w,1)
cla
hold on
imagesc(rescale(mean_dark))
title('Scaled darkfield')

subplot(2*h,2*w,3)
cla
hold on
plot(dark_bins,dark.red.counts,'r')
plot(dark_bins,dark.green.counts,'g')
plot(dark_bins,dark.blue.counts,'b')
% xlim([0,2^16])
% ylim([0,num_pixels])
title('Dark image histograms')
subplot(2*h,2*w,4)
cla
hold on
plot(dark_bins,dark.red.counts,'r')
plot(dark_bins,dark.green.counts,'g')
plot(dark_bins,dark.blue.counts,'b')
% xlim([0,2^16])
% ylim([0.1,num_pixels])
title('Dark image histograms')
set(gca,'Yscale','log')

% %
% When adding uint16s, they saturate at 2^16, don't play nicely. Try
% converting to double in-the-loop
cli_header(2,'Compiling...');
checkval = 1;
for idx= 1:15
    % get next image and align it
    % uses offsets cen_2 = cen_1 + offsets_1     
    this_img  = im2double(imgs{idx});
    this_file_idx = file_idxs(idx);
    this_exposure_time = etimes(etimes(:,1) == this_file_idx,2);
    
    if checkval==1
        checkval = 0;
        net_offset = [0,0];
        img_dark_corrected = this_img-im2double(mean_dark);
        num_samples = img_dark_corrected>0;
        signal = img_dark_corrected;
        signal(signal<0) = 0;
%         signal = signal/this_exposure_time;
        sum_img = signal;
    else
        net_offset = net_offset + offsets.offsets.full(idx-1,:);
        img_dark_corrected = this_img-im2double(mean_dark);
        above_noise_floor = img_dark_corrected>0;
        num_samples = num_samples + imtranslate(above_noise_floor,net_offset);
        signal = img_dark_corrected;
        signal(signal<0) = 0 ;
%         signal = signal/this_exposure_time;
        sum_img = sum_img + imtranslate(signal,net_offset);
    end
    
    
    [bright.red.counts,bright_edges] = histcounts(this_img(:,:,1),num_bins);
    bright.green.counts = histcounts(this_img(:,:,2),num_bins);
    bright.blue.counts = histcounts(this_img(:,:,3),num_bins);
    bright_bins = 0.5*(bright_edges(2:end)+bright_edges(1:end-1));
    
    [dark_sub.red.counts,dark_sub_edges] = histcounts(img_dark_corrected(:,:,1),num_bins);
    dark_sub.green.counts = histcounts(img_dark_corrected(:,:,2),num_bins);
    dark_sub.blue.counts = histcounts(img_dark_corrected(:,:,3),num_bins);
    dark_sub_bins = 0.5*(dark_sub_edges(2:end)+dark_sub_edges(1:end-1));
    
    [flux.red.counts, flux_edges]= histcounts(sum_img(:,:,1),num_bins);
    flux.green.counts = histcounts(sum_img(:,:,2),num_bins);
    flux.blue.counts = histcounts(sum_img(:,:,3),num_bins);
    flux_bins = 0.5*(flux_edges(2:end)+flux_edges(1:end-1));
    
    
    
    subplot(h,w,3)
    cla
    imagesc(imgs{idx})
    title('Current bright image')
    drawnow
    
    subplot(2*h,2*w,9)
    cla
    hold on
    plot(bright_bins,bright.red.counts,'r')
    plot(bright_bins,bright.green.counts,'g')
    plot(bright_bins,bright.blue.counts,'b')
    title('Bright image histograms')
    subplot(2*h,2*w,10)
    cla
    hold on
    plot(bright_bins,bright.red.counts,'r')
    plot(bright_bins,bright.green.counts,'g')
    plot(bright_bins,bright.blue.counts,'b')
    title('Bright log histograms')
    set(gca,'Yscale','log')
    
    subplot(2*h,2*w,15)
    cla
    hold on
    plot(dark_sub_bins,dark_sub.red.counts,'r')
    plot(dark_sub_bins,dark_sub.green.counts,'g')
    plot(dark_sub_bins,dark_sub.blue.counts,'b')
    title('Darkfielded image histograms')
    subplot(2*h,2*w,16)
    cla
    hold on
    plot(dark_sub_bins,dark_sub.red.counts,'r')
    plot(dark_sub_bins,dark_sub.green.counts,'g')
    plot(dark_sub_bins,dark_sub.blue.counts,'b')
    title('Darkfielded log histograms')
    set(gca,'Yscale','log')
    
    
    subplot(h,w,4)
    cla
    imagesc(img_dark_corrected)
    title('Current minus dark')
    
    subplot(h,w,6)
    cla
    imagesc(sum_img)
    title('Cumulative image')
    
    subplot(2*h,2*w,21)
    cla
    hold on
    plot(flux_bins,flux.red.counts,'r')
    plot(flux_bins,flux.green.counts,'g')
    plot(flux_bins,flux.blue.counts,'b')
    title('Sum image histograms')
    subplot(2*h,2*w,22)
    cla
    hold on
    plot(flux_bins,flux.red.counts,'r')
    plot(flux_bins,flux.green.counts,'g')
    plot(flux_bins,flux.blue.counts,'b')
%     xlim([0,2^16])
    title('sum log histograms')
    set(gca,'Yscale','log')
    
    subplot(h,w,7)
    cla
    imagesc(num_samples)
    title('Num samples above noise floor')
    
    subplot(h,w,8)
    cla
    imagesc(signal)
    title('This extracted signal')
end

cli_header(2,'Complete.');

%%
stfig('Final image')
clf
subplot(2,2,1)
imagesc(sum_img)
subplot(2,2,2)
imagesc(sum_img<0)
% profile off
% profile viewer
%%


