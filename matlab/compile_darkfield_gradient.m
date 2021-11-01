% Let's have a look at a few images in detail to figure out how to
% noise-correct this thing...

% Initialize and load some files
clear all
outpath = 'C:\Users\jaker\Pictures\Oregon_Eclipse\code\matlab\out';
dirname = 'C:\Users\jaker\Pictures\Oregon_Eclipse\16bit\png';
offsets = load(fullfile(outpath,'offsets.mat'));
etimes = exposure_times();
%% Import the signal images
fnames = get_files(dirname,'png');
cli_header(2, 'Reading images');
imgs = cellfun(@(x) imread(fullfile(dirname,x)),fnames,'uni',0);
file_idxs = cell2mat(cellfun(@(x) str2double(x(7:8)),fnames,'uni',0));

% Load the dark image

clear dark_mean;
num_pixels = 3457*5194;
sensor.height= 3457;
sensor.width= 5194;
num_bins = 2^6;
num_imgs = length(fnames);
cli_header(2,'Loaded up');

%% darkfield correction
dark = [];
dark_mean= load(fullfile(outpath,'darkfield_out.mat'));
dark.mean_dark = im2double(dark_mean.mean_img);
dark.noise.cutoff = 0*[.99,.99,.99];
dark.hot.cutoff = [.1,.05,.05];
dark.viz = true;
dark.num_bins = 2^8;
dark = dark_noise_correct(dark);


% A dumb version. Kill everything below the ~90th percentile of noise. Then
% subtract the noise floor so no small values are left. IE use pixels with
% SNR of better than e.g. 3?
% Let's see what those values are:

% dark.noise.cutoff.red = 0.;
% dark.noise.cutoff.green = 0.;
% dark.noise.cutoff.blue = 0.;


%% Image manipulation
% -----   Noise floor cutoffs for 0.999 percentile:

% cutoff.red = dark_bins(dark.red.cutoff_idx);
% cutoff.green = dark_bins(dark.green.cutoff_idx);
% cutoff.blue = dark_bins(dark.blue.cutoff_idx);
% cutoff.max = 0.98;
% net_offset = [0,0];
% ctr = 0;
% 
% proc_plots = 0;
% for im_idx = 6:num_imgs
%     ctr = ctr + 1;
%     cli_header('Loading image %u',im_idx);
% %     this_img = im2double(imread(fullfile(dirname,fnames{im_idx})));   
% %     this_img = im2double(imgs{im_idx});
%     
%     signal.red = squeeze(im2double(imgs{im_idx}(:,:,1)));
%     signal.green = squeeze(im2double(imgs{im_idx}(:,:,2)));
%     signal.blue = squeeze(im2double(imgs{im_idx}(:,:,3)));
% 
%     % thresholding
%     signal.red(signal.red<cutoff.red| signal.red > cutoff.max) = 0;
%     signal.blue(signal.blue<cutoff.blue | signal.blue > cutoff.max) = 0;
%     signal.green(signal.green<cutoff.green| signal.green > cutoff.max) = 0;
% 
%     this_file_idx = file_idxs(im_idx);
%     exp_time = etimes(etimes(:,1) == this_file_idx,2);
%     if ctr == 1
%         signal_rgb = cat(3,signal.red,signal.green,signal.blue)-0*mean_dark;
% %         clear signal
%         signal_rgb(signal_rgb<0) = 0;
%         signal_rgb = signal_rgb/exp_time;
%         num_samples = signal_rgb>0;
%     else
%         net_offset = net_offset + offsets.offsets.full(im_idx-1,:);
%         new_rgb = cat(3,signal.red,signal.green,signal.blue)-mean_dark;
% %         clear signal
%         new_rgb(new_rgb<0) = 0;
%         signal_rgb = signal_rgb+imtranslate(new_rgb/exp_time,net_offset);
%         clear new_rgb
%         num_samples = num_samples+(signal_rgb>0);
%     end
% %     num_samples = signal_rgb>0;
% 
% %     if proc_plots
% %         [bright.red.counts,bright_edges] = histcounts(signal.red,num_bins);
% %         bright.green.counts = histcounts(signal.green,num_bins);
% %         bright.blue.counts = histcounts(signal.blue,num_bins);
% %         bright_bins = 0.5*(bright_edges(2:end)+bright_edges(1:end-1));
% % 
% %         [flux.red.counts, final.edges]= histcounts(signal_rgb(:,:,1),num_bins);
% %         flux.green.counts = histcounts(signal_rgb(:,:,2),num_bins);
% %         flux.blue.counts = histcounts(signal_rgb(:,:,3),num_bins);
% %         final.bins = 0.5*(final.edges(2:end)+final.edges(1:end-1));
% % 
% % 
% %         cli_header('Init fig');
% % 
% %         h=2;
% %         w=2;
% %         stfig('Images');
% %         clf
% %         subplot(h,w,1)
% %         cla
% %         imagesc(imgs{im_idx})
% %         title('Current bright image')
% % 
% %         subplot(h,w,2)
% %         cla
% %         imagesc(signal_rgb./im_idx)
% %         colormap(plasma)
% %         title('Scaled signal')
% % 
% %         subplot(h,w,3)
% %         imagesc(rescale(num_samples))
% %         title('Num samples')
% % 
% %         subplot(h,w,4)
% %     %     imagesc((signal_rgb))
% %         drawnow
% % 
% % 
% % 
% %         stfig('Histograms');
% %         clf
% %         h=2;    
% %         w=3;
% %         subplot(h,w,1)
% %         cla
% %         hold on
% %         plot(dark_bins,dark.red.counts,'r:')
% %         plot(dark_bins,dark.green.counts,'g:')
% %         plot(dark_bins,dark.blue.counts,'b:')
% %         plot(bright_bins,bright.red.counts,'r')
% %         plot(bright_bins,bright.green.counts,'g')
% %         plot(bright_bins,bright.blue.counts,'b')
% %         title('Bright \& Dark image histograms')
% % 
% %         subplot(h,w,2)
% %         cla
% %         hold on
% %         plot(dark_bins,dark.red.counts,'r:')
% %         plot(dark_bins,dark.green.counts,'g:')
% %         plot(dark_bins,dark.blue.counts,'b:')
% %         plot(bright_bins,bright.red.counts,'r')
% %         plot(bright_bins,bright.green.counts,'g')
% %         plot(bright_bins,bright.blue.counts,'b')
% %         title('Bright \& Dark image histograms')
% %         set(gca,'Yscale','log')
% % 
% %         subplot(h,w,3)
% %         cla
% %         hold on
% %         plot(dark_bins,dark.red.counts,'r:')
% %         plot(dark_bins,dark.green.counts,'g:')
% %         plot(dark_bins,dark.blue.counts,'b:')
% %         plot(bright_bins,bright.red.counts,'r')
% %         plot(bright_bins,bright.green.counts,'g')
% %         plot(bright_bins,bright.blue.counts,'b')
% %         title('Bright \& Dark image histograms')
% %         set(gca,'Yscale','log')
% %         set(gca,'Xscale','log')
% % 
% %         subplot(h,w,4)
% %         cla
% %         hold on
% %         plot(final.bins,flux.red.counts,'r')
% %         plot(final.bins,flux.green.counts,'g')
% %         plot(final.bins,flux.blue.counts,'b')
% %         title('Flux histograms')
% %         subplot(h,w,5)
% %         cla
% %         hold on
% %         plot(final.bins,flux.red.counts,'r')
% %         plot(final.bins,flux.green.counts,'g')
% %         plot(final.bins,flux.blue.counts,'b')
% %         title('Flux log histograms')
% %         set(gca,'Yscale','log')
% %         drawnow
% %     end
% % 
% end
% 
% cli_header(2,'Plotting done.');
%%
% clear imgs
% clear mean_dark
% clear signal
% final = [];
% cli_header(2,'Compiling registered image');
% % final.img = imgaussfilt(signal_rgb./(ctr*num_samples),1);
% final.img = signal_rgb./(ctr*num_samples);
% final.img(isnan(final.img)) = 0;
% for i=1:3
%     final.img(:,:,i) = imgaussfilt(final.img(:,:,i),5);
% end
% 
% % num_final.bins = 2^8;
% % num_final.bins = logspace(-5,1,300);    
% % [final.red.counts, final.edges]= histcounts(final.img(:,:,1),num_final.bins);
% % final.green.counts = histcounts(final.img(:,:,2),num_final.bins);
% % final.blue.counts = histcounts(final.img(:,:,3),num_final.bins);
% % final.bins = 0.5*(final.edges(2:end)+final.edges(1:end-1));
% % 
% % final.cutoff.red=1e-4;
% % final.cutoff.green=1e-3;
% % final.cutoff.blue=1e-3;
% % 
% % stfig('Final img');
% % clf
% % subplot(2,3,1)
% % imagesc(imresize(final.img,0.5))
% % 
% % subplot(2,3,2)
% % imagesc((num_samples))
% % title('Pixels with data')
% % 
% % subplot(2,3,3)
% % imagesc(rescale(num_samples))
% % title('Number of samples')
% % 
% % subplot(4,2,6)
% % cla
% % hold on
% % plot(final.bins,final.red.counts,'r')
% % plot(final.bins,final.green.counts,'g')
% % plot(final.bins,final.blue.counts,'b')
% % set(gca,'Yscale','log')
% % title('Flux histograms')
% % subplot(4,2,8)
% % cla
% % hold on
% % plot(final.bins,final.red.counts,'r')
% % plot(final.bins,final.green.counts,'g')
% % plot(final.bins,final.blue.counts,'b')
% % title('Flux log histograms')
% % set(gca,'Yscale','log')
% % set(gca,'Xscale','log')
% % 
% % stfig('Final img diagnostics');
% % clf
% % subplot(2,3,1)
% % imagesc(final.img(:,:,1)<final.cutoff.red)
% % title('Red below cutoff')
% % subplot(2,3,2)
% % imagesc(final.img(:,:,2)<final.cutoff.green)
% % title('green below cutoff')
% % subplot(2,3,3)
% % imagesc(final.img(:,:,3)<final.cutoff.blue)
% % colormap(plasma)
% % title('blue below cutoff')

% %% After cutoff
% % cutoff.register = 1e-4;
% % cut.img = rescale(final.img.*(final.img > cutoff.register)/cutoff.register+1);
% % clear final signal_rgb num_samples
% % clear grad_dir imgrad out_img r_grad r_polar th_grad th_polar X x_polar Y y_polar grad_dir_uint grad_sense num_samples
% % 
% % 
% % % cut.img(cut.img <;
% % 
% % cut.max_val = max(cut.img,[],'all');
% % cut.min_val = min(cut.img,[],'all');
% % cut.bins = logspace(log10(cut.min_val+1e-3),log10(cut.max_val),300);    
% % [cut.red.counts, cut.edges]= histcounts(cut.img(:,:,1),cut.bins);
% % cut.green.counts = histcounts(cut.img(:,:,2),cut.bins);
% % cut.blue.counts = histcounts(cut.img(:,:,3),cut.bins);
% % cut.bins = 0.5*(cut.edges(2:end)+cut.edges(1:end-1));
% % 
% % 
% % f_out=stfig('cutoff img');
% % clf
% % subplot(2,3,1)
% % imagesc((cut.img))
% % subplot(4,2,2)
% % cla
% % hold on
% % plot(cut.bins,cut.red.counts,'r')
% % plot(cut.bins,cut.green.counts,'g')
% % plot(cut.bins,cut.blue.counts,'b')
% % set(gca,'Yscale','log')
% % title('Flux histograms')
% % subplot(4,2,4)
% % cla
% % hold on
% % plot(cut.bins,cut.red.counts,'r')
% % plot(cut.bins,cut.green.counts,'g')
% % plot(cut.bins,cut.blue.counts,'b')
% % title('Flux log histograms')
% % set(gca,'Yscale','log')
% % set(gca,'Xscale','log')
% % 
% % cli_header(2,'Plotting done');
% % 
% % 
% % % % Gradient filters
% % close all
% % 
% % cli_header(2,'Applying gradient filters...');
% % 
% % [imgrad,grad_dir] = imgradient(rgb2gray(cut.img));
% % x_len=5194;
% % y_len=3457;
% % x = linspace(1,5194,5194);
% % y = linspace(1,3457,3457);
% % [X,Y] = meshgrid(x,y);
% % % Th = atan2(Y,X);
% % 
% % grad_sense = imgrad>.15;
% % X_min = min(X(grad_sense),[],'all');
% % X_max = max(X(grad_sense),[],'all');
% % Y_min = min(Y(grad_sense),[],'all');
% % Y_max = max(Y(grad_sense),[],'all');
% % X_diam = X_max-X_min;
% % Y_diam = Y_max-Y_min;
% % 
% % x_com=sum(X.*grad_sense,'all')/sum(grad_sense,'all');
% % y_com=sum(Y.*grad_sense,'all')/sum(grad_sense,'all');
% % 
% % x_polar = X - x_com;
% % y_polar = Y - y_com;
% % 
% % r_polar = sqrt(x_polar.^2+y_polar.^2);
% % th_polar = atan2(y_polar,x_polar);
% % 
% % th_grad = sin(pi*grad_dir/180+th_polar).*log(imgrad);
% % r_grad = cos(pi*grad_dir/180+th_polar).*log(imgrad);
% 
% % cli_header('Plotting gradient filters');
% %     figout=stfig('Gradient filters');
% %     clf;
% %     cmap = inferno;
% %     subplot(2,3,1)
% %     imagesc((cut.img))
% %     hold on
% %     plot([x_com,x_com],y_com+Y_diam*[-1,1]/2,'r')
% %     plot(x_com+Y_diam*[-1,1]/2,[y_com,y_com],'r')
% %     plot(x_com,y_com,'rx','MarkerSize',10);
% %     title('Input image')
% %     daspect([1,1,1])
% %     set(gca,'Box','off')
% % 
% % 
% % 
% %     subplot(2,3,2)
% %     imagesc(sqrt(imgrad.*(imgrad<.1)))
% %     title('Grad magnitude')
% %     daspect([1,1,1])
% %     set(gca,'Box','off')
% % 
% %     subplot(2,3,4)
% % 
% %     imagesc((im_spectrum))
% %     title('Fourier transform')
% %     daspect([1,1,1])
% %     set(gca,'Box','off')
% % 
% %     subplot(2,3,3)
% %     imagesc(grad_dir)
% %     title('Grad Direction')
% %     daspect([1,1,1])
% %     set(gca,'Box','off')
% % 
% %     subplot(2,3,5)
% %     imagesc(r_grad.*(grad_sense==0))
% %     title('Radial gradient')
% %     daspect([1,1,1])
% %     set(gca,'Box','off')
% % 
% %     subplot(2,3,6)
% %     imagesc(th_grad.*(grad_sense==0))
% %     title('Polar gradient')
% %     daspect([1,1,1])
% %     set(gca,'Box','off')
% % 
% %     colormap(inferno)
% % cli_header(3,'Done.');

%%
clear r_grad r_polar th_grad th_polar 
clear grad_sense X Y x_polar y_polar

im_spectrum = abs(fft2(cut.img));
im_phase = angle(fft2(cut.img));
% Permute the top and bottom halves
s = size(im_spectrum);
% permute_matrix = im_spectrum([ceil(s(1)/2):s(1),1:floor(s(1)/2)],:,:); 
% permute_matrix = permute_matrix(:,[ceil(s(2)/2):s(2),1:floor(s(2)/2)],:); 
% permute_phase = im_phase([ceil(s(1)/2):s(1),1:floor(s(1)/2)],:,:); 
% permute_phase = permute_phase(:,[ceil(s(2)/2):s(2),1:floor(s(2)/2)],:); 

%%
cli_header('Plotting spectral analysis');
stfig('space spectral analysis');

clf
subplot(2,3,1)
imagesc(cut.img)
title('Input image')

subplot(2,3,2)
imagesc(im_spectrum);
title('Fourier transform')
% 
subplot(2,3,3)
imagesc(abs(permute_matrix))
title('$\lambda$-space')
% 
subplot(2,3,4)
% hold on
surf(log(abs(permute_matrix(1:8:end,1:8:end,1))),'LineStyle','none')
% surf(log(permute_matrix(1:20:end,1:20:end,2)),'LineStyle','none','FaceAlpha',0.4)
% surf(log(permute_matrix(1:20:end,1:20:end,3)),'LineStyle','none','FaceAlpha',0.4)
title('$\lambda$ log-intensity, red channel')


subplot(2,3,5)
imagesc(im_phase)
colormap(plasma)
title('Phase')
% %
subplot(2,3,6)
imagesc(xcorr2_fft(cut.img(1500:3000,1000:3000,1),cut.img(1500:3000,1000:3000,1)))
title('Autocorrelation')
% %
cli_header(2,'Done.');



suptitle('Full image spatial spectral data')

%%
% So what you're looking for is a wavevector expression, right? Well, sort
% of - anyway, first thing would be to take the FT of a patch of sky in the
% corner - that'll catch the blue of the sky, too.

s_region = [2000,3400;
                  4000,4500];

subsample_img = cut.img(s_region(1,1):s_region(1,2),s_region(2,1):s_region(2,2),:);
sample_spectrum = fft2(subsample_img);

s = size(sample_spectrum);
permute_matrix = sample_spectrum([ceil(s(1)/2):s(1),1:floor(s(1)/2)],:,:); 
permute_matrix = permute_matrix(:,[ceil(s(2)/2):s(2),1:floor(s(2)/2)],:); 
cli_header('Plotting subspace stuff')
stfig('Subsampled spectrum');
clf
subplot(2,3,1)
hold on
imagesc(cut.img)
% r=rectangle('Position', [4000  w h] 

subplot(2,3,2)
imagesc(abs(sample_spectrum))
title('Fourier transform')
daspect([1,1,1])
subplot(2,3,3)
imagesc(abs(permute_matrix))
daspect([1,1,1])
title('$\lambda$ space')
subplot(2,3,4)
surf(log(abs(permute_matrix(:,:,1))),'LineStyle','none')
% daspect([1,1,1])
title('$\lambda$ density')
colormap(inferno)
subplot(2,3,5)
imagesc(angle(sample_spectrum))
daspect([1,1,1])
subplot(2,3,6)
imagesc(log(xcorr2_fft(cut.img(1500:3000,1000:3000,1),cut.img(1500:3000,1000:3000,1))))
title('Autocorrelation')
title('Fourier phase')
suptitle(sprintf('Dark-sky sampling [%u,%u]x[%u,%u]',s_region(1,1),s_region(1,2),s_region(2,1),s_region(2,2)));

cli_header('Doneski.');


%% Radial gradients
% [grad_len,grad_dir] = imgradient(cut.img);
% Find the centre

% ax = gcf;
% Requires R2020a or later
% exportgraphic(ax,'Resolution',300)
cli_header(5,'ok');


%% Save output
% out_img = (cut.img);
% whos out_img
% imwrite(out_img,fullfile(outpath,'cutoff_smooth.png'))
% cli_header(2,'saved');
% 
% out_img = (r_grad.*(grad_sense==0));
% imwrite(out_img,fullfile(outpath,'radial_gradient.png'))
% cli_header(2,'saved');
% 
% out_img = (th_grad.*(grad_sense==0));
% imwrite(out_img,fullfile(outpath,'circumferential_gradient.png'))
% cli_header(2,'saved');
% 
% out_img = (sqrt(imgrad.*(imgrad<.1)));
% imwrite(out_img,fullfile(outpath,'intensity_gradient.png'))
% cli_header(2,'saved');
% 
% out_img = (grad_dir.*(imgrad<.1));
% imwrite(out_img,fullfile(outpath,'gradient_direction.png'))
% cli_header(2,'saved');

%%  
% Ok let's have a peek at the FT of these gradients and try some
% filtering...

% grad_ft = fft2(th_grad);

% stfig('Spectrum of gradients');
% plot(sum(isnan(th_grad)))

% saveas(figout)

