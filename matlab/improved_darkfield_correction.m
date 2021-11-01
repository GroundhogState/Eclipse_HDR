clear all
outpath = 'C:\Users\jaker\Pictures\Oregon_Eclipse\code\matlab\out';
dirname = 'C:\Users\jaker\Pictures\Oregon_Eclipse\16bit\png';
offsets = load(fullfile(outpath,'offsets.mat'));
etimes = exposure_times();

% Get the two test images
fi = 14
img_nums = [fi,fi+1];
fnames = get_files(dirname,'png');
cli_header(2, 'Reading images');
imgs = cellfun(@(x) imread(fullfile(dirname,x)),fnames(img_nums),'uni',0);
file_idxs = cell2mat(cellfun(@(x) str2double(x(7:8)),fnames(img_nums),'uni',0));
%%

% mask off zeros


% Do compute
exptime(1) = etimes(etimes(:,1) == file_idxs(1),2);
exptime(2) = etimes(etimes(:,1) == file_idxs(2),2);
imsize = size(imgs{1});

% Align them according to the offset:
% img_out = first_image + imtranslate(second_image)
% therefore the ROI in the second image will be ROI + offset yes?
% A test:
roi = [1000,2000; %Y
            1,1700]; %X
% check boundaries of ROI
if any(roi(2,:) < 1 | roi(2,:) > imsize(2)),warning('X ROI out of bounds');end
if any(roi(1,:) < 1 | roi(1,:) > imsize(2)),warning('Y ROI out of bounds');end

pair_offset = (offsets.offsets.full(img_nums(1),:)); % ordered offsets in X and Y
roi_offset = roi-fliplr(pair_offset)';
% Check boundaries of offset ROI and reduce both if they are out of bounds
[roi,roi_offset] = correct_roi(roi,roi_offset,imsize);
roi_size = diff(roi')+1;

sub_image = [];
sub_image(1,:,:,:) = imgs{1}(roi(1,1):roi(1,2),roi(2,1):roi(2,2),:);
sub_image(2,:,:,:) = imgs{2}(roi_offset(1,1):roi_offset(1,2),roi_offset(2,1):roi_offset(2,2),:);
cli_header('Snip!');

mask = sub_image == 0;
for img_zerovalue = 1:30:500
    sub_image(mask) = img_zerovalue;


% Img processing bit
scaled_img = zeros(2,roi_size(1),roi_size(2),3);
for i=1:2
    scaled_img(i,:,:,:) = sub_image(i,:,:,:)/exptime(i);
end
tau_coef = 1/exptime(1) - 1/exptime(2);
noise_estimate =  (squeeze(scaled_img(1,:,:,:)) - squeeze(scaled_img(2,:,:,:)))/tau_coef;

im_denoise = zeros(2,roi_size(1),roi_size(2),3);
mean_noise = mean(noise_estimate,[1,2]);
for i=1:2
    for chan=1:3
        im_denoise(i,:,:,chan) = sub_image(i,:,:,chan) - mean_noise(chan);
    end
end


cli_header(1,'Making histograms!');

fig_h = 2;
fig_w = 2;
stfig('Refining darkfield estimation');
clf;

    subplot(fig_h,fig_w,1)
    hold on
    imagesc(imgs{1})
    plot([roi(2,1),roi(2,1),roi(2,2),roi(2,2),roi(2,1)],[roi(1,1),roi(1,2),roi(1,2),roi(1,1),roi(1,1)],'r')
    daspect([1,1,1])
    title('Image 1')


    subplot(fig_h,fig_w,2)
    hold on
    imagesc(imgs{2})
    plot([roi_offset(2,1),roi_offset(2,1),roi_offset(2,2),roi_offset(2,2),roi_offset(2,1)],...
                [roi_offset(1,1),roi_offset(1,2),roi_offset(1,2),roi_offset(1,1),roi_offset(1,1)],'r')
    daspect([1,1,1])
    title('Image 2')

    num_bins = 2^8;

    bin_edges = linspace(1,max(sub_image,[],'all'),num_bins+1);
    bin_centres = 0.5*(bin_edges(2:end)+bin_edges(1:end-1));
    counts.red = zeros(2,num_bins);
    counts.blue = zeros(2,num_bins);
    counts.green = zeros(2,num_bins);
    for i=1:2
        counts.red(i,:) = histcounts(sub_image(i,:,:,1),bin_edges);
        counts.blue(i,:) = histcounts(sub_image(i,:,:,2),bin_edges);
        counts.green(i,:) = histcounts(sub_image(i,:,:,3),bin_edges);
    end




    subplot(fig_h,fig_w,3)
    hold on
    plot(bin_centres,counts.red(1,:),'r:');
    plot(bin_centres,counts.green(1,:),'g:');
    p1=plot(bin_centres,counts.blue(1,:),'b:');
    plot(bin_centres,counts.red(2,:),'r-.');
    plot(bin_centres,counts.green(2,:),'g-.');
    p2=plot(bin_centres,counts.blue(2,:),'b-.');
    legend([p1,p2],'Im1','Im2')
    set(gca,'Yscale','log')
    set(gca,'Xscale','log')
    title('Sub-image brightness')


    % Correct both images by their exposure time
    scaled.bin_edges = linspace(1,max(sub_image./exptime',[],'all'),num_bins+1);
    % scaled.bin_edges = linspace(1,15e4,num_bins+1);
    scaled.bin_centres = 0.5*(scaled.bin_edges(2:end)+scaled.bin_edges(1:end-1));
    scaled.counts.red = zeros(2,num_bins);
    scaled.counts.blue = zeros(2,num_bins);
    scaled.counts.green = zeros(2,num_bins);
    for i=1:2
        scaled.counts.red(i,:) = histcounts(sub_image(i,:,:,1)/exptime(i),scaled.bin_edges);
        scaled.counts.blue(i,:) = histcounts(sub_image(i,:,:,2)/exptime(i),scaled.bin_edges);
        scaled.counts.green(i,:) = histcounts(sub_image(i,:,:,3)/exptime(i),scaled.bin_edges);
    end
    bin_centres = 0.5*(bin_edges(2:end)+bin_edges(1:end-1));
    % From this we see that the first scaled image is brighter than the second.
    % This is what we'd expect, right? Because the noise gets scaled up more.

    subplot(fig_h,fig_w,4)
    hold on
    plot(scaled.bin_centres,scaled.counts.red(1,:),'r:')
    plot(scaled.bin_centres,scaled.counts.green(1,:),'g:')
    p21=plot(scaled.bin_centres,scaled.counts.blue(1,:),'b:');
    plot(scaled.bin_centres,scaled.counts.red(2,:),'r-.')
    plot(scaled.bin_centres,scaled.counts.green(2,:),'g-.')
    p22=plot(scaled.bin_centres,scaled.counts.blue(2,:),'b-.');
    set(gca,'Yscale','log')
    set(gca,'Xscale','log')
    title('Scaled by exposure')

    drawnow
    % 

fig_h = 2;
fig_w = 2;
stfig('Minimizing CDF distance - com?');
clf;
% A loop to hist the difference of the sub-imgs
% generalizes to snip hists
    raw_diff.bin_edges = linspace(1,max(sub_image,[],'all'),num_bins+1);
    % scaled.bin_edges = linspace(1,15e4,num_bins+1);
    raw_diff.bin_centres = 0.5*(scaled.bin_edges(2:end)+raw_diff.bin_edges(1:end-1));
    raw_diff.counts.red = zeros(2,num_bins);
    raw_diff.counts.blue = zeros(2,num_bins);
    raw_diff.counts.green = zeros(2,num_bins);
    raw_diff.counts.red = histcounts(sub_image(1,:,:,1)-sub_image(2,:,:,1),raw_diff.bin_edges);
    raw_diff.counts.blue = histcounts(sub_image(1,:,:,2)-sub_image(2,:,:,2),raw_diff.bin_edges);
    raw_diff.counts.green = histcounts(sub_image(1,:,:,3)-sub_image(2,:,:,3),raw_diff.bin_edges);

    subplot(fig_h,fig_w,1)
    hold on
    title('Scaled count diffs')
    plot(raw_diff.bin_centres,raw_diff.counts.red,'r')
    plot(raw_diff.bin_centres,raw_diff.counts.green,'g')
    plot(raw_diff.bin_centres,raw_diff.counts.blue,'b')
    set(gca,'Yscale','log')
    
    denoise_diff.bin_edges = linspace(1,max(sub_image,[],'all'),num_bins+1);
    denoise_diff.bin_centres = 0.5*(scaled.bin_edges(2:end)+denoise_diff.bin_edges(1:end-1));
    denoise_diff.counts.red = zeros(2,num_bins);
    denoise_diff.counts.blue = zeros(2,num_bins);
    denoise_diff.counts.green = zeros(2,num_bins);
    denoise_diff.counts.red = histcounts(sub_image(1,:,:,1)-sub_image(2,:,:,1),denoise_diff.bin_edges);
    denoise_diff.counts.blue = histcounts(sub_image(1,:,:,2)-sub_image(2,:,:,2),denoise_diff.bin_edges);
    denoise_diff.counts.green = histcounts(sub_image(1,:,:,3)-sub_image(2,:,:,3),denoise_diff.bin_edges);


    denoise.bin_edges = linspace(min(im_denoise./exptime',[],'all'),max(im_denoise./exptime',[],'all'),num_bins+1);
    % scaled.bin_edges = linspace(1,15e4,num_bins+1);
    denoise.bin_centres = 0.5*(denoise.bin_edges(2:end)+denoise.bin_edges(1:end-1));
    denoise.counts.red = zeros(2,num_bins);
    denoise.counts.blue = zeros(2,num_bins);
    denoise.counts.green = zeros(2,num_bins);
    for i=1:2
        denoise.counts.red(i,:) = histcounts(im_denoise(i,:,:,1),num_bins);
        denoise.counts.blue(i,:) = histcounts(im_denoise(i,:,:,2),num_bins);
        denoise.counts.green(i,:) = histcounts(im_denoise(i,:,:,3),num_bins);
    end
    subplot(fig_h,fig_w,2)
    hold on
    plot(denoise.bin_centres,denoise.counts.red(1,:),'r:')
    plot(denoise.bin_centres,denoise.counts.green(1,:),'g:')
    p21=plot(denoise.bin_centres,denoise.counts.blue(1,:),'b:');
    plot(denoise.bin_centres,denoise.counts.red(2,:),'r-.')
    plot(denoise.bin_centres,denoise.counts.green(2,:),'g-.')
    p22=plot(denoise.bin_centres,denoise.counts.blue(2,:),'b-.');
    set(gca,'Yscale','log')
    title('image - $\langle\eta\rangle$')
    
    noise_edges = linspace(1,max(noise_estimate,[],'all'),num_bins+1);
    noise_centres = 0.5*(noise_edges(2:end)+noise_edges(1:end-1));
    noise.counts.red = zeros(1,num_bins);
    noise.counts.blue = zeros(1,num_bins);
    noise.counts.green = zeros(1,num_bins);
    for i=1:2
        noise.counts.red = histcounts(noise_estimate(:,:,1),noise_edges);
        noise.counts.blue = histcounts(noise_estimate(:,:,2),noise_edges);
        noise.counts.green = histcounts(noise_estimate(:,:,3),noise_edges);
    end



    subplot(fig_h,fig_w,3)
    hold on
    plot(noise_centres,noise.counts.red(:),'r')
    plot(noise_centres,noise.counts.green(:),'g')
    plot(noise_centres,noise.counts.blue(:),'b');
    set(gca,'Yscale','log')
%     set(gca,'Xscale','log')
    title('Noise estimate')
    




    

    denoise_diff.bin_edges = linspace(1,max(sub_image,[],'all'),num_bins+1);
    % scaled.bin_edges = linspace(1,15e4,num_bins+1);

    denoise_diff.bin_edges = linspace(1,max(sub_image./exptime',[],'all'),num_bins+1);
    denoise_diff.bin_centres = 0.5*(denoise_diff.bin_edges(2:end)+denoise_diff.bin_edges(1:end-1));
    denoise_diff.counts.red = zeros(2,num_bins);
    denoise_diff.counts.blue = zeros(2,num_bins);
    denoise_diff.counts.green = zeros(2,num_bins);

    for i=1:2
    %     cli_header('hey')
        denoise_diff.counts.red = histcounts(im_denoise(1,:,:,1)-im_denoise(2,:,:,1),num_bins);
        denoise_diff.counts.blue = histcounts(im_denoise(1,:,:,2)-im_denoise(2,:,:,2),num_bins);
        denoise_diff.counts.green = histcounts(im_denoise(1,:,:,3)-im_denoise(2,:,:,3),num_bins);
    end


    subplot(fig_h,fig_w,4)
    hold on
    plot(denoise_diff.bin_centres,denoise_diff.counts.red,'r')
    plot(denoise_diff.bin_centres,denoise_diff.counts.green,'g')
    plot(denoise_diff.bin_centres,denoise_diff.counts.blue,'b')
    set(gca,'Yscale','log')
    title('Denoise image difference dn1 - dn2')
    cli_header(1,'Done!');

   drawnow
end
    
% (img_rgb_1/etime_2 - img_rgb_2/etime_2) =
% shot_noise_2/exposure_time_2 - shot_noise_1/exposure_time_1
% assuming the shot noise is some variation around a mean then we can get towards it via
% etime_1*etime_2*(img_1/etime_2 - img_2/etime_2) = etime_1*(mu+d_2) - etime_2*(mu+d_1)
%                = mu*(etime_1-etime_2) + e_1*d_2 - e_2*d_1
% Assuming the d_i are just gonna wind up uncorrelated errors we can get
% the mean from 
% mu = (e_1*img_1 - e_2*img_2)/(e_1-e_2)
% go check this math!!
