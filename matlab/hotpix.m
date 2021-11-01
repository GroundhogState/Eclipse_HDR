clear all
eclipse_init
% outpath = 'C:\Users\jaker\Pictures\Oregon_Eclipse\code\matlab\out';
% dirname = 'C:\Users\jaker\Pictures\Oregon_Eclipse\16bit\png';
% offsets = load(fullfile(outpath,'offsets.mat'));
% etimes = exposure_times();

% Get the two test images
% for fi = 6:2:15
img_num= 5;
img_nums = [img_num,img_num+1];
fnames = get_files(dirname,'png');
cli_header(2, 'Reading images');
% this_img.img = 2^16*im2double(imread(fullfile(dirname,fnames{img_num})));
this_img.img = 2^16*im2double(imread(fullfile(dirname,'IMG_0580.png')));
% imgs = cellfun(@(x) imread(fullfile(dirname,x)),fnames(img_num),'uni',0);
file_idxs = cell2mat(cellfun(@(x) str2double(x(7:8)),fnames(img_num),'uni',0));
cli_header(2,'Loaded up!');

%%

%% Findings:
% It LOOKS like taking the product of filters (like zscores) will work for
% star detection as they are detectable on all channels.
% One could also find a convenient measure for when 1 or 2 channels are hot
% either way - the star seems to have a size of ~10pix which gives us the
% a lower bound on resolvable length scales - disappointingly large, but we
% can look at the deconv attempts later. 
% So the next step here is probably just looking for reliable ways to
% distinguish hotspots from stars - perhaps a high-order gaussian filter to
% act as F(10pix) - F(3pix) or something - which has opposite signs for
% each possibility?


% Extra notes: The z-scores seem to be more reliable at picking out the
% stars (less noisy than the 1pix filter). Of course - the grad filter
% seems a bit better at finding hotpix and ignoring the star

% cool cool - so the px at ~ 3300,2460 in images after IMG_0580 are a very
% hot pixel - they same pix hot in several images - so that's another test
% point. Also means, sadly, that all those spots 'below' the sun are bogus.
% Oh well...

kernel_size = 9;
final_level = 2;


hotpix_kernel = -ones(kernel_size)/kernel_size^2;
hotpix_kernel(ceil(kernel_size/2),ceil(kernel_size/2)) = 1;
imsize = size(this_img.img);



% %
% this_img.roi = this_img.img(300:400,400:500,:);
% full image
roi.x = 1:imsize(2);
roi.y = 1:imsize(1);
% hotspots detected! upper left
% roi.x = 300:500;
% roi.y = 400:600;
% star is here
% roi.x = 1340:1360;
% roi.y = 3335:3370;
% wider star area
% roi.x = 2300:2500;
% roi.y = 3000:3457;
this_img.roi = this_img.img(roi.y,roi.x,:); 

stfig('Hotpixel ROI');
clf
imagesc(smooth_rgb(rescale(this_img.roi),.7))
daspect([1,1,1])
% %
this_img.conv = imfilter(this_img.roi,hotpix_kernel,'replicate');


stfig('Hotpix search by filter');
clf
subplot(2,2,1)
imagesc(roi.x,roi.y,rescale(this_img.conv))
subplot(2,2,2)
imagesc(roi.x,roi.y,rescale(this_img.conv(:,:,1)))
subplot(2,2,3)
imagesc(roi.x,roi.y,rescale(this_img.conv(:,:,2)))
subplot(2,2,4)
imagesc(roi.x,roi.y,rescale(this_img.conv(:,:,3)))
colormap(inferno)

stfig('gradient search');
ig = imgrad_rgb(this_img.roi);
for i=1:3
    subplot(2,2,i)
    imagesc(roi.x,roi.y,ig(:,:,i))
end
subplot(2,2,4)
imagesc(roi.x,roi.y,sum(rescale((this_img.roi)),3))
colormap(inferno)

% % real simple: z scorefor i=1;3

stfig('zscores');
clf
for i=1:3
    subplot(2,2,i)
   this_img.z_scores{i} = zscore(this_img.roi(:,:,i),[],'all');
    imagesc(roi.x,roi.y,this_img.z_scores{i})
end
subplot(2,2,4)
imagesc(this_img.z_scores{1}.*this_img.z_scores{2}.*this_img.z_scores{3})
colormap(inferno)



% %
triple_check = zeros(size(this_img.roi));
for i=1:3
    triple_check(:,:,i) = zscore(this_img.roi(:,:,i),[],'all').*rescale(ig(:,:,i)).*rescale(this_img.conv(:,:,i));
end
colormap(inferno)


stfig('Cross-check');
clf
for i=1:3
    subplot(2,2,i)
   imagesc(roi.x,roi.y,triple_check(:,:,i))
end
subplot(2,2,4)
hotpix_mask = triple_check>final_level;
imagesc(roi.x,roi.y,hotpix_mask)
colormap(inferno)


opts.visual = true;
% opts.label= 'trim ROI';
% rgb_hist(this_img.roi,opts);
% opts.label = 'conv';
% rgb_hist(this_img.conv,opts);
% opts.label = 'grads';
% rgb_hist(imgrad_rgb(this_img.roi),opts);
% opts.label = 'z scores';
% rgb_hist(zscore(this_img.roi,[],'all'),opts);
% rgb_hist(triple_check);
opts.label = 'Cross-checking';
opts.num_bins = 100;
rgb_hist(triple_check,opts)
% stfig('Grad check')
%%


fprintf('Hot pixels detected: (%u,%u,%u)\n',sum(sum(hotpix_mask)))

stfig('Hotpix location');
clf
imagesc(this_img.img)


% Nice! This seems pretty good; find all pixels that take a high value with
% all of the tests. What about if we look at the star, or the corona?

%%

%%


%%

% And can you smooth the result?
%  this is a bandpass, right? Plot transfer function? Better yet, specify
%  how you wanna do this
%%

% sm_width = 1;
% 
% stfig('Hotpixel, texture');
% clf
% imagesc(roi.x,roi.y,smooth_rgb(this_img.hpf,sm_width))
% daspect([1,1,1])
% % Observation: Aside from some correlations - AHA you can look for
% % correlations in the gradient of the channels 
% % This img has bright R/G/B sections in the noise thanks to the hot pix -
% % large peaks in grad ged smoothed out. 
% % where there are white patches of black or white, this is promising - that
% % the grad there is consistent in the channels. 
% this_img.hpf= 2^16*im2double(this_img.img) - this_img.smooth.rgb;
% stfig('Hotpixel, texture breakdown');
% clf
% for i=1:3
%     subplot(2,2,i)
%     high_part{i} = imgaussfilt(this_img.hpf(:,:,i),sm_width)>0;
%     imagesc(roi.x,roi.y,high_part{i})
% end
% subplot(2,2,4)
% imagesc(roi.x,roi.y,high_part{1}.*high_part{2}.*high_part{3})



function rgb_smoothed = smooth_rgb(img,sigma)
rgb_smoothed = zeros(size(img));
    for i=1:3    
        rgb_smoothed(:,:,i) = imgaussfilt(squeeze(img(:,:,i)),sigma);
    end
end


% Find the gradient and look at the peak vals

%Look at the correlation between the img and its gradient - some fractional
%power of s for the laplace transform?


%% Spiral grad align
% use the cost: cos^2(grad_dir(img(1)) - grad_dir(img(2)) - is the squaring
% necessary?
% Start from offsets already obtained, then:
%     Compute cost function for obtained COMS 
%     Update offset/bias vector via autodiff/optimizer
    % Return the gradient of the function wrt the centres also?
    
    