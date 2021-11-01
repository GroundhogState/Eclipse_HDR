clear all
eclipse_init
% outpath = 'C:\Users\jaker\Pictures\Oregon_Eclipse\code\matlab\out\data';
% raw_dir = 'C:\Users\jaker\Pictures\Oregon_Eclipse\16bit\png';
% offsets = load(fullfile(outpath,'offsets.mat'));
% etimes = exposure_times();

% profile on

% img_num= 5;
kernel_size = 9;
final_level = 1.76;
hotpix_detected = [];
hotpix_detected.sum = zeros(size(3457,5194,3));
fnames = get_files(raw_dir,'CR2');
for img_num = 1:length(fnames)
    fname = fnames{img_num};
    cli_header(2, 'Reading image %u/%u, %s',img_num,length(fnames),fname);
    this_img.img = 2^16*im2double(imread(fullfile(raw_dir,fname)));
    file_idxs = cell2mat(cellfun(@(x) str2double(x(7:8)),fnames(img_num),'uni',0));
    cli_header(2,'Loaded up, analysing... ',img_num,length(fnames),fname);
    
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
    this_img.conv = imfilter(this_img.roi,hotpix_kernel,'replicate');
    ig = imgrad_rgb(this_img.roi);
    for i=1:3
        this_img.z_scores{i} = zscore(this_img.roi(:,:,i),[],'all');
    end
    triple_check = zeros(size(this_img.roi));
    for i=1:3
        triple_check(:,:,i) = zscore(this_img.roi(:,:,i),[],'all').*rescale(ig(:,:,i)).*rescale(this_img.conv(:,:,i));
    end

    opts.visual = 0;
    if opts.visual
        stfig('Hotpixel ROI');
        clf
        imagesc(smooth_rgb(rescale(this_img.roi),.7))
        daspect([1,1,1])
        stfig('Hotpix search by filter');
        clf
        subplot(2,2,1)
        imagesc(roi.x,roi.y,rescale(this_img.conv))
        for i = 1:3
            subplot(2,2,i+1)
            imagesc(roi.x,roi.y,rescale(this_img.conv(:,:,i)))
        end
        colormap(inferno)
        stfig('gradient search');
        for i=1:3
            subplot(2,2,i)
            imagesc(roi.x,roi.y,ig(:,:,i))
        end
        subplot(2,2,4)
        imagesc(roi.x,roi.y,sum(rescale((this_img.roi)),3))
        colormap(inferno)
        stfig('zscores');
        clf
        for i=1:3
            subplot(2,2,i)
            imagesc(roi.x,roi.y,this_img.z_scores{i})
        end
        subplot(2,2,4)
        imagesc(this_img.z_scores{1}.*this_img.z_scores{2}.*this_img.z_scores{3})
        colormap(inferno)
        % %
        colormap(inferno)
        stfig('Cross-check');
        clf
        for i=1:3
            subplot(2,2,i)
           imagesc(roi.x,roi.y,triple_check(:,:,i))
        end
        subplot(2,2,4)
    end
    %%

    opts.label = 'Cross-checking';
    opts.num_bins = 100;
    opts.lin = 0;
%     rgb_hist(triple_check,opts);



    %%
    hotpix = [];
    hotpix.mask = triple_check>final_level;
    [hotpix.r(:,1),hotpix.r(:,2)] = ind2sub(imsize(1:2),find(hotpix.mask(:,:,1)));
    [hotpix.g(:,1),hotpix.g(:,2)]= ind2sub(imsize(1:2),find(hotpix.mask(:,:,2)));
    [hotpix.b(:,1),hotpix.b(:,2)]= ind2sub(imsize(1:2),find(hotpix.mask(:,:,3)));

    fprintf('Hot pixels detected: (%u,%u,%u)\n',sum(sum(hotpix.mask)))
    opts.visual = 1;
    if opts.visual
        stfig('Hotpix location');
        clf
        hold on
        imagesc(uint16(this_img.img))
        plot(hotpix.r(:,2),hotpix.r(:,1),'r+')
        plot(hotpix.g(:,2),hotpix.g(:,1),'gx')
        plot(hotpix.b(:,2),hotpix.b(:,1),'bo')
        daspect([1,1,1])
        drawnow
    end
    cli_header(1,'Done.');

    hotpix_detected.(fname(1:8)).r = hotpix.r;
    hotpix_detected.(fname(1:8)).g = hotpix.g;
    hotpix_detected.(fname(1:8)).b = hotpix.b;
    
    hotpix_detected.sum = hotpix_detected.sum + hotpix.mask;

end
%%
% So we have the hot pixels! The ones we would like to kill off are the
% ones that appear in several images. this would help avoid ones that are
% due to, say, Bailey's Beads.
% So what do we want? 
% Basically a histogram of the hotpix?
% Oh just add all the masks together, fool.
%%
opts = [];
opts.label = 'Hotpix mask';
opts.log = 0;
opts.lin = 1;
opts.num_bins = 19;
net_hist = rgb_hist(hotpix_detected.sum,opts); 
xlim([1,20])
ylim([0,20])
xlabel('Number of occurrances among all images')
ylabel('Number of pixels with this occurence')

stfig('Hot mask');
clf
imagesc(hotpix_detected.sum)

%%
cli_header(1,'Saving data...');
save_name = fullfile(data_out,'hotpix');
save(save_name,'hotpix_detected');
cli_header(2,'Saved');


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
    
    
    