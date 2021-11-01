clear all
outpath = 'C:\Users\jaker\Pictures\Oregon_Eclipse\code\matlab\out';
tif_dir = 'C:\Users\jaker\Pictures\Oregon_Eclipse\16bit\tif';
raw_dir = 'C:\Users\jaker\Pictures\Oregon_Eclipse\raw';
dark_raw = 'C:\Users\jaker\Pictures\Oregon_Eclipse\darkfield\raw';
dark_tif= 'C:\Users\jaker\Pictures\Oregon_Eclipse\darkfield\tif';

% offsets = load(fullfile(outpath,'offsets.mat'));
etimes = exposure_times();

% Get the two test images
img_num= 5;
img_nums = [img_num,img_num+1]; 
fnames = get_files(dark_tif,'tif');
cli_header(2, 'Reading images');

for im_idx = 1:15
    imname = fnames{im_idx}(1:8);
%     imname = sprintf('IMG_0018'
    this_img.tif = im2double(imread(fullfile(dark_tif,[imname,'.tif'])));
    this_img.raw= im2double(imread(fullfile(dark_raw,[imname,'.CR2'])));
    cli_header(2,'Loaded up!');


    xsec = 1:5180;
    ysec = 1:3450;

    opts.num_bins = 256;
    opts.max_val = 1;
    opts.label = 'raw';
    opts.loglog = true;
    opts.visual = false;
    opts.lin = false;
    this_img.hist.raw{im_idx} = rgb_hist(this_img.raw(ysec,xsec,:),opts);

    opts.num_bins = 256;
    opts.label = 'tif';
    this_img.hist.tif{im_idx} = rgb_hist(this_img.tif(ysec,xsec,:),opts);
    % %
end

%     stfig(sprintf('comparison %s',imname));
%%
cli_header(1,'Plotting');
stfig('comparison');    
for im_idx = 1:15
%     clf
    subplot(2,2,1)
    title('tif')
    hold on
    plot(this_img.hist.raw{im_idx}.bins,this_img.hist.tif{im_idx}.red.counts,'r-.')
    plot(this_img.hist.raw{im_idx}.bins,this_img.hist.tif{im_idx}.green.counts,'g-.')
    plot(this_img.hist.raw{im_idx}.bins,this_img.hist.tif{im_idx}.blue.counts,'b-.')
    set(gca,'Yscale','log')
    ylim([.1,1e6])
    xlim([1e-3,1])


    subplot(2,2,2)
    hold on
    plot(this_img.hist.raw{im_idx}.bins,this_img.hist.tif{im_idx}.red.counts,'r-.')
    plot(this_img.hist.raw{im_idx}.bins,this_img.hist.tif{im_idx}.green.counts,'g-.')
    plot(this_img.hist.raw{im_idx}.bins,this_img.hist.tif{im_idx}.blue.counts,'b-.')
    set(gca,'Yscale','log')
    set(gca,'Xscale','log')
    ylim([.1,1e6])
    xlim([1e-3,1])
    
    %% raw
    subplot(2,2,3)
    title('raw')
    hold on
    plot(this_img.hist.raw{im_idx}.bins,this_img.hist.raw{im_idx}.red.counts,'r-.')
    plot(this_img.hist.raw{im_idx}.bins,this_img.hist.raw{im_idx}.green.counts,'g-.')
    plot(this_img.hist.raw{im_idx}.bins,this_img.hist.raw{im_idx}.blue.counts,'b-.')
    set(gca,'Yscale','log')
    ylim([.1,1e6])
    xlim([1e-3,1])
    

    subplot(2,2,4)
    hold on
    plot(this_img.hist.raw{im_idx}.bins,this_img.hist.raw{im_idx}.red.counts,'r-.')
    plot(this_img.hist.raw{im_idx}.bins,this_img.hist.raw{im_idx}.green.counts,'g-.')
    plot(this_img.hist.raw{im_idx}.bins,this_img.hist.raw{im_idx}.blue.counts,'b-.')
    set(gca,'Yscale','log')
    set(gca,'Xscale','log')
    ylim([.1,1e6])
    xlim([1e-3,1])
    
%     drawnow
end
    % positive => PNG value higher
% Negative => Raw values higher

% Findin