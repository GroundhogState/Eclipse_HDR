% OK - can you do the eclipse thing faster in matlab?
% Given you've already learned a heap from the Python experience...

dirname = 'C:\Users\jaker\Pictures\Oregon_Eclipse\16bit\png';
fnames = get_files(dirname,'png');
imgs = cellfun(@(x) imread(fullfile(dirname,x)),fnames,'uni',0);
% I just wanna load these into a hypercube, darnit!
cli_header('Done.');
%% Correlating to find offsets
num_imgs = length(fnames);
slice.x = 1600;
slice.y = 1000;
% close all

im_thresholds = [.7e4,.7e4,1e4,1e4,1e4,...
                 1.5e4,1.5e4,2.5e4,4e4,4e4,...
                 5e4,5.5e4,5.5e4,5.5e4,5.5e4,...
                 6.2e4,6.2e4,6.2e4,6.2e4];
im_data = cell(num_imgs,1);
im_opts.num = 1;
im_opts.slice = slice;
im_opts.root = dirname;
im_opts.visual = false;
im_opts.name =  fnames{im_opts.num};
im_opts.brightness_threshold = im_thresholds(im_opts.num);
this_img = get_img(imgs{im_opts.num},im_opts);


corr_opts.visual = true;
corr_opts.slice = slice;
offsets = [];
offsets.crop = nan(num_imgs-1,2);
offsets.full= nan(num_imgs-1,2);
for idx = 1:num_imgs-1
    % returns the delta such that img1 \approx translate(img2,offsets.full)
    % hm - this returns an offset of 1 pixel for identical images, should
    % be 0 - debug this
    cli_header(2,'Getting offsets between %u and %u:',idx,idx+1);
    im_opts.num = idx+1;
    im_opts.brightness_threshold = im_thresholds(im_opts.num);
    that_img = get_img(imgs{im_opts.num},im_opts);
    pair_offsets = find_pair_offset(this_img,that_img,corr_opts);
    offsets.crop(idx,:) = pair_offsets.crop;
    offsets.full(idx,:) = pair_offsets.full;
    this_img = that_img;
end
cli_header(2,'All pairs done.');
%%
outpath = 'C:\Users\jaker\Pictures\Oregon_Eclipse\code\matlab\out';
save(fullfile(outpath,'offsets.mat'),'offsets')
cli_header(3,'Output saved.');
%% Correlating image pairs



% offset_vals = find_pair_offset(this_img,that_img,corr_opts);
% 

%%












