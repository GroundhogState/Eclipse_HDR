dirname = 'C:\Users\jaker\Pictures\Oregon_Eclipse\16bit\png';
fnames = get_files(dirname,'png');
imgs = cellfun(@(x) imread(fullfile(dirname,x)),fnames,'uni',0);
% I just wanna load these into a hypercube, darnit!
cli_header('Done.');
%% ok so you've run the darkfields and you have
% mean_dark = mean_img/2^16;
% and you've run the alignment tool so you've got
% full_offsets = offsets.full;
% so now you take each image, subtract the noise, apply the shift, then sum
% them together

net_offset = [0,0];
stfig('Combining images');
clf
% imagesc(sum_img+imtranslate(imgs{2},offsets.full(1,:)))
net_offset = [0,0];
sum_img = im2double(imgs{1})-0*mean_dark;
for idx=1:num_imgs-1
   net_offset = net_offset + offsets.full(idx,:);
   sum_img = sum_img + imtranslate(im2double(imgs{idx+1})-0*mean_dark,+net_offset);
   imagesc(sum_img)
   drawnow
end


%% Correlating to find offsets
num_imgs = length(fnames);
slice.x = 1600;
slice.y = 1000;
% close all

im_thresholds = [.7e4,.7e4,1e4,1e4,1e4,...
                 1.5e4,1.5e4,2.5e4,4e4,4e4,...
                 5e4,5.5e4,5.5e4,5.5e4,5.5e4,...
                 6.2e4,6.3e4,6.3e4,6.3e4]/2^16;
im_data = cell(num_imgs,1);
for i=1:19
    im_opts.num = i;
    im_opts.slice = slice;
    im_opts.root = dirname;
    im_opts.visual = true;
    im_opts.name =  fnames{im_opts.num};
    im_opts.brightness_threshold = im_thresholds(im_opts.num);
    this_img = get_img(im2double(imgs{im_opts.num}),im_opts);
end
%%
corr_opts.visual = true;
corr_opts.slice = slice;
offsets = [];
offsets.crop = nan(num_imgs-2,2);
offsets.full= nan(num_imgs-2,2);
for idx = 2:num_imgs-1
    % hm - this returns an offset of 1 pixel for identical images, should
    % be 0 - debug this
    cli_header(2,'Getting offsets between %u and %u:',idx-1,idx);
    im_opts.num = idx;
    im_opts.brightness_threshold = im_thresholds(im_opts.num);
    that_img = get_img(im2double(imgs{im_opts.num}),im_opts);
    pair_offsets = find_pair_offset(this_img,that_img,corr_opts);
    offsets.crop(idx-1,:) = pair_offsets.crop;
    offsets.full(idx-1,:) = pair_offsets.full;
    this_img = that_img;
end
cli_header(2,'All pairs done.');