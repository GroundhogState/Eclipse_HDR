% In which we take pairs of images, compute several gradients for
% comparison, and try aligning on them.

clear all
outpath = 'C:\Users\jaker\Pictures\Oregon_Eclipse\code\matlab\out';
dirname = 'C:\Users\jaker\Pictures\Oregon_Eclipse\16bit\png';
offsets = load(fullfile(outpath,'offsets.mat'));
etimes = exposure_times();

% Get the two test images
% for fi = 6:2:15
    fi = 13;
    img_nums = [fi,fi+1];
    fnames = get_files(dirname,'png');
    cli_header(2, 'Reading images');
    imgs = cellfun(@(x) imread(fullfile(dirname,x)),fnames(img_nums),'uni',0);
    file_idxs = cell2mat(cellfun(@(x) str2double(x(7:8)),fnames(img_nums),'uni',0));

    this_img.img = imgs{1};
    this_img.grey = rgb2gray(this_img.img);


    cli_header(2,'finding grads');
    % stfig('Bandpass filtering');
    [this_img.grad.cart,this_img.grad.dir] = imgradient(this_img.grey);

    cli_header(2,'finding com');

    % find rough com for polar gradient.
    im_opts.v.isual = 1;
    im_opts.cutoff = 8;
    im_opts.crop.size = [400,400];
    this_img.com = get_com(this_img.grad.cart,im_opts);


    % smooth out single pixels
    % pixel_


    % find polar grad
    this_img.polar = get_polar_gradient(this_img.img,this_img.com.com);



    cli_header(2,'Done, plotting');

%     stfig('this img');
%     clf
%     h=2;
%     w=3;
%     subplot(h,w,1)
%     imagesc(this_img.grey)
%     hold on
%     % plot(this_img.com.com(2),this_img.com.com(1),'rx')
%     title('Image')
%     daspect([1,1,1])
%     subplot(h,w,2)
%     imagesc(this_img.grad.cart)
%     title('$|\nabla|$')
%     daspect([1,1,1])
%     subplot(h,w,3)
%     title('$\nabla_\theta$')
%     imagesc((this_img.grad.dir))
%     daspect([1,1,1])
% 
%     subplot(h,w,4)
%     imagesc(this_img.grey(this_img.com.crop.bounds(2,1):this_img.com.crop.bounds(2,2),...
%                         this_img.com.crop.bounds(1,1):this_img.com.crop.bounds(1,2)))
%     daspect([1,1,1])
%     subplot(h,w,5)
%     imagesc(this_img.polar.d_r)
%     daspect([1,1,1])
%     subplot(h,w,6)
%     imagesc(this_img.polar.d_theta)
%     daspect([1,1,1])
%     % plot_grads(that_img,'that_img');
%     % 
%     colormap(inferno)


% %
    cli_header(2,'Saving.');
    this_fname = fnames{img_nums(1)}(1:end-4);
    savedir = fullfile(outpath,'im_stash',this_fname);
    if ~exist(savedir,'dir'),mkdir(savedir);end
    cmap = viridis(2^8);
    % Save images!

    imwrite((this_img.polar.d_r),cmap,fullfile(savedir,'radial_rescale.png'))
    imwrite((this_img.polar.d_r),cmap,fullfile(savedir,'radial.png'))
    imwrite(this_img.polar.d_theta,cmap,fullfile(savedir,'circumferential.png'))
    imwrite((this_img.grad.cart),cmap,fullfile(savedir,'cart_mag.png'))
    imwrite((this_img.grad.dir),cmap,fullfile(savedir,'cart_dir.png'))
    imwrite(imgradient(this_img.grad.dir),cmap,fullfile(savedir,'d_dir_d_x.png'))

    cli_header(1,'Done.');
% end

function plot_grads(this_img,label)
    stfig(label);
    clf
    h=2;
    w=2;
    subplot(h,w,1)
    imagesc(this_img.grey)
    hold on
    title('Image')
    daspect([1,1,1])
    subplot(h,w,2)
    imagesc(this_img.grad.cart)
    title('$|\nabla|$')
    daspect([1,1,1])
    subplot(h,w,3)
    title('$\nabla_\theta$')
    imagesc((this_img.grad.dir))
    daspect([1,1,1])
    colormap(inferno)

end
% corr_opts.visual = true;
% corr_opts.nums = [idx,idx+1];
% corr_opts.level = im_levels(idx:idx+1);
% pair_offsets = gradient_align(this_img,that_img,corr_opts);
% offsets.crop(idx,:) = pair_offsets.crop;
% offsets.full(idx,:) = pair_offsets.full;
% this_img = that_img;

    
% end

% cli_header(2,'All done.');
