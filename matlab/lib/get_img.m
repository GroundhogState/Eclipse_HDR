function this_img = get_img(img_in,im_opts)
    % Get the greyscale image and some intensity stats
%     img_in = imread(fullfile(im_opts.root,im_opts.name));
    this_img = get_imdetails(img_in);
    x_len = size(this_img.img,1);
    y_len = size(this_img.img,2);
    % Retrieve the thresholded COM and other data
%     im_opts.brightness_threshold = im_opts.threshold_level*this_img.stats.mean;
    this_img.data = get_img_data(this_img.grey,im_opts.brightness_threshold);
    % crop the image for easier use
    this_img.crop.x = [max(1,ceil(this_img.data.com(1) - im_opts.slice.x/2)),min(floor(this_img.data.com(1) + im_opts.slice.x/2),x_len)];
    this_img.crop.y = [max(1,ceil(this_img.data.com(2) - im_opts.slice.y/2)),min(floor(this_img.data.com(2) + im_opts.slice.y/2),y_len)];
    this_img.crop.grey = this_img.grey(this_img.crop.y(1):this_img.crop.y(2),this_img.crop.x(1):this_img.crop.x(2));
    this_img.crop.mask = this_img.data.mask(this_img.crop.y(1):this_img.crop.y(2),this_img.crop.x(1):this_img.crop.x(2));
    this_img.name = im_opts.name;
    if im_opts.visual
        stfig(im_opts.label);
        clf;
        colormap(viridis)
        subplot(2,3,1)
        imagesc(this_img.grey)
        hold on
        plot(this_img.data.com(1),this_img.data.com(2),'r+','Markersize',10)
        subplot(2,2,2)
        imagesc(this_img.data.mask)
        hold on
        plot(this_img.data.com(1),this_img.data.com(2),'r+','Markersize',10)
        subplot(2,2,4)
        % hold on
        imagesc(this_img.crop.x,this_img.crop.y,this_img.crop.grey)
        subplot(4,2,5)
        plot(this_img.bright.centres,this_img.bright.counts)
        subplot(4,2,7)
        plot(this_img.bright.centres,this_img.bright.counts)
        set(gca,'Yscale','log')
%         subplot(2,3,6)
%         plot(this_img.bright.centres,this_img.stats.cdf)
%         set(gca,'Xscale','log')
        suptitle(sprintf('Img %s',im_opts.name(5:8)))
    end
end

function img_data = get_img_data(grey_img,thresh)
    X = 1:size(grey_img,2);
    Y = 1:size(grey_img,1);
    [X_mesh,Y_mesh] = meshgrid(X,Y);

    brightness_mask = grey_img > thresh;
    total_brightness = sum(sum(grey_img));
    hot_region = brightness_mask>0;
    hot_area = sum(sum(brightness_mask));
    if hot_area < 10
        warning('Mask found nothing - try a lower threshold')
        X_mean = sum(sum(X_mesh.*double(grey_img)))/total_brightness;
        Y_mean = sum(sum(Y_mesh.*double(grey_img)))/total_brightness;
    else
        Y_mean = sum(sum(Y_mesh.*double(hot_region)))/hot_area;
        X_mean = sum(sum(X_mesh.*double(hot_region)))/hot_area;
    end
    img_data.com = [X_mean,Y_mean];
%     img_data.com(2) = Y_mean;
    img_data.mask = brightness_mask;
end