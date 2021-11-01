function com = get_com(grey_img,opts)
    X = 1:size(grey_img,2);
    Y = 1:size(grey_img,1);
    [X_mesh,Y_mesh] = meshgrid(X,Y);
%     total_brightness = sum(sum(grey_img));
    max_val = max(grey_img,[],'all'); 
    median_val = median(median(grey_img));
    hot_region = grey_img>opts.cutoff*median_val;%cutoff is a fraction of max brightness
    hot_area = sum(sum(hot_region));
    if hot_area < 10
        warning('Mask found nothing - try a lower threshold')
        X_mean = sum(sum(X_mesh.*im2double(grey_img)))/hot_area;
        Y_mean = sum(sum(Y_mesh.*im2double(grey_img)))/hot_area;
    else
        Y_mean = sum(sum(Y_mesh.*im2double(hot_region)))/hot_area;
        X_mean = sum(sum(X_mesh.*im2double(hot_region)))/hot_area;
    end
    
%     com.rad = 
    com.com = [X_mean,Y_mean];
    com.hot.region = hot_region;
    com.hot.area = hot_area;    
    
    com.crop.bounds = [ceil(X_mean - opts.crop.size(1)),floor(X_mean + opts.crop.size(1));
                        ceil(Y_mean - opts.crop.size(2)),floor(Y_mean + opts.crop.size(2))];
    
    
    if opts.v.isual
       stfig('COM finding') ;
       clf
       subplot(2,2,1)
       imagesc(grey_img);
       daspect([1,1,1])
       hold on
       plot(com.com(1),com.com(2),'rx')
       subplot(2,2,2)
       imagesc(hot_region)
       daspect([1,1,1])
       subplot(2,2,3)
       imagesc(grey_img(com.crop.bounds(2,1):com.crop.bounds(2,2),com.crop.bounds(1,1):com.crop.bounds(1,2)))
       daspect([1,1,1])
       
       colormap(inferno)
        
    end
end
