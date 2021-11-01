function corr_offset = find_pair_offset(this_img,that_img,corr_opts)
% returns the X-Y offset of two images based on the peak of their
% cross-correlation
    imcorr = xcorr2_fft(this_img.crop.mask,that_img.crop.mask,true);
    [~,peak_loc] = find_impeak(imcorr);

    % the imcorr output runs the translation (including rolling over edges)
    % so size is 2x(crop_dims) - 1
    % runs from -(X-1):0:(X-1) yes?
    % so that would make the *offset* of the cropped image: 
    v_offset = peak_loc(1)-corr_opts.slice.y+1;
    h_offset = peak_loc(2)-corr_opts.slice.x+1;

    % and then correcting back to the offset of the total image:
    h_crop_offset = this_img.crop.x(1)-that_img.crop.x(1);
    v_crop_offset = this_img.crop.y(1)-that_img.crop.y(1);

    %SO. The offset between these images is:
    H_offset = h_offset + h_crop_offset;
    V_offset = v_offset + v_crop_offset;
    
    corr_offset.crop = [h_offset,v_offset];
    corr_offset.full = [H_offset,V_offset];
    % %
    if corr_opts.visual
        downsample = imresize(imcorr,0.02);
        
        stfig('Correlating imgs');
        colormap(viridis)
        clf
        subplot(2,2,1)
        hold on
        imagesc(imcorr)
        plot(peak_loc(2),peak_loc(1),'r+','MarkerSize',10)
        subplot(2,2,2)
        surf(downsample,'EdgeColor',.1*[0.7,0.7,0.7])
        subplot(2,2,3)
        imagesc(this_img.grey+imtranslate(that_img.grey,0*[H_offset,V_offset]))
        subplot(2,2,4)
        imagesc(this_img.grey+imtranslate(that_img.grey,[H_offset,V_offset]))
        suptitle('Correlation between imgs')

%         cli_header(2,'Plots done');
    end
end