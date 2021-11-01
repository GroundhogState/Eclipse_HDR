function signal = correct_and_align_imgs(signal,dark)
    net_offset = [0,0];
    ctr = 0;
    for im_idx = 1:numel(signal.fnames)
        ctr = ctr + 1;
        cli_header('Loading image %u',im_idx);
        this_img = im2double(imread(fullfile(signal.dir.name,signal.fnames{im_idx})));   
    %     that_img = im2double(imread(fullfile(dirname,fnames{im_idx+1})));  
        this_file_idx = signal.file_idxs(im_idx);
        exp_time = signal.exposure_times(signal.exposure_times(:,1) == this_file_idx,2);

        signal_red = squeeze(im2double(this_img(:,:,1)));
        signal_green = squeeze(im2double(this_img(:,:,2)));
        signal_blue = squeeze(im2double(this_img(:,:,3)));

        % thresholding
        signal_red(signal_red<signal.cutoff(1)| signal_red > signal.saturation_level) = 0;
        signal_blue(signal_blue<signal.cutoff(2)| signal_blue > signal.saturation_level) = 0;
        signal_green(signal_green<signal.cutoff(3)| signal_green > signal.saturation_level) = 0;
        
        if ctr == 1
            signal.rgb = cat(3,signal_red,signal_green,signal_blue)-dark.mean_dark;
            signal.rgb(signal.rgb<0) = 0;
            signal.rgb = signal.rgb/exp_time;
            signal.rgb(dark.hot_spots) = 0;
            num_samples = signal.rgb>0;
        else
            net_offset = net_offset + signal.translation_offsets(im_idx-1,:);
            new_rgb = cat(3,signal_red,signal_green,signal_blue)-dark.mean_dark;
            signal.rgb(dark.hot_spots) = 0;
            new_rgb(new_rgb<0) = 0;
            signal.rgb = signal.rgb+imtranslate(new_rgb/exp_time,net_offset);
            clear new_rgb
            num_samples = num_samples+(signal.rgb>0);
        end
    
        % Have a go at correcting hot pixels
        hot_px{1} = dark.hot_spots(dark.hot_spots(:,3)==1,1:2);
        hot_px{2} = dark.hot_spots(dark.hot_spots(:,3)==2,1:2);
        hot_px{3} = dark.hot_spots(dark.hot_spots(:,3)==3,1:2);
        pixel_pinched = this_img;
        imkernel = [1,1,1;
                    1,0,1;
                    1,1,1]/8;
        for chan=1:3
            these_hot_px = hot_px{chan};
            num_hot_pix = length(these_hot_px);
            for pix_idx = 1:num_hot_pix
                swidth=1;
                subsample = pixel_pinched((these_hot_px(1)-swidth):(these_hot_px(1)+swidth),(these_hot_px(2)-swidth):(these_hot_px(2)+swidth),3);
                local_av = imkernel.*subsample;
                pixel_pinched(these_hot_px(1),these_hot_px(2),chan) = sum(local_av,'all');
            end
        end
        cli_header(3,'Smoothed hot pixels');
        
        data_in.image = this_img;
        data_in.grad_sense = signal.grad_sense;
        im_grad = get_polar_gradient(data_in);
        

        if signal.viz.plots
            
            cli_header('Init fig');

            h=3;
            w=2;
            f1=stfig('Single-image plots');
            clf           
            
            
            
            subplot(h,w,1)
            imagesc(sqrt(im_grad.mag))
            hold on
%             plot(hot_px{1},1),dark.hot_spots(hot_px{1},2),'ro')
%             plot(dark.hot_spots(hot_px{2},1),dark.hot_spots(hot_px{2},2),'go')
%             plot(dark.hot_spots(hot_px{3},1),dark.hot_spots(hot_px{3},2),'bo')
            title('sqrt(Intensity gradient)')
            
            daspect([1,1,1])
            xticks([])
            yticks([])
            
            
            subplot(h,w,2)
            imagesc(sqrt(imgradient(rgb2gray(pixel_pinched))))
            daspect([1,1,1])
            xticks([])
            yticks([])
            title('Pixel-pinched gradient','FontSize',10)
            
            subplot(h,w,3)
            imagesc(im_grad.dir)
            daspect([1,1,1])
            xticks([])
            yticks([])
            title('Gradient direction','FontSize',10)
            
            subplot(h,w,5)
            imagesc(im_grad.d_r)
            hold on
            daspect([1,1,1])
            xticks([])
            yticks([])
            
            title('Radial gradient','FontSize',10)
            
            subplot(h,w,6)
            imagesc(im_grad.d_theta)
            daspect([1,1,1])
            xticks([])
            yticks([])
            
            title('Circumferential gradient','FontSize',10)
            
            colormap(inferno)
            
%             [bright.red.counts,bright_edges] = histcounts(signal_red,signal.num_bins);
%             bright.green.counts = histcounts(signal_green,signal.num_bins);
%             bright.blue.counts = histcounts(signal_blue,signal.num_bins);
%             bright_bins = 0.5*(bright_edges(2:end)+bright_edges(1:end-1));
% 
%             [flux.red.counts, final.edges]= histcounts(signal.rgb(:,:,1),signal.num_bins);
%             flux.green.counts = histcounts(signal.rgb(:,:,2),signal.num_bins);
%             flux.blue.counts = histcounts(signal.rgb(:,:,3),signal.num_bins);
%             final.bins = 0.5*(final.edges(2:end)+final.edges(1:end-1));


%             f2=stfig('single-image histograms');
%             clf
%             h=2;    
%             w=3;
%             subplot(h,w,1)
%             cla
%             hold on
%             plot(dark.bins,dark.red.counts,'r:')
%             plot(dark.bins,dark.green.counts,'g:')
%             plot(dark.bins,dark.blue.counts,'b:')
%             plot(bright_bins,bright.red.counts,'r')
%             plot(bright_bins,bright.green.counts,'g')
%             plot(bright_bins,bright.blue.counts,'b')
%             
%             title('Bright \& Dark image histograms')
% 
%             subplot(h,w,2)
%             cla
%             hold on
%             plot(dark.bins,dark.red.counts,'r:')
%             plot(dark.bins,dark.green.counts,'g:')
%             plot(dark.bins,dark.blue.counts,'b:')
%             plot(bright_bins,bright.red.counts,'r')
%             plot(bright_bins,bright.green.counts,'g')
%             plot(bright_bins,bright.blue.counts,'b')
%             title('Bright \& Dark image histograms')
%             
%             set(gca,'Yscale','log')
% 
%             subplot(h,w,3)
%             cla
%             hold on
%             plot(dark.bins,dark.red.counts,'r:')
%             plot(dark.bins,dark.green.counts,'g:')
%             plot(dark.bins,dark.blue.counts,'b:')
%             plot(bright_bins,bright.red.counts,'r')
%             plot(bright_bins,bright.green.counts,'g')
%             plot(bright_bins,bright.blue.counts,'b')
%             title('Bright \& Dark image histograms')
%             
%             set(gca,'Yscale','log')
%             set(gca,'Xscale','log')
% 
%             subplot(h,w,4)
%             cla
%             hold on
%             plot(final.bins,flux.red.counts,'r')
%             plot(final.bins,flux.green.counts,'g')
%             plot(final.bins,flux.blue.counts,'b')
%             title('Flux histograms')
%             
%             subplot(h,w,5)
%             cla
%             hold on
%             plot(final.bins,flux.red.counts,'r')
%             plot(final.bins,flux.green.counts,'g')
%             plot(final.bins,flux.blue.counts,'b')
%             
%             title('Flux log histograms')
%             set(gca,'Yscale','log')
%             drawnow
        
            
            if signal.save.single_img_figs 
                saveas(f1,fullfile(dark.signal.dir.out,'diagnostics','this_img.svg'))
%                 saveas(f2,fullfile(dark.signal.dir.out,'diagnostics','this_img_stats.svg'))
%                 saveas(f3,fullfile(dark.signal.dir.out,'diagnostics','this_img_gradients.svg'))
                cli_header(4,'Figs saved');
            end
            if signal.viz.running_mean 
                f4 = stfig('Mean img');
                clf;
                subplot(2,1,1)
                imagesc(signal.rgb/ctr)
                daspect([1,1,1])
                title('Mean image')

                subplot(2,1,2)
                imagesc(num_samples)
                daspect([1,1,1])
                title('Sample intensity')

                colormap(inferno)

                    if signal.save.figs 
                        saveas(f4,fullfile(dark.signal.dir.out,'diagnostics','this_img_gradients.svg'))
                        cli_header(3,'figssaved');
                    end
            end
            
        end
        if signal.save.single_imgs
            if ~exist(fullfile(signal.dir.out,'single_imgs',signal.fnames{im_idx},'dir'))
                mkdir(fullfile(signal.dir.out,'single_imgs',signal.fnames{im_idx}));
            end
            imwrite(sqrt(imgradient(rgb2gray(pixel_pinched))),fullfile(signal.dir.out,'single_imgs',signal.fnames{im_idx},'pixel_pinched.png'))
            cli_header(2,'saved');
            imwrite(sqrt(im_grad.mag),fullfile(signal.dir.out,'single_imgs',signal.fnames{im_idx},'intensity_gradient.png'))
            cli_header(2,'saved');
            imwrite(im_grad.d_theta,fullfile(signal.dir.out,'single_imgs',signal.fnames{im_idx},'theta_gradient.png'))
            cli_header(2,'saved');
            imwrite(im_grad.dir,fullfile(signal.dir.out,'single_imgs',signal.fnames{im_idx},'grad_direction.png'))
            cli_header(2,'saved');
        end
        
    end
    % plot mean stuff
    if signal.viz.plots
        f4 = stfig('Mean img');
        clf;
        subplot(2,2,1)
        imagesc(signal.rgb/ctr)
        title('Mean image')
        
        subplot(2,2,2)
        imagesc(num_samples)
        title('Sample intensity')
                
        colormap(inferno)

        if signal.save.figs 
            saveas(f4,fullfile(dark.signal.dir.out,'diagnostics','this_img_gradients.svg'))
            cli_header(3,'figssaved');
        end
        
    end
        
    if signal.save.data
        save(fullfile(signal.signal.dir.out,'signal_data.mat'),'dark')
        cli_header(3,'Data saved');
    end
    cli_header(2,'done.');
end