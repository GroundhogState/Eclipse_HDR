function dark = dark_noise_correct(dark)
    im_height = size(dark.mean_dark,1);
    im_width = size(dark.mean_dark,2);
    num_pixels = im_height * im_width;
    dark_edges = linspace(0,0.3,dark.num_bins+1);
    cli_header(2,'Parsing dark img');
    % Histogramming
    [dark.red.counts,dark_edges] = histcounts(dark.mean_dark(:,:,1),dark_edges);
    dark.green.counts = histcounts(dark.mean_dark(:,:,2),dark_edges);
    dark.blue.counts = histcounts(dark.mean_dark(:,:,3),dark_edges);
    dark.bins = 0.5*(dark_edges(2:end)+dark_edges(1:end-1));

    dark.red.cdf = intCDF(dark.red.counts);
    dark.blue.cdf = intCDF(dark.blue.counts);
    dark.green.cdf = intCDF(dark.green.counts);

    dark.red.cutoff_idx = sum(dark.red.cdf/num_pixels < dark.noise.cutoff(1))+1;
    dark.green.cutoff_idx = sum(dark.green.cdf/num_pixels < dark.noise.cutoff(2))+1;
    dark.blue.cutoff_idx = sum(dark.blue.cdf/num_pixels < dark.noise.cutoff(3))+1;

    cli_header(2,'Noise floor cutoffs:');
    cli_header(3,'Red:   %.4f',dark.bins(dark.red.cutoff_idx));
    cli_header(3,'Green: %.4f',dark.bins(dark.green.cutoff_idx));
    cli_header(3,'Blue:  %.4f',dark.bins(dark.blue.cutoff_idx));

    % Find pixels above cutoff and flag them
    dark.hot.pixels = zeros(im_height,im_width,3);
    dark.hot.pixels(:,:,1)= dark.mean_dark(:,:,1)>dark.hot.cutoff(1);
    dark.hot.pixels(:,:,2) = dark.mean_dark(:,:,2)>dark.hot.cutoff(2);
    dark.hot.pixels(:,:,3) = dark.mean_dark(:,:,3)>dark.hot.cutoff(3);
    % %
    [row_,col_,chan_] = ind2sub([im_height,im_width],find(dark.hot.pixels));
    dark.hot_spots = [col_,row_,chan_];

    cli_header(2,'Total hot pixels: [%u,%u,%u].',squeeze(sum(sum(dark.hot.pixels))));

    if dark.viz
        f=stfig('Dark CDFs');
        clf;
        subplot(2,2,1)
        hold on
        plot(dark.bins,dark.red.counts,'r+')
        plot(dark.bins,dark.green.counts,'bx')
        plot(dark.bins,dark.blue.counts,'go')
        title('Pixel values')
        xlabel('Value')
        ylabel('counts')
        set(gca,'Yscale','log')
        set(gca,'FontSize',20)

        subplot(2,2,2)
        hold on
        plot(dark.bins,dark.red.cdf,'r')
        plot(dark.bins,dark.green.cdf,'g')
        plot(dark.bins,dark.blue.cdf,'b')
        plot(dark.bins(dark.red.cutoff_idx)*[1,1],[0,num_pixels],'r:')
        plot(dark.bins(dark.green.cutoff_idx)*[1,1],[0,num_pixels],'g:')
        plot(dark.bins(dark.blue.cutoff_idx)*[1,1],[0,num_pixels],'b:')
        title('CDF')
        xlabel('Intensity')
        ylabel('Num px $\\<$ I')
        set(gca,'FontSize',20)
        
        subplot(2,2,3)
        hold on
        plot(dark.bins,dark.red.cdf,'r')
        plot(dark.bins,dark.green.cdf,'g')
        plot(dark.bins,dark.blue.cdf,'b')
        plot(dark.bins(dark.red.cutoff_idx)*[1,1],[0,num_pixels],'r:')
        plot(dark.bins(dark.green.cutoff_idx)*[1,1],[0,num_pixels],'g:')
        plot(dark.bins(dark.blue.cutoff_idx)*[1,1],[0,num_pixels],'b:')
        xlabel('Intensity')
        title('Log CDF')
        ylabel('Num px $\\<$ I')
        set(gca,'Yscale','log')
        set(gca,'xscale','log')
        set(gca,'FontSize',20)
        
        subplot(2,2,4)
        imagesc(dark.hot.pixels)
        hold on
        plot(dark.hot_spots(chan_==1,1),dark.hot_spots(chan_==1,2),'r+')
        plot(dark.hot_spots(chan_==2,1),dark.hot_spots(chan_==2,2),'gx')
        plot(dark.hot_spots(chan_==3,1),dark.hot_spots(chan_==3,2),'bo')
%         colormap('inferno')
        title('Hot pixels')
        set(gca,'FontSize',20)
    end
    if dark.save.figs
        saveas(f,fullfile(dark.outpath,'diagnostics','hot_pixels.png'))
    end
    if dark.save.data
        save(fullfile(dark.outpath,'dark_data.mat'),'dark')
    end
    cli_header(2,'Done and saved');
end