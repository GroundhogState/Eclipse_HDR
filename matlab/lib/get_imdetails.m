function img_this = get_imdetails(img_in)
    img_this.img = img_in;
    img_this.grey = rgb2gray(img_this.img);
    num_bright_bins = 2^8;
    [img_this.bright.counts,bright_bin_edges] = histcounts(img_this.grey,num_bright_bins);
    img_this.bright.centres = .5*(bright_bin_edges(1:end-1) + bright_bin_edges(2:end));
    img_this.stats.cdf = nIntegrate(img_this.bright.counts);
    img_this.stats.mean = mean(mean(img_this.grey));
    img_this.stats.peak = max(max(img_this.grey));
end

