function this_img = rgb_hist(img_in,varargin)
    
% needs: an RGB image in
% options: num bins?
%     label
%     this_img = get_imdetails(img_in);
    if nargin > 1
        opts = varargin{1};
    else
        opts = [];
    end
    
    if ~isfield(opts,'lin'),opts.lin = true;end
    if ~isfield(opts,'log'),opts.log = true;end
    if ~isfield(opts,'loglog'),opts.loglog = false;end
    if ~isfield(opts,'visual'),opts.visual = true;end
    if ~isfield(opts,'label'),opts.label = 'Img hists';end
    if ~isfield(opts,'num_bins'),opts.num_bins= 150;end
    if ~isfield(opts,'unit_bins'),opts.unit_bins= false;end
    
    if isfield(opts,'max_val'), this_img.max_val=opts.max_val;
    else, this_img.max_val = max(img_in,[],'all');end
    
    if isfield(opts,'min_val'),this_img.min_val=opts.min_val;
    else,this_img.min_val = min(img_in,[],'all');end
    
    

    bin_edges = linspace(double(this_img.min_val),double(this_img.max_val),opts.num_bins+1);    
    [this_img.red.counts, this_img.edges]= histcounts(img_in(:,:,1),bin_edges);
    this_img.green.counts = histcounts(img_in(:,:,2),bin_edges);
    this_img.blue.counts = histcounts(img_in(:,:,3),bin_edges);
    this_img.bins = 0.5*(this_img.edges(2:end)+this_img.edges(1:end-1));
    
    
    num_sf = opts.lin + opts.log + opts.loglog;
    sf = 1; 
    if opts.visual
        stfig(opts.label);
        clf;
        if opts.lin
            subplot(1,num_sf,sf)
            hold on
            plot(this_img.bins,this_img.red.counts,'r.')
            plot(this_img.bins,this_img.green.counts,'g.')
            plot(this_img.bins,this_img.blue.counts,'b.')
%             xlim([0,this_img.max_val])
            sf = sf+1;
        end
        if opts.log
            subplot(1,num_sf,sf)
            hold on
            plot(this_img.bins,this_img.red.counts,'r.')
            plot(this_img.bins,this_img.green.counts,'g.')
            plot(this_img.bins,this_img.blue.counts,'b.')
            m1 = max(this_img.red.counts);
            m2 = max(this_img.blue.counts);
            m3 = max(this_img.green.counts);
            M = max([m1,m2,m3]);
            set(gca,'Yscale','log')
            xlim([this_img.min_val,this_img.max_val])
            sf = sf+1;
        end
        if opts.loglog
            subplot(1,num_sf,sf)
            hold on
            plot(this_img.bins,this_img.red.counts,'r.')
            plot(this_img.bins,this_img.green.counts,'g.')
            plot(this_img.bins,this_img.blue.counts,'b.')
            m1 = max(this_img.red.counts);
            m2 = max(this_img.blue.counts);
            m3 = max(this_img.green.counts);
            M = max([m1,m2,m3]);
            xlim([this_img.min_val,this_img.max_val])
            set(gca,'Yscale','log')
            set(gca,'Xscale','log')
        end
        suptitle(opts.label)
    end
    
end