clear all
dirname = 'C:\Users\jaker\Pictures\Astrophotography\Sensor_calibration\jpg';
fnames = get_files(dirname,'jpg');

im.g = imread(fullfile(dirname,fnames{22}));


% How about a HPF?
im.white_mask = rescale(im.g(:,:,1).*im.g(:,:,2).*im.g(:,:,3));
% starfind = find(white_mask>200);
% [star_row,star_col] = ind2sub(starfind,size(white_mask));

% sme stars appear 5-10 pix in size
%%
im.grey = rgb2gray(im.g);
stfig('starfield');
clf
% subplot(2,2,1)
hold on
imagesc(im.g)
% %
threshold = 10;
fsize = 20;
filt = gausswin(5,fsize).*gausswin(10,fsize)';
filt = filt/sum(filt,'all');
peaks = FastPeakFind(rgb2gray(im.g), threshold, filt , 1);
cli_header('%u peaks with fsize %u and threshold %u', size(peaks,1)/2, fsize,threshold);

% threshold = 5;
% fsize = 10;
% filt = gausswin(5,fsize).*gausswin(10,fsize)';
% filt = filt/sum(filt,'all');
% peaks{2} = FastPeakFind(im.grey, threshold, filt , 1);
% cli_header('%u peaks with fsize %u and threshold %u', size(peaks{2},1)/2, fsize,threshold);

P_x = peaks(1:2:end-1);
P_y = peaks(2:2:end);

plot(P_x,P_y,'bo')


%% Now go fit them!
num_pks = length(P_x);

samplewidth = 12;

% pidx = 12;
p_all = zeros(num_pks,6);
% ffun = 
for pidx = 1:num_pks
    if mod(pidx-1,10) == 0
        cli_header('Fitting peak %u:',pidx);
    end
    try
        
        centre = [P_y(pidx),P_x(pidx)];
        sample.colour = im.g((centre(1)-samplewidth):(centre(1)+samplewidth),(centre(2)-samplewidth):(centre(2)+samplewidth),:);
        sample.grey = im.grey((centre(1)-samplewidth):(centre(1)+samplewidth),(centre(2)-samplewidth):(centre(2)+samplewidth));

        subplot(2,2,2)
        imagesc(sample.colour)


        % %
        imsize = 2*samplewidth+1;
        centre = [0,0];
        amp = double(max(sample.grey,[],'all'));
        sigma_x = 1;
        sigma_y = 1;
        r = .01;

        param_guess = [centre,amp,sigma_x,sigma_y,r];
        img_errfun(param_guess,sample.grey)';
        fit_cost = @(param) img_errfun(param,sample.grey);
        p_opt = fminsearch(fit_cost,param_guess);

    %     subplot(2,2,3)
    %     G = gaussian2d(imsize,centre,amp,sigma_x,sigma_y,r);
    %     imagesc(G)
    %     daspect([1,1,1])
    % 
    %     subplot(2,2,4)
    %     H = gaussian2d(imsize,[p_opt(1),p_opt(2)],p_opt(3),p_opt(4),p_opt(5),p_opt(6));
    %     imagesc(H)
    %     daspect([1,1,1])

        p_all(pidx,:) = p_opt;
    catch
        p_all(pidx,:) = nan(1,6);
    end
end
% 
% subplot(2,2,3)
% G = gaussian2d(imsize,centre,amp,sigma_x,sigma_y,r);
% imagesc(G)
% daspect([1,1,1])
% 
% subplot(2,2,4)
% H = gaussian2d(imsize,[p_opt(1),p_opt(2)],p_opt(3),p_opt(4),p_opt(5),p_opt(6));
% imagesc(H)
% daspect([1,1,1])

%%
% (imsize,centre,amp,sigma_x,sigma_y,r)

sx = 150;
sy = 150;
rmax = 2;

co = viridis(length(p_all));
stfig('Star stats');
clf
subplot(4,2,1)
plot(p_all(:,4),p_all(:,5),'x')
set(gca,'ColorOrder',co)
xlim([-1,sx])
ylim([-1,sy])
xlabel('$\sigma_x$')
ylabel('$\sigma_y$')

subplot(4,2,3)
bin_edges = linspace(0,sqrt(sx*sy),round(length(p_all)/2));
histogram(sqrt(p_all(:,4).^2+p_all(:,4)).^2,bin_edges);
xlabel('$\sqrt{\sigma_x \sigma_y}$')

subplot(4,2,2)
plot(p_all(:,4),p_all(:,6),'rx')
xlabel('$\sigma_x$')
ylabel('r')
xlim([0,sx])
ylim([-rmax,rmax])

subplot(4,2,4)
plot(p_all(:,5),p_all(:,6),'kx')
xlabel('$\sigma_y$')
ylabel('r')
xlim([0,sy])
ylim([-rmax,rmax])

subplot(4,2,5)
hold on
plot(P_x,p_all(:,4),'k.')
plot(P_x,p_all(:,5),'r.')
xlabel('x')
ylabel('$\sigma$')
ylim([0,sx])

subplot(4,2,7)
hold on
plot(P_y,p_all(:,4),'k.')
plot(P_y,p_all(:,5),'r.')
xlabel('y')
ylabel('r')
ylim([0,sy])

subplot(4,2,6)
plot(P_x,p_all(:,6),'kx')
ylabel('r')
ylim([-rmax,rmax])

subplot(4,2,8)
plot(P_y,p_all(:,6),'kx')
ylabel('r')
ylim([-rmax,rmax])


% subplot(2,2,3)
% hold on
% histogram(p_all(:,6),100)
% xlabel('r')
%%
stfig('yeehaw');
% [PX,PY] = meshgrid(P_x,P_y)

scatter3(P_x,P_y,(p_all(:,6)))
zlim([-rmax,rmax])

%%
% What's the correlation between X/Y and r, or other blurs?

function C = img_errfun(p,image)
    % X_Y is a list (all X then all Y)
    
    Z = gaussian2d(length(image)-1,[p(1),p(2)],p(3),p(4),p(5),p(6));
    
    C = norm(Z-im2double(image));
    

end

function Z = gaussian2d(imsize,centre,amp,sigma_x,sigma_y,r)

% want to fit a gaussian with some rotation/skew in it too
    
    [X,Y] = meshgrid(-imsize/2:imsize/2);
%     XY = [X(:),Y(:)];
    
    A = amp/(2*pi*sigma_x*sigma_y*sqrt(1-r^2));
    C = 1/(2*(1-r^2));
    E = (((X-centre(1))/sigma_x).^2 + ((Y-centre(2))/sigma_y).^2 - ...
        2*r* ((X-centre(1)).*(Y-centre(2))) /(sigma_x*sigma_y));
    
    Z = A*exp(-C*E);
    
end

% opts.log = true;
% opts.loglog = true;
% rgb_hist(im.g,opts)
% 
% 
% % clear 
% % spec.stack = hstack(spec.red,spec.green,spec.blue);
% %%
% spec.red = abs(fft2(im.g(:,:,1)));
% spec.green = abs(fft2(im.g(:,:,2)));
% spec.blue = abs(fft2(im.g(:,:,3)));
% stfig('Spectral');
% clf;
% subplot(2,2,1)
% imagesc(log(spec.red));
% subplot(2,2,2)
% imagesc(log(spec.green));
% subplot(2,2,3)
% imagesc(log(spec.blue));


% find some stars


