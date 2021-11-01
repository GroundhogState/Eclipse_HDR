% Playing around with conv/deconv and noise

% the model: A ground-truth image
y = 1:100;
x = 1:100;
[X,Y] = meshgrid(x,y);
centre = [50,50];
source_size = [5,5];
true_image = exp(-0.5*((X-centre(1))/source_size(1)).^2-0.5*((Y-centre(2))/source_size(2)).^2);


% Some camera shake in the form of a parametrized path
t = linspace(0,1,100);
shake_path = [20*t;5*t.^2];
for tt = 1:length(t)
   if tt == 1
       shake_image = true_image;
   else
       shake_image = shake_image + imtranslate(true_image,shake_path(:,tt)');
   end
end
shake_image = shake_image/length(shake_path);


% some optic blur
blur_size = 5;
blur_kernel = gaussian2d(20,blur_size);
blur_image = conv2(shake_image,blur_kernel,'same');




% and some Gaussian noise - in reality there is going to be photon noise
% and sensor noise, but these should probably not be affected by optic
% artefacts
noise_amp = 0.01;
noise_data = noise_amp*randn(size(X));
noisy_image = noise_data + blur_image;



stfig('Imgs');
clf
subplot(2,2,1)
imagesc(true_image)
subplot(2,2,2)
imagesc(shake_image)
subplot(2,2,3)
imagesc(blur_image)
subplot(2,2,4)
imagesc(noisy_image)


colormap(viridis)

% %



 function f=gaussian2d(N,sigma)
      % N is grid size, sigma speaks for itself
     [x, y]=meshgrid(round(-N/2):round(N/2), round(-N/2):round(N/2));
     f=exp(-x.^2/(2*sigma^2)-y.^2/(2*sigma^2));
     f=f./sum(f(:));
 end