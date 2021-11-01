% Testing radial gradients...

imsize = 100;
x = linspace(-imsize,imsize,2*imsize+1);
y = linspace(-imsize,imsize,2*imsize+1);
[X,Y] = meshgrid(x,y);
Th = atan2(Y,X);
R = (X.^2 + Y.^2);
% Im = exp(-(2*R/imsize).^2);
Im = R;


[grad_len,grad_dir] = imgradient(Im);
th_grad = sin(pi*grad_dir/180+Th).*grad_len;
r_grad = cos(pi*grad_dir/180+Th).*grad_len;

stfig('grads');
clf
colormap('inferno')


subplot(1,2,1)
imagesc(Im);
title('Image')
daspect([1,1,1])


subplot(2,4,3)
imagesc(grad_len)
title('Grad magnitude')
daspect([1,1,1])

subplot(2,4,4)
imagesc(grad_dir)
title('grad direction')
daspect([1,1,1])

subplot(2,4,7)
imagesc(th_grad)
title('Theta gradient')
daspect([1,1,1])

subplot(2,4,8)
imagesc(r_grad)
title('R gradient')
daspect([1,1,1])