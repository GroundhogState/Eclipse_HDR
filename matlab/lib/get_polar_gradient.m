function im_grad = get_polar_gradient(image,com)
            if length(size(image)) == 3
                [imgrad,grad_dir] = imgradient(rgb2gray(image));
            else
                [imgrad,grad_dir] = imgradient(im2double(image));
            end
    %         x_len=5194;
    %         y_len=3457;
            x = linspace(1,5194,5194);
            y = linspace(1,3457,3457);
            [X,Y] = meshgrid(x,y);
            x_polar = X - com(1);
            y_polar = Y - com(2);
            th_polar = atan2(y_polar,x_polar);
            th_grad = sin(pi*grad_dir/180+th_polar).*log(imgrad);
            r_grad = cos(pi*grad_dir/180+th_polar).*log(imgrad);

            im_grad.d_theta = th_grad;
            im_grad.d_r = r_grad;
            im_grad.mag = imgrad;
            im_grad.dir = grad_dir;
%             im_grad.com = [x_com,y_com];
end