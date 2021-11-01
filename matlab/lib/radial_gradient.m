function im_grad = get_polar_gradient(data_in)
            [imgrad,grad_dir] = imgradient(rgb2gray(data_in.image));
    %         x_len=5194;
    %         y_len=3457;
            x = linspace(1,5194,5194);
            y = linspace(1,3457,3457);
            [X,Y] = meshgrid(x,y);
            grad_sense = imgrad>data_in.grad_sense;
            x_com=sum(X.*grad_sense,'all')/sum(grad_sense,'all');
            y_com=sum(Y.*grad_sense,'all')/sum(grad_sense,'all');
            x_polar = X - x_com;
            y_polar = Y - y_com;
    %         r_polar = sqrt(x_polar.^2+y_polar.^2);
            th_polar = atan2(y_polar,x_polar);
            th_grad = sin(pi*grad_dir/180+th_polar).*log(imgrad);
            r_grad = cos(pi*grad_dir/180+th_polar).*log(imgrad);

            im_grad.d_theta = th_grad;
            im_grad.d_r = r_grad;
            im_grad.mag = imgrad;
            im_grad.dir = grad_dir;
end