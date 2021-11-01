function rgbgrad = imgrad_rgb(img)
    rgbgrad = zeros(size(img));
    for i=1:3
       rgbgrad(:,:,i) = imgradient(img(:,:,i)); 
    end


end