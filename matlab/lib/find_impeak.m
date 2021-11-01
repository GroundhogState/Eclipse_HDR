function [m,loc] = find_impeak(img)
% Returns the coordinates of the max pixel in 2D array. 
% uses image/matrix coordinates - x is the horizontal direction
% (conventionally 2nd index for images)
%! example
% img = magic(5);
% [m,loc] = find_impeak(img)
% img(loc(1),loc(2)) == m
%!
    [m,i]=max(img,[],'all','linear');
    imsize = size(img);
    x = ceil(i/imsize(1));
    y = i-(x-1)*imsize(1);
    loc = [y,x];
end