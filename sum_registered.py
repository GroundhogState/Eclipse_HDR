import numpy as np
import scipy as scp
import skimage as ski
from skimage import feature
import matplotlib.pyplot as plt
import skimage.io as io
import os
from tqdm import tqdm, tqdm_notebook
from mpl_toolkits.mplot3d import Axes3D  
# Axes3D import has side effects, it enables using projection='3d' in add_subplot
import matplotlib.pyplot as plt
from eclipse_init import *

exposure_times={"65": 1/1250,
"66": 1/800,
"67": 1/640,
"68": 1/500,
"69": 1/400,
"70": 1/320,
"71": 1/250,
"72": 1/160,
"73": 1/100,
"74": 1/60,
"75": 1/40,
"76": 1/25,
"77": 1/20,
"78": 1/15,
"79": 1/13,
"80": 1/10,
"81": 1/8,
"83": 1/6,
"84": 1/5,
"85": 1/4,
"86": 1/3,
"87": 1/2,
"88": 0.6,
"89": 0.8,
"90": 1}

# Estimate the luminance of each pixel.
# We do this by masking out the saturated regions, and those below the noise level.
	# Requires understanding of noise!
# Then, scale the remaining regions by exposure time 
# Take the mean at each pixel 


noise_level=np.r_[628.15629107 ,217.18334989, 251.76106425]
noise_std= np.r_[193.35665065,  34.69773851,  64.33101507]
noise_floor = noise_level+3*noise_std
mask_thresh = 0.98
sigma=3

data_root = "C:\\Users\\jaker\\Pictures\\Oregon_Eclipse\\16bit\\registered\\"
dir_files=os.listdir(data_root)
img_fnames = [fn for fn in dir_files if fn[0:3] == 'IMG']
img_paths = [data_root+fn for fn in dir_files if fn[0:3] == 'IMG']
num_imgs = len(img_paths)
img_tags = [fname[6:8] for fname in img_fnames]
# Import the good stuff
# print('%u images found'%(num_imgs))
# all_images = np.r_[[get_img_with_vips(path) for path in img_paths]]
# print('Images imported')
all_exp = np.r_[[exposure_times[tag] for tag in img_tags]]

# what is the length scale of this noise? 
	# Looks like things are happy with a 3px kernel. That's not so bad, eh?
	# the camera shake/focus is so far out that the resolved star is huge!
	# 

# Estimating pixel luminance:
# Retrieve rectangular limits and compute weighted sum


sum_img_paths = img_paths[7:]
num_sum_imgs = len(sum_img_paths)
sum_exp = all_exp[7:]
for idx in np.r_[0:num_sum_imgs]:
    print('%u/%u'%(idx,num_sum_imgs))
    this_img = get_img_with_vips(sum_img_paths[idx])
    if idx==0:
        # Initialize
        im_counter = np.zeros(this_img.shape)
        mean_img = np.zeros(this_img.shape)
        sqr_img = np.zeros(this_img.shape)
        dark_level = np.dstack((noise_floor[0]*np.ones([this_img.shape[0],this_img.shape[1]]),
            noise_floor[1]*np.ones([this_img.shape[0],this_img.shape[1]]),
            noise_floor[2]*np.ones([this_img.shape[0],this_img.shape[1]])))
        dark_std = np.dstack((noise_std[0]*np.ones([this_img.shape[0],this_img.shape[1]]),
            noise_std[1]*np.ones([this_img.shape[0],this_img.shape[1]]),
            noise_std[2]*np.ones([this_img.shape[0],this_img.shape[1]])))
    # else:
    c_img = this_img-dark_level
    c_img[c_img<0]=0
    this_img = rgb_filter(c_img,5)/sum_exp[idx]
    mean_img+=this_img
    sqr_img+=(this_img)**2
    [idx,idy,idc]=np.nonzero(this_img)
    lims = [[min(idx),max(idx)],[min(idy),max(idy)]]
    im_counter[lims[0][0]:lims[0][1],lims[1][0]:lims[1][1]]+=1
mean_img = mean_img/num_sum_imgs
std_img = sqr_img/num_sum_imgs-mean_img**2

# ref_img = get_img_with_vips(img_paths[0])
# im_counter = np.zeros([ref_img.shape[0],ref_img.shape[1]])
# sum_imgs = np.zeros([ref_img.shape[0],ref_img.shape[1],3])
# for im_idx in np.r_[0:num_imgs]:
#     print(im_idx)
#     test_img = get_img_with_vips(img_paths[im_idx])
#     test_grey = ski.color.rgb2gray(test_img)
#     test_mask = test_grey<mask_thresh
#     test_img = test_img*test_mask.reshape(test_img.shape[0],test_img.shape[1],1)
#     [idx,idy,idc]=np.nonzero(test_img)
#     lims = [[min(idx),max(idx)],[min(idy),max(idy)]]
#     im_counter[lims[0][0]:lims[0][1],lims[1][0]:lims[1][1]]+=1
#     sum_imgs+=test_img/all_exp[im_idx]

# Crop and smooth the image
cnr = [1350,4800,300,5200]
mean_img = mean_img[cnr[0]:cnr[1],cnr[2]:cnr[3]]
im_counter = im_counter[cnr[0]:cnr[1],cnr[2]:cnr[3]]

mean_img = mean_img/im_counter.reshape(mean_img.shape[0],mean_img.shape[1],1)
std_img = std_img/im_counter.reshape(std_img.shape[0],std_img.shape[1],1)

print('Saving output...')
np.savetxt(data_root+"txt_r_data.txt",mean_img[:,:,0])
np.savetxt(data_root+"txt_g_data.txt",mean_img[:,:,1])
np.savetxt(data_root+"txt_b_data.txt",mean_img[:,:,2])

np.savetxt(data_root+"txt_r_std.txt",std_img[:,:,0])
np.savetxt(data_root+"txt_g_std.txt",std_img[:,:,1])
np.savetxt(data_root+"txt_b_std.txt",std_img[:,:,2])
print('Done.')
[h,b]=rgb_histograms(crop_mean)
image_diagnostic(crop_mean)