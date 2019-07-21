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

# If I need to write this again, then instead of saving all the images the first time,
# I'll compute the pixel registration shifts and save them to disk.
# Then, you can work out exactly how big your frame needs to be, and iterate accordingly.

print('Setting up...')
data_root = "C:\\Users\\jaker\\Pictures\\Oregon_Eclipse\\16bit\\"
dir_files=os.listdir(data_root+"\\crop\\")
crop_paths = [data_root+fn for fn in dir_files if fn[0:3] == 'IMG']
frame_size = [5000,6000]

[rgb_sum,n_sum]=vips_channel_registration(data_root,crop_paths,frame_size,sigma=5,
	mask_level=0.45,verbose=2,subsampwidth=1500)

r_sum = rgb_sum[0]
g_sum = rgb_sum[1]
b_sum = rgb_sum[2]
zoom_range = [1000,3000,2000,4000]

rhist=np.histogram(r_sum,300)
rcounts = rhist[0]
ghist=np.histogram(g_sum,300)
gcounts = ghist[0]
bhist=np.histogram(b_sum,300)
bcounts = bhist[0]

hist_bins = 0.5*(rhist[1][1:]+rhist[1][:-1])


plt.figure(figsize=(9,9))
plt.subplot(221)
plt.imshow(r_sum[zoom_range[0]:zoom_range[1],zoom_range[2]:zoom_range[3]])
plt.subplot(222)
plt.imshow(g_sum[zoom_range[0]:zoom_range[1],zoom_range[2]:zoom_range[3]])
plt.subplot(223)
plt.imshow(b_sum[zoom_range[0]:zoom_range[1],zoom_range[2]:zoom_range[3]])
plt.subplot(224)
plt.imshow(n_sum)

plt.show()

plt.figure(figsize=(7,7))
plt.subplot(2,1,1)
plt.plot(hist_bins[1:],rcounts[1:],color='red')
plt.plot(hist_bins[1:],gcounts[1:],color='green')
plt.plot(hist_bins[1:],bcounts[1:],color='blue')
plt.subplot(2,1,2)
plt.plot(hist_bins[1:],rcounts[1:],color='red')
plt.plot(hist_bins[1:],gcounts[1:],color='green')
plt.plot(hist_bins[1:],bcounts[1:],color='blue')
plt.yscale('log')
plt.show()