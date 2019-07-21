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
import pyvips
format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

dtype_to_format = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
}


data_root = "C:\\Users\\jaker\\Pictures\\Oregon_Eclipse\\16bit\\"
dir_files=os.listdir(data_root)
img_paths = [data_root+fn for fn in dir_files if fn[-3:] == 'png']
# img_paths=img_paths[2:]
img_imports = io.imread_collection(img_paths)
num_imgs = len(img_imports)



## Single image processing
# show the edges of the lunar disk and zoom
winsize = 600;
sm_sigma = 0.1
reopt_cutoff = 350 
opt_verbose = False
visual_out = False

def cost_fn(x,pts):
    return np.var(np.sqrt(np.sum((x.reshape(2,1)-pts)**2,axis=0)))

im_cens = np.zeros([num_imgs,2])
edge_seps = np.zeros([num_imgs,2,2])
disk_radii = np.zeros([num_imgs,2])

for im_idx in np.r_[0:num_imgs]:
    print('Starting img %u/%u'%(im_idx,num_imgs))
    filename = img_paths[im_idx][-12:]
    img = pyvips.Image.new_from_file(img_paths[im_idx], access='sequential')
    this_img = np.ndarray(buffer=img.write_to_memory(),
                       dtype=format_to_dtype[img.format],
                       shape=[img.height, img.width, img.bands])

    # this_img = img_imports[im_idx]
    im_greyscale = ski.color.rgb2gray(this_img) 
    
    # Detect the edges of the lunar disk
    edges = feature.canny(im_greyscale,sigma=sm_sigma)
    edge_pts = np.nonzero(edges)
    my = int(np.mean(edge_pts[0]))+np.r_[-winsize,winsize]
    mx = int(np.mean(edge_pts[1]))+np.r_[-winsize,winsize]
    print('Edges detected')
    # Find the centre by minimizing the variance of the radius function
    x0 = np.mean(edge_pts,axis=1)
    opt_res=scp.optimize.minimize(lambda x: cost_fn(x,edge_pts),x0,
                                  method='nelder-mead',options={'xtol': 1e-8, 'disp': opt_verbose})

    R = np.sqrt(np.sum((opt_res.x.reshape(2,1)-edge_pts)**2,axis=0))
    print('Centre optimized')
    # Cull points too far from disk (noise etc) and recentre
    # R_mask = R<reopt_cutoff
    # trim_pts=[edge_pts[0][R_mask],edge_pts[1][R_mask]]

    # opt_res2=scp.optimize.minimize(lambda x: cost_fn(x,trim_pts),x0,
    #                                method='nelder-mead',options={'xtol': 1e-8, 'disp': opt_verbose})

    # R2 = np.sqrt(np.sum((opt_res.x.reshape(2,1)-trim_pts)**2,axis=0))

    # Done! Write calculations for outputs
    cen_disk = np.int_(opt_res.x)
    ysep = [-cen_disk[0]+im_greyscale.shape[0],cen_disk[0]]
    xsep = [-cen_disk[1]+im_greyscale.shape[1],cen_disk[1]]
    
    [ymin,ymax] = [cen_disk[0]-min(ysep),cen_disk[0]+min(ysep)]
    [xmin,xmax] = [cen_disk[1]-min(xsep),cen_disk[1]+min(xsep)]

    ## Write the Outputs
    im_cens[im_idx] = cen_disk
    edge_seps[im_idx] = [ysep,xsep]
    disk_radii[im_idx] = [np.mean(R),np.std(R)]
    
    img_crop = this_img[ymin:ymax,xmin:xmax]
    out_fname = data_root+'\\crop\\'+filename
    # io.imsave(out_fname, img_crop)?
    height, width, bands = img_crop.shape
    linear = img_crop.reshape(width * height * bands)
    vi = pyvips.Image.new_from_memory(linear.data, width, height, bands,
                                      dtype_to_format[str(img_crop.dtype)])
    vi.write_to_file(out_fname)
    
    ## Visual/verbose out
    if visual_out:
        theta = np.r_[0:2*np.pi:0.01]
        X = np.cos(theta)
        Y = np.sin(theta)
        r2=np.median(R)
        arc2 = [r2*X,r2*Y]+opt_res.x.reshape(2,1)

        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(edges[my[0]:my[1],mx[0]:mx[1]],extent=[mx[0],mx[1],my[1],my[0]])
        plt.subplot(1,3,2)
        plt.imshow(this_img[my[0]:my[1],mx[0]:mx[1]],extent=[mx[0],mx[1],my[1],my[0]])
        plt.plot(opt_res.x[1],opt_res.x[0],'o')
        plt.plot(arc2[1],arc2[0],linewidth=3,linestyle='dotted')
        plt.subplot(1,3,3)
        plt.hist(R,500)
        plt.show()
        
        plt.figure()
        plt.imshow(img_crop,extent=[xmin,xmax,ymax,ymin])
        plt.show()
    # print('img %u of %u:'%(im_idx,num_imgs))
    print('Disk radius        (%.2f +- %.2f)'%(np.mean(R),np.std(R)))
    print('Disk centre (Y,X)  (%u,%u)'%(cen_disk[0],cen_disk[1]))
    print('Dist to edge (X)  (%u,%u)'%(xsep[0],xsep[1]))
    print('Dist to edge (Y)  (%u,%u)'%(ysep[0],ysep[1]))


x_pts = np.r_[0:disk_radii.shape[0]]
plt.subplot(2,2,1)
plt.errorbar(x_pts,disk_radii[:,0],disk_radii[:,1],fmt='x')
plt.ylim([0,350])
plt.subplot(2,2,2)
plt.plot(im_cens[:,1],im_cens[:,0],'.')
plt.xlim([0,5194])
plt.ylim([0,3457])
plt.subplot(2,2,3)
plt.plot(edge_seps[:,0])
plt.subplot(2,2,4)
plt.plot(edge_seps[:,1])
plt.show()