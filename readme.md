# High dynamic range eclipse photography with Python

## About this project

In this notebook and (and additional scripts), I combine several exposures of the 2017 solar eclipse into a single high dynamic range image.

### Preview:

A snapshot of work in progress:

![Preview](/demo/16bit_master_gimp_compress.jpg "Okay, so I got a little lazy and made a contrast adjustment in GIMP, sue me.")


I use darkfield analysis to estimate the noise floor of the Canon 600-D camera I used. I use an edge detection algorithm to find a first estimate of the centre of the solar disk. I crop the images down by a factor of $\approx 10$ to dramatically speed up the calculation of the correlation between images, the peak of which allows for registration of the images to within a few pixels, which is smaller than the point spread function of the camera optics. I then adjust the images by exposure time and compute the average RGB brightness of each point in the sky around the solar eclipse, including the solar wind whose brightness varies by four orders of magnitude. Images are not included in this repository (the raw CR2 files, plus the darkfield calibration, add up to some 10GB.)

### Remaining goals

The objective of this project is to produce a true-to-the-eye image of the solar eclipse. The remaining challenges are:
* **Enhancing the contrast:** Human vision is particularly sensitive to *local* contrast, rather than total dynamic range. This means that much more detail is visible in the solar wind than simply the combination of the images. Things to try here are gradient-based and fourier-based methods.
* **Colour recovery:** During the eclipse, stars are visible in a gently blue sky around the disk of the sun, a truly remarkable vision. Reproducing this image requires careful adjustment of the brightness levels whilst retaining as much of the global contrast as possible. Further, the bright red prominences are visible jutting out around the disk of the moon, which are washed out by the white light in the presently composed image. One way to make these visible is detailed work with image editing software, but this isn't desirable. It may be possible to enhance the visibility of these coronal mass ejections using the contrast enhancement methods above.
* **Deblurring:** The camera optics have an intrinsic point spread function and spherical aberration. Worse, I also failed to ensure these images were perfectly focused, and the tripod I used was a lightweight aluminum one, which led to considerable camera shake. It may be possible to employ techniques from super-resolution imaging to overcome these and increase detail in the recovered image.



### General remarks

In this project, I iterated over several different styles of workflow and algorithmic approaches. If I were to start it again, I'd do it differently and, I think, more efficiently (for one, using OpenCV from the start). However, in this case, done is better than perfect, and I am more motivated to solve the outstanding challenges than fiddling with the existing process. 
