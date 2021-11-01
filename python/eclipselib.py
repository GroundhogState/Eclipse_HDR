import numpy as np
import scipy as scp
import skimage as ski
from skimage import feature
import matplotlib.pyplot as plt
import skimage.io as io
import os
from tqdm import tqdm, tqdm_notebook
from mpl_toolkits.mplot3d import Axes3D
import pyvips


def examine_smoothing(sample, kmax=10):
    fmeans = np.zeros([kmax, 3])
    fstds = np.zeros([kmax, 3])
    krange = np.r_[0:kmax]
    for kernel in krange:
        filter_sample = rgb_filter(sample, kernel)
        [fmeans[kernel], fstds[kernel]] = channel_stats(filter_sample)
    plt.subplot(211)
    plt.plot(krange, [mn[0] for mn in fmeans], color="red")
    plt.plot(krange, [mn[1] for mn in fmeans], color="green")
    plt.plot(krange, [mn[2] for mn in fmeans], color="blue")
    plt.title("Sample mean")
    plt.subplot(212)
    plt.plot(krange, [st[0] for st in fstds], color="red")
    plt.plot(krange, [st[1] for st in fstds], color="green")
    plt.plot(krange, [st[2] for st in fstds], color="blue")
    plt.title("Sample std")
    plt.show()


def rgb_filter(img, sigma):
    r = scp.ndimage.gaussian_filter(img[:, :, 0], sigma)
    g = scp.ndimage.gaussian_filter(img[:, :, 1], sigma)
    b = scp.ndimage.gaussian_filter(img[:, :, 2], sigma)
    return np.dstack((r, g, b))


def imshow16(image):
    plt.imshow(imnorm(ski.color.rgb2gray(image)))


def channel_stats(img):
    chan_means = [
        np.mean(np.ndarray.flatten(img[:, :, chan]))
        for chan in np.r_[0 : img.shape[-1]]
    ]
    chan_stds = [
        np.std(np.ndarray.flatten(img[:, :, chan])) for chan in np.r_[0 : img.shape[-1]]
    ]
    return np.r_[chan_means], np.r_[chan_stds]


def np_to_vips(np_3d):
    dtype_to_format = {
        "uint8": "uchar",
        "int8": "char",
        "uint16": "ushort",
        "int16": "short",
        "uint32": "uint",
        "int32": "int",
        "float32": "float",
        "float64": "double",
        "complex64": "complex",
        "complex128": "dpcomplex",
    }

    height, width, bands = np_3d.shape
    linear = np_3d.reshape(width * height * bands)
    vi = pyvips.Image.new_from_memory(
        linear.data, width, height, bands, dtype_to_format[str(np_3d.dtype)]
    )
    return vi


def exposure_scaling(img_paths, rgn, all_exp, sigma=0):
    num_imgs = len(img_paths)
    mn = np.zeros([num_imgs, 3])
    st = np.zeros([num_imgs, 3])
    for im_idx in np.r_[0:num_imgs]:
        test_img = get_img_with_vips(img_paths[im_idx])
        if sigma > 0:
            test_img = rgb_filter(test_img, sigma)
        [mn[im_idx], st[im_idx]] = channel_stats(
            test_img[rgn[0][0] : rgn[0][1], rgn[1][0] : rgn[1][1]]
        )
    plt.figure(figsize=(9, 6))
    plt.subplot(231)
    plt.plot(all_exp, mn[:, 0], color="r")
    plt.plot(all_exp, mn[:, 1], color="g")
    plt.plot(all_exp, mn[:, 2], color="b")
    plt.title("Regional mean")
    plt.xlabel("Exposure")
    plt.subplot(232)
    plt.plot(all_exp, st[:, 0], color="r")
    plt.plot(all_exp, st[:, 1], color="g")
    plt.plot(all_exp, st[:, 2], color="b")
    plt.title("Regional std")
    plt.xlabel("Exposure")
    plt.subplot(234)
    plt.plot(all_exp, mn[:, 0], color="r")
    plt.plot(all_exp, mn[:, 1], color="g")
    plt.plot(all_exp, mn[:, 2], color="b")
    plt.yscale("log")
    plt.title("Regional mean")
    plt.xlabel("Exposure")
    plt.subplot(235)
    plt.plot(all_exp, st[:, 0], color="r")
    plt.plot(all_exp, st[:, 1], color="g")
    plt.plot(all_exp, st[:, 2], color="b")
    plt.yscale("log")
    plt.title("Regional std")
    plt.xlabel("Exposure")
    plt.subplot(233)
    plt.plot(all_exp, mn[:, 0] / st[:, 0], color="r")
    plt.plot(all_exp, mn[:, 1] / st[:, 1], color="g")
    plt.plot(all_exp, mn[:, 2] / st[:, 2], color="b")
    plt.title("Regional SNR")
    plt.xlabel("Exposure")
    plt.subplot(236)
    plt.plot(all_exp, mn[:, 0] / st[:, 0], color="r")
    plt.plot(all_exp, mn[:, 1] / st[:, 1], color="g")
    plt.plot(all_exp, mn[:, 2] / st[:, 2], color="b")
    plt.yscale("log")
    plt.title("Regional SNR")
    plt.xlabel("Exposure")
    plt.show()
    return [mn, st]


def show_darkfield(dark_img, xlims=[]):
    [r, g, b] = [dark_img[:, :, 0], dark_img[:, :, 1], dark_img[:, :, 2]]
    mns = list(map(lambda x: np.mean(x), [r, g, b]))
    sts = list(map(lambda x: np.std(x), [r, g, b]))
    print("Means: ", mns)
    print(" Stds: ", sts)
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.hist(np.ndarray.flatten(r), 300, color="red", histtype="step")
    plt.hist(np.ndarray.flatten(g), 300, color="green", histtype="step")
    plt.hist(np.ndarray.flatten(b), 300, color="blue", histtype="step")
    plt.xlim([0, 6e3])
    if len(xlims) > 0:
        plt.xlim(xlims)
    plt.ylim([0, 1e6])
    plt.subplot(122)
    plt.hist(np.ndarray.flatten(r), 300, color="red", log="True", histtype="step")
    plt.hist(np.ndarray.flatten(g), 300, color="green", log="True", histtype="step")
    plt.hist(np.ndarray.flatten(b), 300, color="blue", log="True", histtype="step")
    plt.xlim([1, 1e4])
    if len(xlims) > 0:
        plt.xlim(xlims)
    plt.ylim([0.1, 1e6])
    plt.show()
    return [mns, sts]


def examine_smoothing(sample, kmax=10):
    fmeans = np.zeros([kmax, 3])
    fstds = np.zeros([kmax, 3])
    krange = np.r_[0:kmax]
    for kernel in krange:
        filter_sample = rgb_filter(sample, kernel)
        [fmeans[kernel], fstds[kernel]] = channel_stats(filter_sample)
    plt.subplot(221)
    plt.plot(krange, [mn[0] for mn in fmeans], color="red")
    plt.plot(krange, [mn[1] for mn in fmeans], color="green")
    plt.plot(krange, [mn[2] for mn in fmeans], color="blue")
    plt.title("Sample mean")
    plt.subplot(222)
    plt.plot(krange, [st[0] for st in fstds], color="red")
    plt.plot(krange, [st[1] for st in fstds], color="green")
    plt.plot(krange, [st[2] for st in fstds], color="blue")
    plt.title("Sample std")
    plt.subplot(223)
    plt.plot(krange, [st[0] / fstds[0][0] for st in fstds], color="red")
    plt.plot(krange, [st[1] / fstds[0][1] for st in fstds], color="green")
    plt.plot(krange, [st[2] / fstds[0][2] for st in fstds], color="blue")
    plt.yscale("log")
    plt.subplot(224)
    plt.plot(krange, [m[0] / s[0] for (m, s) in zip(fmeans, fstds)], color="red")
    plt.plot(krange, [m[1] / s[1] for (m, s) in zip(fmeans, fstds)], color="green")
    plt.plot(krange, [m[2] / s[2] for (m, s) in zip(fmeans, fstds)], color="blue")
    plt.title("Sample SNR")
    plt.show()


def show_channel_stats(path_in, region, exposures, visual=True, sigma=0):
    num_imgs = len(path_in)
    chan_stats = np.zeros([num_imgs, 2, 3])  # img, stat,channel

    for im_idx in np.r_[0:num_imgs]:
        newsample = get_img_with_vips(path_in[im_idx])
        newsample = newsample[region[0][0] : region[0][1], region[1][0] : region[1][1]]
        if sigma > 0:
            newsample = rgb_filter(newsample, sigma)
        chan_stats[im_idx][0], chan_stats[im_idx][1] = channel_stats(newsample)

    r_SNR = [img_stat[0][0] / img_stat[1][0] for img_stat in chan_stats]
    g_SNR = [img_stat[0][1] / img_stat[1][1] for img_stat in chan_stats]
    b_SNR = [img_stat[0][2] / img_stat[1][2] for img_stat in chan_stats]

    if visual:
        plt.figure(figsize=(8, 8))
        plt.subplot(221)
        plt.plot(exposures, [img_stat[0][0] for img_stat in chan_stats], color="red")
        plt.plot(exposures, [img_stat[0][1] for img_stat in chan_stats], color="green")
        plt.plot(exposures, [img_stat[0][2] for img_stat in chan_stats], color="blue")
        plt.title("Channel mean")
        plt.subplot(222)
        plt.plot(exposures, [img_stat[1][0] for img_stat in chan_stats], color="red")
        plt.plot(exposures, [img_stat[1][1] for img_stat in chan_stats], color="green")
        plt.plot(exposures, [img_stat[1][2] for img_stat in chan_stats], color="blue")
        plt.title("Channel std")
        plt.subplot(223)
        plt.plot(exposures, r_SNR, color="red")
        plt.plot(exposures, g_SNR, color="green")
        plt.plot(exposures, b_SNR, color="blue")
        plt.title("Channel SNR")
        plt.subplot(224)
        plt.plot(exposures, r_SNR, color="red")
        plt.plot(exposures, g_SNR, color="green")
        plt.plot(exposures, b_SNR, color="blue")
        plt.yscale("log")
        #         plt.xscale('log')
        plt.title("Channel SNR")
        plt.show()


def save_img_with_vips(img, fname):
    format_to_dtype = {
        "uchar": np.uint8,
        "char": np.int8,
        "ushort": np.uint16,
        "short": np.int16,
        "uint": np.uint32,
        "int": np.int32,
        "float": np.float32,
        "double": np.float64,
        "complex": np.complex64,
        "dpcomplex": np.complex128,
    }

    dtype_to_format = {
        "uint8": "uchar",
        "int8": "char",
        "uint16": "ushort",
        "int16": "short",
        "uint32": "uint",
        "int32": "int",
        "float32": "float",
        "float64": "double",
        "complex64": "complex",
        "complex128": "dpcomplex",
    }
    height, width, bands = img.shape
    linear = img.reshape(width * height * bands)
    vi = pyvips.Image.new_from_memory(
        linear.data, width, height, bands, dtype_to_format[str(img.dtype)]
    )
    vi.write_to_file(fname)


def get_img_with_vips(img_path):
    format_to_dtype = {
        "uchar": np.uint8,
        "char": np.int8,
        "ushort": np.uint16,
        "short": np.int16,
        "uint": np.uint32,
        "int": np.int32,
        "float": np.float32,
        "double": np.float64,
        "complex": np.complex64,
        "dpcomplex": np.complex128,
    }

    dtype_to_format = {
        "uint8": "uchar",
        "int8": "char",
        "uint16": "ushort",
        "int16": "short",
        "uint32": "uint",
        "int32": "int",
        "float32": "float",
        "float64": "double",
        "complex64": "complex",
        "complex128": "dpcomplex",
    }

    img = pyvips.Image.new_from_file(img_path, access="sequential")
    this_img = np.ndarray(
        buffer=img.write_to_memory(),
        dtype=format_to_dtype[img.format],
        shape=[img.height, img.width, img.bands],
    )
    return this_img


def image_diagnostic(img_for_show, mid_y=-1, mid_x=-1, hists=False):
    is_rgb = len(img_for_show.shape) > 2
    # if is_rgb:
    #     img_greyscale = imnorm(ski.color.rgb2gray(img_for_show))
    # else:
    #     img_greyscale = imnorm(img_for_show)
    if mid_x < 0:
        mid_x = int(0.5 * img_for_show.shape[0])
    if mid_y < 0:
        mid_y = int(0.5 * img_for_show.shape[1])

    print(mid_x, mid_y)
    print(img_for_show.shape)
    #     if is_rgb:
    x_slice = img_for_show[:, mid_y]
    x_len = len(x_slice)
    x_shift = 0.5 * x_len
    XX = np.r_[0:x_len] - x_shift

    y_slice = img_for_show[mid_x, :]
    y_len = len(y_slice)
    y_shift = 0.5 * y_len
    YY = np.r_[0:y_len] - y_shift

    # if
    colors = ["r", "g", "b"]
    plt.figure(figsize=(10, 4.5), dpi=80, facecolor="w", edgecolor="k")
    plt.subplot(2, 2, 1)
    if is_rgb:
        for channel in np.r_[0:3]:
            plt.plot(XX, x_slice[:, channel], color=colors[channel])
    else:
        plt.plot(XX, x_slice, color="k")
    plt.subplot(2, 2, 2)
    if is_rgb:
        for channel in np.r_[0:3]:
            plt.plot(YY, y_slice[:, channel], color=colors[channel])
    else:
        plt.plot(YY, y_slice, color="k")
    plt.subplot(2, 2, 3)
    if is_rgb:
        for channel in np.r_[0:3]:
            plt.plot(XX, x_slice[:, channel], color=colors[channel])
    else:
        plt.plot(XX, x_slice, color="k")
    plt.yscale("log")
    plt.subplot(2, 2, 4)
    if is_rgb:
        for channel in np.r_[0:3]:
            plt.plot(YY, y_slice[:, channel], color=colors[channel])
    else:
        plt.plot(YY, y_slice, color="k")
    plt.yscale("log")
    plt.show()
    if hists == True:
        if is_rgb:
            rgb_histograms(img_for_show)
        else:
            plt.hist(img_for_show)
            plt.show()


def imnorm(img):
    return img / max(np.ndarray.flatten(img))


def rgb_rescale(image, range=255):
    max_value = max(np.ndarray.flatten((image)))
    return range * image / max_value


## Functions for image registration via autocorrelation


def d_framing(this_img, frame_size, shift=[0, 0], verbose=0):
    dims = this_img.shape
    # Start by adding to the blank frame
    hdiff = frame_size[0] - dims[0]
    hhdiff = np.int(0.5 * hdiff + shift[0])
    vdiff = frame_size[1] - dims[1]
    hvdiff = np.int(0.5 * vdiff + shift[1])
    framed_img = np.zeros([frame_size[0], frame_size[1]])
    if verbose > 0:
        print("standard_framing:")
        print(" Frame size     (%u,%u)" % (framed_img.shape))
        print(" this_img dim   (%u,%u)" % (dims[0], dims[1]))
        print(" Corner offsets (%u,%u)" % (hhdiff, hvdiff))

    framed_img[hhdiff : hhdiff + dims[0], hvdiff : hvdiff + dims[1]] += this_img
    if verbose > 0:
        print(" Out size       (%u,%u)" % (framed_img.shape))
        print("------- return -------")
    return framed_img


def subimage_registration(
    img1,
    img2,
    frame_size,
    subsampwidth=500,
    x_offset=0,
    y_offset=0,
    visual=False,
    verbose=0,
):

    # Only necessary if not already same size
    # Doesn't require standardized image size
    blank_frame = np.int_(np.zeros([frame_size[0], frame_size[1]]))
    blank_dim = blank_frame.shape

    first_framed = standard_framing(img1, blank_dim, verbose=verbose - 1)
    second_framed = standard_framing(img2, blank_dim, verbose=verbose - 1)
    # There's a lot of blank space. Subsample for speed gains
    # Assumed
    yup = np.int_(0.5 * blank_dim[0] - subsampwidth)
    ybt = np.int_(0.5 * blank_dim[0] + subsampwidth)
    xup = np.int_(0.5 * blank_dim[1] - subsampwidth)
    xbt = np.int_(0.5 * blank_dim[1] + subsampwidth)
    first_zoom = first_framed[yup:ybt, xup:xbt]
    second_zoom = second_framed[
        yup + y_offset : ybt + y_offset, xup + x_offset : xbt + x_offset
    ]
    first_zoom = first_zoom / max(np.ndarray.flatten(first_zoom))
    second_zoom = second_zoom / max(np.ndarray.flatten(second_zoom))
    first_fft = np.fft.fft2(first_zoom)
    second_fft = np.fft.fft2(second_zoom)

    cross_fft = first_fft * second_fft.conj()
    cross_corr = np.real(np.fft.ifft2(cross_fft))

    # Locate maximum
    #     search_region = cross_corr[250:750,250:750]

    maxima = np.unravel_index(np.argmax(np.abs(cross_corr)), cross_corr.shape)
    midpoints = np.array([np.fix(axis_size / 2) for axis_size in cross_corr.shape])

    shifts = np.array(maxima, dtype=np.float64)
    shifts[shifts > midpoints] -= np.array(cross_corr.shape)[shifts > midpoints]
    if visual:
        plt.figure(figsize=(9, 6))
        plt.subplot(231)
        plt.imshow(first_zoom)
        plt.subplot(232)
        plt.imshow(np.log(abs(first_fft)))

        plt.subplot(234)
        plt.imshow(second_zoom)
        plt.subplot(235)
        plt.imshow(np.log(abs(second_fft)))

        plt.subplot(233)
        plt.imshow((abs(cross_corr)))
        plt.subplot(236)
        plt.plot(cross_corr)
        plt.show()

    return shifts, maxima, cross_corr


def channel_registration(
    data_root,
    img_array,
    grey_imgs,
    frame_size,
    sigma=5,
    mask_level=0.4,
    verbose=0,
    subsampwidth=500,
    img_names=[],
):

    # height, width, bands = img_crop.shape
    # linear = img_crop.reshape(width * height * bands)
    # vi = pyvips.Image.new_from_memory(linear.data, width, height, bands,
    #                                   dtype_to_format[str(img_crop.dtype)])
    # vi.write_to_file(out_fname)

    num_imgs = len(img_array)

    if verbose > 0:
        print("Frame size  (%u,%u)" % (frame_size[0], frame_size[1]))
    for im_idx in np.r_[0:num_imgs]:

        # img = pyvips.Image.new_from_file(img_paths[im_idx], access='sequential')
        # this_img = np.ndarray(buffer=img.write_to_memory(),
        #                    dtype=format_to_dtype[img.format],
        #                    shape=[img.height, img.width, img.bands])

        print("IMG %u/%u" % (im_idx, num_imgs))
        if len(img_names) == num_imgs:
            this_name = img_names[im_idx][:-4]
        else:
            this_name = str(im_idx)
        fname = data_root + "\\register\\" + this_name + ".jpg"

        if im_idx == 0:
            old_img = imnorm(ski.filters.gaussian(grey_imgs[im_idx], sigma=sigma))
            sum_out = standard_framing(old_img, frame_size, verbose=verbose - 1)
            r_sum = standard_framing(img_array[im_idx][:, :, 0], frame_size)
            g_sum = standard_framing(img_array[im_idx][:, :, 1], frame_size)
            b_sum = standard_framing(img_array[im_idx][:, :, 2], frame_size)
            n_sum = standard_framing(
                np.ones(old_img.shape), frame_size, verbose=verbose - 1
            )
            this_rgb = np.int_(np.dstack((r_sum, g_sum, b_sum)))
            old_bin = sum_out > 0.4
            if verbose > 0:
                print("Setup:")
                print(" Sum shape:  (%u,%u)" % sum_out.shape)
            print("Saving as %s" % (fname))
            io.imsave(fname, this_rgb)
        else:
            new_img = imnorm(ski.filters.gaussian(grey_imgs[im_idx], sigma=sigma))
            new_bin = new_img > mask_level
            [shifts, maxima, cross_corr] = subimage_registration(
                sum_out, new_bin, frame_size, subsampwidth=subsampwidth
            )
            if verbose > 0:
                print("IMG %u" % (im_idx))
                print(" Shifts:    (%u,%u)" % (shifts[0], shifts[1]))
                print(" sum shape: (%u,%u)" % sum_out.shape)
            new_out = standard_framing(
                new_img, frame_size, shift=shifts, verbose=verbose - 1
            )

            if verbose > 0:
                print(" New shape: (%u,%u)" % new_out.shape)
            sum_out += new_out
            r_new = standard_framing(
                img_array[im_idx][:, :, 0], frame_size, shift=shifts
            )
            g_new = standard_framing(
                img_array[im_idx][:, :, 1], frame_size, shift=shifts
            )
            b_new = standard_framing(
                img_array[im_idx][:, :, 2], frame_size, shift=shifts
            )
            n_new = standard_framing(
                np.ones(new_img.shape), frame_size, shift=shifts, verbose=verbose - 1
            )
            this_rgb = np.int_(np.dstack((r_new, g_new, b_new)))
            r_sum += r_new
            g_sum += g_new
            b_sum += b_new
            n_sum += n_new
            old_img = np.copy(new_out)
            old_bin = old_img > mask_level
            print("Saving as %s" % (fname))
            io.imsave(fname, this_rgb)
    rgb_sum = np.int_(np.r_[[r_sum], [g_sum], [b_sum]])
    io.imsave(data_root + "N_profile" + ".jpg", n_sum)
    return rgb_sum, n_sum


def rgb_histograms(img, nbins=255, x_log=0):
    r_sum = img[:, :, 0]
    g_sum = img[:, :, 1]
    b_sum = img[:, :, 2]

    rhist = np.histogram(r_sum, nbins)
    rcounts = rhist[0]
    ghist = np.histogram(g_sum, nbins)
    gcounts = ghist[0]
    bhist = np.histogram(b_sum, nbins)
    bcounts = bhist[0]

    if x_log > 0:
        log_rhist = np.histogram(np.log10(r_sum[r_sum > 0]), nbins)
        log_rcounts = log_rhist[0]
        log_ghist = np.histogram(np.log10(g_sum[g_sum > 0]), nbins)
        log_gcounts = log_ghist[0]
        log_bhist = np.histogram(np.log10(b_sum[b_sum > 0]), nbins)
        log_bcounts = log_bhist[0]
        log_hist_bins = 0.5 * (log_rhist[1][1:] + log_rhist[1][:-1])

    hist_bins = 0.5 * (rhist[1][1:] + rhist[1][:-1])

    plt.figure(figsize=(10, 4 * (1 + x_log)))

    plt.subplot(1 + x_log, 3, 1)
    plt.plot(hist_bins[1:], rcounts[1:], color="red")
    plt.yscale("log")
    # if ~(xrng==[]):
    #     plt.xlim(xrng)
    plt.subplot(1 + x_log, 3, 2)
    plt.plot(hist_bins[1:], gcounts[1:], color="green")
    plt.yscale("log")
    # if ~(xrng==[]):
    #     plt.xlim(xrng)
    plt.subplot(1 + x_log, 3, 3)
    plt.plot(hist_bins[1:], bcounts[1:], color="blue")
    plt.yscale("log")
    # if ~(xrng==[]):
    #     plt.xlim(xrng)
    if x_log > 0:
        plt.subplot(1 + x_log, 3, 4)
        plt.plot(log_hist_bins[1:], log_rcounts[1:], color="red")
        plt.yscale("log")
        #         plt.xscale('log')
        # if ~(xrng==[]):
        #     plt.xlim(xrng)
        plt.subplot(1 + x_log, 3, 5)
        plt.plot(log_hist_bins[1:], log_gcounts[1:], color="green")
        plt.yscale("log")
        #         plt.xscale('log')
        # if ~(xrng==[]):
        #     plt.xlim(xrng)
        plt.subplot(1 + x_log, 3, 6)
        plt.plot(log_hist_bins[1:], log_bcounts[1:], color="blue")
        plt.yscale("log")
    #         plt.xscale('log')
    # if ~(xrng==[]):
    #     plt.xlim(xrng)
    plt.show()
    return [np.r_[[rcounts], [gcounts], [bcounts]], rhist[1]]


def vips_channel_registration(
    data_root,
    img_paths,
    frame_size,
    sigma=5,
    mask_level=0.4,
    verbose=0,
    subsampwidth=500,
    img_names=[],
):

    import pyvips

    format_to_dtype = {
        "uchar": np.uint8,
        "char": np.int8,
        "ushort": np.uint16,
        "short": np.int16,
        "uint": np.uint32,
        "int": np.int32,
        "float": np.float32,
        "double": np.float64,
        "complex": np.complex64,
        "dpcomplex": np.complex128,
    }

    dtype_to_format = {
        "uint8": "uchar",
        "int8": "char",
        "uint16": "ushort",
        "int16": "short",
        "uint32": "uint",
        "int32": "int",
        "float32": "float",
        "float64": "double",
        "complex64": "complex",
        "complex128": "dpcomplex",
    }

    num_imgs = len(img_paths)

    if verbose > 0:
        print("Frame size  (%u,%u)" % (frame_size[0], frame_size[1]))
    for im_idx in np.r_[0:num_imgs]:

        img = pyvips.Image.new_from_file(img_paths[im_idx], access="sequential")
        this_img = np.ndarray(
            buffer=img.write_to_memory(),
            dtype=format_to_dtype[img.format],
            shape=[img.height, img.width, img.bands],
        )

        print("IMG %u/%u" % (im_idx, num_imgs))
        this_name = img_paths[im_idx][-12:]
        fname = data_root + "registered\\" + this_name

        img = pyvips.Image.new_from_file(img_paths[im_idx], access="sequential")
        this_img = np.ndarray(
            buffer=img.write_to_memory(),
            dtype=format_to_dtype[img.format],
            shape=[img.height, img.width, img.bands],
        )
        this_grey = ski.color.rgb2gray(this_img)

        if im_idx == 0:
            # img = pyvips.Image.new_from_file(img_paths[im_idx], access='sequential')
            # this_img = np.ndarray(buffer=img.write_to_memory(),
            #                dtype=format_to_dtype[img.format],
            #                shape=[img.height, img.width, img.bands])
            # this_grey = ski.color.rgb2gray(this_img)
            old_img = imnorm(ski.filters.gaussian(this_grey, sigma=sigma))
            sum_out = standard_framing(old_img, frame_size, verbose=verbose - 1)
            r_sum = standard_framing(this_img[:, :, 0], frame_size)
            g_sum = standard_framing(this_img[:, :, 1], frame_size)
            b_sum = standard_framing(this_img[:, :, 2], frame_size)
            n_sum = standard_framing(np.ones(old_img.shape), frame_size)
            this_rgb = np.array(np.dstack((r_sum, g_sum, b_sum)), dtype="uint16")
            old_bin = sum_out > 0.4
            if verbose > 0:
                print("Setup:")
                print(" Sum shape:  (%u,%u)" % sum_out.shape)
            # print('Saving as %s'%(fname))
            # io.imsave(fname,this_rgb)
        else:
            # img = pyvips.Image.new_from_file(img_paths[im_idx], access='sequential')
            # this_img = np.ndarray(buffer=img.write_to_memory(),
            #                dtype=format_to_dtype[img.format],
            #                shape=[img.height, img.width, img.bands])
            # this_grey = ski.color.rgb2gray(this_img)
            new_img = imnorm(ski.filters.gaussian(this_grey, sigma=sigma))
            new_bin = new_img > mask_level
            [shifts, maxima, cross_corr] = subimage_registration(
                sum_out, new_bin, frame_size, subsampwidth=subsampwidth
            )
            if verbose > 0:
                print("IMG %u" % (im_idx))
                print(" Shifts:    (%u,%u)" % (shifts[0], shifts[1]))
                print(" sum shape: (%u,%u)" % sum_out.shape)
            new_out = standard_framing(
                new_img, frame_size, verbose=verbose - 1, shift=shifts
            )

            if verbose > 0:
                print(" New shape: (%u,%u)" % new_out.shape)
            sum_out += new_out
            r_new = standard_framing(this_img[:, :, 0], frame_size, shift=shifts)
            g_new = standard_framing(this_img[:, :, 1], frame_size, shift=shifts)
            b_new = standard_framing(this_img[:, :, 2], frame_size, shift=shifts)
            n_new = standard_framing(np.ones(new_img.shape), frame_size, shift=shifts)
            this_rgb = np.array(np.dstack((r_new, g_new, b_new)), dtype="uint16")
            r_sum += r_new
            g_sum += g_new
            b_sum += b_new
            n_sum += n_new
            old_img = np.copy(new_out)
            old_bin = old_img > mask_level
        print("Saving as %s" % (fname))
        # io.imsave(fname,this_rgb)
        height, width, bands = this_rgb.shape
        linear = this_rgb.reshape(width * height * bands)
        vi = pyvips.Image.new_from_memory(
            linear.data, width, height, bands, dtype_to_format[str(this_rgb.dtype)]
        )
        vi.write_to_file(fname)
    rgb_sum = np.int_(np.r_[[r_sum], [g_sum], [b_sum]])
    io.imsave(data_root + "N_profile" + ".jpg", n_sum)
    return rgb_sum, n_sum
