import numpy as np
import phasepack.phasecong as pc
import cv2
import glob

def im2col(img, k, stride=1):
    # Parameters
    m, n = img.shape
    s0, s1 = img.strides
    nrows = m - k + 1
    ncols = n - k + 1
    shape = (k, k, nrows, ncols)
    arr_stride = (s0, s1, s0, s1)

    ret = np.lib.stride_tricks.as_strided(img, shape=shape, strides=arr_stride)
    return ret[:, :, ::stride, ::stride].reshape(k * k, -1)


def integral_image(x):
    M, N = x.shape
    int_x = np.zeros((M + 1, N + 1))
    int_x[1:, 1:] = np.cumsum(np.cumsum(x, 0), 1)
    return int_x


def moments(x, y, k, stride):
    kh = kw = k

    k_norm = k ** 2

    x_pad = np.pad(x, int((kh - stride) / 2), mode='reflect')
    y_pad = np.pad(y, int((kw - stride) / 2), mode='reflect')

    int_1_x = integral_image(x_pad)
    int_1_y = integral_image(y_pad)

    int_2_x = integral_image(x_pad * x_pad)
    int_2_y = integral_image(y_pad * y_pad)

    int_xy = integral_image(x_pad * y_pad)

    mu_x = (int_1_x[:-kh:stride, :-kw:stride] - int_1_x[:-kh:stride, kw::stride] - int_1_x[kh::stride,
                                                                                   :-kw:stride] + int_1_x[kh::stride,
                                                                                                  kw::stride]) / k_norm
    mu_y = (int_1_y[:-kh:stride, :-kw:stride] - int_1_y[:-kh:stride, kw::stride] - int_1_y[kh::stride,
                                                                                   :-kw:stride] + int_1_y[kh::stride,
                                                                                                  kw::stride]) / k_norm

    var_x = (int_2_x[:-kh:stride, :-kw:stride] - int_2_x[:-kh:stride, kw::stride] - int_2_x[kh::stride,
                                                                                    :-kw:stride] + int_2_x[kh::stride,
                                                                                                   kw::stride]) / k_norm - mu_x ** 2
    var_y = (int_2_y[:-kh:stride, :-kw:stride] - int_2_y[:-kh:stride, kw::stride] - int_2_y[kh::stride,
                                                                                    :-kw:stride] + int_2_y[kh::stride,
                                                                                                   kw::stride]) / k_norm - mu_y ** 2

    cov_xy = (int_xy[:-kh:stride, :-kw:stride] - int_xy[:-kh:stride, kw::stride] - int_xy[kh::stride,
                                                                                   :-kw:stride] + int_xy[kh::stride,
                                                                                                  kw::stride]) / k_norm - mu_x * mu_y

    mask_x = (var_x < 0)
    mask_y = (var_y < 0)

    var_x[mask_x] = 0
    var_y[mask_y] = 0

    cov_xy[mask_x + mask_y] = 0

    return (mu_x, mu_y, var_x, var_y, cov_xy)


def vif_gsm_model(pyr, subband_keys, M):
    tol = 1e-15
    s_all = []
    lamda_all = []

    for subband_key in subband_keys:
        y = pyr[subband_key]
        y_size = (int(y.shape[0] / M) * M, int(y.shape[1] / M) * M)
        y = y[:y_size[0], :y_size[1]]

        y_vecs = im2col(y, M, 1)
        cov = np.cov(y_vecs)
        lamda, V = np.linalg.eigh(cov)
        lamda[lamda < tol] = tol
        cov = V @ np.diag(lamda) @ V.T

        y_vecs = im2col(y, M, M)

        s = np.linalg.inv(cov) @ y_vecs
        s = np.sum(s * y_vecs, 0) / (M * M)
        s = s.reshape((int(y_size[0] / M), int(y_size[1] / M)))

        s_all.append(s)
        lamda_all.append(lamda)

    return s_all, lamda_all


def vif_channel_est(pyr_ref, pyr_dist, subband_keys, M):
    tol = 1e-15
    g_all = []
    sigma_vsq_all = []

    for i, subband_key in enumerate(subband_keys):
        y_ref = pyr_ref[subband_key]
        y_dist = pyr_dist[subband_key]

        lev = int(np.ceil((i + 1) / 2))
        winsize = 2 ** lev + 1

        y_size = (int(y_ref.shape[0] / M) * M, int(y_ref.shape[1] / M) * M)
        y_ref = y_ref[:y_size[0], :y_size[1]]
        y_dist = y_dist[:y_size[0], :y_size[1]]

        mu_x, mu_y, var_x, var_y, cov_xy = moments(y_ref, y_dist, winsize, M)

        g = cov_xy / (var_x + tol)
        sigma_vsq = var_y - g * cov_xy

        g[var_x < tol] = 0
        sigma_vsq[var_x < tol] = var_y[var_x < tol]
        var_x[var_x < tol] = 0

        g[var_y < tol] = 0
        sigma_vsq[var_y < tol] = 0

        sigma_vsq[g < 0] = var_y[g < 0]
        g[g < 0] = 0

        sigma_vsq[sigma_vsq < tol] = tol

        g_all.append(g)
        sigma_vsq_all.append(sigma_vsq)

    return g_all, sigma_vsq_all


def vif(img_ref, img_dist, wavelet='steerable', full=False):
    assert wavelet in ['steerable', 'haar', 'db2', 'bio2.2'], 'Invalid choice of wavelet'
    M = 3
    sigma_nsq = 0.1

    if wavelet == 'steerable':
        from pyrtools.pyramids import SteerablePyramidSpace as SPyr
        pyr_ref = SPyr(img_ref, 4, 5, 'reflect1').pyr_coeffs
        pyr_dist = SPyr(img_dist, 4, 5, 'reflect1').pyr_coeffs
        subband_keys = []
        for key in list(pyr_ref.keys())[1:-2:3]:
            subband_keys.append(key)
    else:
        from pywt import wavedec2
        ret_ref = wavedec2(img_ref, wavelet, 'reflect', 4)
        ret_dist = wavedec2(img_dist, wavelet, 'reflect', 4)
        pyr_ref = {}
        pyr_dist = {}
        subband_keys = []
        for i in range(4):
            pyr_ref[(3 - i, 0)] = ret_ref[i + 1][0]
            pyr_ref[(3 - i, 1)] = ret_ref[i + 1][1]
            pyr_dist[(3 - i, 0)] = ret_dist[i + 1][0]
            pyr_dist[(3 - i, 1)] = ret_dist[i + 1][1]
            subband_keys.append((3 - i, 0))
            subband_keys.append((3 - i, 1))
        pyr_ref[4] = ret_ref[0]
        pyr_dist[4] = ret_dist[0]

    subband_keys.reverse()
    n_subbands = len(subband_keys)

    [g_all, sigma_vsq_all] = vif_channel_est(pyr_ref, pyr_dist, subband_keys, M)

    [s_all, lamda_all] = vif_gsm_model(pyr_ref, subband_keys, M)

    nums = np.zeros((n_subbands,))
    dens = np.zeros((n_subbands,))
    for i in range(n_subbands):
        g = g_all[i]
        sigma_vsq = sigma_vsq_all[i]
        s = s_all[i]
        lamda = lamda_all[i]

        n_eigs = len(lamda)

        lev = int(np.ceil((i + 1) / 2))
        winsize = 2 ** lev + 1
        offset = (winsize - 1) / 2
        offset = int(np.ceil(offset / M))

        g = g[offset:-offset, offset:-offset]
        sigma_vsq = sigma_vsq[offset:-offset, offset:-offset]
        s = s[offset:-offset, offset:-offset]

        for j in range(n_eigs):
            nums[i] += np.mean(np.log(1 + g * g * s * lamda[j] / (sigma_vsq + sigma_nsq)))
            dens[i] += np.mean(np.log(1 + s * lamda[j] / sigma_nsq))

    if not full:
        return np.mean(nums + 1e-4) / np.mean(dens + 1e-4)
    else:
        return np.mean(nums + 1e-4) / np.mean(dens + 1e-4), (nums + 1e-4), (dens + 1e-4)


def vif_spatial(img_ref, img_dist, k=11, sigma_nsq=0.1, stride=1, full=False):
    x = img_ref.astype('float32')
    y = img_dist.astype('float32')

    mu_x, mu_y, var_x, var_y, cov_xy = moments(x, y, k, stride)

    g = cov_xy / (var_x + 1e-10)
    sv_sq = var_y - g * cov_xy

    g[var_x < 1e-10] = 0
    sv_sq[var_x < 1e-10] = var_y[var_x < 1e-10]
    var_x[var_x < 1e-10] = 0

    g[var_y < 1e-10] = 0
    sv_sq[var_y < 1e-10] = 0

    sv_sq[g < 0] = var_x[g < 0]
    g[g < 0] = 0
    sv_sq[sv_sq < 1e-10] = 1e-10

    vif_val = np.sum(np.log(1 + g ** 2 * var_x / (sv_sq + sigma_nsq)) + 1e-4) / np.sum(
        np.log(1 + var_x / sigma_nsq) + 1e-4)
    if (full):
        # vif_map = (np.log(1 + g**2 * var_x / (sv_sq + sigma_nsq)) + 1e-4)/(np.log(1 + var_x / sigma_nsq) + 1e-4)
        # return (vif_val, vif_map)
        return (
        np.sum(np.log(1 + g ** 2 * var_x / (sv_sq + sigma_nsq)) + 1e-4), np.sum(np.log(1 + var_x / sigma_nsq) + 1e-4),
        vif_val)
    else:
        return vif_val


def msvif_spatial(img_ref, img_dist, k=11, sigma_nsq=0.1, stride=1, full=False):
    x = img_ref.astype('float32')
    y = img_dist.astype('float32')

    n_levels = 5
    nums = np.ones((n_levels,))
    dens = np.ones((n_levels,))
    for i in range(n_levels - 1):
        if np.min(x.shape) <= k:
            break
        nums[i], dens[i], _ = vif_spatial(x, y, k, sigma_nsq, stride, full=True)
        x = x[:(x.shape[0] // 2) * 2, :(x.shape[1] // 2) * 2]
        y = y[:(y.shape[0] // 2) * 2, :(y.shape[1] // 2) * 2]
        x = (x[::2, ::2] + x[1::2, ::2] + x[1::2, 1::2] + x[::2, 1::2]) / 4
        y = (y[::2, ::2] + y[1::2, ::2] + y[1::2, 1::2] + y[::2, 1::2]) / 4

    if np.min(x.shape) > k:
        nums[-1], dens[-1], _ = vif_spatial(x, y, k, sigma_nsq, stride, full=True)
    msvifval = np.sum(nums) / np.sum(dens)

    if full:
        return msvifval, nums, dens
    else:
        return msvifval


def _assert_image_shapes_equal(org_img: np.ndarray, pred_img: np.ndarray, metric: str):
    # shape of the image should be like this (rows, cols, bands)
    # Please note that: The interpretation of a 3-dimension array read from rasterio is: (bands, rows, columns) while
    # image processing software like scikit-image, pillow and matplotlib are generally ordered: (rows, columns, bands)
    # in order efficiently swap the axis order one can use reshape_as_raster, reshape_as_image from rasterio.plot
    msg = (
        f"Cannot calculate {metric}. Input shapes not identical. y_true shape ="
        f"{str(org_img.shape)}, y_pred shape = {str(pred_img.shape)}"
    )

    assert org_img.shape == pred_img.shape, msg


def _gradient_magnitude(img: np.ndarray, img_depth: int):
    """
    Calculate gradient magnitude based on Scharr operator.
    """
    scharrx = cv2.Scharr(img, img_depth, 1, 0)
    scharry = cv2.Scharr(img, img_depth, 0, 1)

    return np.sqrt(scharrx ** 2 + scharry ** 2)


def _similarity_measure(x: np.array, y: np.array, constant: float):
    """
    Calculate feature similarity measurement between two images
    """
    numerator = 2 * x * y + constant
    denominator = x ** 2 + y ** 2 + constant

    return numerator / denominator


def fsim(org_img: np.ndarray, pred_img: np.ndarray, T1: float = 0.85, T2: float = 160) -> float:
    """
    Feature-based similarity index, based on phase congruency (PC) and image gradient magnitude (GM)
    There are different ways to implement PC, the authors of the original FSIM paper use the method
    defined by Kovesi (1999). The Python phasepack project fortunately provides an implementation
    of the approach.
    There are also alternatives to implement GM, the FSIM authors suggest to use the Scharr
    operation which is implemented in OpenCV.
    Note that FSIM is defined in the original papers for grayscale as well as for RGB images. Our use cases
    are mostly multi-band images e.g. RGB + NIR. To accommodate for this fact, we compute FSIM for each individual
    band and then take the average.
    Note also that T1 and T2 are constants depending on the dynamic range of PC/GM values. In theory this parameters
    would benefit from fine-tuning based on the used data, we use the values found in the original paper as defaults.
    Args:
        org_img -- numpy array containing the original image
        pred_img -- predicted image
        T1 -- constant based on the dynamic range of PC values
        T2 -- constant based on the dynamic range of GM values
    """
    _assert_image_shapes_equal(org_img, pred_img, "FSIM")

    alpha = (
        beta
    ) = 1  # parameters used to adjust the relative importance of PC and GM features
    fsim_list = []
    for i in range(org_img.shape[2]):
        # Calculate the PC for original and predicted images
        pc1_2dim = pc(
            org_img[:, :, i], nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978
        )
        pc2_2dim = pc(
            pred_img[:, :, i], nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978
        )

        # pc1_2dim and pc2_2dim are tuples with the length 7, we only need the 4th element which is the PC.
        # The PC itself is a list with the size of 6 (number of orientation). Therefore, we need to
        # calculate the sum of all these 6 arrays.
        pc1_2dim_sum = np.zeros((org_img.shape[0], org_img.shape[1]), dtype=np.float64)
        pc2_2dim_sum = np.zeros(
            (pred_img.shape[0], pred_img.shape[1]), dtype=np.float64
        )
        for orientation in range(6):
            pc1_2dim_sum += pc1_2dim[4][orientation]
            pc2_2dim_sum += pc2_2dim[4][orientation]

        # Calculate GM for original and predicted images based on Scharr operator
        gm1 = _gradient_magnitude(org_img[:, :, i], cv2.CV_16U)
        gm2 = _gradient_magnitude(pred_img[:, :, i], cv2.CV_16U)

        # Calculate similarity measure for PC1 and PC2
        S_pc = _similarity_measure(pc1_2dim_sum, pc2_2dim_sum, T1)
        # Calculate similarity measure for GM1 and GM2
        S_g = _similarity_measure(gm1, gm2, T2)

        S_l = (S_pc ** alpha) * (S_g ** beta)

        numerator = np.sum(S_l * np.maximum(pc1_2dim_sum, pc2_2dim_sum))
        denominator = np.sum(np.maximum(pc1_2dim_sum, pc2_2dim_sum))
        fsim_list.append(numerator / denominator)

    return np.mean(fsim_list)


