from __future__ import print_function
import numpy as np


# PREPROCESSING
def preprocess(pre_scaled_x, image_padding=0):
    """
    preprocessing image consists of 3 parts
    1. Scaling: scale original input image to twice scale, using nearest neighbor method.
    2. Normalization: normalize each pixel's value from 0~256 to 0~1
    3. Padding: pad edge using np.pad.
                because image size will reduce during convolution in neural network
    :param pre_scaled_x:  original input image array
    :param image_padding: value of pixel to be padded
    :return:
    """
    print('preprocess...')
    print('pre_scaled_x.shape', pre_scaled_x.shape)
    scaled_x = np.empty((pre_scaled_x.shape[0],
                         pre_scaled_x.shape[1],
                         pre_scaled_x.shape[2] * 2,
                         pre_scaled_x.shape[3] * 2),
                        )
    # dtype=train_set_x.dtype)
    for k in np.arange(pre_scaled_x.shape[2]):
        for l in np.arange(pre_scaled_x.shape[3]):
            # scaled_x[i][j, 2 * k: 2 * k + 1, 2 * l: 2 * l + 1] = pre_scaled_x[i, j, k, l] / 256.
            scaled_x[:, :, 2 * k, 2 * l] = pre_scaled_x[:, :, k, l] / 256.
            scaled_x[:, :, 2 * k, 2 * l + 1] = pre_scaled_x[:, :, k, l] / 256.
            scaled_x[:, :, 2 * k + 1, 2 * l] = pre_scaled_x[:, :, k, l] / 256.
            scaled_x[:, :, 2 * k + 1, 2 * l + 1] = pre_scaled_x[:, :, k, l] / 256.

    # print('pre_scaled_x = ', pre_scaled_x)
    # print('scaled_x = ', scaled_x)

    # PADDING
    if image_padding > 0:
        print('image padding ', image_padding, ' pixels...')
        new_img = np.empty((scaled_x.shape[0], scaled_x.shape[1],
                            scaled_x.shape[2] + 2 * image_padding,
                            scaled_x.shape[3] + 2 * image_padding), dtype=scaled_x.dtype)
        for i in np.arange(scaled_x.shape[0]):
            for j in np.arange(scaled_x.shape[1]):
                new_img[i, j, :, :] = np.pad(scaled_x[i, j, :, :], image_padding, "edge")
        scaled_x = new_img

    return scaled_x

