from __future__ import print_function

try:
    import cPickle as pickle
except:
    import pickle

import os
import cv2
import numpy as np
import theano

# constant
# output size for training images, input size will be crop_size // 2
crop_size = 232

# global variable
crop_height = crop_size
crop_width = crop_size
image_channel_number = 1  # Y only
filepath = os.path.dirname(os.path.realpath(__file__))
input_directory = os.path.join(filepath, '../../data/training_images')
cropped_directory = os.path.join(filepath, '../../data/training_images-232-cropped')
half_directory = os.path.join(filepath, '../../data/training_images-116-cropped')
logfile_name = os.path.join(filepath, '../../data/prepare_data.log')


def build_data(image_save_flag=False):
    if not os.path.exists(input_directory):
        print(input_directory, ' does not exist! please put training image files in this folder')
        return

    index = 0
    logfile = open(logfile_name, 'w')

    if image_save_flag:
        if not os.path.exists(cropped_directory):
            os.makedirs(cropped_directory)
        if not os.path.exists(half_directory):
            os.makedirs(half_directory)

    for root, dirs, files in os.walk(input_directory):
        # print root, dirs, files
        batch_size = len(files)
        print('file size', len(files))
        data_x = np.empty((batch_size, image_channel_number, crop_height / 2, crop_width / 2))
        data_y = np.empty((batch_size, image_channel_number, crop_height, crop_width))

        for file in files:
            img = cv2.imread(os.path.join(root, file))

            # crop largest square image from center
            height = img.shape[0]
            width = img.shape[1]
            shorter_edge = min(height, width)
            if shorter_edge < crop_height:
                print('skip', file)
                print('skip', file, file=logfile)
                continue
            # print 'shorter_edge', shorter_edge  # (h, w, 3) = (height, width, channel RGB)  where h == w
            # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
            square_img = img[
                       height // 2 - shorter_edge // 2: height // 2 + shorter_edge // 2,
                       width // 2 - shorter_edge // 2: width // 2 + shorter_edge // 2,
                       :]

            # print square_img.shape  # (h, w, 3) = (height, width, channel RGB)  where h == w
            # crop_img = cv2.resize(square_img, (crop_height, crop_width))
            crop_img = img[
                       height // 2 - crop_height // 2: height // 2 + crop_height // 2,
                       width // 2 - crop_width // 2: width // 2 + crop_width // 2,
                       :]

            # print crop_img.shape  # (h, w, 3) = (height, width, channel RGB)  where h == w = 256
            # save y image
            if image_save_flag:
                print('saving to ', os.path.join(cropped_directory, file))
                cv2.imwrite(os.path.join(cropped_directory, file), crop_img)

            # Construct answer data y
            # convert from RGB to YCbCr
            ycc_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2YCR_CB)
            transposed_y_img = np.transpose(crop_img[:, :, 0:1], (2, 0, 1))  # (ch, height, width)

            data_y[index, :, :, :] = transposed_y_img
            # print transposed_y_img.shape  # (3, 420, 640) = (channel, height, width)

            # resize image half. NOTE: size order is (width, height)
            half_img = cv2.resize(ycc_img, (crop_width // 2, crop_height // 2))
            # print half_img.shape
            if image_save_flag:
                print('saving to ', os.path.join(half_directory, file))
                cv2.imwrite(os.path.join(half_directory, file), half_img)

            ycc_half_img = cv2.cvtColor(half_img, cv2.COLOR_BGR2YCR_CB)
            transposed_y_half_img = np.transpose(ycc_half_img[:, :, 0:1], (2, 0, 1))  # (ch, height, width)
            data_x[index] = transposed_y_half_img
            # cv2.imwrite(os.path.join(half_directory, file), half_img)
            index += 1

        # print 'data_x.shape', data_x.shape
        # print 'data_y.shape', data_y.shape
        # dataset = (shared_x, shared_y)
        dataset = (data_x, data_y)

        return dataset


def format_data(dataset):
    ''' data_x, data_y (batch_size, channel, height, width)
    data_x: (1000, 1, 128, 128)
    data_y: (1000, 1, 256, 256)
    '''
    # print 'dataset', dataset
    data_x, data_y = dataset

    total_batch_size = data_x.shape[0]
    train_index = int(total_batch_size * 0.7)  # 70% of data is used for training
    valid_index = int(total_batch_size * 0.9)  # 20% of data is used for validation
    # 10% of data is used for test

    np_train_data_x = data_x[:train_index]
    np_valid_data_x = data_x[train_index: valid_index]
    np_test_data_x = data_x[valid_index:]
    np_train_data_y = data_y[:train_index]
    np_valid_data_y = data_y[train_index: valid_index]
    np_test_data_y = data_y[valid_index:]

    return [(np_train_data_x, np_train_data_y),
            (np_valid_data_x, np_valid_data_y),
            (np_test_data_x, np_test_data_y)]

def load_data():
    dataset = build_data()
    return format_data(dataset)


if __name__ == '__main__':
    # Save photo data
    build_data(True)


