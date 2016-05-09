"""
Inference twice scaled image using SRCNN
"""

from __future__ import print_function
import argparse
try:
    import cPickle as pickle
except:
    import pickle as pickle

import os
import cv2
import json

import numpy as np
import theano
import theano.tensor as T

from layer import ConvLayer
from tools.image_processing import preprocess
from tools.prepare_data import load_data

model_name = '3x3x3_1x3x3'

# Parse
parser = argparse.ArgumentParser(description=
                                 'Super Resolution using Convolutional Neural Network with theano.\n'
                                 'This upscales 2 times of size of input image, save to output image')
parser.add_argument('input', help='path to input image')
parser.add_argument('output', nargs='?', default=None, help='path to output image')
parser.add_argument('--model', '-m', default=model_name, help='model (directory path)')
parser.add_argument('--compare', '-c', default=False,
                    help='compare flag. if set true, outputs xxx-conventional.jpg to compare the performance of xxx-theanonSR.jpg')
args = parser.parse_args()

if args.output == None:
    file_name_with_ext = os.path.basename(args.input) # returns filename from path
    filename_wo_ext, ext = os.path.splitext(file_name_with_ext)
    output_file_path = os.path.join(os.path.dirname(args.input), filename_wo_ext + '-theanonSR.jpg')
    conventional_file_path = os.path.join(os.path.dirname(args.input), filename_wo_ext + '-conventional.jpg')
else:
    file_name_with_ext = os.path.basename(args.output) # returns filename from path
    filename_wo_ext, ext = os.path.splitext(file_name_with_ext)
    output_file_path = args.output
    conventional_file_path = os.path.join(os.path.dirname(args.output), filename_wo_ext + '-conventional.jpg')

if args.model is not None:
    model_name = args.model

# Model
filepath = os.path.dirname(os.path.realpath(__file__))
model_folder = os.path.join(filepath, '../model', model_name)

def predict_test_set(image_height=232,
                     image_width=232,
                     model_folder=model_folder):
    print('predict...')
    if not os.path.exists(model_folder):
        print('os.getcwd() ', os.getcwd())
        print(model_folder + ' does not exist!')
        return

    training_process_folder = os.path.join(model_folder, 'training_process')
    f = open(os.path.join(model_folder, 'train.json'), 'r')
    model_data = json.load(f)

    rng = np.random.RandomState(model_data["rng_seed"])
    input_channel_number = model_data["input_channel"]
    layer_count = model_data["layer_count"]
    batch_size = model_data["minibatch_size"]
    layers = model_data["layers"]

    image_padding = 0
    for i in np.arange(layer_count):
        # currently it only supports filter height == filter width
        assert layers[i]["filter_height"] == layers[i]["filter_width"], "filter height and filter width must be same!"
        assert layers[i]["filter_height"] % 2 == 1, "Only odd number filter is supported!"
        image_padding += layers[i]["filter_height"] - 1

    total_image_padding = image_padding

    index = T.lscalar()  # index to a minibatch
    x = T.tensor4(name='x')  # input data (rasterized images): (batch_size, ch, image_height, image_width)
    y = T.tensor4(name='y')  # output data (rasterized images): (batch_size, ch, image_height, image_width)
    layer0_input = x # x.reshape((batch_size, input_channel_number, image_height, image_width))  # shape(batch_size, #of feature map, image height, image width)

    param_lists = pickle.load(open(os.path.join(model_folder, 'best_model.pkl')))
    #print('param_lists', param_lists)
    #print('param_lists1', param_lists[1])
    ConvLayers = []

    for i in np.arange(layer_count):
        if i == 0:
            previous_layer_channel = input_channel_number
            layer_input = layer0_input
        else:
            previous_layer_channel = layers[i-1]["channel"]
            layer_input = ConvLayers[-1].output

        print('[DEBUG], i = %i, layers[i]["channel"] = %i, layers[i]["filter_height"] = %i, layers[i]["filter_width"] = %i' %
              (i, layers[i]["channel"], layers[i]["filter_height"], layers[i]["filter_width"]))
        layer = ConvLayer(
            rng,
            input=layer_input,
            image_shape=(batch_size, previous_layer_channel, image_height + image_padding, image_width + image_padding),
            filter_shape=(layers[i]["channel"], previous_layer_channel, layers[i]["filter_height"], layers[i]["filter_width"]),
            W_values=param_lists[i][0],
            b_values=param_lists[i][1]
        )
        #print('layer.W', layer.W, layer.W.get_value())
        #print('layer.b', layer.b, layer.b.get_value())
        ConvLayers.append(layer)

    #print('ConvLayers', ConvLayers)
    # crete train model
    cost = ConvLayers[-1].cost(y)

    datasets = load_data()
    train_dataset, valid_dataset, test_dataset = datasets
    # train_set_x, train_set_y = train_dataset
    # valid_set_x, valid_set_y = valid_dataset
    np_test_set_x, np_test_set_y = test_dataset

    # PREPROCESSING
    test_scaled_x = preprocess(np_test_set_x, total_image_padding // 2)
    test_set_x = theano.shared(np.asarray(test_scaled_x, dtype=theano.config.floatX), borrow=True)
    #print('test_scaled_x', test_scaled_x)

    construct_photo_predict = theano.function(
        [index],
        ConvLayers[-1].output,
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    photo_num = 0
    loop = 0
    while photo_num < 5:
        img_batch = construct_photo_predict(loop)
        for j in np.arange(batch_size):
            if photo_num == 0:
                print('output_img0: ', img_batch[j].transpose(1, 2, 0) * 256.)
            cv2.imwrite(os.path.join(training_process_folder,
                                     'photo' + str(photo_num) + '_predict.jpg'),
                        img_batch[j].transpose(1, 2, 0) * 256.)
            photo_num += 1
            if photo_num == 5:
                break
        loop += 1


def srcnn2x(photo_file_path, output_file_path, model_folder=model_folder, compare=False):
    print('srcnn2x: ', photo_file_path, ' with model ', model_folder)
    if not os.path.exists(model_folder):
        print('os.getcwd() ', os.getcwd())
        print(model_folder + ' does not exist!')
        return

    input_img = cv2.imread(photo_file_path, cv2.IMREAD_COLOR)
    input_image_height = input_img.shape[0]
    input_image_width = input_img.shape[1]
    output_image_height = 2 * input_image_height
    output_image_width = 2 * input_image_width
    f = open(os.path.join(model_folder, 'train.json'), 'r')
    model_data = json.load(f)

    rng = np.random.RandomState(model_data["rng_seed"])
    input_channel_number = model_data["input_channel"]
    layer_count = model_data["layer_count"]
    #batch_size = model_data["minibatch_size"]
    batch_size = 1  # We will convert one photo file
    layers = model_data["layers"]

    image_padding = 0
    for i in np.arange(layer_count):
        # currently it only supports filter height == filter width
        assert layers[i]["filter_height"] == layers[i]["filter_width"], "filter height and filter width must be same!"
        assert layers[i]["filter_height"] % 2 == 1, "Only odd number filter is supported!"
        image_padding += layers[i]["filter_height"] - 1

    total_image_padding = image_padding

    index = T.lscalar()  # index to a minibatch
    x = T.tensor4(name='x')  # input data (rasterized images): (batch_size, ch, image_height, image_width)
    y = T.tensor4(name='y')  # output data (rasterized images): (batch_size, ch, image_height, image_width)
    layer0_input = x # x.reshape((batch_size, input_channel_number, output_image_height, output_image_width))
    # shape(batch_size, #of feature map, image height, image width)

    param_lists = pickle.load(open(os.path.join(model_folder, 'best_model.pkl')))
    #print('param_lists1', param_lists[1])
    #print('param_lists.shape', param_lists.shape)
    conv_layers = []

    for i in np.arange(layer_count):
        if i == 0:
            previous_layer_channel = input_channel_number
            layer_input = layer0_input
        else:
            previous_layer_channel = layers[i-1]["channel"]
            layer_input = conv_layers[-1].output

        # print('[DEBUG], i = %i, layers[i]["channel"] = %i, layers[i]["filter_height"] = %i, layers[i]["filter_width"] = %i' %
        #       (i, layers[i]["channel"], layers[i]["filter_height"], layers[i]["filter_width"]))
        layer = ConvLayer(
            rng,
            input=layer_input,
            image_shape=(batch_size, previous_layer_channel, output_image_height + image_padding, output_image_width + image_padding),
            filter_shape=(layers[i]["channel"], previous_layer_channel, layers[i]["filter_height"], layers[i]["filter_width"]),
            W_values=param_lists[i][0],
            b_values=param_lists[i][1]
        )
        conv_layers.append(layer)
    # crete train model
    cost = conv_layers[-1].cost(y)

    ycc_input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2YCR_CB)

    data_prescaled_x = np.empty((1, input_channel_number, input_image_height, input_image_width))
    data_prescaled_x[0, :, :, :] = np.transpose(ycc_input_img[:, :, 0:1], (2, 0, 1))  # # (ch, height, width)

    # data_scaled_x = np.empty((1, input_channel_number, output_image_height, output_image_width))
    # PREPROCESSING
    data_scaled_x = preprocess(data_prescaled_x, total_image_padding // 2)
    input_x = theano.shared(np.asarray(data_scaled_x, dtype=theano.config.floatX), borrow=True)

    srcnn_photo_predict = theano.function(
        [],
        conv_layers[-1].output,
        givens={
            x: input_x[:]
        }
    )

    output_img_y = srcnn_photo_predict()  # (ch, width, height)
    #print('output_img.shape', output_img_y.shape)
    #print('output_img: ', output_img_y)

    img0 = output_img_y[0].transpose(1, 2, 0) * 256. # (width, height, ch)
    #print('img0.shape', img0.shape)
    #print('img0: ', img0)

    scaled_input_img = cv2.resize(input_img, (output_image_width, output_image_height))
    if compare:
        cv2.imwrite(conventional_file_path, scaled_input_img)

    ycc_scaled_input_img = cv2.cvtColor(scaled_input_img, cv2.COLOR_BGR2YCR_CB)
    ycc_scaled_input_img[:, :, 0:1] = img0  # (width, height, ch)
    rgb_scaled_img = cv2.cvtColor(ycc_scaled_input_img, cv2.COLOR_YCR_CB2BGR)
    cv2.imwrite(output_file_path, rgb_scaled_img)


if __name__ == '__main__':
    #predict(model_folder=training_model_folder)
    # input = '../data/small-320-cropped/attractive-beautiful-body-smiling-41248.jpeg'
    input_file_path = args.input
    if not os.path.exists(input_file_path):
        print('Error: input file ', os.path.dirname(input_file_path), ' not exist')
        exit()

    if not os.path.exists(os.path.dirname(output_file_path)):
        print('Error: output file directory ', os.path.dirname(output_file_path), ' not exist')
        exit()

    flag_compare = args.compare
    srcnn2x(photo_file_path=input_file_path,
            output_file_path=output_file_path,
            model_folder=model_folder,
            compare=flag_compare)

