""" Training SRCNN
SRCNN: Super-Resolution Convolutional Neural Network
"""
from __future__ import print_function

try:
    import cPickle as pickle
except:
    import pickle as pickle

import os
import sys
import timeit
import cv2
import json

import numpy as np

import theano
import theano.tensor as T

from tools.prepare_data import load_data
from tools.image_processing import preprocess

# from mlp import HiddenLayer
from layer import ConvLayer

#model_name = '3x3x3_1x3x3'
model_name = '16x3x3_32x3x3_32x3x3_64x3x3_1x3x3'
# model_name = '32x3x3_32x3x3_32x3x3_1x3x3'

# Model
filepath = os.path.dirname(os.path.realpath(__file__))
# input_directory = os.path.join(filepath, '../data/training_images') # fixed: not used for now
training_model_folder = os.path.join(filepath, '../model/', model_name)
training_process_folder = os.path.join(training_model_folder, 'training_process')
train_log_file_name = 'train.log'

def execute_srcnn(n_epochs=40000,
                  image_height=232,
                  image_width=232,
                  resume=False):
    """
    :param n_epochs:
    :param batch_size: minibatch_size to train
    :param image_height:
    :param image_width:
    :return:
    """

    # Pre setup
    if not os.path.exists(training_model_folder):
        print('os.getcwd() ', os.getcwd())
        print(training_model_folder + ' does not exist!')
        return
    print('start training for', training_model_folder)
    if not os.path.exists(training_process_folder):
        os.makedirs(training_process_folder)

    if resume:
        train_log_file = open(os.path.join(training_model_folder, train_log_file_name), 'a')
        print('resuming from here!')
        print('\n resuming from here!', file=train_log_file)
    else:
        train_log_file = open(os.path.join(training_model_folder, train_log_file_name), 'w')

    # model data loading
    f = open(os.path.join(training_model_folder, 'train.json'))
    model_data = json.load(f)
    rng = np.random.RandomState(model_data["rng_seed"])

    # #of feature map = 1 - Y only from YCbCr, you need to modify a bit to support 3 - RGB channel
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

    # training data loading
    datasets = load_data()

    np_train_dataset, np_valid_dataset, np_test_dataset = datasets
    np_train_set_x, np_train_set_y = np_train_dataset
    np_valid_set_x, np_valid_set_y = np_valid_dataset
    np_test_set_x, np_test_set_y = np_test_dataset

    # print('train_set_x.shape[0]', train_set_x.shape[0].eval())
    # PREPROCESSING
    start_time = timeit.default_timer()
    train_scaled_x = preprocess(np_train_set_x, total_image_padding // 2)
    valid_scaled_x = preprocess(np_valid_set_x, total_image_padding // 2)
    test_scaled_x = preprocess(np_test_set_x, total_image_padding // 2)
    end_time = timeit.default_timer()
    print('preprocess time %i sec' % (end_time - start_time))
    print('preprocess time %i sec' % (end_time - start_time), file=train_log_file)

    # print('scaled_x', scaled_x)

    def shared_dataset(data, borrow=True):
        shared_data = theano.shared(np.asarray(data, dtype=theano.config.floatX), borrow=borrow)
        return shared_data

    train_set_x = shared_dataset(train_scaled_x)
    valid_set_x = shared_dataset(valid_scaled_x)
    test_set_x = shared_dataset(test_scaled_x)
    train_set_y = shared_dataset(np_train_set_y / 256.) # normalize
    valid_set_y = shared_dataset(np_valid_set_y / 256.)
    test_set_y = shared_dataset(np_test_set_y / 256.)

    train_set_batch_size = np_train_set_x.shape[0]
    n_train_batches = np_train_set_x.shape[0] // batch_size
    n_valid_batches = np_valid_set_x.shape[0] // batch_size
    n_test_batches = np_test_set_x.shape[0] // batch_size

    # SHOW Test images (0~5)
    test_img_batch = test_set_x.get_value(borrow=False)[0: 5]  # OK.
    for i in np.arange(5):
        cv2.imwrite(os.path.join(training_process_folder, 'photo' + str(i) + '_input.jpg'),
                    test_img_batch[i].transpose(1, 2, 0) * 256.)

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a minibatch
    indexes = T.lvector()  # index randomizer
    x = T.tensor4(name='x')  # input data (rasterized images): (batch_size, ch, image_height, image_width)
    y = T.tensor4(name='y')  # output data (rasterized images): (batch_size, ch, image_height, image_width)

    # BUILD MODEL
    print('... building the model')
    if resume:
        param_lists = pickle.load(open(os.path.join(training_model_folder, 'best_model.pkl')))
    layer0_input = x.reshape((batch_size,
                              input_channel_number,
                              image_height + total_image_padding,
                              image_width + total_image_padding))  # shape(batch_size, #of feature map, image height, image width)

    ConvLayers = []
    updates = []

    for i in np.arange(layer_count):
        if i == 0:
            previous_layer_channel = input_channel_number
            layer_input = layer0_input
        else:
            previous_layer_channel = layers[i-1]["channel"]
            layer_input = ConvLayers[-1].output

        #print('[DEBUG], i = %i, layers[i]["channel"] = %i, layers[i]["filter_height"] = %i, layers[i]["filter_width"] = %i' %
        #      (i, layers[i]["channel"], layers[i]["filter_height"], layers[i]["filter_width"]))
        if resume:
            layer = ConvLayer(
                rng,
                input=layer_input,
                image_shape=(batch_size, previous_layer_channel, image_height + image_padding, image_width + image_padding),
                filter_shape=(layers[i]["channel"], previous_layer_channel, layers[i]["filter_height"], layers[i]["filter_width"]),
                use_adam=True,
                W_values=param_lists[i][0],
                b_values=param_lists[i][1]
            )
        else:
            layer = ConvLayer(
                rng,
                input=layer_input,
                image_shape=(batch_size, previous_layer_channel, image_height + image_padding, image_width + image_padding),
                filter_shape=(layers[i]["channel"], previous_layer_channel, layers[i]["filter_height"], layers[i]["filter_width"]),
                use_adam=True
            )
        ConvLayers.append(layer)
        image_padding -= (layers[i]["filter_height"] - 1)
        # print('[DEBUG] image_padding = ', image_padding)

    # crete train model
    # alpha = 0.001
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 0.1  # 0.000001
    beta1_t = theano.shared(value=0.0, borrow=True)
    beta2_t = theano.shared(value=.9, borrow=True)

    cost = ConvLayers[-1].cost(y)

    updates.append((beta1_t, beta1_t * beta1))
    updates.append((beta2_t, beta1_t * beta2))
    for i in np.arange(layer_count):
        params = ConvLayers[i].params
        params_m = ConvLayers[i].params_m # for adam
        params_v = ConvLayers[i].params_v # for adam
        gparams = T.grad(cost, params)
        for param, gparam, param_m, param_v in zip(params, gparams, params_m, params_v):
            # Adam
            updates.append((param_m, beta1 * param_m + (1 - beta1) * gparam))
            updates.append((param_v, beta2 * param_v + (1 - beta2) * gparam * gparam))
            updates.append((param, param - layers[i]["learning_rate"] * param_m / (1. - beta1_t) / (T.sqrt(param_v / (1 - beta2_t)) + epsilon)))
            # Normal SGD
            # updates.append((param, param - layers[i]["learning_rate"] * gparam))

    # indexes is used for image sequence randomizer for training
    train_model = theano.function(
        [index, indexes],
        cost,
        updates=updates,
        givens={
            x: train_set_x[indexes[index * batch_size: (index + 1) * batch_size]],
            y: train_set_y[indexes[index * batch_size: (index + 1) * batch_size]]
        }
    )

    # create a test function
    # TODO: implement error function (test evaluation function, e.g. PSNR), instead of using cost function
    test_model = theano.function(
        [index],
        ConvLayers[-1].cost(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        ConvLayers[-1].cost(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    construct_photo = theano.function(
        [index],
        ConvLayers[-1].output,
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    # TRAIN MODEL
    print('... training')
    patience = 30000  # 10000
    patience_increase = 2
    improvement_threshold = 0.998  # 0.995

    validation_frequency = min(n_train_batches, patience // 2) * 2

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        start_time_for_each_epoch = timeit.default_timer()
        epoch += 1
        mean_cost = []
        random_indexes = np.random.permutation(train_set_batch_size)  # index randomizer
        for minibatch_index in range(n_train_batches):
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter < 10:
                img_batch = construct_photo(0)
                img0 = img_batch[0].transpose(1, 2, 0) * 256.
                print('[DEBUG] iter ', iter, ' img0: ', img0)

            if iter % 100 == 0:
                print('training @ iter ', iter)
            mean_cost += [train_model(minibatch_index, random_indexes)]

            if (iter + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i) for i in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation cost %f' %
                      (epoch, minibatch_index + 1, n_train_batches, this_validation_loss))
                print('epoch %i, minibatch %i/%i, validation cost %f' %
                      (epoch, minibatch_index + 1, n_train_batches, this_validation_loss),
                      file=train_log_file)

                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                        print('update patience -> ', patience, ' iter')
                        # we will execute at least patience iter.

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    test_losses = [test_model(i) for i in range(n_test_batches)]
                    test_score = np.mean(test_losses)
                    print('     epoch %i, minibatch %i/%i, test cost of best model %f' %
                          (epoch, minibatch_index + 1, n_train_batches, test_score))
                    print('     epoch %i, minibatch %i/%i, test cost of best model %f' %
                          (epoch, minibatch_index + 1, n_train_batches, test_score),
                          file=train_log_file)
                    # Save best model
                    with open(os.path.join(training_model_folder, 'best_model.pkl'), 'wb') as f:
                        param_lists = []
                        for i in np.arange(layer_count):
                            param_lists.append([ConvLayers[i].W.get_value(), ConvLayers[i].b.get_value()])
                        pickle.dump(param_lists, f)

            if patience <= iter:
                done_looping = True
                break
        end_time_for_each_epoch = timeit.default_timer()
        diff_time = end_time_for_each_epoch - start_time_for_each_epoch
        print('Training epoch %d cost is %f, took %i min %i sec' %
              (epoch, np.mean(mean_cost), diff_time / 60., diff_time % 60))
        print('Training epoch %d cost is %f, took %i min %i sec' %
              (epoch, np.mean(mean_cost), diff_time / 60., diff_time % 60),
              file=train_log_file)

        # for checking/monitoring the training process
        if epoch // 10 == 0 or epoch % 10 == 0:
            photo_num = 0
            loop = 0
            while photo_num < 5:
                img_batch = construct_photo(loop)
                for j in np.arange(batch_size):
                    if photo_num == 0:
                        print('output_img0: ', img_batch[j].transpose(1, 2, 0) * 256.)
                    cv2.imwrite(os.path.join(training_process_folder,
                                             'photo' + str(photo_num) + '_epoch' + str(epoch) + '.jpg'),
                                img_batch[j].transpose(1, 2, 0) * 256.)
                    photo_num += 1
                    if photo_num == 5:
                        break
                loop += 1

    end_time = timeit.default_timer()

    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f' %
          (best_validation_loss, best_iter + 1, test_score))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

    print('Optimization complete.',
          file=train_log_file)
    print('Best validation score of %f obtained at iteration %i, '
          'with test performance %f' %
          (best_validation_loss, best_iter + 1, test_score),
          file=train_log_file)
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)),
          file=train_log_file)


def predict_test_set(image_height=232,
                     image_width=232,
                     model_folder=training_model_folder):
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


if __name__ == '__main__':
    execute_srcnn()
    predict_test_set(model_folder=training_model_folder,
                     image_height=232,
                     image_width=232)



