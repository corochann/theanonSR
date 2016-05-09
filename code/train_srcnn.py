""" Convolutional Neural Network
used for Super-Resolution Convolutional Neural Network (SRCNN)
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
from theano.ifelse import ifelse
from theano.tensor.nnet import conv2d

from tools.prepare_data import load_data
from srcnn import predict
from preprocess import nearest_neighbor_2x
# from mlp import HiddenLayer
from layer import ConvLayer

# Model
training_model_folder = os.path.join('./model', '32x3x3_32x3x3_32x3x3_1x3x3')
#training_model_folder = './model/64x5x5_32x5x5_1x3x3/'
#training_model_folder = os.path.join('./model', '2x3x3_1x3x3')
training_process_folder = os.path.join(training_model_folder, 'training_process')
train_log_file_name = 'train.log'

def execute_srcnn(lr=0.005, n_epochs=10000,
                  dataset='',
                  nkerns=[2, 2], batch_size=4,
                  image_height=232, image_width=232,
                  resume=False):
    """
    :param lr: float - learning rate
    :param n_epochs:
    :param dataset:
    :param nkerns:
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

    # #of feature map = 3 - RGB
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
    #dataset = load_photo_data()

    train_dataset, valid_dataset, test_dataset = datasets
    train_set_x, train_set_y = train_dataset
    valid_set_x, valid_set_y = valid_dataset
    test_set_x, test_set_y = test_dataset

    # print('train_set_x.shape[0]', train_set_x.shape[0].eval())
    # PREPROCESSING
    start_time = timeit.default_timer()
    train_scaled_x = nearest_neighbor_2x(train_set_x.get_value(), total_image_padding // 2)
    valid_scaled_x = nearest_neighbor_2x(valid_set_x.get_value(), total_image_padding // 2)
    test_scaled_x = nearest_neighbor_2x(test_set_x.get_value(), total_image_padding // 2)
    end_time = timeit.default_timer()
    print('preprocess time %i sec' % (end_time - start_time))
    print('preprocess time %i sec' % (end_time - start_time), file=train_log_file)

    # print('scaled_x', scaled_x)
    train_set_x.set_value(train_scaled_x)
    train_set_y.set_value(train_set_y.get_value(borrow=True) / 256.)
    valid_set_x.set_value(valid_scaled_x)
    valid_set_y.set_value(valid_set_y.get_value(borrow=True) / 256.)
    test_set_x.set_value(test_scaled_x)
    test_set_y.set_value(test_set_y.get_value(borrow=True) / 256.)

    train_set_batch_size = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    # SHOW Test images
    test_img_batch1 = test_set_x.get_value(borrow=False)[1 * batch_size: (1 + 2) * batch_size]  # OK.
    cv2.imwrite(os.path.join(training_process_folder, 'photo0_input.jpg'), test_img_batch1[0].transpose(1, 2, 0) * 256.)
    cv2.imwrite(os.path.join(training_process_folder, 'photo1_input.jpg'), test_img_batch1[1].transpose(1, 2, 0) * 256.)
    cv2.imwrite(os.path.join(training_process_folder, 'photo2_input.jpg'), test_img_batch1[2].transpose(1, 2, 0) * 256.)
    cv2.imwrite(os.path.join(training_process_folder, 'photo3_input.jpg'), test_img_batch1[3].transpose(1, 2, 0) * 256.)
    cv2.imwrite(os.path.join(training_process_folder, 'photo4_input.jpg'), test_img_batch1[4].transpose(1, 2, 0) * 256.)
    cv2.imwrite(os.path.join(training_process_folder, 'photo5_input.jpg'), test_img_batch1[5].transpose(1, 2, 0) * 256.)
    print('test_input_img: ', test_img_batch1)

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

        print('[DEBUG], i = %i, layers[i]["channel"] = %i, layers[i]["filter_height"] = %i, layers[i]["filter_width"] = %i' %
              (i, layers[i]["channel"], layers[i]["filter_height"], layers[i]["filter_width"]))
        if resume:
            layer = ConvLayer(
                rng,
                input=layer_input,
                image_shape=(batch_size, previous_layer_channel, image_height + image_padding, image_width + image_padding),
                filter_shape=(layers[i]["channel"], previous_layer_channel, layers[i]["filter_height"], layers[i]["filter_width"]),
                use_momentum=False,
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
                use_momentum=False,
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
        #prev_params = ConvLayers[i].prev_params
        #delta_params = ConvLayers[i].delta_params
        params_m = ConvLayers[i].params_m
        params_v = ConvLayers[i].params_v
        gparams = T.grad(cost, params)
        #for param, gparam, prev_param, delta_param in zip(params, gparams, prev_params, delta_params):
        #for param, gparam in zip(params, gparams):
        for param, gparam, param_m, param_v in zip(params, gparams, params_m, params_v):
            # Adam
            updates.append((param_m, beta1 * param_m + (1 - beta1) * gparam))
            updates.append((param_v, beta2 * param_v + (1 - beta2) * gparam * gparam))
            updates.append((param, param - layers[i]["learning_rate"] * param_m / (1. - beta1_t) / (T.sqrt(param_v / (1 - beta2_t)) + epsilon)))
            #updates.append((param, param - alpha * param_m))

            # Momentum
            #updates.append((prev_param, param))
            #updates.append((param, param - layers[i]["learning_rate"] * gparam + 0 * delta_param))
            #updates.append((delta_param, param - prev_param))

            # Normal SGD
            #updates.append((param, param - layers[i]["learning_rate"] * gparam))


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

    construct_first_layer = theano.function(
        [index],
        ConvLayers[0].output,
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
                img_batch1 = construct_photo(1)
                img0 = img_batch1[0].transpose(1, 2, 0) * 256.
                print('[DEBUG] iter ', iter ,' img0: ', img0)

            if iter % 100 == 0:
                print('training @ iter ', iter)
            mean_cost += [train_model(minibatch_index, random_indexes)]

            if (iter + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i) for i in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation cost %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches, this_validation_loss))
                print('epoch %i, minibatch %i/%i, validation cost %f %%' %
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
                    print('     epoch %i, minibatch %i/%i, test cost of best model %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches, test_score))
                    print('     epoch %i, minibatch %i/%i, test cost of best model %f %%' %
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

        if epoch // 10 == 0 or epoch % 10 == 0:
            # img_layer0 = construct_first_layer(1)
            img_batch1 = construct_photo(1)
            img_batch2 = construct_photo(2)
            # print('img_layer0.shape: ', img_layer0.shape)
            # print('img_layer0: ', img_layer0)

            #print('img_batch1.shape', img_batch1.shape)
            #print('img_batch1: ', img_batch1)

            img0 = img_batch1[0].transpose(1, 2, 0) * 256.
            #print('img0.shape', img0.shape)
            #print('img0: ', img0)

            print('img0: ', img0)
            cv2.imwrite(os.path.join(training_process_folder, 'photo0_epoch' + str(epoch) + '.jpg'), img0)
            cv2.imwrite(os.path.join(training_process_folder, 'photo1_epoch' + str(epoch) + '.jpg'), img_batch1[1].transpose(1, 2, 0) * 256.)
            cv2.imwrite(os.path.join(training_process_folder, 'photo2_epoch' + str(epoch) + '.jpg'), img_batch1[2].transpose(1, 2, 0) * 256.)
            cv2.imwrite(os.path.join(training_process_folder, 'photo3_epoch' + str(epoch) + '.jpg'), img_batch1[3].transpose(1, 2, 0) * 256.)
            cv2.imwrite(os.path.join(training_process_folder, 'photo4_epoch' + str(epoch) + '.jpg'), img_batch2[0].transpose(1, 2, 0) * 256.)
            cv2.imwrite(os.path.join(training_process_folder, 'photo5_epoch' + str(epoch) + '.jpg'), img_batch2[1].transpose(1, 2, 0) * 256.)

    end_time = timeit.default_timer()

    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss, best_iter + 1, test_score))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

    print('Optimization complete.',
          file=train_log_file)
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss, best_iter + 1, test_score),
          file=train_log_file)
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)),
          file=train_log_file)


if __name__ == '__main__':
    execute_srcnn()
    predict(model_folder=training_model_folder,
            image_height=232,
            image_width=232)



