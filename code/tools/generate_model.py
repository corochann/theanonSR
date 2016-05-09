import json
import os

filepath = os.path.dirname(os.path.realpath(__file__))
model_root_directory = os.path.join(filepath, '../../model')


def generate_train_json():
    # Training json data
    model_name = "3x3x3_1x3x3"
    model_directory = os.path.join(model_root_directory, model_name)
    file_name = "train.json"
    # layer0 = {"filter_height": -1, "filter_width": -1, "channel": 1, "learning_rate": 0.002, "rnd": }
    layer1 = {"filter_height": 3, "filter_width": 3, "channel": 3, "learning_rate": 0.0009}
    #layer2 = {"filter_height": 3, "filter_width": 3, "channel": 32, "learning_rate": 0.0006}
    #layer3 = {"filter_height": 3, "filter_width": 3, "channel": 32, "learning_rate": 0.00036}
    layer4 = {"filter_height": 3, "filter_width": 3, "channel": 1, "learning_rate": 0.00020}

    layers = [layer1, layer4]
    # input_channel = 1 -> Y only
    model_dict = {"input_channel": 1, "rng_seed": 13266, "layer_count": layers.__len__(), "minibatch_size": 8, "layers": layers}
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    f = open(os.path.join(model_directory, file_name), "w")
    json.dump(model_dict, f, indent=4, sort_keys=True)
    #print json.dumps(model_dict, indent=4, sort_keys=True)

if __name__ == '__main__':
    generate_train_json()

