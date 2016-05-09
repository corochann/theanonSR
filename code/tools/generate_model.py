import json
import os


def generate_train_json():
    # Training json data
    directory_name = "32x3x3_32x3x3_32x3x3_1x3x3"
    file_name = "train.json"
    # layer0 = {"filter_height": -1, "filter_width": -1, "channel": 1, "learning_rate": 0.002, "rnd": }
    layer1 = {"filter_height": 3, "filter_width": 3, "channel": 32, "learning_rate": 0.0009}
    layer2 = {"filter_height": 3, "filter_width": 3, "channel": 32, "learning_rate": 0.0006}
    layer3 = {"filter_height": 3, "filter_width": 3, "channel": 32, "learning_rate": 0.00036}
    layer4 = {"filter_height": 3, "filter_width": 3, "channel": 1, "learning_rate": 0.00020}

    layers = [layer1, layer2, layer3, layer4]
    # input_channel = 1 -> Y only
    model_dict = {"input_channel": 1, "rng_seed": 13266, "layer_count": layers.__len__(), "minibatch_size": 4, "layers": layers}
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    f = open(os.path.join(directory_name, file_name), "w")
    json.dump(model_dict, f, indent=4, sort_keys=True)
    #print json.dumps(model_dict, indent=4, sort_keys=True)

if __name__ == '__main__':
    generate_train_json()

