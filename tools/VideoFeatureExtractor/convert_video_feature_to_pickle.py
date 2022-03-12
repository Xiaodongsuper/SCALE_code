import os
import argparse
import numpy as np
import pickle

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Package video feature')

    parser.add_argument('--feature_root_path', type=str, help='feature path')
    parser.add_argument('--pickle_root_path', type=str, help='pickle path', default='.')
    parser.add_argument('--pickle_name', type=str, help='pickle name')
    args = parser.parse_args()

    save_file = os.path.join(args.pickle_root_path, args.pickle_name)

    feature_dict = {}
    for root, dirs, files in os.walk(args.feature_root_path):
        for file_name in files:
            if file_name.find(".npy") > 0:
                file_name_split = file_name.split(".")
                if len(file_name_split) == 2:
                    key = file_name_split[0]
                    feature_file = os.path.join(root, file_name)
                    features = np.load(feature_file)
                    print("features: ",features.shape)
                    feature_dict[key] = features
                else:
                    print("{} is error.".format(file_name))
    print("Total num: {}".format(len(feature_dict)))
    pickle.dump(feature_dict, open(save_file, 'wb'))
    print("pickle is saved in: {}".format(save_file))
