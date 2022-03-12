import os
current_dir_path = os.path.dirname(os.path.realpath(__file__))
print(current_dir_path)

import pdb
import torch as th
import math
import numpy as np
from video_loader import VideoLoader
from torch.utils.data import DataLoader
import argparse
from model import get_model
from preprocessing import Preprocessing
from random_sequence_shuffler import RandomSequenceSampler
import torch.nn.functional as F
from tqdm import tqdm

FRAMERATE_DICT = {'2d':1, '3d':24, 's3dg':16, 'raw_data':16}
SIZE_DICT = {'2d':224, '3d':112, 's3dg':224, 'raw_data':224}
CENTERCROP_DICT = {'2d':False, '3d':True, 's3dg':True, 'raw_data':True}
FEATURE_LENGTH = {'2d':2048, '3d':2048, 's3dg':1024, 'raw_data':1024}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Easy video feature extractor')

    parser.add_argument('--csv', type=str, help='input csv with video input path')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--type', type=str, default='2d', help='CNN type', choices=['2d','3d','s3dg','raw_data'])
    parser.add_argument('--half_precision', type=int, default=1, help='output half precision float')
    parser.add_argument('--num_decoding_thread', type=int, default=4, help='Num parallel thread for video decoding')
    parser.add_argument('--l2_normalize', type=int, default=1, help='l2 normalize feature')
    parser.add_argument('--resnext101_model_path', type=str, default='model/resnext101.pth', help='Resnext model path')
    parser.add_argument('--s3d_model_path', type=str, default='model/s3d_howto100m.pth', help='S3GD model path')
    parser.add_argument('--datastore_base', type=str, default='.')
    args = parser.parse_args()


    dataset = VideoLoader(
        args.csv, # args.datastore_base,
        framerate=FRAMERATE_DICT[args.type],
        size=SIZE_DICT[args.type],
        centercrop=CENTERCROP_DICT[args.type]
    )
    n_dataset = len(dataset)
    sampler = RandomSequenceSampler(n_dataset, 10)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_decoding_thread,
        sampler=sampler if n_dataset > 10 else None,
    )
    preprocess = Preprocessing(args.type, FRAMERATE_DICT)

    if args.type == "raw_data":
        model = None
    else:
        model = get_model(args)

    with th.no_grad():
        k = 0
        for data in tqdm(loader):
            k += 1
            input_file = data['input'][0]
            output_file = data['output'][0]

            if os.path.exists(output_file):
                continue
            #
            # pdb.set_trace()
            if len(data['video'].shape) > 3:
                print('Computing features of video {}/{}: {}'.format(k + 1, n_dataset, input_file))
                video = data['video'].squeeze()
                if len(video.shape) == 4:
                    video = preprocess(video)
                    # Batch x 3 x T x H x W
                    if args.type == "raw_data":
                        features = video
                    else:
                        n_chunk = len(video)
                        features = th.cuda.FloatTensor(n_chunk, FEATURE_LENGTH[args.type]).fill_(0)
                        n_iter = int(math.ceil(n_chunk / float(args.batch_size)))
                        for i in range(n_iter):
                            min_ind = i * args.batch_size
                            max_ind = (i + 1) * args.batch_size
                            video_batch = video[min_ind:max_ind].cuda()
                            batch_features = model(video_batch)
                            if args.l2_normalize:
                                batch_features = F.normalize(batch_features, dim=1)
                            features[min_ind:max_ind] = batch_features
                    features = features.cpu().numpy()
                    if args.half_precision:
                        features = features.astype('float16')
                    os.makedirs('/'.join(output_file.split('/')[:-1]), exist_ok=True)
                    np.save(output_file, features)
            else:
                print("data['video'].shape: ",data['video'].shape)
                print('Video {} already processed.'.format(input_file))