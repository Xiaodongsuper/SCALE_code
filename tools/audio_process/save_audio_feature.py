import torch
import torchaudio
import requests
from io import BytesIO
from moviepy.editor import *
import time
from multiprocessing import Pool
from tqdm import tqdm
import json
import random
import pdb
import numpy as np


def read_json(file):
    f=open(file,"r",encoding="utf-8").read()
    return json.loads(f)

def write_json(file,data):
    f=open(file,"w",encoding="utf-8")
    json.dump(data,f,indent=2,ensure_ascii=False)
    return



audio_len=12
devices=torch.device("cuda" if torch.cuda.is_available() else "cpu")
audio_dir = "/multi_modal/data/audios"
audio_feature_dir= "/multi_modal/data/audio_feature"
def extract_audio_feature(item_id):
    # try:
    if not os.path.exists("{}/{}.mp3".format(audio_dir,item_id)):
        return None

    if os.path.exists("{}/{}.npy".format(audio_feature_dir,item_id)):
        return None

    try:
        audios,sample_rate = torchaudio.load("{}/{}.mp3".format(audio_dir,item_id))

        audios=audios.to(devices)
        resample = torchaudio.transforms.Resample(sample_rate, 16000).to(devices)
        audios=resample(audios)
        audio_data = torch.sum(torch.as_tensor(audios), dim=0) / 2


        if (len(audio_data) / 16000 < audio_len):
            new_audio_data = torch.zeros([audio_len * 16000]).to(devices)
            new_audio_data[0:len(audio_data)] = audio_data
            audio_data = new_audio_data.to(devices)
        else:
            # audio_data = torch.as_tensor(audio_data.cpu().numpy())[:16000 * audio_len].to(devices)
            audio_data=audio_data[:16000 * audio_len]

        transform=torchaudio.transforms.MelSpectrogram(n_mels=80,
                                                       n_fft=1024,
                                                       win_length=1024,
                                                       hop_length=256).to(devices)

        audio_feature = transform(audio_data)
        # (channel, n_mels, time)
        print("audio_feature1: ",audio_feature.size()) # 80 x 751
        cur_mean, cur_std = audio_feature.mean(dim=0), audio_feature.std(dim=0)
        audio_feature = (audio_feature - cur_mean) / (cur_std + 1e-9)
        # audio_feature = audio_feature.permute(1, 0)
        # num_audio = audio_feature.shape[0]
        # print("audio_feature2: ",audio_feature.size())
        audio_feature_np=audio_feature.cpu().numpy()
        print("audio_feature_np: ",audio_feature_np.shape)

        np.save("{}/{}.npy".format(audio_feature_dir,item_id),audio_feature_np)

    except Exception as e:
        print(e)


def extract_audios_feature(item_id_list):
    for item_id in tqdm(item_id_list):
        extract_audio_feature(item_id)
        # break


if __name__ == '__main__':
    print()

    # item_id="123456"
    # video_url="http://cloud.video.taobao.com/play/u/1745634433/p/1/e/6/t/1/50152114431.mp4"
    # audio_processor.get_audio_feature(item_id,video_url)

    # one thread
    subset_v2_id_list=list(read_json("/multi_modal/data/product5m_v2/subset_v2_id_label.json").keys())
    extract_audios_feature(subset_v2_id_list)

    # multi thread
    # subset_v2_id_list=list(read_json("/multi_modal/data/product5m_v2/subset_v2_id_label.json").keys())
    # filter_data=subset_v2_id_list
    # random.shuffle(filter_data)
    #
    # thread_num = 10
    # chunk_size = int(len(filter_data) / thread_num)
    # chunk_data = [filter_data[i:i + chunk_size] for i in range(0, len(filter_data), chunk_size)]
    #
    # pool = Pool(thread_num)
    # # download_all_data_images(data)
    # multi_data = pool.map(extract_audios_feature, chunk_data)
    #










