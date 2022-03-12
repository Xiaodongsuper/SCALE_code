from moviepy.editor import *
import os
from tqdm import tqdm
from multiprocessing import Pool
import requests
import os
from _md5 import md5
from tqdm import tqdm
from multiprocessing import Pool
import random
import json
import json
import jsonlines
import pickle
import csv
import re
import os.path
import random
from tqdm import tqdm


class IOProcessor():
    def read_jsonline(self,file):
        file=open(file,"r",encoding="utf-8")
        data=[json.loads(line) for line in file.readlines()]
        return  data

    def write_jsonline(self,file,data):
        f=jsonlines.open(file,"w")
        for each in data:
            jsonlines.Writer.write(f,each)
        return

    def read_json(self,file):
        f=open(file,"r",encoding="utf-8").read()
        return json.loads(f)

    def write_json(self,file,data):
        f=open(file,"w",encoding="utf-8")
        json.dump(data,f,indent=2,ensure_ascii=False)
        return

    def read_pickle(self,filename):
        return pickle.loads(open(filename,"rb").read())

    def write_pickle(self,filename,data):
        open(filename,"wb").write(pickle.dumps(data))
        return

def convert(file_list):
    for each in tqdm(file_list):
        try:
            id = each
            if id == "544053141067":
                print(id)
            if os.path.exists('/multi_modal/data/audios/{}.mp3'.format(id)):
                continue
            video = VideoFileClip("/multi_modal/data/videos/{}.mp4".format(each))
            audio = video.audio
            audio.write_audiofile('/multi_modal/data/audios/{}.mp3'.format(id))
        except Exception as e:
            print("error: ", e)

    return


import random
if __name__ == '__main__':
    print()
    io_pro=IOProcessor()

    ava_video_ids=list(io_pro.read_json("/multi_modal/data/product5m_v2/subset_v2_id_label.json").keys())

    filter_data=ava_video_ids
    random.shuffle(filter_data)

    thread_num = 10
    chunk_size = int(len(filter_data) / thread_num)
    chunk_data = [filter_data[i:i + chunk_size] for i in range(0, len(filter_data), chunk_size)]

    pool = Pool(thread_num)
    # download_all_data_images(data)
    multi_data = pool.map(convert, chunk_data)









