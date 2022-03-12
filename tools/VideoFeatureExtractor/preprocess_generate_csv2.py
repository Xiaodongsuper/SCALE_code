'''
First of all you need to generate a csv containing the list of videos you want to process. For instance, if you have video1.mp4 and video2.webm to process, you will need to generate a csv of this form:

video_path, feature_path
video1.mp4, path_of_video1_features.npy
video2.webm, path_of_video2_features.npy

And then just simply run:

python extract.py --csv=input.csv --type=s3dg --batch_size=64 --num_decoding_thread=4
'''

import os
import argparse


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




if __name__ == "__main__":

    io_process=IOProcessor()

    parser = argparse.ArgumentParser(description='Generate CSV')

    parser.add_argument("--ids_file",type=str)
    parser.add_argument('--csv', type=str, help='input csv with video input path')
    parser.add_argument('--video_root_path', type=str, help='video path')
    parser.add_argument('--feature_root_path', type=str, help='feature path')
    parser.add_argument('--csv_save_path', type=str, help='csv path', default='.')
    args = parser.parse_args()

    video_root_path = args.video_root_path
    feature_root_path = args.feature_root_path

    csv_save_path = os.path.join(args.csv_save_path, args.csv)
    fp_wt = open(csv_save_path, 'w')
    line = "video_path,feature_path"
    fp_wt.write(line + "\n")

    ids=io_process.read_json("{}".format(args.ids_file))

    # all_files = os.walk(video_root_path)
    # for path, d, filelist in all_files:
    #     for file_name in filelist:
    #         video_path = os.path.join(path, file_name)
    #         video_id = video_path.replace("\\","/").split("/")[-1].split(".")[0]
    #         feature_path = os.path.join(feature_root_path, "{}.npy".format(video_id))
    #         line = ",".join([video_path, feature_path])
    #         fp_wt.write(line + "\n")

    for id in ids:
        video_path = os.path.join(video_root_path, "{}.mp4".format(id))
        video_id =id
        feature_path = os.path.join(feature_root_path, "{}.npy".format(video_id))
        line = ",".join([video_path, feature_path])
        fp_wt.write(line + "\n")


    fp_wt.close()
    print("csv is saved in: {}".format(csv_save_path))