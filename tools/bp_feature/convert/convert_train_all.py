import os
import numpy as np
# from tensorpack.dataflow import RNGDataFlow, PrefetchDataZMQ
from tensorpack.dataflow import *
import lmdb
import json
import pdb
import csv
FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features',"title"]
import sys
import pandas as pd
import zlib
import base64

csv.field_size_limit(sys.maxsize)


def read_json(file):
    f=open(file,"r",encoding="utf-8").read()
    return json.loads(f)

def write_json(file,data):
    f=open(file,"w",encoding="utf-8")
    json.dump(data,f,indent=2,ensure_ascii=False)
    return


def open_tsv(fname, folder):
    print("Opening %s Data File..." % fname)
    df = pd.read_csv(fname, sep=',', names=["caption", "url"], usecols=range(1, 3))
    df['folder'] = folder
    print("Processing", len(df), " Images:")
    return df


def _file_name(row):
    return "%s/%s" % (row['folder'], (zlib.crc32(row['url'].encode('utf-8')) & 0xffffffff))

class Conceptual_Caption(RNGDataFlow):
    """
    """
    def __init__(self, corpus_path, shuffle=False):
        """
        Same as in :class:`ILSVRC12`.
        """
        self.shuffle = shuffle
        self.num_file = 30
        # self.name = os.path.join(corpus_path, 'conceptual_caption_trainsubset_resnet101_faster_rcnn_genome.tsv.%d')

        file_name_list=[
          "train_2.5m_3.5m_50_62.tsv",
          "train_1m_2m_25_37.tsv",
          "train_3.5m_5m_75_87.tsv",
          "train_3.5m_5m_12_25.tsv",
          "train_1m_75_87.tsv",
          "train_2.5m_3.5m_62_75.tsv",
          "train_2m_2.5m_37_50.tsv",
          "train_3.5m_5m_37_50.tsv",
          "train_2.5m_3.5m_87_100.tsv",
          "train_1m_2m_50_62.tsv",
          "train_3.5m_5m_50_62.tsv",
          "train_1m_25_37.tsv",
          "train_1m_37_50.tsv",
          "train_1m_2m_75_87.tsv",
          "train_2.5m_3.5m_0_12.tsv",
          "train_1m_2m_12_25.tsv",
          "train_1m_12_25.tsv",
          "train_2m_2.5m_12_25.tsv",
          "train_1m_50_62.tsv",
          "train_2.5m_3.5m_75_87.tsv",
          "train_3.5m_5m_0_12.tsv",
          "train_1m_87_100.tsv",
          "train_1m_0_12.tsv",
          "train_3.5m_5m_87_100.tsv",
          "train_3.5m_5m_62_75.tsv",
          "train_1m_2m_87_100.tsv",
          "train_1m_2m_62_75.tsv",
          "train_2m_2.5m_25_37.tsv",
          "train_2m_2.5m_0_12.tsv",
          "train_2.5m_3.5m_12_25.tsv",
          "train_1m_62_75.tsv",
          "train_2.5m_3.5m_37_50.tsv",
          "train_1m_2m_37_50.tsv",
          "train_2.5m_3.5m_25_37.tsv",
          "train_3.5m_5m_25_37.tsv",
          "train_1m_2m_0_12.tsv"
        ]

        self.infiles = ["{}/{}".format("/multi_modal/data/tsv_features",i) for i in file_name_list]
        self.counts = []
        # self.num_caps = 3136161
        # self.num_caps = 15027 + 113970 # TODO: modify each time
        self.num_caps = 4363122






    def __len__(self):
        return self.num_caps

    def __iter__(self):
        cnt = 0
        for infile in self.infiles:
            count = 0
            with open(infile) as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
                for item in reader:
                    cnt+=1

                    # if cnt>521820 and cnt<521830:
                    #     continue

                    try:
                        image_id = item['image_id']
                        # if image_id not in self.id_list: continue

                        image_h = int(item['image_h'])
                        image_w = int(item['image_w'])
                        num_boxes = item['num_boxes']
                        boxes = np.frombuffer(base64.b64decode(item['boxes']), dtype=np.float32).reshape(
                            int(num_boxes), 4)
                        features = np.frombuffer(base64.b64decode(item['features']), dtype=np.float32).reshape(
                            int(num_boxes), 2048)
                        # print("image_id: ",image_id)
                        caption = item["title"]
                    except:
                        print("error: ",cnt,image_id)
                        continue
                    yield [features, boxes, num_boxes, image_h, image_w, image_id, caption]




import time
if __name__ == '__main__':
    import time
    # time.sleep(3600*3)
    corpus_path = ''

    # time.sleep(25200)
    ds = Conceptual_Caption(corpus_path)

    # for each in ds:
    #     print(each)
    #     break

    ds1 = PrefetchDataZMQ(ds)
    LMDBSerializer.save(ds1, '/multi_modal/data/lmdb_features/train_feature.lmdb')












