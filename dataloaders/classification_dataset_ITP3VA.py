

'''




'''



import copy
import json
import logging
import os
import random

import lmdb
import numpy as np
import tensorpack.dataflow as td

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import torch.distributed as dist
import sys
import pdb
# import torchaudio

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

devices=torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_json(file):
    f=open(file,"r",encoding="utf-8").read()
    return json.loads(f)

def write_json(file,data):
    f=open(file,"w",encoding="utf-8")
    json.dump(data,f,indent=2,ensure_ascii=False)
    return


class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(
            self,
            image_feat=None,
            image_target=None,
            caption=None,
            pv=None,
            video_feature=None,
            audio_feature=None,
            is_next=None,
            lm_labels=None,
            image_loc=None,
            num_boxes=None,
            num_frames=None,
            num_audio=None,
            label=None
    ):
        self.image_feat = image_feat
        self.caption = caption
        self.pv=pv
        self.video_feature=video_feature
        self.audio_feature=audio_feature
        self.is_next = is_next  # nextSentence
        self.lm_labels = lm_labels  # masked words for language model
        self.image_loc = image_loc
        self.image_target = image_target
        self.num_boxes = num_boxes
        self.num_frames=num_frames
        self.num_audio=num_audio
        self.label=label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
            self,
            input_ids=None,
            input_mask=None,
            segment_ids=None,
            is_next=None,
            lm_label_ids=None,
            pv_input_ids=None,
            pv_input_mask=None,
            pv_segment_ids=None,
            em_label_ids=None,
            image_feat=None,
            image_target=None,
            image_loc=None,
            image_label=None,
            image_mask=None,
            video_feat=None,
            video_target=None,
            video_label=None,
            video_mask=None,
            audio_feat=None,
            audio_target=None,
            audio_label=None,
            audio_mask=None,
            label=None
        ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_next = is_next
        self.lm_label_ids = lm_label_ids
        self.pv_input_ids=pv_input_ids
        self.pv_input_mask=pv_input_mask
        self.pv_segment_ids=pv_segment_ids
        self.em_label_ids=em_label_ids
        self.image_feat = image_feat
        self.image_loc = image_loc
        self.image_label = image_label
        self.image_target = image_target
        self.image_mask = image_mask
        self.video_feat = video_feat
        self.video_target = video_target
        self.video_label = video_label
        self.video_mask = video_mask
        self.audio_feat=audio_feat
        self.audio_target=audio_target
        self.audio_label=audio_label
        self.audio_mask=audio_mask
        self.label=label


class Classification_DataSet_Train(object):
    def __init__(
            self,
            tokenizer,
            seq_len,
            predict_feature=False,
            batch_size=512,
            num_workers=25,
            lmdb_file=None,
            caption_path=None,
            video_feature_dir=None,
            video_len=12,
            pv_seq_len=64,
            audio_file_dir=None,
            audio_len=12,
            label_list_file=None,
            MLM=True,
            MRM=True,
            MEM=True,
            ITM=True,
            MFM=True,
            MAM=True
        ):

        lmdb_file=lmdb_file
        caption_path=caption_path
        print("Loading from %s" % lmdb_file)

        ds = td.LMDBSerializer.load(lmdb_file, shuffle=False)
        self.num_dataset = len(ds)

        print("len: ",len(ds))

        preprocess_function = BertPreprocessBatch(
            caption_path,
            video_feature_dir,
            audio_file_dir,
            tokenizer,
            seq_len,
            36,
            video_len,
            pv_seq_len,
            audio_len,
            self.num_dataset,
            encoding="utf-8",
            predict_feature=predict_feature,
            label_list_file=label_list_file,
            MLM=MLM,
            MRM=MRM,
            MEM=MEM,
            ITM=ITM,
            MFM=MFM,
            MAM=MAM
        )

        # ds = td.LocallyShuffleData(ds, cache)
        # ds = td.PrefetchData(ds, 5000, 1)
        ds = td.MapData(ds, preprocess_function)
        # self.ds = td.PrefetchData(ds, 1)
        # ds = td.PrefetchDataZMQ(ds, num_workers)
        self.ds = td.BatchData(ds, batch_size)
        # self.ds = ds
        self.ds.reset_state() # TODO: it is retained in the original version

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.MLM=MLM
        self.MRM=MRM
        self.ITM=ITM
        self.MFM=MFM
        self.MEM=MEM
        self.MAM=MAM

    def __iter__(self):

        for batch in self.ds.get_data():
            input_ids, input_mask, segment_ids, lm_label_ids, is_next, \
            pv_input_ids, pv_input_mask, pv_segment_ids, em_label_ids,\
            image_feat, image_loc, image_target, image_label, image_mask, video_feat, \
            video_target,video_label,video_mask,\
            audio_feat,audio_target,audio_label,audio_mask,\
            label, image_id = batch

            # image
            batch_size = input_ids.shape[0]
            g_image_feat = np.sum(image_feat, axis=1) / np.sum(image_mask, axis=1, keepdims=True)
            image_feat = np.concatenate([np.expand_dims(g_image_feat, axis=1), image_feat], axis=1)
            image_feat = np.array(image_feat, dtype=np.float32)

            g_image_loc = np.repeat(np.array([[0,0,1,1,1]], dtype=np.float32), batch_size, axis=0)
            image_loc = np.concatenate([np.expand_dims(g_image_loc, axis=1), image_loc], axis=1)
            image_loc = np.array(image_loc, dtype=np.float32)

            g_image_mask = np.repeat(np.array([[1]]), batch_size, axis=0)
            image_mask = np.concatenate([g_image_mask, image_mask], axis=1)

            # video
            g_video_feat=np.sum(video_feat,axis=1)/np.sum(video_mask,axis=1,keepdims=True)
            video_feat=np.concatenate([np.expand_dims(g_video_feat,axis=1),video_feat],axis=1)
            video_feat=np.array(video_feat,dtype=np.float32)
            g_video_mask = np.repeat(np.array([[1]]), batch_size, axis=0)
            video_mask = np.concatenate([g_video_mask, video_mask], axis=1)

            # audio
            # pdb.set_trace()
            g_audio_feat=np.sum(audio_feat,axis=1)/np.sum(audio_mask,axis=1,keepdims=True)
            audio_feat=np.concatenate([np.expand_dims(g_audio_feat,axis=1),audio_feat],axis=1)
            audio_feat=np.array(audio_feat,dtype=np.float32)
            g_audio_mask = np.repeat(np.array([[1]]), batch_size, axis=0)
            audio_mask = np.concatenate([g_audio_mask, audio_mask], axis=1)


            batch = (input_ids, input_mask, segment_ids, lm_label_ids, is_next,
                     pv_input_ids, pv_input_mask, pv_segment_ids, em_label_ids,
                     image_feat,image_loc, image_target, image_label, image_mask,
                     video_feat,video_target,video_label,video_mask,
                     audio_feat, audio_target, audio_label, audio_mask
                     )

            yield tuple([torch.tensor(data) for data in batch]+[torch.tensor(label).long()]+ [image_id])

    def __len__(self):
        return self.ds.size()

class Classification_DataSet_Val(object):
    def __init__(
            self,
            tokenizer,
            seq_len,
            predict_feature=False,
            batch_size=512,
            num_workers=25,
            lmdb_file=None,
            caption_path=None,
            video_feature_dir=None,
            video_len=12,
            pv_seq_len=64,
            audio_file_dir=None,
            audio_len=12,
            label_list_file=None,
            MLM=True,
            MRM=True,
            MEM=True,
            ITM=True,
            MFM=True,
            MAM=True
        ):
        lmdb_file = lmdb_file
        caption_path = caption_path
        print("Loading from %s" % lmdb_file)

        ds = td.LMDBSerializer.load(lmdb_file, shuffle=True)
        self.num_dataset = len(ds)
        preprocess_function = BertPreprocessBatch(
            caption_path,
            video_feature_dir,
            audio_file_dir,
            tokenizer,
            seq_len,
            36,
            video_len,
            pv_seq_len,
            audio_len,
            self.num_dataset,
            encoding="utf-8",
            predict_feature=predict_feature,
            label_list_file=label_list_file,
            MLM=MLM,
            MRM=MRM,
            MEM=MEM,
            ITM=ITM,
            MFM=MFM,
            MAM=MAM
        )

        ds = td.MapData(ds, preprocess_function)
        self.ds = td.BatchData(ds, batch_size)
        self.ds.reset_state()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.MLM=MLM
        self.MRM=MRM
        self.ITM=ITM
        self.MFM=MFM
        self.MEM=MEM
        self.MAM=MAM

    def __iter__(self):
        for batch in self.ds.get_data():
            input_ids, input_mask, segment_ids, lm_label_ids, is_next, \
            pv_input_ids, pv_input_mask, pv_segment_ids, em_label_ids, \
            image_feat,image_loc, image_target, image_label, image_mask, video_feat, \
            video_target,video_label,video_mask,\
            audio_feat,audio_target,audio_label,audio_mask,\
            label,image_id = batch

            batch_size = input_ids.shape[0]
            g_image_feat = np.sum(image_feat, axis=1) / np.sum(image_mask, axis=1, keepdims=True)
            image_feat = np.concatenate([np.expand_dims(g_image_feat, axis=1), image_feat], axis=1)
            image_feat = np.array(image_feat, dtype=np.float32)

            g_image_loc = np.repeat(np.array([[0,0,1,1,1]], dtype=np.float32), batch_size, axis=0)
            image_loc = np.concatenate([np.expand_dims(g_image_loc, axis=1), image_loc], axis=1)
            
            image_loc = np.array(image_loc, dtype=np.float32)
            g_image_mask = np.repeat(np.array([[1]]), batch_size, axis=0)
            image_mask = np.concatenate([g_image_mask, image_mask], axis=1)

            # video
            g_video_feat=np.sum(video_feat,axis=1)/np.sum(video_mask,axis=1,keepdims=True)
            video_feat=np.concatenate([np.expand_dims(g_video_feat,axis=1),video_feat],axis=1)
            video_feat=np.array(video_feat,dtype=np.float32)
            g_video_mask = np.repeat(np.array([[1]]), batch_size, axis=0)
            video_mask = np.concatenate([g_video_mask, video_mask], axis=1)

            # audio
            g_audio_feat=np.sum(audio_feat,axis=1)/np.sum(audio_mask,axis=1,keepdims=True)
            audio_feat=np.concatenate([np.expand_dims(g_audio_feat,axis=1),audio_feat],axis=1)
            audio_feat=np.array(audio_feat,dtype=np.float32)
            g_audio_mask = np.repeat(np.array([[1]]), batch_size, axis=0)
            audio_mask = np.concatenate([g_audio_mask, audio_mask], axis=1)


            # batch = (input_ids, input_mask, segment_ids, lm_label_ids, is_next, image_feat, \
            # image_loc, image_target, image_label, image_mask, image_id)
            batch = (input_ids, input_mask, segment_ids, lm_label_ids, is_next,
                     pv_input_ids, pv_input_mask,pv_segment_ids,em_label_ids,
                     image_feat, image_loc, image_target, image_label, image_mask,
                     video_feat,video_target,video_label,video_mask,
                     audio_feat, audio_target, audio_label, audio_mask
                     )

            yield tuple([torch.tensor(data) for data in batch]+[torch.tensor(label).long()]+ [image_id])



    def __len__(self):
        return self.ds.size()


class BertPreprocessBatch(object):
    def __init__(
            self,
            caption_path,
            video_feature_dir,
            audio_file_dir,
            tokenizer,
            seq_len,
            region_len,
            video_len,
            pv_seq_len,
            audio_len,
            data_size,
            split="Train",
            encoding="utf-8",
            predict_feature=False,
            visualization=False,
            label_list_file=None,
            MLM=True,
            MRM=True,
            MEM=True,
            ITM=True,
            MFM=True,
            MAM=True
        ):

        self.MLM=MLM
        self.MRM=MRM
        self.MEM=MEM
        self.ITM=ITM
        self.MFM=MFM
        self.MAM=MAM

        self.split = split
        self.seq_len = seq_len
        self.region_len = region_len
        self.tokenizer = tokenizer
        self.predict_feature = predict_feature
        self.video_feature_dir=video_feature_dir
        self.video_len=video_len
        self.pv_seq_len=pv_seq_len
        self.audio_file_dir=audio_file_dir
        self.audio_len=audio_len

        self.label_list=read_json(label_list_file) #TODO: MODIFY
        self.label_map = {label: i for i, label in enumerate(self.label_list)}


        # self.captions = list(json.load(open(caption_path, 'r')).values())
        self.id_info_dict=json.load(open(caption_path, 'r'))

        self.captions=[]  # TODO: change
        for each in self.id_info_dict:
            # self.captions.append(id_info_dict[each]["title"])

            pv_pairs=self.id_info_dict[each]["pv"]
            if len(pv_pairs)==1:
                pv_pairs_str=""
            else:
                # pv_pair_list = [":".join(each.split("#:#")) for each in pv_pairs.split("#;#")]
                pv_pairs_str = pv_pairs
            self.captions.append(self.id_info_dict[each]["title"])
        # print("hello here! ")
        self.num_caps=len(self.captions)
        self.visualization = visualization

    def __call__(self, data):

        image_feature_wp, image_location_wp, num_boxes,  image_h, image_w, image_id, caption = data
        
        image_feature = np.zeros((self.region_len, 2048), dtype=np.float32)
        image_target = np.zeros((self.region_len, 1601), dtype=np.float32)
        image_location = np.zeros((self.region_len, 5), dtype=np.float32)

        num_boxes = int(num_boxes)
        image_feature[:num_boxes] = image_feature_wp
        # image_target[:num_boxes] = image_target_wp
        image_location[:num_boxes,:4] = image_location_wp

        image_location[:,4] = (image_location[:,3] - image_location[:,1]) * (image_location[:,2] - image_location[:,0]) / (float(image_w) * float(image_h))
        
        image_location[:,0] = image_location[:,0] / float(image_w)
        image_location[:,1] = image_location[:,1] / float(image_h)
        image_location[:,2] = image_location[:,2] / float(image_w)
        image_location[:,3] = image_location[:,3] / float(image_h)

        if self.predict_feature:
            image_feature = copy.deepcopy(image_feature)
            image_target = copy.deepcopy(image_feature)
        else:
            image_feature = copy.deepcopy(image_feature)
            image_target = copy.deepcopy(image_target)


        # pv
        pv_pairs = self.id_info_dict[image_id]["pv"]
        if len(pv_pairs) == 1:
            pv_pairs_str = ""
        else:
            # pv_pair_list = ["".join(each.split("#:#")) for each in pv_pairs.split("#;#")]
            pv_pairs_str = pv_pairs

        # caption
        caption=caption
        pv=pv_pairs_str
        caption, label = self.random_cap(caption)

        # video
        video_feature=np.zeros((self.video_len,1024),dtype=np.float32)
        num_frames=self.video_len
        try:
            video_feature_file="{}/{}.npy".format(self.video_feature_dir,image_id)
            video_feature_ = np.load(video_feature_file)
            # print("video_feature_: ",video_feature_.shape)
            num_frames=min(video_feature_.shape[0],self.video_len)
            video_feature[:num_frames]=video_feature_[:num_frames]
        except:
            # print("no video feature")
            pass

        # audio
        # audio_feature,num_audio=self.get_audio_feature(image_id)
        num_audio=80
        self.audio_feature_dir="audio_feature" #audio_path
        audio_feature=np.zeros((num_audio,751),dtype=np.float32)
        try:
            audio_feature_file="{}/{}.npy".format(self.audio_feature_dir,image_id)
            audio_feature=np.load(audio_feature_file)
        except:
            print("read audio feature error")

        classification_label=self.label_map[self.id_info_dict[image_id]["label"]]

        cur_example = InputExample(
            image_feat=image_feature,
            image_target=image_target,
            caption=caption,
            pv=pv,
            video_feature=video_feature,
            audio_feature=audio_feature,
            is_next=label,
            image_loc=image_location,
            num_boxes=num_boxes,
            num_frames=num_frames,
            num_audio=num_audio,
            label=classification_label
        )

        # transform sample to features
        cur_features = self.convert_example_to_features(cur_example,
                                                        self.seq_len,
                                                        self.tokenizer,
                                                        self.region_len,
                                                        self.video_len,
                                                        self.pv_seq_len,
                                                        num_audio)
        
        cur_tensors = (
            cur_features.input_ids,
            cur_features.input_mask,
            cur_features.segment_ids,
            cur_features.lm_label_ids,
            cur_features.is_next,
            cur_features.pv_input_ids,
            cur_features.pv_input_mask,
            cur_features.pv_segment_ids,
            cur_features.em_label_ids,
            cur_features.image_feat,
            cur_features.image_loc,
            cur_features.image_target,
            cur_features.image_label,
            cur_features.image_mask,
            cur_features.video_feat,
            cur_features.video_target,
            cur_features.video_label,
            cur_features.video_mask,
            cur_features.audio_feat,
            cur_features.audio_target,
            cur_features.audio_label,
            cur_features.audio_mask,
            cur_features.label,
            image_id,
        )
        return cur_tensors

    def random_cap(self, caption):
        if self.visualization:
            return caption, 0

        if self.ITM:
            if random.random() > 0.5:
                label = 0
            else:
                caption = self.get_random_caption()
                label = 1
        else:
            label = 0

        return caption, label

    def get_random_caption(self):
        rand_doc_idx = random.randint(0, self.num_caps - 1)
        caption = self.captions[rand_doc_idx]

        return caption


    def convert_example_to_features(self, example,
                                    max_seq_length,
                                    tokenizer,
                                    max_region_length,
                                    max_frame_length,
                                    max_pv_seq_length,
                                    max_audio_length):
        image_feat = example.image_feat
        caption = example.caption
        pv = example.pv
        image_loc = example.image_loc
        image_target = example.image_target
        num_boxes = int(example.num_boxes)

        # video
        video_feature=example.video_feature[:self.video_len]
        video_target = copy.deepcopy(video_feature)
        num_frames=example.num_frames

        # audio
        audio_feature=example.audio_feature
        audio_target=copy.deepcopy(audio_feature)
        num_audio=example.num_audio

        caption = self.tokenizer.tokenize(caption)
        self._truncate_seq_pair(caption, max_seq_length - 2)

        pv = pv.replace("#:#", ":")
        pv = pv.replace("#;#", ";")
        pv = self.tokenizer.tokenize(pv)
        self._truncate_seq_pair(pv, max_pv_seq_length - 2)


        # random mask
        caption, caption_label = self.random_word(caption, tokenizer)
        pv, pv_label = self.random_word(pv, tokenizer)
        image_feat, image_loc, image_label = self.random_region(image_feat, image_loc, num_boxes)
        video_feature,video_label=self.random_frame(video_feature,num_frames)
        audio_feature,audio_label=self.random_audio(audio_feature,num_audio)

        # concatenate lm labels
        lm_label_ids = [-1] + caption_label + [-1]
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in caption:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        # PV
        # print("pv: ",len(pv),pv)
        em_label_ids=[-1]+pv_label+[-1]
        pv_tokens=[]
        pv_segment_ids=[]
        pv_tokens.append("[CLS]")
        pv_segment_ids.append(0)
        for pv_token in pv:
            pv_tokens.append(pv_token)
            pv_segment_ids.append(0)
        pv_tokens.append("[SEP]")
        pv_segment_ids.append(0)

        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]
            lm_label_ids = lm_label_ids[:max_seq_length]
            segment_ids = segment_ids[:max_seq_length]

        if len(pv_tokens) > max_pv_seq_length:
            pv_tokens = pv_tokens[:max_pv_seq_length]
            em_label_ids = em_label_ids[:max_pv_seq_length]
            pv_segment_ids = pv_segment_ids[:max_pv_seq_length]

        input_ids=tokenizer.convert_tokens_to_ids(tokens)
        pv_input_ids=tokenizer.convert_tokens_to_ids(pv_tokens)

        input_mask = [1] * (len(input_ids))
        pv_input_mask=[1]*(len(pv_input_ids))
        image_mask = [1] * (num_boxes)
        video_mask = [1] * (num_frames)
        audio_mask = [1] * (num_audio)

        # Zero-pad up to the visual sequence length.
        while len(image_mask) < max_region_length:
            image_mask.append(0)
            image_label.append(-1)

        while len(video_mask)<max_frame_length:
            video_mask.append(0)
            video_label.append(-1)

        while len(audio_mask)<max_audio_length:
            audio_mask.append(0)
            audio_label.append(-1)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            lm_label_ids.append(-1)

        while len(pv_input_ids) < max_pv_seq_length:
            pv_input_ids.append(0)
            pv_input_mask.append(0)
            pv_segment_ids.append(0)
            em_label_ids.append(-1)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(lm_label_ids) == max_seq_length
        assert len(pv_input_ids) == max_pv_seq_length
        assert len(pv_input_mask) == max_pv_seq_length
        assert len(pv_segment_ids) == max_pv_seq_length
        assert len(image_mask) == max_region_length
        assert len(image_label) == max_region_length
        assert len(video_feature) == max_frame_length
        assert len(video_mask) == max_frame_length
        assert len(video_label) == max_frame_length
        assert len(audio_feature) == max_audio_length
        assert len(audio_mask) == max_audio_length
        assert len(audio_label) == max_audio_length

        features = InputFeatures(
            input_ids=np.array(input_ids),
            input_mask=np.array(input_mask),
            segment_ids=np.array(segment_ids),
            lm_label_ids=np.array(lm_label_ids),
            is_next=np.array(example.is_next),
            pv_input_ids=np.array(pv_input_ids),
            pv_input_mask=np.array(pv_input_mask),
            pv_segment_ids=np.array(pv_segment_ids),
            em_label_ids=np.array(em_label_ids),
            image_feat=image_feat,
            image_target=image_target,
            image_loc=image_loc,
            image_label=np.array(image_label),
            image_mask = np.array(image_mask),
            video_feat=np.array(video_feature),
            video_target=np.array(video_target),
            video_label=np.array(video_label),
            video_mask=np.array(video_mask),
            audio_feat=np.array(audio_feature),
            audio_target=np.array(audio_target),
            audio_label=np.array(audio_label),
            audio_mask=np.array(audio_mask),
            label=example.label
        )
        return features
    #
    # def get_audio_feature(self,item_id):
    #     # audio
    #     audio_data = torch.zeros([self.audio_len * 16000]).to(devices)
    #     if not os.path.exists("{}/{}.mp3".format(self.audio_file_dir, item_id)):
    #         print("no audio file")
    #         audio_data = torch.zeros([self.audio_len * 16000]).to(devices)
    #     else:
    #         try:
    #             audio_file = "{}/{}.mp3".format(self.audio_file_dir, item_id)
    #             audios,sample_rate = torchaudio.load(audio_file)
    #             audios = audios.to(devices)
    #             resample = torchaudio.transforms.Resample(sample_rate, 16000).to(devices)
    #             audios = resample(audios)
    #             audio_data = torch.sum(torch.as_tensor(audios), dim=0) / 2
    #         except:
    #             print("no audio file")
    #
    #     if (len(audio_data) / 16000 < self.audio_len):
    #         new_audio_data = torch.zeros([self.audio_len * 16000])
    #         new_audio_data[0:len(audio_data)] = audio_data
    #         audio_data = new_audio_data.to(devices)
    #     else:
    #         audio_data = audio_data[:16000 * self.audio_len]
    #
    #     transform = torchaudio.transforms.MelSpectrogram(n_mels=80,
    #                                                      n_fft=1024,
    #                                                      win_length=1024,
    #                                                      hop_length=256).to(devices)
    #
    #     audio_feature = transform(audio_data)
    #     cur_mean, cur_std = audio_feature.mean(dim=0), audio_feature.std(dim=0)
    #     audio_feature = (audio_feature - cur_mean) / (cur_std + 1e-9)
    #     audio_feature = audio_feature.permute(1, 0)
    #     num_audio = audio_feature.shape[0]
    #
    #     return audio_feature,num_audio

    def _truncate_seq_pair(self, tokens_b, max_length):
        while True:
            total_length = len(tokens_b)
            if total_length <= max_length:
                break

            tokens_b.pop()

    def random_word(self, tokens, tokenizer):
        output_label = []

        if self.MLM:
            for i, token in enumerate(tokens):
                prob = random.random()
                # mask token with 15% probability

                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = "[MASK]"

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    try:
                        output_label.append(tokenizer.vocab[token])
                    except KeyError:
                        # For unknown words (should not occur with BPE vocab)
                        output_label.append(tokenizer.vocab["[UNK]"])
                        logger.warning(
                            "Cannot find token '{}' in vocab. Using [UNK] insetad".format(token)
                        )
                else:
                    # no masking token (will be ignored by loss function later)
                    output_label.append(-1)
        else:
            for i, token in enumerate(tokens):
                output_label.append(-1)

        return tokens, output_label

    def mask_pv(self,tokens, tokenizer, need_mask=False):
        output_label = []
        if self.MEM:
            for i, token in enumerate(tokens):
                if need_mask:
                    tokens[i] = "[MASK]"
                    try:
                        output_label.append(tokenizer.vocab[token])
                    except KeyError:
                        # For unknown words (should not occur with BPE vocab)
                        output_label.append(tokenizer.vocab["[UNK]"])
                        logger.warning(
                            "Cannot find token '{}' in vocab. Using [UNK] insetad".format(token)
                        )
                else:
                    output_label.append(-1)
        else:
            for i, token in enumerate(tokens):
                output_label.append(-1)

        return tokens, output_label

    def random_region(self, image_feat, image_loc, num_boxes):
        """
        TODO: maybe all the patch is not masked, and the loss is nan, to be done
        """
        output_label = []

        if self.MRM:
            for i in range(num_boxes):
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.9:
                        image_feat[i] = 0
                    output_label.append(1)
                else:
                    # no masking token (will be ignored by loss function later)
                    output_label.append(-1)
        else:
            for i in range(num_boxes):
                output_label.append(-1)

        return image_feat, image_loc, output_label

    def random_frame(self,video_feature,num_frames):
        output_label=[]
        if self.MFM:
            for i in range(num_frames):
                prob=random.random()
                if prob<0.15:
                    video_feature[i]=0
                    output_label.append(1)
                else:
                    output_label.append(-1)

        else:
            for i in range(num_frames):
                output_label.append(-1)

        return video_feature,output_label

    def random_audio(self,audio_feature,num_audio):
        output_label=[]
        if self.MAM:
            for i in range(num_audio):
                prob=random.random()
                if prob<0.15:
                    audio_feature[i]=0
                    output_label.append(1)
                else:
                    output_label.append(-1)
        else:
            for i in range(num_audio):
                output_label.append(-1)

        return audio_feature,output_label

