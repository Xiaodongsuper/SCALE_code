import argparse
import json
import logging
import os
import random
from io import open
import numpy as np

import sys
sys.path.append("../../../")

from tensorboardX import SummaryWriter
from tqdm import tqdm
from bisect import bisect
import yaml
from easydict import EasyDict as edict
import sys

import torch
import torch.nn.functional as F
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader, Dataset, RandomSampler


from pytorch_pretrained_bert.tokenization import BertTokenizer

from dataloaders.pretrain_dataset_ITP3VA import Pretrain_DataSet_Train
# from model.capture_ITP3V import BertForMultiModalPreTraining, BertConfig
from model.capture_ITP3VA_v3_dyctr_dymask_per_sample import Capture_ITPV_ForClassification, BertConfig


from utils_args import get_args

import torch.distributed as dist

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def read_pickle(filename):
    return pickle.loads(open(filename,"rb").read())

def write_pickle(filename,data):
    open(filename,"wb").write(pickle.dumps(data))
    return


def main():
    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    # timeStamp = '-'.join(task_names) + '_' + args.config_file.split('/')[1].split('.')[0]
    if '/' in args.from_pretrained:
        timeStamp = args.from_pretrained.split('/')[1]
    else:
        timeStamp = args.from_pretrained

    savePath = os.path.join(args.output_dir, timeStamp)

    config = BertConfig.from_json_file(args.config_file)
    # bert_weight_name = json.load(open("config/" + args.bert_model + "_weight_name.json", "r"))

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend="nccl")

    logger.info(
        "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16
        )
    )

    default_gpu = False
    if dist.is_available() and args.local_rank != -1:
        rank = dist.get_rank()
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case
    )

    train_dataset = Pretrain_DataSet_Train(
        tokenizer,
        seq_len=args.max_seq_length,
        batch_size=args.train_batch_size,
        predict_feature=args.predict_feature,
        num_workers=args.num_workers,
        lmdb_file=args.lmdb_file, # '/train/training_feat_all_v2.lmdb'
        caption_path=args.caption_path ,# "/id_info_dict.json"
        video_feature_dir=args.video_feature_dir,
        video_len=args.video_len,
        pv_seq_len=args.pv_seq_len,
        audio_file_dir=args.audio_file_dir,
        audio_len=args.audio_len,
        MLM=args.MLM,
        MRM=args.MRM,
        MEM=args.MEM,
        ITM=args.ITM,
        MFM=args.MFM,
        MAM=args.MAM
    )

    print("all image batch num: ", len(train_dataset))

    config.fast_mode = True
    if args.predict_feature:
        print("predict_feature")
        config.v_target_size = 2048
        config.predict_feature = True
    else:
        print("no predict_feature")
        config.v_target_size = 1601
        config.predict_feature = False

    model = Capture_ITPV_ForClassification.from_pretrained(args.from_pretrained, config)

    model.to(device)
    if args.local_rank != -1:
        try:
            from apex import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model = DDP(model, deay_allreduce=True)

    elif n_gpu > 1:
        model = nn.DataParallel(model)

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    print("Prepare to generate feature! ready!")
    model.eval()

    # lib
    lib_vil_id=[]
    dense_feature_list=[]

    for step, batch in enumerate(tqdm(train_dataset)):
        image_id = batch[-1]
        batch = batch[:-1]
        batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

        input_ids, input_mask, segment_ids, lm_label_ids, is_next, \
        pv_input_ids, pv_input_mask, pv_segment_ids, em_label_ids, \
        image_feat, image_loc, image_target, image_label, image_mask, \
        video_feat, video_target, video_label, video_mask, \
        audio_feat, audio_target, audio_label, audio_mask= (
            batch
        )

        # lib_vil_id.append(image_ids)
        lib_vil_id+=list(image_id)

        with torch.no_grad():
            _,_,pooled_output_dense=model(
                input_ids,
                pv_input_ids,
                image_feat,
                video_feat,
                audio_feat,
                image_loc,
                segment_ids,
                pv_segment_ids,
                input_mask,
                pv_input_mask,
                image_mask,
                video_mask,
                audio_mask,
            )

        # ##########################
        pooled_output_dense = pooled_output_dense.detach().cpu().numpy()
        ##############################33
        dense_feature_list.append(pooled_output_dense)

    dense_feature_np=np.vstack(dense_feature_list)
    print("dense_feature_np: ",dense_feature_np.shape)
    if not os.path.exists(args.feature_dir):
        os.makedirs(args.feature_dir)
    np.save("{}/dense_feature_np.npy".format(args.feature_dir),dense_feature_np)
    np.save("{}/id.npy".format(args.feature_dir),lib_vil_id)


if __name__ == "__main__":
    main()
