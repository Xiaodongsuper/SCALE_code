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
from model.capture_ITP3VA_v3_dyctr_dymask_per_sample import BertForMultiModalPreTraining, BertConfig


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

    # if default_gpu and not os.path.exists(savePath):
    #     os.makedirs(savePath)

    # task_batch_size, task_num_iters, task_ids, task_datasets_val, task_dataloader_val \
    #     = LoadDatasetEval(args, task_cfg, args.tasks.split('-'))
    #

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
        MFM=args.MFM
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

    model = BertForMultiModalPreTraining.from_pretrained(args.from_pretrained, config)

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

    t_feature_list=[]
    p_feature_list=[]
    i_feature_list=[]
    v_feature_list=[]
    a_feature_list=[]

    tp_feature_list=[]
    ti_feature_list=[]
    tv_feature_list=[]
    pi_feature_list=[]
    pv_feature_list=[]
    iv_feature_list=[]
    ta_feature_list=[]
    pa_feature_list=[]
    ia_feature_list=[]
    va_feature_list=[]

    tpi_feature_list=[]
    tpv_feature_list=[]
    tiv_feature_list=[]
    piv_feature_list=[]

    tpiv_feature_list=[]

    tpiva_feature_list=[]

    graph_list=[]
    modality_weight_list=[]



    for step, batch in enumerate(tqdm(train_dataset)):
        image_id = batch[-1]
        batch = batch[:-1]
        batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

        input_ids, input_mask, segment_ids, lm_label_ids, is_next, \
        pv_input_ids, pv_input_mask, pv_segment_ids, em_label_ids, \
        image_feat, image_loc, image_target, image_label, image_mask, \
        video_feat, video_target, video_label, video_mask,\
        audio_feat, audio_target, audio_label, audio_mask = (
            batch
        )

        # lib_vil_id.append(image_ids)
        lib_vil_id+=list(image_id)

        with torch.no_grad():
            _,_,_,_,_,_,\
            pooled_output_t,\
            pooled_output_p,\
            pooled_output_i,\
            pooled_output_v,\
            pooled_output_a,graph,modality_weight=model(
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
                lm_label_ids,
                em_label_ids,
                image_label,
                image_target,
                video_label,
                video_target,
                audio_label,
                audio_target,
                is_next,
                return_features=True
            )

        pooled_output_tp=pooled_output_t+pooled_output_p
        pooled_output_ti=pooled_output_t+pooled_output_i
        pooled_output_tv=pooled_output_t+pooled_output_v
        pooled_output_pi=pooled_output_p+pooled_output_i
        pooled_output_pv=pooled_output_p+pooled_output_v
        pooled_output_iv=pooled_output_i+pooled_output_v
        pooled_output_ta=pooled_output_t+pooled_output_a
        pooled_output_pa=pooled_output_p+pooled_output_a
        pooled_output_ia=pooled_output_i+pooled_output_a
        pooled_output_va=pooled_output_v+pooled_output_a

        pooled_output_tpi= pooled_output_t + pooled_output_p+pooled_output_i
        pooled_output_tpv = pooled_output_t + pooled_output_p + pooled_output_v
        pooled_output_tiv = pooled_output_t + pooled_output_i + pooled_output_v
        pooled_output_piv = pooled_output_p + pooled_output_i + pooled_output_v

        pooled_output_tpiv = pooled_output_t + pooled_output_p + pooled_output_i + pooled_output_v

        pooled_output_tpiva=pooled_output_t+pooled_output_p\
                            +pooled_output_i+pooled_output_v\
                            +pooled_output_a

        # ##########################

        pooled_output_t = pooled_output_t.detach().cpu().numpy()
        pooled_output_p = pooled_output_p.detach().cpu().numpy()
        pooled_output_i = pooled_output_i.detach().cpu().numpy()
        pooled_output_v = pooled_output_v.detach().cpu().numpy()
        pooled_output_a = pooled_output_a.detach().cpu().numpy()

        pooled_output_tp = pooled_output_tp.detach().cpu().numpy()
        pooled_output_ti = pooled_output_ti.detach().cpu().numpy()
        pooled_output_tv = pooled_output_tv.detach().cpu().numpy()
        pooled_output_pi = pooled_output_pi.detach().cpu().numpy()
        pooled_output_pv = pooled_output_pv.detach().cpu().numpy()
        pooled_output_iv = pooled_output_iv.detach().cpu().numpy()
        pooled_output_ta = pooled_output_ta.detach().cpu().numpy()
        pooled_output_pa = pooled_output_pa.detach().cpu().numpy()
        pooled_output_ia = pooled_output_ia.detach().cpu().numpy()
        pooled_output_va = pooled_output_va.detach().cpu().numpy()

        pooled_output_tpi = pooled_output_tpi.detach().cpu().numpy()
        pooled_output_tpv = pooled_output_tpv.detach().cpu().numpy()
        pooled_output_tiv = pooled_output_tiv.detach().cpu().numpy()
        pooled_output_piv = pooled_output_piv.detach().cpu().numpy()

        pooled_output_tpiv = pooled_output_tpiv.detach().cpu().numpy()
        pooled_output_tpiva = pooled_output_tpiva.detach().cpu().numpy()
        graph=graph.detach().cpu().numpy()
        modality_weight=modality_weight.detach().cpu().numpy()


        ##############################33
        t_feature_list.append(pooled_output_t)
        p_feature_list.append(pooled_output_p)
        i_feature_list.append(pooled_output_i)
        v_feature_list.append(pooled_output_v)
        a_feature_list.append(pooled_output_a)

        tp_feature_list.append(pooled_output_tp)
        ti_feature_list.append(pooled_output_ti)
        tv_feature_list.append(pooled_output_tv)
        pi_feature_list.append(pooled_output_pi)
        pv_feature_list.append(pooled_output_pv)
        iv_feature_list.append(pooled_output_iv)
        ta_feature_list.append(pooled_output_ta)
        pa_feature_list.append(pooled_output_pa)
        ia_feature_list.append(pooled_output_ia)
        va_feature_list.append(pooled_output_va)

        tpi_feature_list.append(pooled_output_tpi)
        tpv_feature_list.append(pooled_output_tpv)
        tiv_feature_list.append(pooled_output_tiv)
        piv_feature_list.append(pooled_output_piv)

        tpiv_feature_list.append(pooled_output_tpiv)
        tpiva_feature_list.append(pooled_output_tpiva)

        graph_list.append(graph)
        modality_weight_list.append(modality_weight)

        # if step==10000:
        #     break

    t_feature_np=np.vstack(t_feature_list)
    p_feature_np=np.vstack(p_feature_list)
    i_feature_np=np.vstack(i_feature_list)
    v_feature_np=np.vstack(v_feature_list)
    a_feature_np=np.vstack(a_feature_list)

    tp_feature_np=np.vstack(tp_feature_list)
    ti_feature_np=np.vstack(ti_feature_list)
    tv_feature_np = np.vstack(tv_feature_list)
    pi_feature_np = np.vstack(pi_feature_list)
    pv_feature_np = np.vstack(pv_feature_list)
    iv_feature_np = np.vstack(iv_feature_list)
    ta_feature_np = np.vstack(ta_feature_list)
    pa_feature_np = np.vstack(pa_feature_list)
    ia_feature_np = np.vstack(ia_feature_list)
    va_feature_np = np.vstack(va_feature_list)

    tpi_feature_np=np.vstack(tpi_feature_list)
    tpv_feature_np = np.vstack(tpv_feature_list)
    tiv_feature_np = np.vstack(tiv_feature_list)
    piv_feature_np = np.vstack(piv_feature_list)

    tpiv_feature_np = np.vstack(tpiv_feature_list)
    tpiva_feature_np = np.vstack(tpiva_feature_list)

    graph_np=np.vstack(graph_list)
    modality_weight_np=np.vstack(modality_weight_list)


    print("t_feature_np: ",t_feature_np.shape)
    print("p_feature_np: ",p_feature_np.shape)
    print("i_feature_np: ",i_feature_np.shape)
    print("v_feature_np: ", v_feature_np.shape)
    print("a_feature_np: ", a_feature_np.shape)

    print("tp_feature_np: ", tp_feature_np.shape)
    print("ti_feature_np: ", ti_feature_np.shape)
    print("tv_feature_np: ", tv_feature_np.shape)
    print("pi_feature_np: ", pi_feature_np.shape)
    print("pv_feature_np: ", pv_feature_np.shape)
    print("iv_feature_np: ", iv_feature_np.shape)
    print("ta_feature_np: ", ta_feature_np.shape)
    print("pa_feature_np: ", pa_feature_np.shape)
    print("ia_feature_np: ", ia_feature_np.shape)
    print("va_feature_np: ", va_feature_np.shape)

    print("tpi_feature_np: ", tpi_feature_np.shape)
    print("tpv_feature_np: ", tpv_feature_np.shape)
    print("tiv_feature_np: ", tiv_feature_np.shape)
    print("piv_feature_np: ", piv_feature_np.shape)

    print("tpiv_feature_np: ", tpiv_feature_np.shape)
    print("tpiva_feature_np: ", tpiva_feature_np.shape)

    print("graph_np: ",graph_np.shape)
    print("modality_weight_np: ",modality_weight_np.shape)

    if not os.path.exists(args.feature_dir):
        os.makedirs(args.feature_dir)

    np.save("{}/t_feature_np.npy".format(args.feature_dir),t_feature_np)
    np.save("{}/p_feature_np.npy".format(args.feature_dir),p_feature_np)
    np.save("{}/i_feature_np.npy".format(args.feature_dir),i_feature_np)
    np.save("{}/v_feature_np.npy".format(args.feature_dir),v_feature_np)
    np.save("{}/a_feature_np.npy".format(args.feature_dir),a_feature_np)

    np.save("{}/tp_feature_np.npy".format(args.feature_dir),tp_feature_np)
    np.save("{}/ti_feature_np.npy".format(args.feature_dir),ti_feature_np)
    np.save("{}/tv_feature_np.npy".format(args.feature_dir),tv_feature_np)
    np.save("{}/pi_feature_np.npy".format(args.feature_dir),pi_feature_np)
    np.save("{}/pv_feature_np.npy".format(args.feature_dir),pv_feature_np)
    np.save("{}/iv_feature_np.npy".format(args.feature_dir),iv_feature_np)
    np.save("{}/ta_feature_np.npy".format(args.feature_dir),ta_feature_np)
    np.save("{}/pa_feature_np.npy".format(args.feature_dir),pa_feature_np)
    np.save("{}/ia_feature_np.npy".format(args.feature_dir),ia_feature_np)
    np.save("{}/va_feature_np.npy".format(args.feature_dir),va_feature_np)

    np.save("{}/tpi_feature_np.npy".format(args.feature_dir),tpi_feature_np)
    np.save("{}/tpv_feature_np.npy".format(args.feature_dir),tpv_feature_np)
    np.save("{}/tiv_feature_np.npy".format(args.feature_dir),tiv_feature_np)
    np.save("{}/piv_feature_np.npy".format(args.feature_dir),piv_feature_np)

    np.save("{}/tpiv_feature_np.npy".format(args.feature_dir), tpiv_feature_np)
    np.save("{}/tpiva_feature_np.npy".format(args.feature_dir), tpiva_feature_np)
    np.save("{}/graph_np.npy".format(args.feature_dir),graph_np)
    np.save("{}/modality_weight_np.npy".format(args.feature_dir),modality_weight_np)
    np.save("{}/id.npy".format(args.feature_dir),lib_vil_id)


if __name__ == "__main__":
    main()
