import argparse
import json
import logging
import os
import random
from io import open
import math
import sys
sys.path.append("../../")

from time import gmtime, strftime
from timeit import default_timer as timer

import numpy as np
from tqdm import tqdm, trange

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from dataloaders.classification_dataset_ITP3VA import Classification_DataSet_Train,Classification_DataSet_Val
from model.capture_ITP3VA_v3_dyctr_dymask import Capture_ITPV_ForClassification, BertConfig

# from ProductBert.ProductBert3_CLR import HybridMMTRForClassification, BertConfig  # just for eproduct

import torch.distributed as dist

import pdb
from utils_args import get_args
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


f = open("cls_result.txt", "a")
def test(Test_Dataset,tokenizer,args,model,epoch):
    model.eval()
    correct=0
    total=0

    for step, batch in enumerate(Test_Dataset):

        # batch = iter_dataloader.next()
        image_id = batch[-1]
        batch = batch[:-1]
        batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

        input_ids, input_mask, segment_ids, lm_label_ids, is_next, \
        pv_input_ids, pv_input_mask, pv_segment_ids, em_label_ids, \
        image_feat, image_loc, image_target, image_label, image_mask, \
        video_feat, video_target, video_label, video_mask,\
        audio_feat, audio_target, audio_label, audio_mask, label = (
            batch
        )

        if torch.sum(torch.isnan(image_feat)) > 0:
            continue

        if torch.sum(torch.isnan(image_feat)) > 0:
            print("is nan")
            continue
        with torch.no_grad():
            loss,output_logits,_ = model(
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
                label
            )
        step += 1

        _,predicted=torch.max(output_logits.data,1)

        total+=output_logits.size(0)
        correct+=(predicted==label).sum().item()

    print("correct: {} total:{} ".format(correct,total))
    print("ACC: ",100*correct/total)

    f.write("{}: {}\n\n".format(epoch,100*correct/total))
    return  100*correct/total

def main():

    args = get_args()


    print(args)
    if args.save_name is not '':
        timeStamp = args.save_name
    else:
        timeStamp = strftime("%d-%b-%y-%X-%a", gmtime())
        timeStamp += "_{:0>6d}".format(random.randint(0, 10e6))

    savePath = os.path.join(args.output_dir, timeStamp)

    if not os.path.exists(savePath):
        os.makedirs(savePath)
    
    config = BertConfig.from_json_file(args.config_file)
    
    if args.freeze > config.t_biattention_id[0]:
        config.fixed_t_layer = config.t_biattention_id[0]

    if args.without_coattention:
        config.with_coattention = False
    # save all the hidden parameters. 
    with open(os.path.join(savePath, 'command.txt'), 'w') as f:
        print(args, file=f)  # Python 3.x
        print('\n', file=f)
        print(config, file=f)

    bert_weight_name = json.load(open("../../config/bert-base-uncased_weight_name.json", "r"))
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
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

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps
            )
        )

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case
    )

    num_train_optimization_steps = None

    viz = TBlogger("logs", timeStamp)

    print("MLM: {} MRM:{} ITM:{}".format(args.MLM, args.MRM, args.ITM))


    train_dataset = Classification_DataSet_Train(
        tokenizer,
        seq_len=args.max_seq_length,
        batch_size=args.train_batch_size,
        predict_feature=args.predict_feature,
        num_workers=args.num_workers,
        lmdb_file=args.train_lmdb_file, # '/train/training_feat_all_v2.lmdb'
        caption_path=args.caption_path ,# "id_info_dict.json"
        video_feature_dir=args.video_feature_dir,
        video_len=args.video_len,
        pv_seq_len=args.pv_seq_len,
        audio_file_dir=args.audio_file_dir,
        audio_len=args.audio_len,
        label_list_file=args.label_list_file,
        MLM=args.MLM,
        MRM=args.MRM,
        MEM=args.MEM,
        ITM=args.ITM,
        MFM=args.MFM,
        MAM=args.MAM
    )
    Test_Dataset = Classification_DataSet_Val(
        tokenizer,
        seq_len=args.max_seq_length,
        batch_size=args.train_batch_size,
        predict_feature=args.predict_feature,
        num_workers=args.num_workers,
        lmdb_file=args.test_lmdb_file,  # '/train/training_feat_all_v2.lmdb'
        caption_path=args.caption_path,  # "/id_info_dict.json"
        video_feature_dir=args.video_feature_dir,
        video_len=args.video_len,
        pv_seq_len=args.pv_seq_len,
        audio_file_dir=args.audio_file_dir,
        audio_len=args.audio_len,
        label_list_file=args.label_list_file,
        MLM=args.MLM,
        MRM=args.MRM,
        MEM=args.MEM,
        ITM=args.ITM,
        MFM=args.MFM,
        MAM=args.MAM
    )



    num_train_optimization_steps = (
        int(
            train_dataset.num_dataset
            / args.train_batch_size
            / args.gradient_accumulation_steps
        )
        * (args.num_train_epochs - args.start_epoch)
    )
    # if args.local_rank != -1:
    #     num_train_optimization_steps = (
    #         num_train_optimization_steps // torch.distributed.get_world_size()
    #     )

    default_gpu = False
    if dist.is_available() and args.distributed:
        rank = dist.get_rank()
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True

    # pdb.set_trace()
    if args.predict_feature:
        config.v_target_size = 2048
        config.predict_feature = True
    else:
        config.v_target_size = 1601
        config.predict_feature = False

    if args.from_pretrained:
        model = Capture_ITPV_ForClassification.from_pretrained(args.from_pretrained, config)
    else:
        model = Capture_ITPV_ForClassification(config)

    model.cuda()

    if args.fp16:
        model.half()
    if args.local_rank != -1:
        try:
            from apex import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    if args.freeze != -1:
        bert_weight_name_filtered = []
        for name in bert_weight_name:
            if 'embeddings' in name:
                bert_weight_name_filtered.append(name)
            elif 'encoder' in name:
                layer_num = name.split('.')[2]
                if int(layer_num) <= args.freeze:
                    bert_weight_name_filtered.append(name)

        optimizer_grouped_parameters = []
        for key, value in dict(model.named_parameters()).items():
            if key[12:] in bert_weight_name_filtered:
                value.requires_grad = False

        if default_gpu:
            print("filtered weight")
            print(bert_weight_name_filtered)

    if not args.from_pretrained:
        param_optimizer = list(model.named_parameters())
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
    else:
        optimizer_grouped_parameters = []
        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                if key[12:] in bert_weight_name:
                    lr = args.learning_rate * 0.1
                else:
                    lr = args.learning_rate

                if any(nd in key for nd in no_decay):
                    optimizer_grouped_parameters += [
                        {"params": [value], "lr": lr, "weight_decay": 0.01}
                    ]

                if not any(nd in key for nd in no_decay):
                    optimizer_grouped_parameters += [
                        {"params": [value], "lr": lr, "weight_decay": 0.0}
                    ]
        if default_gpu:
            print(len(list(model.named_parameters())), len(optimizer_grouped_parameters))

    # set different parameters for vision branch and lanugage branch.
    if args.fp16:
        try:
            from apex import FP16_Optimizer
            from apex import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )

        optimizer = FusedAdam(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            bias_correction=False,
            max_grad_norm=1.0,
        )
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        if args.from_pretrained:
            optimizer = BertAdam(
                optimizer_grouped_parameters,
                warmup=args.warmup_proportion,
                t_total=num_train_optimization_steps,

            )

        else:
            optimizer = BertAdam(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                warmup=args.warmup_proportion,
                t_total=num_train_optimization_steps,
            )

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", train_dataset.num_dataset)
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    startIterID = 0
    global_step = 0
    masked_loss_v_tmp = 0
    masked_loss_t_tmp = 0
    next_sentence_loss_tmp = 0
    loss_tmp = 0
    start_t = timer()


    print("Prepare to training!")
    print("MLM: {} MRM:{} ITM:{}".format(args.MLM, args.MRM, args.ITM))
    right_temp=0
    all_num_temp=0

    acc_file=open("{}/acc.txt".format(savePath),"w",encoding="utf-8")


    # t1 = timer()
    # args.num_train_epochs=100
    for epochId in range(int(args.start_epoch), int(args.num_train_epochs)):
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        for step, batch in enumerate(train_dataset):
            iterId = startIterID + step + (epochId * len(train_dataset))
            # batch = iter_dataloader.next()
            image_id=batch[-1]
            batch=batch[:-1]
            batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

            input_ids, input_mask, segment_ids, lm_label_ids, is_next,\
            pv_input_ids,pv_input_mask, pv_segment_ids,em_label_ids,\
            image_feat, image_loc, image_target, image_label, image_mask, \
            video_feat, video_target, video_label, video_mask,\
            audio_feat, audio_target, audio_label, audio_mask, label= (
                batch
            )

            if torch.sum(torch.isnan(image_feat))>0:
                continue

            loss, output_logits,_ = model(
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
                label
            )

            right=0
            all_num=0
            right_temp+=right
            all_num_temp+=all_num

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            if math.isnan(loss.item()):
                pdb.set_trace()

            tr_loss += loss.item()

            rank = 0

            if dist.is_available() and args.distributed:
                rank = dist.get_rank()
            else:
                rank = 0
                
            viz.linePlot(iterId, loss.item(), "loss_"+str(rank), "train")

            loss_tmp += loss.item()


            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = args.learning_rate * warmup_linear(
                        global_step / num_train_optimization_steps,
                        args.warmup_proportion,
                    )
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr_this_step

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if step % 20 == 0 and step != 0:
                masked_loss_t_tmp = masked_loss_t_tmp / 20.0
                masked_loss_v_tmp = masked_loss_v_tmp / 20.0
                next_sentence_loss_tmp = next_sentence_loss_tmp / 20.0
                loss_tmp = loss_tmp / 20.0

                end_t = timer()
                timeStamp = strftime("%a %d %b %y %X", gmtime())

                Ep = epochId + nb_tr_steps / float(len(train_dataset))
                printFormat = "[%s][Ep: %.2f][Iter: %d][Time: %5.2fs][Loss: %.5g][Loss_v: %.5g][Loss_t: %.5g][Loss_n: %.5g][LR: %.8g][epoch: %d][step: %d]"

                printInfo = [
                    timeStamp,
                    Ep,
                    nb_tr_steps,
                    end_t - start_t,
                    loss_tmp,
                    masked_loss_v_tmp,
                    masked_loss_t_tmp,
                    next_sentence_loss_tmp,
                    optimizer.get_lr()[0],
                    epochId,
                    step
                ]
                
                start_t = end_t
                print(printFormat % tuple(printInfo))

                # print(" right/all_num: {}/{}  ---   ACC:{}".format(right_temp,all_num_temp,right_temp/all_num_temp))

                masked_loss_v_tmp = 0
                masked_loss_t_tmp = 0
                next_sentence_loss_tmp = 0
                loss_tmp = 0

        acc=test(Test_Dataset,tokenizer,args,model,epochId)
        acc_file.write("{}\n".format(acc))
        if default_gpu:
            # Save a trained model
            logger.info("** ** * Saving fine - tuned model ** ** * ")
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Only save the model it-self
            output_model_file = os.path.join(
                savePath, "pytorch_model_" + str(epochId) + ".bin"
            )

            torch.save(model_to_save.state_dict(), output_model_file)


class TBlogger:
    def __init__(self, log_dir, exp_name):
        log_dir = log_dir + "/" + exp_name
        print("logging file at: " + log_dir)
        self.logger = SummaryWriter(log_dir=log_dir)

    def linePlot(self, step, val, split, key, xlabel="None"):
        self.logger.add_scalar(split + "/" + key, val, step)

if __name__ == "__main__":
    # import time
    # time.sleep(3600*3)

    main()
