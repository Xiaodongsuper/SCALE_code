#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=7 python pretrain_task.py \
 --from_pretrained /bert_model/bert_base_chinese \
 --bert_model bert-base-chinese \
 --config_file ../../config/bert_base_6layer_6conect_capture_itp3va.json\
 --predict_feature \
 --learning_rate 1e-4 \
 --video_feature_dir  video_path \
 --audio_file_dir audio_path \
 --train_batch_size 64 \
 --max_seq_length 36 \
 --video_len 12 \
 --pv_seq_len 64 \
 --audio_len 12 \
 --lmdb_file image_path \
 --caption_path  caption_path \
 --save_name capture_subset_v2_MLM_MRM_MEM_MFM_MAM_CLR \
 --MLM \
 --MRM \
 --MFM \
 --MEM \
 --MAM \
 --CLR