#!/usr/bin/env bash
TRAIN_TYPE=ITP3VA_capture_v3_dyctr_dymask_per_sample
MODEL_TYPE=capture_subset_v2_MLM_MRM_MEM_MFM_MAM_CLR

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_cls.py \
 --from_pretrained /multi_modal/CAPTURE/examples/${TRAIN_TYPE}/save/${MODEL_TYPE}/pytorch_model_9.bin \
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
 --train_lmdb_file /data/lmdb_features/subset_v2_train_feature.lmdb \
 --test_lmdb_file /data/lmdb_features/subset_v2_test_feature.lmdb \
 --caption_path /data/product5m_v2/subset_v2_id_label.json \
 --label_list_file /data/product5m_v2/subset_v2_label_list.json \
 --save_name CLS