#!/usr/bin/env bash
TRAIN_TYPE=ITP3VA_capture_v3_dyctr_dymask_per_sample
MODEL_TYPE=CLS
QUERY_FEATURE=subset_v2_test_feature
GALLERY_FEATURE=subset_v2_test_feature
QUERY_FEATURE_DIR=examples/${TRAIN_TYPE}/eval/feature_data/${MODEL_TYPE}/${QUERY_FEATURE}/return_hidden
GALLERY_FEATURE_DIR=examples/${TRAIN_TYPE}/eval/feature_data/${MODEL_TYPE}/${GALLERY_FEATURE}/return_hidden
RETRIEVAL_RESULTS_DIR=examples/${TRAIN_TYPE}/eval/retrieval_id_list/${MODEL_TYPE}/${QUERY_FEATURE}/return_hidden


GALLERY_FILE=/multi_modal/data/train.txt
QUERY_FILE=/multi_modal/data/query.txt
IMAGE_PATH=/multi_modal/data/images
VIDEO_PATH=/multi_modal/data/videos


# remenber to change the dir and filename
# gallery
CUDA_VISIBLE_DEVICES=7 python extract_features_cls.py \
 --bert_model bert-base-chinese \
 --from_pretrained ../save/${MODEL_TYPE}/pytorch_model_9.bin \
 --config_file ../../../config/bert_base_6layer_6conect_capture_itp3va.json \
 --predict_feature \
 --video_feature_dir  /multi_modal/data/video_feature \
 --split test \
 --train_batch_size 32 \
 --max_seq_length 36 \
 --zero_shot \
 --video_len 12 \
 --pv_seq_len 64 \
 --lmdb_file /multi_modal/data/lmdb_features/${GALLERY_FEATURE}.lmdb \
 --caption_path /multi_modal/data/product5m_v2/subset_v2_id_label.json \
 --feature_dir ./feature_data/${MODEL_TYPE}/${GALLERY_FEATURE}/return_hidden \
 --return_hidden


cd ../../..


python retrieval_unit_id_list_v2.py \
  --query_feature_path ${QUERY_FEATURE_DIR} \
  --gallery_feature_path ${GALLERY_FEATURE_DIR} \
  --retrieval_results_path ${RETRIEVAL_RESULTS_DIR} \
  --max_topk 10 \
  --dense


GT_file=/multi_modal/data/product5m_v2/subset_v2_id_label.json
OUTPUT_METRIC_DIR=examples/${TRAIN_TYPE}/eval/retrieval_metric/${MODEL_TYPE}/${QUERY_FEATURE}/return_hidden
python evaluate_unit_v2.py \
  --retrieval_result_dir ${RETRIEVAL_RESULTS_DIR} \
  --GT_file ${GT_file} \
  --output_metric_dir ${OUTPUT_METRIC_DIR} \
  --dense

#
#python eval_spearman.py \
#  --query_feature_path ${QUERY_FEATURE_DIR} \
#  --gallery_feature_path ${GALLERY_FEATURE_DIR} \
#  --retrieval_results_path ${RETRIEVAL_RESULTS_DIR}


## cross-modal
#python retrieval_unit_id_list_cross_modal.py \
#  --query_feature_path ${QUERY_FEATURE_DIR} \
#  --gallery_feature_path ${GALLERY_FEATURE_DIR} \
#  --retrieval_results_path ${RETRIEVAL_RESULTS_DIR} \
#  --max_topk 10 \
#  --cross_modal vt
#
#
#GT_file=/multi_modal/data/product5m_v2/product1m_product5m_id_label.json
#OUTPUT_METRIC_DIR=examples/${TRAIN_TYPE}/eval/retrieval_metric/${MODEL_TYPE}/${QUERY_FEATURE}/return_hidden
#python evaluate_unit_cross_modal.py \
#  --retrieval_result_dir ${RETRIEVAL_RESULTS_DIR} \
#  --GT_file ${GT_file} \
#  --output_metric_dir ${OUTPUT_METRIC_DIR} \
#  --cross_modal vt


#RETRIEVAL_IMAGES_DIR=examples/${TRAIN_TYPE}/eval/retrieval_images/${MODEL_TYPE}/${QUERY_FEATURE}
#python retrieval_unit_images.py \
#  --retrieval_ids_path ${RETRIEVAL_RESULTS_DIR} \
#  --retrieval_images_path ${RETRIEVAL_IMAGES_DIR} \
#  --query_image_prefix  /multi_modal/data/images \
#  --gallery_image_prefix /multi_modal/data/images \
#  --t \
#  --v














