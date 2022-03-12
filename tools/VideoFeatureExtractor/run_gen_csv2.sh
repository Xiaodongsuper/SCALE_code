#!/usr/bin/env bash

DATA_FILE=/multi_modal/data/product5m_v2/product1m_product5m_train_id_label.json
IMAGE_PATH=/multi_modal/data/images
VIDEO_PATH=/multi_modal/data/videos
VIDEO_FEATURE_PATH=/multi_modal/data/video_feature
CSV_SAVE_PATH=/multi_modal/data/product5m_v2


python preprocess_generate_csv2.py \
--ids_file /multi_modal/data/product5m_v2/ava_video_ids.json \
--csv=input.csv \
--video_root_path ${VIDEO_PATH} \
--feature_root_path ${VIDEO_FEATURE_PATH} \
--csv_save_path ${CSV_SAVE_PATH}


CUDA_VISIBLE_DEVICES=6 python extract.py \
--csv=${CSV_SAVE_PATH}/input.csv \
--type=s3dg \
--batch_size=32 \
--num_decoding_thread=16


#python convert_video_feature_to_pickle.py \
#--feature_root_path ${VIDEO_FEATURE_PATH} \
#--pickle_root_path . \
#--pickle_name input.pickle





