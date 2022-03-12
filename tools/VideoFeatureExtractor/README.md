
This repo is forked from [video_feature_extractor](https://github.com/antoine77340/video_feature_extractor) to extract S3D feature ([S3D_HowTo100M](https://github.com/antoine77340/S3D_HowTo100M)) pretraied on HowTo100M. Read more details in [video_feature_extractor](https://github.com/antoine77340/video_feature_extractor).

This repo is also as a preprocess in video-language pretrain model [UniVL](https://github.com/microsoft/UniVL).

## Requirements

IMPORTANT: The video decode process depends on the FFmpeg (https://www.ffmpeg.org/download.html), install it first and run `ffmpeg` and `ffprobe` command derectly to make them work well.

- Python 3
- PyTorch (>= 1.0)
- python-ffmpeg (https://github.com/kkroening/ffmpeg-python)

## Downloading pretrained models
This will download the pretrained S3D model:

```sh
mkdir -p model
cd model
wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_howto100m.pth
cd ..
```

## Extract S3D Feature

First of all you need to generate a csv containing the list of videos you
want to process. For instance, if you have absolute_path_video1.mp4 and absolute_path_video2.webm to process,
you will need to generate a csv of this form:

```sh
video_path,feature_path
absolute_path_video1.mp4,absolute_path_of_video1_features.npy
absolute_path_video2.webm,absolute_path_of_video2_features.npy
```

Refer to below command to generate such a csv file:
```sh
python preprocess_generate_csv.py --csv=input.csv --video_root_path [VIDEO_PATH] --feature_root_path [FEATURE_PATH] --csv_save_path .
```
*Note: the video file should have a suffix, modify the code for your customization*


And then just simply run:

```sh
python extract.py --csv=./input.csv --type=s3dg --batch_size=64 --num_decoding_thread=4
```
This command will extract s3d-g video feature in a form of a numpy array.

If you want to pickle all generated npy files:
```sh
python convert_video_feature_to_pickle.py --feature_root_path [FEATURE_PATH] --pickle_root_path . --pickle_name input.pickle
```
*The key is set as the video name in the pickle file*

## Acknowledgements
The code re-used code from https://github.com/kenshohara/3D-ResNets-PyTorch
for 3D CNN. And modified from https://github.com/antoine77340/video_feature_extractor.
