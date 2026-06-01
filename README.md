# Video Anomaly Detection with Semantics-Aware Information Bottleneck [ICME 2026]

## Dataset
### Download Dataset
[ShanghaiTech](https://svip-lab.github.io/dataset/campus_dataset.html)  
[Ped2](http://www.svcl.ucsd.edu/projects/anomaly/dataset.html)  
[Avenue](https://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html)

### Get Frames
Next, you need to cut the video frame by frame into pictures and save them. For the specific naming method, please refer to the **/data** folder.

For example, the file structure is as follows:
```
my_dataset/
├── training/
│   ├── 01_0014/
│   │   ├── 0000.jpg
│   │   ├── 0001.jpg
│   │   ├── ...
│   │   └── 00NN.jpg
│   ├── 02_0005/
│   │   └── ...
│   └── .../
├── testing/
│   ├── 01_0038/
│   │   ├── 0000.jpg
│   │   ├── ...
│   │   └── 00NN.jpg
│   ├── 02_0012/
│   │   └── ...
│   └── .../
```

### Build GT Annotation
The format of the ground truth (GT) annotations is as follows: Each video corresponds to one .npy file with shape (num_frames, ), where 0 indicates normal frames and 1 indicates anomalous frames.

For the Ped2 and Avenue datasets, we need to manually build frame-level GT annotations:

* For the Ped2 dataset, we simply check whether the original mask images contain white foreground objects. If present, the corresponding frames are labeled as anomalous.

* For the Avenue dataset, we determine anomaly by checking the volLabel count in the original .mat files. Frames are labeled as anomalous if the count is non-zero. Reference implementation can be found at: ```/preprocess/avenue_build_gt.py```.


### Get Semantic Features
**/preprocess/video_caption_extract_demo.py** provides a demo script for pre-extracting video captions and extracting semantic features. After completing the aforementioned video frame splitting step, you need to pre-extract semantic features. Note that this script processes data in a single-threaded manner which is relatively slow. We recommend dividing the target videos into batches and processing them through multi-threaded/multi-process parallelism. 

We now provide the pre-processed frame-centric semantic features and ground-truth (GT) files at: [\[URL\]](https://drive.google.com/drive/folders/1_3XKZMFw5zJLoQT5Tsv5hO_nP7MQpPA6?usp=sharing).

##  Frame-centric VAD
### 1. config
You can modify the configuration file in **/config/xxx.yaml** to set training parameters, **data paths**, etc.
### 2. train
```
CUDA_VISIBLE_DEVICES=6,7 python train_w_diff.py --config config/shtech.yaml
```
The training results will be saved in **output_dir/experiment_name** which can be modified in config file.
### 3. eval
You need to set the ckpt_path field in the configuration file /config/xxx.yaml to the path of the trained checkpoint file.
```
CUDA_VISIBLE_DEVICES=6,7 python eval.py --config config/shtech.yaml
```
The frame error and similarity error for each inference computation will be saved in the **/save_path**  to avoid excessively long inference times from repeated computations. If you need to recalculate metrics using pre-computed inference results, simply set use_saved_result to True without rerunning the inference process.


**Thank you for taking notice of this work!** The code will be released here soon. Thank you for your patience!

```
@inproceedings{li2025video,
    title={Video Anomaly Detection with Semantics-Aware Information Bottleneck}, 
    author={Li, Juntong and Dang, Lingwei and Xiao, Qingxin and Shang, Shishuo and Cheng, Jiajia and Wu, Haomin and Hao, Yun and Wu, Qingyao},
    booktitle={ICME}, 
    year={2026}
}
```