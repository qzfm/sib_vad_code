import sys
sys.path.append('.')

from torchvision import transforms
import torch

import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
from PIL import Image
import json
import random
import time

from dataset.image_utils import load_image

class Ped2Dataset(Dataset):
    def __init__(self, json_path, image_path, text_path, train=True, ext='.tif'):
        super().__init__()
        
        
        self.json_file = open(json_path, 'r')
        self.json_file = json.load(self.json_file)
        self.train = train
        self.ext = ext
        self.image_size = 448

        self.video_names = list(self.json_file.keys())# 按视频名称顺序组织数据
        self.video_names = sorted(self.video_names)

   
        self.total_num_frames = 0 # 所有视频的总帧数
        self.num_frames_for_each_video = [] # 每个视频的帧数
        for video_name in self.video_names:
            self.num_frames_for_each_video.append(len(self.json_file[video_name]) - 4) # 视频长度 - 4 丢弃前4帧 
            self.total_num_frames += len(self.json_file[video_name]) - 4 

        # 求累计和，用于定为某个帧属于哪个图片 例如[1, 3, 2, 4] -> [0, 1, 4, 6, 10] 
        self.num_frames_for_accumulate = [0]
        for i in range(len(self.num_frames_for_each_video)):
            self.num_frames_for_accumulate.append(self.num_frames_for_accumulate[i] 
                                                  + self.num_frames_for_each_video[i])

        # 设置图片路径
        self.image_path = image_path
        self.image_path = os.path.join(self.image_path, 'Train' if self.train else 'Test')

        # 设置文本路径
        self.text_path = text_path
        self.text_path = os.path.join(self.text_path, 'Train' if self.train else 'Test')


    def find_video_and_frame(self, index):
        # 根据全局索引找到对应的视频和帧索引
        for i in range(1, len(self.num_frames_for_accumulate)):
            if index < self.num_frames_for_accumulate[i]:
                return i-1, index - self.num_frames_for_accumulate[i - 1] + 4


    def __len__(self):
        return self.total_num_frames
    

    def __getitem__(self, index):

        # 根据全局索引找到对应的视频索引和帧索引
        video_index, frame_index = self.find_video_and_frame(index)
        
        # 获取视频、帧相关信息
        video_name = self.video_names[video_index] # 01_0025
        video_lenght = len(self.json_file[video_name]) # 视频长度
        start_frame = frame_index 
        frame_name = self.json_file[video_name][start_frame] # 001

        # 取出连续5帧的片段
        len_format = len(frame_name) # 文件名长度，如001为3
        frame_name_list = [] 
        frame_name_list.append(frame_name)
        for i in range(1, 5):
            frame_name_list.append(str(int(frame_name)-i).zfill(len_format))
        frame_name_list.reverse() # 例如['000', '001', '002', '003', '004']       

        images = []
        for frame_name in frame_name_list:

            frame_path = os.path.join(self.image_path, video_name, frame_name + self.ext)
            image = load_image(frame_path, self.image_size)
            images.append(image)

        # 应该改成t-1 的text feature
        text_path = os.path.join(self.text_path, video_name, frame_name_list[-2] + '.npy')
        text_feature = np.load(text_path)

        images = torch.stack(images)

        return images, text_feature


if __name__ == '__main__':
    dataset = Ped2Dataset(json_path='/home/lijuntong/workplace/internvl_vad/data/ped2_test.json', 
                          image_path='/home/lijuntong/dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2', 
                          text_path='/home/lijuntong/dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2/internvl_video_simsce', 
                          train=False)
    
    print(len(dataset))

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for i, data in enumerate(dataloader):
        print(data[0].shape)
        print(data[1].shape)
        break

        


    