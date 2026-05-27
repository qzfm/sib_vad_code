import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
from torch.utils.tensorboard import SummaryWriter
import datetime
import torch.optim as optim
from tqdm import tqdm
import time
import json
from sklearn.metrics import roc_auc_score

from model.unet import UNet
from utils.metrics import gaussian_smooth, normalize
from utils.config_utils import load_config, parse_args

def denormalize(image, mean=1, std=1):
    """
    反归一化图像
    :param image: 归一化后的图像 (Tensor)
    :param mean: 归一化时使用的均值
    :param std: 归一化时使用的标准差
    :return: 反归一化后的图像 (Tensor)
    """
    if isinstance(mean, list) and isinstance(std, list):
        mean = torch.tensor(mean, dtype=image.dtype, device=image.device).view(1, 3, 1, 1)
        std = torch.tensor(std, dtype=image.dtype, device=image.device).view(1, 3, 1, 1)
    return image * std + mean

def get_test_files(json_path):
    f = open(json_path, 'r')
    datas = json.load(f)
    print("测试文件包括", datas.keys())
    video_names = list(datas.keys())
    video_names.sort()
    return video_names

def eval(config):


    save_path = config['testing']['save_path']
    # save_path = os.path.join(save_path, config['exp']['experiment_name'])

    mse_list = []
    sim_t_list = []
    all_gt = []
    all_pred = []
    if not config['testing']['use_saved_result']:
    
        model = UNet(bottle_dim=config['dataset']["bottleneck_dim"], decoder_layers=config['dataset']["decoder_layers"]).cuda()
        ckpt_path = config['testing']['ckpt_path']
        model.load_state_dict(torch.load(ckpt_path)["model_state_dict"], strict=True)
        model = nn.DataParallel(model)
        model.eval()

        if config['dataset']['name'] == 'shtech':
            from dataset.shtech_dataset import SHTechDataset as Dataset
        elif config['dataset']['name'] == 'ped2':
            from dataset.ped2_dataset import Ped2Dataset as Dataset
        elif config['dataset']['name'] == 'avenue':
            from dataset.avenue_dataset import AvenueDataset as Dataset
        else:
            raise NotImplementedError

        test_dataset = Dataset(json_path=config['dataset']['testset_json_path'], 
                                     image_path=config['dataset']['image_path'], 
                                     text_path=config['dataset']['text_path'], 
                                     train=False, 
                                     )
        test_loader = DataLoader(test_dataset, 
                                 batch_size=config['testing']['batch_size'], 
                                 shuffle=False, 
                                 num_workers=config['testing']['num_workers'], 
                                 drop_last=False)
        with torch.no_grad():

            for data in tqdm(test_loader):

                x = data[0].cuda()
                texts = data[1].cuda()

                frames_input = x[:, :4, :, :, :]
                frame_gt = x[:, 4, :, :, :]

                x_hat, t_hat = model(frames_input, texts) 

                texts = texts.squeeze(1).squeeze(1) # [b, 768]
                mse = torch.mean((frame_gt - x_hat) ** 2, dim=[1, 2, 3])
                sim_t = 1 - F.cosine_similarity(texts, t_hat, dim=1)

                mse = mse.cpu().numpy() # [b]
                sim_t = sim_t.cpu().numpy()

                mse_list.extend(mse)
                sim_t_list.extend(sim_t)


            mse_list = np.array(mse_list)
            sim_t_list = np.array(sim_t_list)

            # 保存推理结果
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            np.save(os.path.join(save_path, 'mse.npy'), mse_list) 
            np.save(os.path.join(save_path, 'simt.npy'), sim_t_list)
    else:
        # 直接加载推理好的结果
        mse_list = np.load(os.path.join(save_path, 'mse.npy'))
        sim_t_list = np.load(os.path.join(save_path, 'simt.npy'))



    test_files = get_test_files(json_path=config['dataset']['testset_json_path'])
    



    start_idx = 0
    for test_file in test_files: 
        gt = np.load(f"{config['dataset']['gt_files_dir']}/{test_file}.npy") # 加载gt
        video_length = gt.shape[0] - 4 
        gt = gt[4:]
        pred = mse_list[start_idx : start_idx + video_length]
        pred_t = sim_t_list[start_idx : start_idx + video_length]
            
        start_idx += video_length

        pred = gaussian_smooth(pred, config['testing']['sigma'], config['testing']['pre_norm'])
        pred_t = gaussian_smooth(pred_t, config['testing']['sigma'], config['testing']['pre_norm'])

        w1 = config['dataset']['weight_rgb'] 
        w2 = config['dataset']['weight_text']
        anomaly_score = w1*pred + w2*pred_t

        all_gt.append(gt)
        all_pred.append(anomaly_score)
                    
    all_gt = np.concatenate(all_gt)
    all_pred = np.concatenate(all_pred)
    assert len(all_gt) == len(all_pred), "all_gt and all_pred must have the same length"

    auc = roc_auc_score(all_gt , all_pred)
    print(f'AUC: {auc}')


if __name__ == '__main__':

    args = parse_args()
    config = load_config(args.config)

    eval(config)

    print('done')

