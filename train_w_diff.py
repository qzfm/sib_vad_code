import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
from torch.utils.tensorboard import SummaryWriter# Create an instance of the object 
import datetime
import torch.optim as optim
from tqdm import tqdm
import time

from model.unet import UNet
from utils.loss import loss_sparsity, loss_similarity, loss_diff
from utils.config_utils import load_config, parse_args

def get_time_experiment_name(output_dir="outputs", experiment_name="experiment"):
    
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_dir = os.path.join(output_dir, f"{experiment_name}_{time_str}")
    
    return log_dir

def set_seed(seed=None):

    if not seed:
        print("不设置随机种子")
        return

    # 设置 Python 的随机种子
    random.seed(seed)
    
    # 设置 NumPy 的随机种子
    np.random.seed(seed)
    
    # 设置 PyTorch 的随机种子
    torch.manual_seed(seed)
    
    # 如果使用的是 GPU，需要固定 CUDA 的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多卡，固定所有设备的种子
    

def train(config, writer):

    model = UNet(bottle_dim=config['dataset']["bottleneck_dim"], decoder_layers=config['dataset']["decoder_layers"])
    model = nn.DataParallel(model)
    model = model.cuda()
    model.train()
    model.module.internvl_model.eval()

    print("模型参数：", sum(p.numel() for p in model.parameters())/1e6)
    print("可训练参数量：", sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6)

    if config['dataset']['name'] == 'shtech':
        from dataset.shtech_dataset import SHTechDataset as Dataset
    elif config['dataset']['name'] == 'ped2':
        from dataset.ped2_dataset import Ped2Dataset as Dataset
    elif config['dataset']['name'] == 'avenue':
        from dataset.avenue_dataset import AvenueDataset as Dataset
    else:
        raise NotImplementedError

    train_dataset = Dataset(json_path=config['dataset']['trainset_json_path'], 
                                  image_path=config['dataset']['image_path'], 
                                  text_path=config['dataset']['text_path'], 
                                  train=True, 
                                  )
    
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=config['training']['batch_size'], 
                                  shuffle=True, 
                                  num_workers=config['training']['num_workers'])
    optimizer = optim.Adam(model.parameters(), 
                           lr=float(config['training']['lr']), 
                           betas=(0.9, 0.999), weight_decay=1e-5)

    if config['training']['load_pretrained']:
        load_pretrained = config['training']['load_pretrained']
        print("加载预训练模型：", load_pretrained)
        checkpoint = torch.load(load_pretrained)
        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            model.module.load_state_dict(checkpoint["model_state_dict"], strict=True)
        else:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    step=0
    for epoch in tqdm(range(config['training']['epochs'])):

        epoch_loss = 0
        
        for batch_idx, (images,texts) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

            images = images.cuda() # [b, 5, 3, 448, 448]
            texts = texts.cuda() # [b, 1, 1, 768]

            frames_input = images[:, :4, :, :, :]
            frame_gt = images[:, 4, :, :, :]

            
            frame_pred, texts_recon = model(frames_input , texts)
            

            texts = texts.squeeze(1).squeeze(1)
            mse_loss = F.mse_loss(frame_pred, frame_gt)
            sim_t_loss = loss_similarity(texts_recon, texts)
            diff_loss = loss_diff(frame_pred, frame_gt, frames_input[:, -1, :, :, :])

            loss = mse_loss*config['training']['lambda_rgb'] + sim_t_loss*config['training']['lambda_text'] \
                + diff_loss*config['training']['lambda_diff']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            step += 1
            writer.add_scalar('Loss/train_batch_rgb', mse_loss.item(), step)
            writer.add_scalar('Loss/train_batch_text', sim_t_loss.item(), step)
            writer.add_scalar('Loss/train_batch_motion', diff_loss.item(), step)
            writer.add_scalar('lr/train_batch', optimizer.param_groups[0]['lr'], step)


        writer.add_scalar('Loss/train', epoch_loss / len(train_dataloader), epoch)
        print(f"Epoch {epoch}: Loss = {epoch_loss / len(train_dataloader)}")


        if (epoch+1) % config['training']['save_interval'] == 0:
            model_state = model.module.state_dict() if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else model.state_dict()
            checkpoint = {
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
            }
            
            torch.save(checkpoint, os.path.join(config['exp']['ckpt_path'], 
                                                f'checkpoint_{epoch}.pth'))



if __name__ == '__main__':

    args = parse_args()
    config = load_config(args.config)
    print(config)


    print(config['training']['seed'])
    set_seed(config['training']['seed'])


    log_dir = get_time_experiment_name(output_dir=config['exp']['output_dir'], 
                                       experiment_name=config['exp']['experiment_name'])
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs saved in: {log_dir}")


    ckpt_path = os.path.join(log_dir, 'ckpt')
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    config['exp']['ckpt_path'] = ckpt_path

    train(config, writer)



