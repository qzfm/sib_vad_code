import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.nn.functional as F


from model.internvl_model import InternVLModel
from model.decoder import Decoder, Decoderv3
from model.motion_encoder import MotionEncoder
from model.attention import JointCrossAttn

class UNet(nn.Module):

    def __init__(self, decoder_layers=4, bottle_dim=128):
        super(UNet, self).__init__()

        self.internvl_model = InternVLModel()
        for param in self.internvl_model.parameters():
            param.requires_grad = False  

        # 统一投影到统一维度
        self.visual_project = nn.Linear(1024, 256)
        self.patch_emb_project = nn.Linear(1024, 256)
        self.temporal_project = nn.Linear(1024, 256)
        self.text_project = nn.Linear(768, 256)

        # 运动模块
        self.motion_encoder = MotionEncoder(input_dim=3, embed_dim=128, decoder_depth=4)
        self.frame_diff_project = nn.Linear(128*3, 256)

        self.joint_cross_attn = JointCrossAttn(dim=256, layers=4, heads=4, dropout=0.0, ff_mult=2)

        # 解码器
        self.decoder = Decoder(embed_dim=256, 
                               decoder_embed_dim=256, 
                               decoder_depth=decoder_layers, 
                               num_patches=32 ** 2, 
                               bottle_dim=bottle_dim)

    def forward(self, x, text_features):
        # x [b, 4, 3, 448, 448] 前4帧
        # text_features [b, 1, 1, 768] 前4帧的文本特征


        b, t, c, h, w = x.shape
        x_diff = (x[:, 1:] - x[:, :-1]).reshape(-1, c, h, w) # 计算帧差[b, t-1, c, h, w]
        x = x.reshape(b*t, c, h, w) # [b*t, c, h, w]

        # 提取patch编码和视觉编码
        results = self.internvl_model(x) 
        visual_feature = results['visual_feature'].to(torch.float).reshape(b, t, 1025, 1024) # [b, t, 1025, 1024]
        patch_embed = results['patch_embed'].to(torch.float).reshape(b, t, 1025, 1024) #  [b, t, 1025, 1024]

        # 运动编码
        x_diff = self.motion_encoder(x_diff) # [b*3, 32*32+1, 128]
        x_diff = x_diff.reshape(b, 3, 1+32*32, 128).permute(0, 2, 1, 3).reshape(b, 1+32*32, 3*128) # [b, 32*32+1, 3*128]
        x_diff = self.frame_diff_project(x_diff) # [b, 32*32+1, 256]

        # 对视觉编码进行proj
        visual_feature = self.visual_project(visual_feature) # [b, t, 1025, 256]
        visual_feature = visual_feature.permute(0, 2, 1, 3).reshape(b, 1025, 4*256)
        visual_feature = self.temporal_project(visual_feature) # [b, 1025, 256]
        patch_embed = patch_embed[:, -1, :, :] # 只取t-1帧做外观编码 [b, 1025, 1024]
        patch_embed = self.patch_emb_project(patch_embed) # [b, 1025, 256]



        # 文本特征proj
        text_features = text_features.squeeze(1).squeeze(1) # [b, 1, 1, 768] -> [b, 768]
        text_features = self.text_project(text_features) # [b, 256]
        text_features = text_features.unsqueeze(1) # [b, 1, 256]


        visual_feature = self.joint_cross_attn(visual_feature, text_features, x_diff)  # [b, 1025, 256]

        x, text = self.decoder(visual_feature, patch_embed, text_features, x_diff) # x [b, 3, 448, 448], text [b, 256]

        return x, text
        
if __name__ == '__main__':
    model = UNet()
    print(model)
    x = torch.randn(4, 4, 3, 448, 448)
    x_gt = torch.randn(4, 3, 448, 448)
    text = torch.randn(4, 1, 1, 768)
    out = model(x, text)
    print(out[0].shape)
    print(out[1].shape)
    # print(out[2].shape)

    print(sum(p.numel() for p in model.parameters())/1e6)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6)