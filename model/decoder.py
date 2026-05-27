import sys
sys.path.append('.')
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from timm.models.vision_transformer import Block
# from model.moe import MoEVitBlock as Block
from model.moe import MoE

from model.model_utils import get_2d_sincos_pos_embed

class Decoder(nn.Module):

    def __init__(self, embed_dim=1024, decoder_embed_dim=768, decoder_depth=2, num_patches=32 ** 2, bottle_dim=512):
        super().__init__()

        # 投影层
        self.rgb_project = nn.Sequential(
            nn.Linear(2*embed_dim, 4*decoder_embed_dim, bias=True),
            nn.GELU(),
            nn.Linear(4*decoder_embed_dim, decoder_embed_dim, bias=True),
        )

        # 固定位置编码
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding +1是cls +2是text token
        
        # vit解码块
        self.decoder_blocks = nn.ModuleList([
            Block(dim=decoder_embed_dim, num_heads=4, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm
                  )
            for i in range(decoder_depth)])
        
        # RGB解码
        self.decoder_pred = nn.Linear(decoder_embed_dim*2, 14 ** 2 * 3, bias=True)
        self.final_conv = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(3, 3, kernel_size=1, stride=1),
        )

        # 文本特征重构
        self.text_recon = nn.Sequential(
            nn.Linear(decoder_embed_dim, 4*decoder_embed_dim),
            nn.GELU(),
            nn.Linear(4*decoder_embed_dim, 768),
        )
        
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)


        # 初始化位置编码
        decoder_pos_embed = get_2d_sincos_pos_embed(embed_dim=self.decoder_pos_embed.shape[-1], 
                                                    grid_size=int(num_patches**.5), 
                                                    cls_token=True,
                                                    text_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        self.moes = nn.ModuleList([
            MoE(num_experts_per_tok=7, 
                    n_routed_experts=63, 
                    n_shared_experts=1,
                    hidden_size=decoder_embed_dim,
                    moe_intermediate_size=bottle_dim, # 
                    )
            for i in range(decoder_depth)])
        
    def forward(self, x, skip, text, motion):

        
        # 按通道拼接rgb和motion
        x = torch.concat([x, motion], dim=2) # x[b, 1025, dim*2]
        x = self.rgb_project(x) # [b, 1025, dim]
        
        # 按seq拼接text feature token和rgb
        x = torch.concat([text, x], dim=1) # 按seq连接 [b, 1025 + 1, dim]
        x = x + self.decoder_pos_embed # pos


        # for blk in self.decoder_blocks:
        #     x = blk(x)
        for i in range(len(self.decoder_blocks)):
            x = self.decoder_blocks[i](x)
            x = self.moes[i](x)

        x = self.decoder_norm(x)

        # 分别取出结果
        text_token = x[:, 0, :]
        cls_token = x[:, 1, :]
        x = x[:, 2:1026, :] # 移除cls # [b, 32*32 + 1, dim]
        skip = skip[:, 1:, :] # 也移除cls
        
        # 结合patch emb解码RGB
        x = torch.cat([x, skip], dim=2) # 按通道连接
        x = self.decoder_pred(x)
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=14, p2=14, h=32, w=32)
        x = self.final_conv(x)

        # 文本特征重构
        text = self.text_recon(text_token)

        return x, text

class Decoderv3(nn.Module):

    def __init__(self, embed_dim=1024, decoder_embed_dim=768, decoder_depth=2, num_patches=32 ** 2, bottle_dim=512):
        super().__init__()


        # 固定位置编码
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding +1是cls +2是text token
        
        # vit解码块
        self.decoder_blocks = nn.ModuleList([
            Block(dim=decoder_embed_dim, num_heads=4, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm
                  )
            for i in range(decoder_depth)])
        
        # RGB解码
        self.decoder_pred = nn.Linear(decoder_embed_dim*2, 14 ** 2 * 3, bias=True)
        self.final_conv = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(3, 3, kernel_size=1, stride=1),
        )

        # 文本特征重构
        self.text_recon = nn.Sequential(
            nn.Linear(decoder_embed_dim, 4*decoder_embed_dim),
            nn.GELU(),
            nn.Linear(4*decoder_embed_dim, 768),
        )
        
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)


        # 初始化位置编码
        decoder_pos_embed = get_2d_sincos_pos_embed(embed_dim=self.decoder_pos_embed.shape[-1], 
                                                    grid_size=int(num_patches**.5), 
                                                    cls_token=True,
                                                    text_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        self.moes = nn.ModuleList([
            MoE(num_experts_per_tok=7, 
                    n_routed_experts=63, 
                    n_shared_experts=1,
                    hidden_size=decoder_embed_dim,
                    moe_intermediate_size=bottle_dim, # 
                    )
            for i in range(decoder_depth)])
        
    def forward(self, x, skip, text):


        # 按seq拼接text feature token和rgb
        x = torch.concat([text, x], dim=1) # 按seq连接 [b, 1025 + 1, dim]
        x = x + self.decoder_pos_embed # pos


        # for blk in self.decoder_blocks:
        #     x = blk(x)
        for i in range(len(self.decoder_blocks)):
            x = self.decoder_blocks[i](x)
            x = self.moes[i](x)

        x = self.decoder_norm(x)

        # 分别取出结果
        text_token = x[:, 0, :]
        cls_token = x[:, 1, :]
        x = x[:, 2:1026, :] # 移除cls # [b, 32*32 + 1, dim]
        skip = skip[:, 1:, :] # 也移除cls
        
        # 结合patch emb解码RGB
        x = torch.cat([x, skip], dim=2) # 按通道连接
        x = self.decoder_pred(x)
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=14, p2=14, h=32, w=32)
        x = self.final_conv(x)

        # 文本特征重构
        text = self.text_recon(text_token)

        return x, text
        
if __name__ == '__main__':

    x = torch.randn(4, 1025, 256) # b, 32*32 1024
    skip = torch.randn(4, 1025, 256)
    text = torch.randn(4, 1, 256)
    motion = torch.randn(4, 1025, 256)
    model = Decoder(embed_dim=256, decoder_embed_dim=256, decoder_depth=2)

    out = model(x, skip, text, motion)   
    print(out[0].shape)
    print(out[1].shape)


    # print(sum(p.numel() for p in patch_expand.parameters())/1e6)

