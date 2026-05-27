import sys
sys.path.append('.')
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Block
from model.model_utils import get_2d_sincos_pos_embed
class MotionEncoder(nn.Module): 

    def __init__(self, input_dim=3, embed_dim=256, decoder_depth=2, num_patches=32 ** 2):
        super(MotionEncoder, self).__init__()
        self.patch_emb = PatchEmbed(img_size=448, patch_size=14, in_chans=input_dim, embed_dim=embed_dim)
        self.encoder_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=4, mlp_ratio=2, qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(decoder_depth)])
        
        self.motion_pos_embed = nn.Parameter(torch.zeros(1, 32 ** 2 + 1, embed_dim), requires_grad=False) # 运动特征位置编码
        motion_pos_embed = get_2d_sincos_pos_embed(embed_dim=self.motion_pos_embed.shape[-1], 
                                                    grid_size=int(num_patches**.5), 
                                                    cls_token=True,
                                                    text_token=False)
        
        self.motion_pos_embed.data.copy_(torch.from_numpy(motion_pos_embed).float().unsqueeze(0))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
    def forward(self, x):
        x = self.patch_emb(x) 
        
        # cls token
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1) # [B, 32**2+1, 256]
        x = x + self.motion_pos_embed
        
        for block in self.encoder_blocks:
            x = block(x)
        return x
    

if __name__ == '__main__':
    x = torch.randn(2, 3, 448, 448)
    model = MotionEncoder()
    y = model(x)
    print(y.shape)