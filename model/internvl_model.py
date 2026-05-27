import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from transformers import BertTokenizer, BertModel
import time

class InternVLModel(nn.Module):

    def __init__(self, ):
        super().__init__()
        path = 'OpenGVLab/InternVL2_5-1B'

        print("加载Internvl模型")
        self.model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=True)
        self.generation_config = dict(max_new_tokens=1024, do_sample=True)
        self.model = self.model.vision_model
        # patchemb 
        self.patch_embbing = {}
        self.hook_hidden_states()
        print("加载Internvl模型完成")

    def hook_hidden_states(self,):
        # 定义一个钩子函数来捕获中间层的输出
        # ！指定device是因为避免dp训练时候device错乱
        def hook_fn_patchemb(module, input, output):
            self.patch_embbing[input[0].device] = output
        # 注册钩子
        hook_patchemb = self.model.embeddings.register_forward_hook(hook_fn_patchemb)
        return hook_patchemb

    def forward(self, pixel_values):
        '''
        pixel_values: [B, 3, 448, 448] float16
        num_patches_list: [B]
        '''
        results = {}
        with torch.no_grad():
            pixel_values = pixel_values.to(torch.bfloat16)

            # 只计算视觉编码器
            results['visual_feature'] = self.model(pixel_values).last_hidden_state
            results['patch_embed'] = self.patch_embbing[pixel_values.device]
        return results


if __name__ == '__main__':
    model = InternVLModel().cuda()
    for param in model.parameters():
        param.requires_grad = False  
    x = torch.randn(32, 3, 448, 448).to(torch.bfloat16).cuda()
    num_patches_list = [1 for _ in range(x.shape[0])]

    results = model(x, num_patches_list)

    print(results.keys())
    
    print(results['visual_feature'].shape)
    print(results['patch_embed'].shape)
    # print(results['responses'])
    # print(results['text_embed'].shape)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6
    # print(sum(p.numel() for p in model.parameters())/1e6)
    print(f"可训练参数量: {num_params}")