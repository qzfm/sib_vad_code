import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from transformers import BertTokenizer, BertModel
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import numpy as np
import json

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def load_image(image_file, input_size=448):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    pixel_values = transform(image)
    pixel_values = pixel_values.unsqueeze(0)
    return pixel_values



class InternVLModel(nn.Module):

    def __init__(self, ):
        super().__init__()
        path = 'OpenGVLab/InternVL2_5-1B'

        print("加载Internvl模型")
        self.model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
        self.generation_config = dict(max_new_tokens=1024, do_sample=False) # 采用固定答案

        print("加载Internvl模型完成")

        # bert
        print("加载simsce bert模型")
        # self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # self.bert_model= BertModel.from_pretrained("bert-base-uncased").cuda().eval()
        self.bert_tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/sup-simcse-bert-base-uncased')
        self.bert_model= AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased").cuda().eval()
        print("加载simsce bert模型完成")


    def encode_text_results(self, text):
        with torch.no_grad():
            encoded_input = self.bert_tokenizer(text,padding=True, truncation=True,return_tensors="pt")

            for k, v in encoded_input.items():
                encoded_input[k] = v.cuda()
            # output = self.bert_model(**encoded_input) #改成下面的simsce模式
            output = self.bert_model(**encoded_input, output_hidden_states=True, return_dict=True)
        return output.pooler_output


    def forward(self, pixel_values, num_patches_list=None):
        '''
        pixel_values: [B, 3, 448, 448] float16
        num_patches_list: [B]
        '''
        if num_patches_list is None:
            # 用于标注每个图片被切分成多少patch，用于动态分辨率处理
            num_patches_list = [1 for _ in range(pixel_values.shape[0])]

        with torch.no_grad():
            pixel_values = pixel_values.to(torch.bfloat16)

            questions = ['<image>\nDescribe the scene in the video, what objects are present, and what is each person doing?'] * len(num_patches_list)
            responses = self.model.batch_chat(self.tokenizer, pixel_values,
                                        num_patches_list=num_patches_list,
                                        questions=questions,
                                        generation_config=self.generation_config)
            
            results = {}
            results['responses'] = responses
            results['text_embed'] = self.encode_text_results(responses)

        return results
    
    def forward_video(self, pixel_values, num_patches_list=None):
        '''
        pixel_values: [B, 3, 448, 448] float16
        num_patches_list: [B]
        '''
        if num_patches_list is None:
            # 用于标注每个图片被切分成多少patch，用于动态分辨率处理
            num_patches_list = [1 for _ in range(pixel_values.shape[0])]

        with torch.no_grad():

            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
            question = video_prefix + 'Describe the scene in the video segment, what objects are present, and what is each person doing?'
            # Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
            response, history = self.model.chat(self.tokenizer, pixel_values, question, self.generation_config,
                                        num_patches_list=num_patches_list, history=None, return_history=True)
            
            results = {}
            results['question'] = question
            results['responses'] = response
            results['text_embed'] = self.encode_text_results(response)

        return results
    
def test_single_image():
    model = InternVLModel().cuda()
    x = load_image('/home/lijuntong/workplace/internvl/186.jpg')
    x = x.cuda()

    results = model.forward(x)

    print(results['responses'])



def test_contiguous_frames():
    model = InternVLModel()
    video_frames=[]
    video_frames.append(load_image('/home/lijuntong/workplace/internvl/183.jpg')) # [1, 3, 448, 448]
    video_frames.append(load_image('/home/lijuntong/workplace/internvl/184.jpg'))
    video_frames.append(load_image('/home/lijuntong/workplace/internvl/185.jpg'))
    video_frames.append(load_image('/home/lijuntong/workplace/internvl/186.jpg'))

    pixel_values = torch.cat(video_frames, dim=0) # [4, 3, 448, 448]
    num_patches_list = [1 for _ in range(pixel_values.shape[0])]


    res = model.forward_video(pixel_values, num_patches_list)
    print(res['question'])
    print(res['responses'])
    print(res['text_embed'].shape)


def extract_text_feature(dir_prefix, file_dir_list, output_dir):
    model = InternVLModel().cuda().eval()

    for file_dir in file_dir_list:
        file_list = os.listdir(os.path.join(dir_prefix, file_dir))

        file_list = sorted(file_list)
        file_list = file_list[3:] # 跳过前3帧, 从第4帧开始提取

        if not os.path.exists(os.path.join(output_dir, file_dir)):
            os.makedirs(os.path.join(output_dir, file_dir))
        else: 
            print(f"video {file_dir} already extracted")
            continue

        for file_name in file_list:
            file_path = os.path.join(dir_prefix, file_dir, file_name)
            assert os.path.exists(file_path), f'{file_path} not exists'

            # 取出连续4帧的片段
            file_name, ext = os.path.splitext(file_name) # .jpg
            len_format = len(file_name) # 文件名长度，如001为3
            frame_name_list = [] 
            frame_name_list.append(file_name)
            for i in range(1, 4):
                frame_name_list.append(str(int(file_name)-i).zfill(len_format))
            frame_name_list.reverse() # 例如['000', '001', '002', '003']       

            # 读取图片
            video_frames=[]
            for frame_name in frame_name_list:
                frame_path = os.path.join(dir_prefix, file_dir, frame_name+ext)
                video_frames.append(load_image(frame_path))

            pixel_values = torch.cat(video_frames, dim=0) # [4, 3, 448, 448]
            num_patches_list = [1 for _ in range(pixel_values.shape[0])]

            with torch.no_grad():
                res = model.forward_video(pixel_values, num_patches_list)

            text_embed = res['text_embed'].unsqueeze(0).cpu().numpy()

            np.save(os.path.join(output_dir, file_dir, file_name.replace('.jpg', '.npy')), text_embed)
            print(f'{file_path} done')
    print('done')


if __name__ == '__main__':


    # 测试函数
    # test_single_image()
    test_contiguous_frames()


    mode = 'testing'
    json_path = f'/home/lijuntong/workplace/internvl/shtech_test.json'
    dir_prefix = f'/home/lijuntong/dataset/SHTech/shanghaitech/{mode}/frames'
    output_dir = f'/home/datasets/lijuntong_datasets/internvl_video_simsce/{mode}'
    

    json_file = open(json_path, 'r')
    json_file = json.load(json_file)
    video_names = list(json_file.keys())# 按视频名称顺序组织数据
    video_names = sorted(video_names) 
    file_dir_list = video_names
    print(file_dir_list)
    extract_text_feature(dir_prefix, file_dir_list, output_dir)
