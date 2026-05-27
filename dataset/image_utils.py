import torch
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import os

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
    pixel_values = pixel_values
    return pixel_values

def read_video_dir(video_dir_path, text_feature_dir=None):
    video_dir_list = os.listdir(video_dir_path)
    video_dir_list = sorted(video_dir_list)

    images = []
    for i in range(len(video_dir_list)):
        image_file = os.path.join(video_dir_path, video_dir_list[i])
        data = load_image(image_file)
        images.append(data)
    images = torch.stack(images)

    if text_feature_dir is not None:
        text_features = []
        for i in range(len(video_dir_list)):
            text_file = os.path.join(text_feature_dir, video_dir_list[i].replace('.jpg', '.npy'))
            text_feature = np.load(text_file)
            text_features.append(text_feature)
        text_features = torch.from_numpy(np.stack(text_features))
        print(text_features.shape)
        print(images.shape)
        return images, text_features

    return images


if __name__ == '__main__':

    image_file = '/home/lijuntong/workplace/internvl/170.jpg'
    data = load_image(image_file)
    print(data.shape)

    video_dir = '/home/lijuntong/dataset/SHTech/shanghaitech/testing/frames/01_0014'
    images = read_video_dir(video_dir)
    print(images.shape)