import scipy.io
import matplotlib.pyplot as plt
import os
import numpy as np


def build_gt_npy(mat_path, save_path):
    mat_data = scipy.io.loadmat(mat_path)
    vol_label = mat_data.get("volLabel")
    if vol_label is None:
        raise ValueError("volLabel not found in the .mat file")
    assert len(vol_label) == 1

    mask = vol_label[0]  # 读取当前帧的掩码

    gt = []
    for frame_idx in range(len(mask)):

        gt.append(0 if np.sum(mask[frame_idx]) == 0 else 1)
    gt = np.array(gt)

    np.save(save_path, gt)

if __name__ == "__main__":
    mat_dir = r'D:\workplace\dataset\Avenue\ground_truth_demo\ground_truth_demo\testing_label_mask'
    save_dir = r'D:\workplace\dataset\Avenue\Avenue19\Avenue19\gt'

    mat_list = os.listdir(mat_dir)

    for mat_name in mat_list:
        mat_path = os.path.join(mat_dir, mat_name)

        video_name = mat_name.split('_')[0]
        npy_name = f"{int(video_name):02d}.npy"
        
        save_path = os.path.join(save_dir, npy_name)
        build_gt_npy(mat_path, save_path)
