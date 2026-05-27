import torch
import numpy as np
from scipy.ndimage import gaussian_filter1d
from matplotlib import pyplot as plt

def normalize(data, ):
    return (data - data.min()) / (data.max() - data.min())

def plot_and_save(scores: np.ndarray, gt: np.ndarray, save_path: str = "plot.png"):
    """
    绘制折线图并保存到本地。

    参数：
    - scores: np.ndarray，表示得分数组。
    - gt: np.ndarray，表示ground truth数组。
    - save_path: str，保存图像的路径，默认值为"plot.png"。
    """
    if scores.shape != gt.shape:
        raise ValueError("The shapes of scores and gt must be the same.")

    # 创建一个x轴范围，假设每个点对应一个索引
    x = np.arange(len(scores))

    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(x, scores, label="Scores", color="blue", marker='o')
    plt.plot(x, gt, label="Ground Truth", color="orange", linestyle='--', marker='x')

    # 设置图表标题和标签
    plt.title("Scores vs Ground Truth", fontsize=16)
    plt.xlabel("Index", fontsize=14)
    plt.ylabel("Value", fontsize=14)

    # 添加图例
    plt.legend(fontsize=12)

    # 添加网格线
    plt.grid(alpha=0.5)

    # 保存图片到本地
    plt.savefig(save_path, format='png', dpi=300)
    plt.close()
    print(f"Plot saved to {save_path}")

def calculate_psnr(predicted, original, max_pixel_value):
    """
    计算预测图像和原始图像之间的 PSNR。
    
    参数:
        predicted (torch.Tensor): 预测图像，形状为 [b, c, h, w]。
        original (torch.Tensor): 原始图像，形状为 [b, c, h, w]。
        max_pixel_value (float): 图像像素的最大值（默认为 1.0。
    
    返回:
        torch.Tensor: PSNR 值，形状为 [b]，表示每个批次的 PSNR。
    """
    # 确保输入维度一致
    assert predicted.shape == original.shape, "Predicted and original images must have the same shape."
    
    mse = torch.mean((predicted - original) ** 2, dim=(1, 2, 3))  # 按 [c, h, w] 计算均方误差

    # 避免分母为 0 的情况
    mse = torch.clamp(mse, min=1e-12)

    # 计算 PSNR
    psnr = 10 * torch.log10((max_pixel_value ** 2) / mse)
    
    return psnr

def psnr_to_anomaly_score(psnr_values):
    """
    将 PSNR 值转换为异常分数 (0~1)，值越接近 1 表示越可能异常，越接近 0 表示正常。
    
    参数:
        psnr_values (np array): 各帧的 PSNR 值，形状为 [n]。
    
    返回:
        np array: 异常分数 (0~1)，形状为 [n]。
    """
    # 确定最小值和最大值
    min_psnr = np.min(psnr_values).item()
    max_psnr = np.max(psnr_values).item()
    
    # 归一化 PSNR 值
    normalized_psnr = (psnr_values - min_psnr) / (max_psnr - min_psnr + 1e-15)
    
    # 计算异常分数
    anomaly_score = 1 - normalized_psnr
    return anomaly_score

def calculate_l2(predicted, original):
    return torch.mean((predicted - original) ** 2, dim=(1, 2, 3))

def calculate_epe(pred, gt):
    """
    计算光流的 Endpoint Error (EPE) 指标。

    参数:
    pred (torch.Tensor): 预测的光流，形状为 [b, 2, h, w]
    gt (torch.Tensor): 真实的光流，形状为 [b, 2, h, w]

    返回:
    torch.Tensor: 每个样本的 EPE, 形状为 [b]
    """
    # 计算每个像素点的光流向量的欧几里得距离
    diff = pred - gt
    epe_per_pixel = torch.sqrt(diff[:, 0, :, :] ** 2 + diff[:, 1, :, :] ** 2)

    # 对每个样本的所有像素点的距离求均值
    epe_per_sample = epe_per_pixel.mean(dim=(1, 2))

    return epe_per_sample

def gaussian_smooth(data, sigma=1, pre_norm=False):
    """
    对一维数据进行高斯平滑。
    
    参数:
        data (list or numpy.ndarray): 输入数据。
        sigma (float): 高斯核的标准差，控制平滑程度。
        kernel_size (int): 高斯核的大小。
    
    返回:
        numpy.ndarray: 平滑后的数据。
    """
    if pre_norm:
        data = normalize(data)
    smoothed_data = gaussian_filter1d(data, sigma=sigma)
    
    return smoothed_data

def calculate_average_psnr(psnr_list, gt):
    # 分别计算 gt中0和1的PSNR
    # 找出gt中标记为0和1的索引
    psnr_0 = psnr_list[gt == 0]
    psnr_1 = psnr_list[gt == 1]

    # 计算它们的平均值
    avg_psnr_0 = np.mean(psnr_0) if len(psnr_0) > 0 else 0
    avg_psnr_1 = np.mean(psnr_1) if len(psnr_1) > 0 else 0

    return avg_psnr_0, avg_psnr_1

if __name__ == '__main__':
    predicted = torch.randn(4, 3, 256, 256)
    original = torch.randn(4, 3, 256, 256)
    # predicted = torch.zeros(4, 3, 256, 256)
    # original = torch.zeros(4, 3, 256, 256)
    psnr = calculate_psnr(predicted, original, max_pixel_value=1.0)
    print(psnr)
    psnr = psnr.cpu().numpy()
    anomaly_score = psnr_to_anomaly_score(psnr)
    print(anomaly_score)
