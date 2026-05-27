import torch
import torch.nn.functional as F

def loss_sparsity(attn_weights):
    return torch.mean(torch.sum(-attn_weights * torch.log(attn_weights + 1e-12), dim=1))

def loss_similarity(pred, gt):
    '''
    pred: [b, c]
    gt: [b, c]
    '''
    cosine_sim = F.cosine_similarity(pred, gt, dim=1)
    loss = 1 - cosine_sim  # 余弦相似度越大，损失越小
    return torch.mean(loss)

def loss_diff(pred, gt, pre_gt):
    '''
    pred: [b, c, h, w]
    gt: [b, c, h, w]
    pre_gt: [b, c, h, w]
    '''

    diff_pre = (pred - pre_gt)**2  # |pred - I_t|
    diff_gt = (pred - gt)**2       # |pred - I_{t+1}|
    # diff_pre = torch.abs(pred - pre_gt)  # |pred - I_t|
    # diff_gt = torch.abs(pred - gt)       # |pred - I_{t+1}|
    
    loss = torch.clamp(diff_gt - diff_pre, min=0)
    
    return loss.mean()    

if __name__ == '__main__':

    attn_weights = torch.randn(32, 30, 1)

    print(loss_sparsity(attn_weights))
