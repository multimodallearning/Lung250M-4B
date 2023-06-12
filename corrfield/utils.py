import numpy as np
import torch
import torch.nn.functional as F

def gpu_usage():
    print('gpu usage (current/max): {:.2f} / {:.2f} GB'.format(torch.cuda.memory_allocated()*1e-9, torch.cuda.max_memory_allocated()*1e-9))

def kpts_pt(kpts_world, shape, align_corners=None):
    device = kpts_world.device
    D, H, W = shape
   
    kpts_pt_ = (kpts_world.flip(-1) / (torch.tensor([W, H, D], device=device) - 1)) * 2 - 1
    if not align_corners:
        kpts_pt_ *= (torch.tensor([W, H, D], device=device) - 1)/torch.tensor([W, H, D], device=device)
    
    return kpts_pt_

def kpts_world(kpts_pt, shape, align_corners=None):
    device = kpts_pt.device
    D, H, W = shape
    
    if not align_corners:
        kpts_pt /= (torch.tensor([W, H, D], device=device) - 1)/torch.tensor([W, H, D], device=device)
    kpts_world_ = (((kpts_pt + 1) / 2) * (torch.tensor([W, H, D], device=device) - 1)).flip(-1) 
    
    return kpts_world_

def flow_pt(flow_world, shape, align_corners=None):
    device = flow_world.device
    D, H, W = shape
    
    flow_pt_ = (flow_world.flip(-1) / (torch.tensor([W, H, D], device=device) - 1)) * 2
    if not align_corners:
        flow_pt_ *= (torch.tensor([W, H, D], device=device) - 1)/torch.tensor([W, H, D], device=device)
    
    return flow_pt_

def flow_world(flow_pt, shape, align_corners=None):
    device = flow_pt.device
    D, H, W = shape
    
    if not align_corners:
        flow_pt /= (torch.tensor([W, H, D], device=device) - 1)/torch.tensor([W, H, D], device=device)
    flow_world_ = ((flow_pt / 2) * (torch.tensor([W, H, D], device=device) - 1)).flip(-1)
    
    return flow_world_

def get_disp(disp_step, disp_radius, shape, device):
    D, H, W = shape
    
    disp = torch.stack(torch.meshgrid(torch.arange(- disp_step * disp_radius, disp_step * disp_radius + 1, disp_step, device=device),
                                      torch.arange(- disp_step * disp_radius, disp_step * disp_radius + 1, disp_step, device=device),
                                      torch.arange(- disp_step * disp_radius, disp_step * disp_radius + 1, disp_step, device=device)), dim=3).view(1, -1, 3)
    
    disp = flow_pt(disp, (D, H, W), align_corners=True)
    return disp

def get_patch(patch_step, patch_radius, shape, device):
    D, H, W = shape
    
    patch = torch.stack(torch.meshgrid(torch.arange(0, 2 * patch_radius + 1, patch_step, device=device),
                                       torch.arange(0, 2 * patch_radius + 1, patch_step, device=device),
                                       torch.arange(0, 2 * patch_radius + 1, patch_step, device=device)), dim=3).view(1, -1, 3) - patch_radius
    patch = flow_pt(patch, (D, H, W), align_corners=True)
    return patch

def pdist(x, p=2):
    if p==1:
        dist = torch.abs(x.unsqueeze(2) - x.unsqueeze(1)).sum(dim=3)
    elif p==2:
        xx = (x**2).sum(dim=2).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x, x.permute(0, 2, 1))
        dist[:, torch.arange(dist.shape[1]), torch.arange(dist.shape[2])] = 0
    return dist

def pdist2(x, y, p=2):
    if p==1:
        dist = torch.abs(x.unsqueeze(2) - y.unsqueeze(1)).sum(dim=3)
    elif p==2:
        xx = (x**2).sum(dim=2).unsqueeze(2)
        yy = (y**2).sum(dim=2).unsqueeze(1)
        dist = xx + yy - 2.0 * torch.bmm(x, y.permute(0, 2, 1))
    return dist

def find_rigid_3d(x, y):
    device = x.device
    x_mean = x[:, :3].mean(0)
    y_mean = y[:, :3].mean(0)
    u, s, v = torch.svd(torch.matmul((x[:, :3]-x_mean).t(), (y[:, :3]-y_mean)))
    m = torch.eye(v.shape[0], v.shape[0], device=device)
    m[-1,-1] = torch.det(torch.matmul(v, u.t()))
    rotation = torch.matmul(torch.matmul(v, m), u.t())
    translation = y_mean - torch.matmul(rotation, x_mean)
    T = torch.eye(4, device=device)
    T[:3,:3] = rotation
    T[:3, 3] = translation
    return T

def compute_rigid_transform(kpts_fixed, kpts_moving, iter=5):
    import time
    device = kpts_fixed.device
    kpts_fixed = torch.cat((kpts_fixed, torch.ones(1, kpts_fixed.shape[1], 1, device=device)), 2)
    kpts_moving = torch.cat((kpts_moving, torch.ones(1, kpts_moving.shape[1], 1, device=device)), 2)
    idx = torch.arange(kpts_fixed.shape[1]).to(kpts_fixed.device)[torch.randperm(kpts_fixed.shape[1])[:kpts_fixed.shape[1]//2]]
    for i in range(iter):
        x = find_rigid_3d(kpts_fixed[0, idx, :], kpts_moving[0, idx, :]).t()
        residual = torch.sqrt(torch.sum(torch.pow(kpts_moving[0] - torch.mm(kpts_fixed[0], x), 2), 1))
        _, idx = torch.topk(residual, kpts_fixed.shape[1]//2, largest=False)
    return x.t().unsqueeze(0)