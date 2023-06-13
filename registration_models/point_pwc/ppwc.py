"""
PointPWC-Net model and losses
Author: Wenxuan Wu
Date: May 2020
"""

import torch.nn as nn
import torch
from pointconv_util import PointConvD, PointWarping, UpsampleFlow, PointConvFlow
from pointconv_util import SceneFlowEstimatorPointConv
from pointconv_util import index_points_gather as index_points, index_points_group, Conv1d, square_distance


scale = 1.0


class PointConvSceneFlowPWC8192(nn.Module):
    def __init__(self, cfg):
        super(PointConvSceneFlowPWC8192, self).__init__()

        flow_nei = 32
        feat_nei = 16
        self.scale = scale
        # l0: 8192
        self.level0 = Conv1d(3, 32)
        self.level0_1 = Conv1d(32, 32)
        self.cost0 = PointConvFlow(flow_nei, 32 + 32 + 32 + 32 + 3, [32, 32])
        self.flow0 = SceneFlowEstimatorPointConv(32 + 64, 32, use_bn=cfg.MODEL.PPWC_SFPC_BN)
        self.level0_2 = Conv1d(32, 64)

        # l1: 2048
        self.level1 = PointConvD(2048, feat_nei, 64 + 3, 64)
        self.cost1 = PointConvFlow(flow_nei, 64 + 32 + 64 + 32 + 3, [64, 64])
        self.flow1 = SceneFlowEstimatorPointConv(64 + 64, 64, use_bn=cfg.MODEL.PPWC_SFPC_BN)
        self.level1_0 = Conv1d(64, 64)
        self.level1_1 = Conv1d(64, 128)

        # l2: 512
        self.level2 = PointConvD(512, feat_nei, 128 + 3, 128)
        self.cost2 = PointConvFlow(flow_nei, 128 + 64 + 128 + 64 + 3, [128, 128])
        self.flow2 = SceneFlowEstimatorPointConv(128 + 64, 128, use_bn=cfg.MODEL.PPWC_SFPC_BN)
        self.level2_0 = Conv1d(128, 128)
        self.level2_1 = Conv1d(128, 256)

        # l3: 256
        self.level3 = PointConvD(256, feat_nei, 256 + 3, 256)
        self.cost3 = PointConvFlow(flow_nei, 256 + 64 + 256 + 64 + 3, [256, 256])
        self.flow3 = SceneFlowEstimatorPointConv(256, 256, flow_ch=0, use_bn=cfg.MODEL.PPWC_SFPC_BN)
        self.level3_0 = Conv1d(256, 256)
        self.level3_1 = Conv1d(256, 512)

        # l4: 64
        self.level4 = PointConvD(64, feat_nei, 512 + 3, 256)

        # deconv
        self.deconv4_3 = Conv1d(256, 64)
        self.deconv3_2 = Conv1d(256, 64)
        self.deconv2_1 = Conv1d(128, 32)
        self.deconv1_0 = Conv1d(64, 32)

        # warping
        self.warping = PointWarping()

        # upsample
        self.upsample = UpsampleFlow()

    def forward(self, xyz1, xyz2, color1, color2):
        # xyz1, xyz2: B, N, 3
        # color1, color2: B, N, 3

        # l0
        pc1_l0 = xyz1.permute(0, 2, 1)
        pc2_l0 = xyz2.permute(0, 2, 1)
        color1 = color1.permute(0, 2, 1)  # B 3 N
        color2 = color2.permute(0, 2, 1)  # B 3 N
        feat1_l0 = self.level0(color1)
        feat1_l0 = self.level0_1(feat1_l0)
        feat1_l0_1 = self.level0_2(feat1_l0)

        feat2_l0 = self.level0(color2)
        feat2_l0 = self.level0_1(feat2_l0)
        feat2_l0_1 = self.level0_2(feat2_l0)

        # l1
        pc1_l1, feat1_l1, fps_pc1_l1 = self.level1(pc1_l0, feat1_l0_1)
        feat1_l1_2 = self.level1_0(feat1_l1)
        feat1_l1_2 = self.level1_1(feat1_l1_2)

        pc2_l1, feat2_l1, fps_pc2_l1 = self.level1(pc2_l0, feat2_l0_1)
        feat2_l1_2 = self.level1_0(feat2_l1)
        feat2_l1_2 = self.level1_1(feat2_l1_2)

        # l2
        pc1_l2, feat1_l2, fps_pc1_l2 = self.level2(pc1_l1, feat1_l1_2)
        feat1_l2_3 = self.level2_0(feat1_l2)
        feat1_l2_3 = self.level2_1(feat1_l2_3)

        pc2_l2, feat2_l2, fps_pc2_l2 = self.level2(pc2_l1, feat2_l1_2)
        feat2_l2_3 = self.level2_0(feat2_l2)
        feat2_l2_3 = self.level2_1(feat2_l2_3)

        # l3
        pc1_l3, feat1_l3, fps_pc1_l3 = self.level3(pc1_l2, feat1_l2_3)
        feat1_l3_4 = self.level3_0(feat1_l3)
        feat1_l3_4 = self.level3_1(feat1_l3_4)

        pc2_l3, feat2_l3, fps_pc2_l3 = self.level3(pc2_l2, feat2_l2_3)
        feat2_l3_4 = self.level3_0(feat2_l3)
        feat2_l3_4 = self.level3_1(feat2_l3_4)

        # l4
        pc1_l4, feat1_l4, _ = self.level4(pc1_l3, feat1_l3_4)
        feat1_l4_3 = self.upsample(pc1_l3, pc1_l4, feat1_l4)
        feat1_l4_3 = self.deconv4_3(feat1_l4_3)

        pc2_l4, feat2_l4, _ = self.level4(pc2_l3, feat2_l3_4)
        feat2_l4_3 = self.upsample(pc2_l3, pc2_l4, feat2_l4)
        feat2_l4_3 = self.deconv4_3(feat2_l4_3)

        # l3
        c_feat1_l3 = torch.cat([feat1_l3, feat1_l4_3], dim=1)
        c_feat2_l3 = torch.cat([feat2_l3, feat2_l4_3], dim=1)
        cost3 = self.cost3(pc1_l3, pc2_l3, c_feat1_l3, c_feat2_l3)
        feat3, flow3 = self.flow3(pc1_l3, feat1_l3, cost3)

        feat1_l3_2 = self.upsample(pc1_l2, pc1_l3, feat1_l3)
        feat1_l3_2 = self.deconv3_2(feat1_l3_2)

        feat2_l3_2 = self.upsample(pc2_l2, pc2_l3, feat2_l3)
        feat2_l3_2 = self.deconv3_2(feat2_l3_2)

        c_feat1_l2 = torch.cat([feat1_l2, feat1_l3_2], dim=1)
        c_feat2_l2 = torch.cat([feat2_l2, feat2_l3_2], dim=1)

        feat1_l2_1 = self.upsample(pc1_l1, pc1_l2, feat1_l2)
        feat1_l2_1 = self.deconv2_1(feat1_l2_1)

        feat2_l2_1 = self.upsample(pc2_l1, pc2_l2, feat2_l2)
        feat2_l2_1 = self.deconv2_1(feat2_l2_1)

        c_feat1_l1 = torch.cat([feat1_l1, feat1_l2_1], dim=1)
        c_feat2_l1 = torch.cat([feat2_l1, feat2_l2_1], dim=1)

        feat1_l1_0 = self.upsample(pc1_l0, pc1_l1, feat1_l1)
        feat1_l1_0 = self.deconv1_0(feat1_l1_0)

        feat2_l1_0 = self.upsample(pc2_l0, pc2_l1, feat2_l1)
        feat2_l1_0 = self.deconv1_0(feat2_l1_0)

        c_feat1_l0 = torch.cat([feat1_l0, feat1_l1_0], dim=1)
        c_feat2_l0 = torch.cat([feat2_l0, feat2_l1_0], dim=1)

        # l2
        up_flow2 = self.upsample(pc1_l2, pc1_l3, self.scale * flow3)
        pc2_l2_warp = self.warping(pc1_l2, pc2_l2, up_flow2)
        cost2 = self.cost2(pc1_l2, pc2_l2_warp, c_feat1_l2, c_feat2_l2)

        feat3_up = self.upsample(pc1_l2, pc1_l3, feat3)
        new_feat1_l2 = torch.cat([feat1_l2, feat3_up], dim=1)
        feat2, flow2 = self.flow2(pc1_l2, new_feat1_l2, cost2, up_flow2)

        # l1
        up_flow1 = self.upsample(pc1_l1, pc1_l2, self.scale * flow2)
        pc2_l1_warp = self.warping(pc1_l1, pc2_l1, up_flow1)
        cost1 = self.cost1(pc1_l1, pc2_l1_warp, c_feat1_l1, c_feat2_l1)

        feat2_up = self.upsample(pc1_l1, pc1_l2, feat2)
        new_feat1_l1 = torch.cat([feat1_l1, feat2_up], dim=1)
        feat1, flow1 = self.flow1(pc1_l1, new_feat1_l1, cost1, up_flow1)

        # l0
        up_flow0 = self.upsample(pc1_l0, pc1_l1, self.scale * flow1)
        pc2_l0_warp = self.warping(pc1_l0, pc2_l0, up_flow0)
        cost0 = self.cost0(pc1_l0, pc2_l0_warp, c_feat1_l0, c_feat2_l0)

        feat1_up = self.upsample(pc1_l0, pc1_l1, feat1)
        new_feat1_l0 = torch.cat([feat1_l0, feat1_up], dim=1)
        _, flow0 = self.flow0(pc1_l0, new_feat1_l0, cost0, up_flow0)

        flows = [flow0, flow1, flow2, flow3]
        pc1 = [pc1_l0, pc1_l1, pc1_l2, pc1_l3]
        pc2 = [pc2_l0, pc2_l1, pc2_l2, pc2_l3]
        fps_pc1_idxs = [fps_pc1_l1, fps_pc1_l2, fps_pc1_l3]
        fps_pc2_idxs = [fps_pc2_l1, fps_pc2_l2, fps_pc2_l3]

        return flows, fps_pc1_idxs, fps_pc2_idxs, pc1, pc2


def multiScaleLoss(pred_flows, gt_flow, fps_idxs):
    # num of scale
    num_scale = len(pred_flows)
    offset = len(fps_idxs) - num_scale + 1

    # generate GT list and mask1s
    gt_flows = [gt_flow]
    for i in range(1, len(fps_idxs) + 1):
        fps_idx = fps_idxs[i - 1]
        sub_gt_flow = index_points(gt_flows[-1], fps_idx) / scale
        gt_flows.append(sub_gt_flow)

    total_loss = torch.zeros(1).cuda()
    for i in range(num_scale):
        diff_flow = pred_flows[i].permute(0, 2, 1) - gt_flows[i + offset]
        total_loss += torch.square(diff_flow).mean()

    return total_loss


def curvature(pc):
    # pc: B 3 N
    pc = pc.permute(0, 2, 1)
    sqrdist = square_distance(pc, pc)
    _, kidx = torch.topk(sqrdist, 10, dim = -1, largest=False, sorted=False) # B N 10 3
    grouped_pc = index_points_group(pc, kidx)
    pc_curvature = torch.sum(grouped_pc - pc.unsqueeze(2), dim = 2) / 9.0
    return pc_curvature # B N 3

def computeChamfer(pc1, pc2):
    '''
    pc1: B 3 N
    pc2: B 3 M
    '''
    pc1 = pc1.permute(0, 2, 1)
    pc2 = pc2.permute(0, 2, 1)
    sqrdist12 = square_distance(pc1, pc2) # B N M

    #chamferDist
    dist1, _ = torch.topk(sqrdist12, 1, dim = -1, largest=False, sorted=False)
    dist2, _ = torch.topk(sqrdist12, 1, dim = 1, largest=False, sorted=False)
    dist1 = dist1.squeeze(2)
    dist2 = dist2.squeeze(1)

    return dist1, dist2

def curvatureWarp(pc, warped_pc):
    warped_pc = warped_pc.permute(0, 2, 1)
    pc = pc.permute(0, 2, 1)
    sqrdist = square_distance(pc, pc)
    _, kidx = torch.topk(sqrdist, 10, dim = -1, largest=False, sorted=False) # B N 10 3
    grouped_pc = index_points_group(warped_pc, kidx)
    pc_curvature = torch.sum(grouped_pc - warped_pc.unsqueeze(2), dim = 2) / 9.0
    return pc_curvature # B N 3

def computeSmooth(pc1, pred_flow):
    '''
    pc1: B 3 N
    pred_flow: B 3 N
    '''

    pc1 = pc1.permute(0, 2, 1)
    pred_flow = pred_flow.permute(0, 2, 1)
    sqrdist = square_distance(pc1, pc1) # B N N

    #Smoothness
    _, kidx = torch.topk(sqrdist, 9, dim = -1, largest=False, sorted=False)
    grouped_flow = index_points_group(pred_flow, kidx) # B N 9 3
    diff_flow = torch.norm(grouped_flow - pred_flow.unsqueeze(2), dim = 3).sum(dim = 2) / 8.0

    return diff_flow


def interpolateCurvature(pc1, pc2, pc2_curvature):
    '''
    pc1: B 3 N
    pc2: B 3 M
    pc2_curvature: B 3 M
    '''

    B, _, N = pc1.shape
    pc1 = pc1.permute(0, 2, 1)
    pc2 = pc2.permute(0, 2, 1)
    pc2_curvature = pc2_curvature

    sqrdist12 = square_distance(pc1, pc2) # B N M
    dist, knn_idx = torch.topk(sqrdist12, 5, dim = -1, largest=False, sorted=False)
    grouped_pc2_curvature = index_points_group(pc2_curvature, knn_idx) # B N 5 3
    norm = torch.sum(1.0 / (dist + 1e-6), dim = 2, keepdim = True)
    weight = (1.0 / (dist + 1e-6)) / norm

    inter_pc2_curvature = torch.sum(weight.view(B, N, 5, 1) * grouped_pc2_curvature, dim = 2)
    return inter_pc2_curvature

def multiScaleChamferSmoothCurvature(cfg, pc1, pc2, pred_flows):
    f_curvature = cfg.DA.SELF_SUP.CURVATURE_FAC
    f_smoothness = cfg.DA.SELF_SUP.SMOOTHNESS_FAC
    f_chamfer = cfg.DA.SELF_SUP.CHAMFER_FAC

    #num of scale
    num_scale = len(pred_flows)

    chamfer_loss = torch.zeros(1).cuda()
    smoothness_loss = torch.zeros(1).cuda()
    curvature_loss = torch.zeros(1).cuda()
    for i in range(num_scale):
        cur_pc1 = pc1[i] # B 3 N
        cur_pc2 = pc2[i]
        cur_flow = pred_flows[i] # B 3 N
        cur_pc1_warp = cur_pc1 + cur_flow

        # chamfer
        if f_chamfer > 0:
            dist1, dist2 = computeChamfer(cur_pc1_warp, cur_pc2)
            chamferLoss = dist1.mean() + dist2.mean()
            chamfer_loss += chamferLoss

        #smoothness
        if f_smoothness > 0:
            smoothnessLoss = computeSmooth(cur_pc1, cur_flow).mean()
            smoothness_loss += smoothnessLoss

        #curvature
        if f_curvature > 0:
            cur_pc2_curvature = curvature(cur_pc2)
            moved_pc1_curvature = curvatureWarp(cur_pc1, cur_pc1_warp)
            inter_pc2_curvature = interpolateCurvature(cur_pc1_warp, cur_pc2, cur_pc2_curvature)
            curvatureLoss = torch.sum((inter_pc2_curvature - moved_pc1_curvature) ** 2, dim = 2).mean()
            curvature_loss += curvatureLoss

    total_loss = f_chamfer * chamfer_loss + f_curvature * curvature_loss + f_smoothness * smoothness_loss

    return total_loss, chamfer_loss, curvature_loss, smoothness_loss


def singleScaleChamferSmoothCurvature(pc1, pc2, pred_flows):
    f_curvature = 0.3
    f_smoothness = 1.0
    f_chamfer = 1.0

    cur_pc1 = pc1.permute(0,2,1) # B 3 N
    cur_pc2 = pc2.permute(0,2,1)
    cur_flow = pred_flows.permute(0,2,1) # B 3 N

    #compute curvature
    cur_pc2_curvature = curvature(cur_pc2)
    cur_pc1_warp = cur_pc1 + cur_flow
    dist1, dist2 = computeChamfer(cur_pc1_warp, cur_pc2)
    moved_pc1_curvature = curvatureWarp(cur_pc1, cur_pc1_warp)

    # chamferLoss = dist1.sum(dim = 1).mean() + dist2.sum(dim = 1).mean()
    chamferLoss = dist1.mean() + dist2.mean()

    #smoothness
    smoothnessLoss = computeSmooth(cur_pc1, cur_flow).mean()

    #curvature
    inter_pc2_curvature = interpolateCurvature(cur_pc1_warp, cur_pc2, cur_pc2_curvature)
    curvatureLoss = torch.sum((inter_pc2_curvature - moved_pc1_curvature) ** 2, dim = 2).mean()

    total_loss = f_chamfer * chamferLoss + f_curvature * curvatureLoss + f_smoothness * smoothnessLoss
    if torch.isnan(total_loss):
        print(chamferLoss.item(), curvatureLoss.item(), smoothnessLoss.item())
    return total_loss


def computeCurvature(pc1, pc2, pred_flow):
    pc1_warp = pc1 + pred_flow
    pc2_curvature = curvature(pc2)
    moved_pc1_curvature = curvatureWarp(pc1, pc1_warp)
    inter_pc2_curvature = interpolateCurvature(pc1_warp, pc2, pc2_curvature)
    curvatureLoss = torch.sum((inter_pc2_curvature - moved_pc1_curvature) ** 2, dim=2)
    return curvatureLoss