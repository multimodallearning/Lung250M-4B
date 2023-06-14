# Create point clouds from vessel segmentations

import nibabel as nib
import torch
import numpy as np
import os
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

def foerstner_nms(pcd, sigma, neigh_1, neigh_2, min_points, sigma_interval):
    pcd = pcd.cuda()
    knn = torch.zeros(len(pcd), neigh_1).long().cuda()
    knn_dist = torch.zeros(len(pcd), neigh_1).float().cuda()
    with torch.no_grad():
        chk = torch.chunk(torch.arange(len(pcd)).cuda(), 192)
        for i in range(len(chk)):
            dist = (pcd[chk[i]].unsqueeze(1) - pcd.unsqueeze(0)).pow(2).sum(-1).sqrt()
            q = torch.topk(dist, neigh_1, dim=1, largest=False)
            knn[chk[i]] = q[1][:, :]
            knn_dist[chk[i]] = q[0][:, :]

    curr_points = 0
    curr_sigma = sigma
    while curr_points < min_points:
        exp_score = torch.exp(-knn_dist[:, :].pow(2) * curr_sigma ** 2).mean(1)
        knn_score = torch.max(exp_score[knn[:, :neigh_2]], 1)[0]
        valid_idx = (knn_score == exp_score).nonzero(as_tuple=True)[0]
        curr_points = valid_idx.shape[0]
        curr_sigma += sigma_interval
    return valid_idx,knn,knn_dist#rand_idx[valid_idx]


if not os.path.exists('../cloudsTr'):
    os.mkdir('../cloudsTr')
    os.mkdir('../cloudsTr/coordinates')
    os.mkdir('../cloudsTr/artery_vein')
    os.mkdir('../cloudsTr/distance')
    
if not os.path.exists('../cloudsTs'):
    os.mkdir('../cloudsTs')
    os.mkdir('../cloudsTs/coordinates')
    os.mkdir('../cloudsTs/artery_vein')
    os.mkdir('../cloudsTs/distance')

for file_name in os.listdir("../segTr"):
    if file_name.endswith(".nii.gz"):
        file_name_0 = file_name[:-7]
        vessel_seg = torch.from_numpy(nib.load('../segTr/'+file_name_0+'.nii.gz').get_fdata())
        mask = torch.from_numpy(nib.load('../masksTr/'+file_name_0+'.nii.gz').get_fdata())
        vessel_resample = torch.nn.functional.interpolate(1.0*(vessel_seg>0)*(mask>0).unsqueeze(0).unsqueeze(0), scale_factor=2, mode='nearest').squeeze(0).squeeze(0)
        vessel_full = torch.nonzero(vessel_resample)/2
        seg_bin = vessel_resample>0
        skel = torch.from_numpy(1.0*skeletonize(seg_bin))
        vessel_skel = torch.nonzero(skel)/2

        valid_idx,_,_ = foerstner_nms(vessel_skel, 0.7, 29,15,8192,0.1)

        valid_idx2 = valid_idx[torch.randperm(valid_idx.shape[0])][:8192]

        vessel_8k = vessel_skel[valid_idx2.cpu()]
        vessel_full_classes = vessel_seg[vessel_full[:,0].long(),vessel_full[:,1].long(),vessel_full[:,2].long()]
        vessel_8k_classes = vessel_seg[vessel_8k[:,0].long(),(vessel_8k[:,1]).long(),(vessel_8k[:,2]).long()]
        vessel_skel_classes = vessel_seg[vessel_skel[:,0].long(),(vessel_skel[:,1]).long(),(vessel_skel[:,2]).long()]

        torch.save([vessel_8k, vessel_skel , vessel_full], '../cloudsTr/coordinates/'+file_name_0+'.pth')
        torch.save([vessel_8k_classes, vessel_skel_classes, vessel_full_classes], '../cloudsTr/artery_vein/'+file_name_0+'.pth')
        
        seg_edt = distance_transform_edt(1.0*(vessel_seg>0))

        vessel_8k_edt = seg_edt[vessel_8k[:,0].long(), vessel_8k[:,1].long(), vessel_8k[:,2].long()]
        vessel_full_edt = seg_edt[vessel_full[:,0].long(), vessel_full[:,1].long(), vessel_full[:,2].long()]
        vessel_skel_edt = seg_edt[vessel_skel[:,0].long(), vessel_skel[:,1].long(), vessel_skel[:,2].long()]
        
        torch.save([vessel_8k_edt, vessel_skel_edt, vessel_full_edt], '../cloudsTr/distance/'+file_name_0+'.pth')

for file_name in os.listdir("../segTs"):
    if file_name.endswith(".nii.gz"):
        file_name_0 = file_name[:-7]
        vessel_seg = torch.from_numpy(nib.load('../segTs/'+file_name_0+'.nii.gz').get_fdata())
        mask = torch.from_numpy(nib.load('../masksTs/'+file_name_0+'.nii.gz').get_fdata())
        vessel_resample = torch.nn.functional.interpolate(1.0*(vessel_seg>0)*(mask>0).unsqueeze(0).unsqueeze(0), scale_factor=2, mode='nearest').squeeze(0).squeeze(0)
        vessel_full = torch.nonzero(vessel_resample)/2
        seg_bin = vessel_resample>0
        skel = torch.from_numpy(1.0*skeletonize(seg_bin))
        vessel_skel = torch.nonzero(skel)/2

        valid_idx,_,_ = foerstner_nms(vessel_skel, 0.7, 29,15,8192,0.1)

        valid_idx2 = valid_idx[torch.randperm(valid_idx.shape[0])][:8192]

        vessel_8k = vessel_skel[valid_idx2.cpu()]
        vessel_full_classes = vessel_seg[vessel_full[:,0].long(),vessel_full[:,1].long(),vessel_full[:,2].long()]
        vessel_8k_classes = vessel_seg[vessel_8k[:,0].long(),(vessel_8k[:,1]).long(),(vessel_8k[:,2]).long()]
        vessel_skel_classes = vessel_seg[vessel_skel[:,0].long(),(vessel_skel[:,1]).long(),(vessel_skel[:,2]).long()]

        torch.save([vessel_8k, vessel_skel , vessel_full], '../cloudsTs/coordinates/'+file_name_0+'.pth')
        torch.save([vessel_8k_classes, vessel_skel_classes, vessel_full_classes], '../cloudsTs/artery_vein/'+file_name_0+'.pth')
        
        seg_edt = distance_transform_edt(1.0*(vessel_seg>0))

        vessel_8k_edt = seg_edt[vessel_8k[:,0].long(), vessel_8k[:,1].long(), vessel_8k[:,2].long()]
        vessel_full_edt = seg_edt[vessel_full[:,0].long(), vessel_full[:,1].long(), vessel_full[:,2].long()]
        vessel_skel_edt = seg_edt[vessel_skel[:,0].long(), vessel_skel[:,1].long(), vessel_skel[:,2].long()]
        
        torch.save([vessel_8k_edt, vessel_skel_edt, vessel_full_edt], '../cloudsTs/distance/'+file_name_0+'.pth')
        