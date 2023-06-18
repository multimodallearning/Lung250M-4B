#!/usr/bin/env python
# coding: utf-8

# In[3]:


import nibabel as nib
import numpy as np
import struct
import torch
from tqdm.notebook import tqdm,trange
import torch.nn.functional as F


# In[5]:


def bbox(mask_fixed,mask_moving):
    H,W,D = mask_fixed.shape
    where = torch.where(mask_fixed)
    x_min = int(torch.min(where[0]))
    y_min = int(torch.min(where[1]))
    z_min = int(torch.min(where[2]))
    x_max = int(torch.max(where[0]))
    y_max = int(torch.max(where[1]))
    z_max = int(torch.max(where[2]))

    H1,W1,D1 = mask_moving.shape
    where = torch.where(mask_moving)
    x1_min = int(torch.min(where[0]))
    y1_min = int(torch.min(where[1]))
    z1_min = int(torch.min(where[2]))
    x1_max = int(torch.max(where[0]))
    y1_max = int(torch.max(where[1]))
    z1_max = int(torch.max(where[2]))

    H_new = max(x1_max-x1_min,x_max-x_min)+10
    W_new = max(y1_max-y1_min,y_max-y_min)+10
    D_new = max(z1_max-z1_min,z_max-z_min)+10
    #print('new',int(H_new),int(W_new),int(D_new))

    x_start = ((x_max+x_min)-H_new)//2
    x_end = ((x_max+x_min)+H_new)//2
    y_start = ((y_max+y_min)-W_new)//2
    y_end = ((y_max+y_min)+W_new)//2
    z_start = ((z_max+z_min)-D_new)//2
    z_end = ((z_max+z_min)+D_new)//2
    
    x1_start = ((x1_max+x1_min)-H_new)//2
    x1_end = ((x1_max+x1_min)+H_new)//2
    y1_start = ((y1_max+y1_min)-W_new)//2
    y1_end = ((y1_max+y1_min)+W_new)//2
    z1_start = ((z1_max+z1_min)-D_new)//2
    z1_end = ((z1_max+z1_min)+D_new)//2
    return x_start,y_start,z_start,x_end,y_end,z_end,x1_start,y1_start,z1_start,x1_end,y1_end,z1_end


# In[7]:


idx_select = torch.arange(1,11)
case_num = torch.arange(104,114)


get_ipython().system('mkdir imagesTs')
get_ipython().system('mkdir masksTs')
copd_bbox_fixed = []
copd_bbox_moving = []
shapes_1mm = []

mask_crop_fixed_all = []
mask_crop_moving_all = []
for i in trange(10):
    case = int(idx_select[i])

    hdr_fixed = nib.load('COPD_1mm/COPD_0'+str(case).zfill(2)+'_fixed_0000.nii.gz').header
    img_fixed = -1024+torch.from_numpy(nib.load('COPD_1mm/COPD_0'+str(case).zfill(2)+'_fixed_0000.nii.gz').get_fdata()).float()

    hdr_moving = nib.load('COPD_1mm/COPD_0'+str(case).zfill(2)+'_moving_0000.nii.gz').header
    img_moving = -1024+torch.from_numpy(nib.load('COPD_1mm/COPD_0'+str(case).zfill(2)+'_moving_0000.nii.gz').get_fdata()).float()

    mask_fixed = torch.from_numpy(nib.load('COPD_1mm_mask/COPD_0'+str(case).zfill(2)+'_fixed.nii.gz').get_fdata()).float()
    mask_moving = torch.from_numpy(nib.load('COPD_1mm_mask/COPD_0'+str(case).zfill(2)+'_moving.nii.gz').get_fdata()).float()
    H,W,D = mask_fixed.shape
    H1,W1,D1 = mask_moving.shape

    shapes_1mm.append(mask_moving.shape)
    x_start,y_start,z_start,x_end,y_end,z_end,x1_start,y1_start,z1_start,x1_end,y1_end,z1_end = bbox(mask_fixed,mask_moving)
    mask_fixed_crop = F.pad(mask_fixed,(-z_start,z_end-D,-y_start,y_end-W,-x_start,x_end-H))
    img_fixed_crop = F.pad(img_fixed,(-z_start,z_end-D,-y_start,y_end-W,-x_start,x_end-H),value=-1024)
    #print(mask_fixed_crop.shape,mask_fixed.sum(),mask_fixed_crop.sum())

    copd_bbox_fixed.append(torch.tensor([x_start,y_start,z_start,x_end,y_end,z_end]))
    
    mask_moving_crop = F.pad(mask_moving,(-z1_start,z1_end-D1,-y1_start,y1_end-W1,-x1_start,x1_end-H1))
    img_moving_crop = F.pad(img_moving,(-z1_start,z1_end-D1,-y1_start,y1_end-W1,-x1_start,x1_end-H1),value=-1024)
    #print(mask_moving_crop.shape,mask_moving.sum(),mask_moving_crop.sum())
    
    mask_crop_moving_all.append(mask_moving_crop)
    mask_crop_fixed_all.append(mask_fixed_crop)
    
    copd_bbox_moving.append(torch.tensor([x1_start,y1_start,z1_start,x1_end,y1_end,z1_end]))
    nib.save(nib.Nifti1Image(img_fixed_crop,None,header=hdr_moving),'imagesTs/case_'+str(int(case_num[i])).zfill(3)+'_1.nii.gz')
    nib.save(nib.Nifti1Image(img_moving_crop,None,header=hdr_moving),'imagesTs/case_'+str(int(case_num[i])).zfill(3)+'_2.nii.gz')
    nib.save(nib.Nifti1Image(mask_fixed_crop,None,header=hdr_fixed),'masksTs/case_'+str(int(case_num[i])).zfill(3)+'_1.nii.gz')
    nib.save(nib.Nifti1Image(mask_moving_crop,None,header=hdr_fixed),'masksTs/case_'+str(int(case_num[i])).zfill(3)+'_2.nii.gz')
#'copd_lms_insp_fixed':copd_lms_insp_fixed,'copd_lms_exp_moving':copd_lms_exp_moving,


# In[8]:



torch.save({'copd_bbox_fixed':copd_bbox_fixed,'copd_bbox_moving':copd_bbox_moving,'shapes_1mm':shapes_1mm},'copd_bboxes_1mm.pth')

