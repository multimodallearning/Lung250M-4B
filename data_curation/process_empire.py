import SimpleITK as sitk
import numpy as np
import torch.nn.functional as F
import nibabel as nib
from tqdm import trange
import torch
idx_select = torch.tensor([1,7,8,13,14,16,17,18,20,21,23,28])
order_swap = torch.tensor([0,0,0,1,0,1,1,0,0,0,1,0,0])

base = './'
empire_bbox_crop = torch.load(base+'empire_bbox_crop.pth')
empire_bbox_fixed = empire_bbox_crop['empire_bbox_fixed']
empire_bbox_moving = empire_bbox_crop['empire_bbox_moving']
orig_dim_mask_fixed = empire_bbox_crop['orig_dim_mask_fixed']
orig_dim_mask_moving = empire_bbox_crop['orig_dim_mask_moving']

#base = 'Lung250M-4B/data_curation/'


for i in trange(12):

    for phase in range(2):
        if(order_swap[i]==(1-phase)):
            endmhd = '_Fixed.mhd'
        else:
            endmhd =  '_Moving.mhd'

        online = 'Online'
        if(int(idx_select[i])>=21):
            online = ''
        itkimage = sitk.ReadImage(base+'/'+online+'/scans/'+str(int(idx_select[i])).zfill(2)+endmhd)
        ct_scan = sitk.GetArrayFromImage(itkimage)
    
        if(order_swap[i]==(1-phase)):
            dim = orig_dim_mask_fixed[i]
            x_start,y_start,z_start,x_end,y_end,z_end = empire_bbox_fixed[i]
            H,W,D = orig_dim_mask_fixed[i]
        else:
            dim = orig_dim_mask_moving[i]
            x_start,y_start,z_start,x_end,y_end,z_end = empire_bbox_moving[i]
            H,W,D = orig_dim_mask_moving[i]

        ct_1mm = -24+F.interpolate(torch.from_numpy(ct_scan).permute(2,1,0).float().unsqueeze(0).unsqueeze(0),size=dim,mode='trilinear').squeeze()
        if(i==3):
            ct_1mm = ct_1mm.flip((0,1))
        img_2 = F.pad(ct_1mm,(-z_start,z_end-D,-y_start,y_end-W,-x_start,x_end-H),value=-1024)

        folder = 'masksTr'
        if((i==2)|(i==8)):
            folder = 'masksTs'

        hdr = nib.load(base+'../'+folder+'/case_'+str(int(i)).zfill(3)+'_'+str(phase+1)+'.nii.gz').header
        folder = 'imagesTr'
        if((i==2)|(i==8)):
            folder = 'imagesTs'

            nib.save(nib.Nifti1Image(img_2.numpy(),None,header=hdr),base+'../'+folder+'/case_'+str(int(i)).zfill(3)+'_'+str(phase+1)+'.nii.gz')
        
