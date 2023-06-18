import numpy as np
import torch.nn.functional as F
import nibabel as nib
from tqdm import trange
import torch
#import struct

crops = torch.load('copd_bboxes_and_lms_1mm.pth')


base = './'

for i in trange(1,11):

    
    x_start,y_start,z_start,x_end,y_end,z_end = crops['copd_bbox_moving'][i-1]
    H,W,D = crops['shapes_1mm'][i-1]

    #with open(base+'copd'+str(i)+'/copd'+str(i)+'_eBHCT.img', 'rb') as content_file:
    #        content = content_file.read()
    data = np.fromfile(base+'copd'+str(i)+'/copd'+str(i)+'_eBHCT.img', dtype='<h')

    image = torch.clamp(-1000+torch.from_numpy(data).reshape(-1,512,512),-1000)
    image_1mm = F.interpolate(image.permute(2,1,0).flip(-1).float().unsqueeze(0).unsqueeze(0),size=(H,W,D),mode='trilinear').squeeze()
    img_2 = F.pad(image_1mm,(-z_start,z_end-D,-y_start,y_end-W,-x_start,x_end-H),value=-1024)


    
    hdr = nib.load(base+'../masksTs/case_'+str(int(i+103)).zfill(3)+'_2.nii.gz').header
    nib.save(nib.Nifti1Image(img_2.numpy(),None,header=hdr),base+'../imagesTs/case_'+str(int(i+103)).zfill(3)+'_2.nii.gz')
    
    x_start,y_start,z_start,x_end,y_end,z_end = crops['copd_bbox_fixed'][i-1]
    H,W,D = crops['shapes_1mm'][i-1]

    #with open(base+'copd'+str(i)+'/copd'+str(i)+'_iBHCT.img', 'rb') as content_file:
    #        content = content_file.read()
    data = np.fromfile(base+'copd'+str(i)+'/copd'+str(i)+'_iBHCT.img', dtype='<h')

    image = torch.clamp(-1000+torch.from_numpy(data).reshape(-1,512,512),-1000)
    image_1mm = F.interpolate(image.permute(2,1,0).flip(-1).float().unsqueeze(0).unsqueeze(0),size=(H,W,D),mode='trilinear').squeeze()
    img_2 = F.pad(image_1mm,(-z_start,z_end-D,-y_start,y_end-W,-x_start,x_end-H),value=-1024)

    hdr = nib.load(base+'../masksTs/case_'+str(int(i+103)).zfill(3)+'_1.nii.gz').header
    nib.save(nib.Nifti1Image(img_2.numpy(),None,header=hdr),base+'../imagesTs/case_'+str(int(i+103)).zfill(3)+'_1.nii.gz')
    

