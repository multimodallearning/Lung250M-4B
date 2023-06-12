import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from glob import glob
from tqdm import trange,tqdm
import sys
sys.path.insert(0,'../corrfield/')
from foerstner import foerstner_kpts 
from vxmplusplus_utils import MINDSSC

def read_image_folder(base_img,base_mask,do_MIND = True):
    

    img_list = [i.split('/')[1] for i in sorted(glob(base_img+'/*.nii.gz'))]
    mask_list = [i.split('/')[1] for i in sorted(glob(base_mask+'/*.nii.gz'))]
    if(img_list != mask_list):
        raise Exception("Masks and images do not correctly match. Please check folder content.")

    img_list1 = [i.replace('_1.nii.gz','').replace('_2.nii.gz','') for i in img_list]
    if(len(img_list1) != 2*len(set(img_list1))):
        raise Exception("Not all images seem to have both _1.nii.gz and _2.nii.gz")
    case_list = sorted(list(set(img_list1)))
    img_insp_all = []
    img_exp_all = []
    keypts_insp_all = []
    mind_insp_all = []
    mind_exp_all = []
    orig_shapes_all = []
    print('Loading '+str(len(case_list))+' scan pairs')

    for ii in trange(len(case_list)):
        i = int(ii)

        img_exp = torch.from_numpy(nib.load(base_img+'/'+case_list[ii]+'_2.nii.gz').get_fdata()).float()
        mask_exp = torch.from_numpy(nib.load(base_mask+'/'+case_list[ii]+'_2.nii.gz').get_fdata()).float()
        masked_exp = F.interpolate(((img_exp+1024)*mask_exp).unsqueeze(0).unsqueeze(0),scale_factor=.5,mode='trilinear').squeeze()

        img_insp = torch.from_numpy(nib.load(base_img+'/'+case_list[ii]+'_1.nii.gz').get_fdata()).float()
        mask_insp = torch.from_numpy(nib.load(base_mask+'/'+case_list[ii]+'_1.nii.gz').get_fdata()).float()
        
        kpts_fix = foerstner_kpts(img_insp.unsqueeze(0).unsqueeze(0).cuda(), mask_insp.unsqueeze(0).unsqueeze(0).cuda(), 1.4, 3).cpu()
        keypts_insp_all.append(kpts_fix)
            
        masked_insp = F.interpolate(((img_insp+1024)*mask_insp).unsqueeze(0).unsqueeze(0),scale_factor=.5,mode='trilinear').squeeze()

        shape1 = mask_insp.shape
        

        img_exp_all.append(masked_exp)
        orig_shapes_all.append(shape1)

        img_insp_all.append(masked_insp)

        if(do_MIND):
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    mind_insp = F.avg_pool3d(mask_insp.unsqueeze(0).unsqueeze(0).cuda().half()*\
                            MINDSSC(img_insp.unsqueeze(0).unsqueeze(0).cuda(),1,2).half(),2).cpu()
                    mind_exp = F.avg_pool3d(mask_exp.unsqueeze(0).unsqueeze(0).cuda().half()*\
                            MINDSSC(img_exp.unsqueeze(0).unsqueeze(0).cuda(),1,2).half(),2).cpu()

            mind_insp_all.append(mind_insp)
            mind_exp_all.append(mind_exp)
            del mind_insp
            del mind_exp
            
    return img_insp_all,img_exp_all,keypts_insp_all,mind_insp_all,mind_exp_all,orig_shapes_all,case_list