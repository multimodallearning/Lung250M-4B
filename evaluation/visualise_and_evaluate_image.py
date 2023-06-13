#!/usr/bin/env python
import streamlit as st
import torch
import argparse

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys
from glob import glob
from io import BytesIO
import copy
sys.path.insert(0,'corrfield/')
from thin_plate_spline import thin_plate_dense
#torch.backends.cuda.matmul.allow_tf32 = False
#torch.backends.cudnn.allow_tf32 = False

@st.cache_data
def get_warped_pair(_cf,_img_moving):
    H,W,D = _img_moving.shape[-3:]
    
    kpts_fixed = torch.flip((_cf[:,:3]-torch.tensor([H/2,W/2,D/2]).view(1,-1)).div(torch.tensor([H/2,W/2,D/2]).view(1,-1)),(-1,))
    kpts_moving = torch.flip((_cf[:,3:]-torch.tensor([H/2,W/2,D/2]).view(1,-1)).div(torch.tensor([H/2,W/2,D/2]).view(1,-1)),(-1,))


    with torch.no_grad():
        dense_flow = thin_plate_dense(kpts_fixed.unsqueeze(0).cuda(), (kpts_moving-kpts_fixed).unsqueeze(0).cuda(), (H, W, D), 4, 0.01).cpu()
    warped_img = F.grid_sample(_img_moving.view(1,1,H,W,D),dense_flow+F.affine_grid(torch.eye(3,4).unsqueeze(0),(1,1,H,W,D))).squeeze()

        
    return warped_img,dense_flow
    
def main(args):

    st.title('Visualise lung registration')

    st.write('passed argument ',args.imgfolder,args.outfolder)
    base_img = args.imgfolder
    img_list = [i.split('/')[1] for i in sorted(glob(base_img+'/*.nii.gz'))]

    img_list1 = [i.replace('_1.nii.gz','').replace('_2.nii.gz','') for i in img_list]
    if(len(img_list1) != 2*len(set(img_list1))):
        raise Exception("Not all images seem to have both _1.nii.gz and _2.nii.gz")
    case_list = sorted(list(set(img_list1)))

    case = st.select_slider("Select #case from "+base_img, options=case_list)
        
    img_fixed = torch.from_numpy(nib.load(base_img+'/'+case+'_1.nii.gz').get_fdata()).float()
    img_moving = torch.from_numpy(nib.load(base_img+'/'+case+'_2.nii.gz').get_fdata()).float()

    dense_flow = None
    cf = None
    if(args.outfolder!='None'):
        cf = torch.from_numpy(np.loadtxt(args.outfolder+'/'+case+'.csv',delimiter=',')).float()
        img_warped,dense_flow = get_warped_pair(cf,img_moving)

    H,W,D = img_fixed.shape

    c1,c2 = st.columns(2)
    with c1:
        y = st.slider("slide in AP", min_value=20, max_value=W-20, value=W//2, step=10)
    with c2:
        int_max = st.slider("HU window (max)", min_value=-800, max_value=1200, value=-200, step=50)
#
    col1,col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        st.write('before alignment')
        ax.imshow(torch.clamp(img_fixed[:,y].t().flip(0),-1000,int_max),'Blues')
        ax.imshow(torch.clamp(img_moving[:,y].t().flip(0),-1000,int_max),'Oranges',alpha=.5)
        ax.axis('off')
        st.pyplot(fig)
    
    with col2:
        str_after = ''
        if(args.outfolder!='None'):
            str_after = 'after alignment'
        st.write(str_after)
        if(args.outfolder!='None'):

            fig, ax = plt.subplots()
            ax.imshow(torch.clamp(img_fixed[:,y].t().flip(0),-1000,int_max),'Blues')
            ax.imshow(torch.clamp(img_warped[:,y].t().flip(0),-1000,int_max),'Oranges',alpha=.5)
            ax.axis('off')
            st.pyplot(fig)

    if(args.validation!='None'):
        ii = int(case.split('case_')[1])

        lms_validation = torch.load(args.validation)
        tre0 = (lms_validation[str(ii)][:,:3]-lms_validation[str(ii)][:,3:]).pow(2).sum(-1).sqrt()
        st.write('statistics on landmark errors')
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(torch.sort(tre0,descending=False).values.numpy(),torch.linspace(0,1,tre0.shape[0]).numpy(),label='before '+str('%0.2f'%(tre0.mean()))+' mm')
        H,W,D = img_fixed.shape

        if(dense_flow is not None):
            lms1 = torch.flip((lms_validation[str(ii)][:,:3]-torch.tensor([H/2,W/2,D/2]))/torch.tensor([H/2,W/2,D/2]),(1,))
            lms_disp1 = F.grid_sample(dense_flow.permute(0,4,1,2,3).cpu(),lms1.view(1,-1,1,1,3)).squeeze().t()
            lms_disp = torch.flip(lms_disp1,(1,))*torch.tensor([H/2,W/2,D/2])
            tre2 = (lms_validation[str(ii)][:,:3]+lms_disp-lms_validation[str(ii)][:,3:]).pow(2).sum(-1).sqrt()
            

            ax.plot(torch.sort(tre2,descending=False).values.numpy(),torch.linspace(0,1,tre2.shape[0]).numpy(),label='registered '+str('%0.2f'%(tre2.mean()))+' mm')
            

        ax.set_xlim([0,30])
        ax.set_xlabel('TRE in mm')
        ax.set_ylabel('cumulative distribution')
        ax.legend()
        st.pyplot(fig)

        #buf = BytesIO()
        #fig.savefig(buf, format="png")
        #st.image(buf,width=400)




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = 'visualise Lung250M-4B registrations')
    parser.add_argument('-I',  '--imgfolder',    default='imagesTs', help="image folder containing (/case_???_{1,2}.nii.gz)")
    parser.add_argument('-o',  '--outfolder',    default='None', help="output folder for individual keypoint displacement predictions")
    parser.add_argument('-v',  '--validation',    default='None', help="pth file with validation landmarks")

    args = parser.parse_args()

    main(args)
