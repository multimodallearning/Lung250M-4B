#!/usr/bin/env python
import pydicom
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
import os
import argparse
import subprocess
from tqdm import trange,tqdm
from scipy.ndimage.filters import gaussian_filter
import nibabel as nib

from preprocess_utils import *

header_nlst = """
downloadServerUrl=https://nlst.cancerimagingarchive.net/nbia-download/servlet/DownloadServlet
includeAnnotation=true
noOfrRetry=10
databasketId=manifest-NLST_allCT.tcia
manifestVersion=3.0
ListOfSeriesToDownload=
"""

#case_ids=(212202 212587 213510 214907 216160 216422 216978 218307 100108 100124)
#num=(114 115 116 117 118 119 120 121 122 123)
def main(args):
    tmp_dir = 'tmp_tcia'#+str(int(torch.randint(99999,(1,)))).zfill(5)
    os.makedirs(tmp_dir, exist_ok = True)

    with open(args.manifest, "r") as f:
        file_list = f.readlines()
        
    nlst_list1 = {}
    nlst_list2 = {}
    nlst_case = {}
    for i in range(len(file_list)):
        line1 = file_list[i].strip('\\n').split(',')
        nlst_case[str(line1[1])] = str(line1[0])
        nlst_list1[str(line1[0])] = line1[2]
        nlst_list2[str(line1[0])] = line1[3]
        
    #print(nlst_case)
    with open(tmp_dir+'/manifest.tcia', 'w') as f:
        f.write(header_nlst)
        for case in nlst_case.keys():
            series_id = str(nlst_case[str(case)])
            f.writelines([nlst_list1[series_id],'\n',nlst_list2[series_id]])

    print("Downloading "+len(file_list)+" 3D images up to "+series_id+". this may take up to several minutes")

    cmd = '/opt/nbia-data-retriever/nbia-data-retriever --cli '+os.getcwd()+'/'+tmp_dir+'/manifest.tcia -d '+os.getcwd()+'/'+tmp_dir+' –f -q -v'
    tcia_proc = subprocess.Popen(['echo Y | '+cmd], shell=True)
    tcia_proc.communicate()
    
    for case in nlst_case.keys():
        series_id = str(nlst_case[str(case)])
        #series_ids = []
        for i in range(2):
            path0 = tmp_dir+'/manifest/NLST/'+series_id+'/'
            path0 += os.listdir(path0)[i]+'/'
            path0 += os.listdir(path0)[0]+'/'

            file0 = sorted(os.listdir(path0))[0]
            #series_ids.append(pydicom.dcmread(os.path.join(path0,file0)).SeriesInstanceUID)
            
            img3d,ps,ss,orient,position = load_dcm(path0)
            ss = np.abs((position[5]-position[2]))
            if(np.percentile(img3d[256],15)>0):
                img3d -= 1024
            if((position[-1]-position[2])<0):
                img3d = np.array(np.flip(img3d,2))
                
            scaling = list(np.array([ps[0], ps[1], ss])/1)
            
            img_post = F.interpolate(torch.from_numpy(img3d.transpose(1,0,2)).float().unsqueeze(0).unsqueeze(0),scale_factor=scaling,mode='trilinear').squeeze()
            
            end_letter = 'A'
            if(i==1):
                end_letter = 'B'
            nib.save(nib.Nifti1Image(1024+img_post.numpy(),np.eye(4)),args.imgfolder+'/case_'+case+'_'+end_letter+'_0000.nii.gz')

        #dicoms.append({'img3d':img3d.transpose(1,0,2),'spacing':np.array([ps[0], ps[1], ss])})
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'preprocessing of Lung250M-4B')

    parser.add_argument('-M',  '--manifest',        default='nlst_test_tcia.csv', help="csv-file with list list of TCIA cases with (paired) patients to process")
    parser.add_argument('-I',  '--imgfolder',    default='imagesTemp', help="image folder output for resampled but uncropped images (/case_???_{A,B}_0000.nii.gz)")
    args = parser.parse_args()
    main(args)
