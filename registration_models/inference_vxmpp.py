#!/usr/bin/env python
import torch
import sys
sys.path.insert(0,'src/')
from vxmplusplus_utils import get_vxmpp_models,return_crops
sys.path.insert(0,'corrfield/')
from thin_plate_spline import *
from tqdm import trange
from vxmplusplus_utils import adam_mind
import argparse

from data_utils import read_image_folder

def main(args):
    
    img_insp_all,img_exp_all,keypts_insp_all,mind_insp_all,mind_exp_all,orig_shapes_all,case_list = read_image_folder(args.imgfolder,args.maskfolder,do_MIND=True)

    unet_model,heatmap,mesh = get_vxmpp_models()


    lms_validation = torch.load('lms_validation.pth')
    datasets = ['EMPIRE10']*12+['NLSTtrain']*22+['4D-Lung-1']*20+['LungCT-L2R']*30+['VentilCT']*20+['COPDgene']*10+['NLSTtest']*10



    state_dicts = torch.load(args.model)
    unet_model.load_state_dict(state_dicts[1])
    heatmap.load_state_dict(state_dicts[0])

    print('inference for validation scans with TRE computation ')

    predictions = []
    
    for case in trange(len(case_list)):
        ii = int(case_list[case].split('case_')[1])

        ##MASKED INPUT IMAGES ARE HALF-RESOLUTION
        dataset = datasets[ii]
        with torch.no_grad():
            fixed_img = img_insp_all[case]
            moving_img = img_exp_all[case]
            keypts_fix = keypts_insp_all[case].squeeze().cuda()


            H,W,D = fixed_img.shape[-3:]

            fixed_img = fixed_img.view(1,1,H,W,D).cuda()
            moving_img = moving_img.view(1,1,H,W,D).cuda()

            with torch.cuda.amp.autocast():
                #VoxelMorph requires some padding
                input,x_start,y_start,z_start,x_end,y_end,z_end = return_crops(torch.cat((fixed_img,moving_img),1).cuda())
                output = F.pad(F.interpolate(unet_model(input),scale_factor=2),(z_start,(-z_end+D),y_start,(-y_end+W),x_start,(-x_end+H)))
                disp_est = torch.zeros_like(keypts_fix)
                for idx in torch.split(torch.arange(len(keypts_fix)),1024):
                    sample_xyz = keypts_fix[idx]
                    sampled = F.grid_sample(output,sample_xyz.cuda().view(1,-1,1,1,3),mode='bilinear')
                    disp_pred = heatmap(sampled.permute(2,1,0,3,4))
                    disp_est[idx] = torch.sum(torch.softmax(disp_pred.view(-1,11**3,1),1)*mesh.view(1,11**3,3),1)


        ##NOW EVERYTHING FULL-RESOLUTION
        H,W,D = orig_shapes_all[case]#.shape[-3:]

        fixed_mind = mind_insp_all[case].view(1,-1,H//2,W//2,D//2).cuda()
        moving_mind = mind_exp_all[case].view(1,-1,H//2,W//2,D//2).cuda()

        pred_xyz,disp_smooth,dense_flow = adam_mind(keypts_fix,disp_est,fixed_mind,moving_mind,H,W,D)
        predictions.append(pred_xyz.cpu())
        ##EVALUATION WITH MANUAL LANDMARKS PROVIDED WITH LUNG-250M-4B
        tre0 = (lms_validation[str(ii)][:,:3]-lms_validation[str(ii)][:,3:]).pow(2).sum(-1).sqrt()
        lms1 = torch.flip((lms_validation[str(ii)][:,:3]-torch.tensor([H/2,W/2,D/2]))/torch.tensor([H/2,W/2,D/2]),(1,))
        lms_disp = F.grid_sample(disp_smooth.cpu(),lms1.view(1,-1,1,1,3)).squeeze().t()

        tre2 = (lms_validation[str(ii)][:,:3]+lms_disp-lms_validation[str(ii)][:,3:]).pow(2).sum(-1).sqrt()
        print(dataset+'-TRE init: '+str('%0.3f'%tre0.mean())+'mm; net+adam: '+str('%0.3f'%tre2.mean())+'mm;')

    torch.save({'predictions':predictions,'case_list':case_list,'keypts_insp_all':keypts_insp_all},args.outfile)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'inference of VoxelMorph++ on Lung250M-4B')

    parser.add_argument('-M',  '--model',        default='models/voxelmorphplusplus.pth', help="model file (pth)")
    parser.add_argument('-m',  '--maskfolder',   default='masksTs', help="mask folder containing (/case_???_{1,2}.nii.gz)")
    parser.add_argument('-I',  '--imgfolder',    default='imagesTs', help="image folder containing (/case_???_{1,2}.nii.gz)")
    parser.add_argument('-O',  '--outfile',      default='predictions.pth', help="output file for keypoint displacement predictions")

    #args = parser.parse_args(args=[])
    args = parser.parse_args()
    print(args)
    main(args)






