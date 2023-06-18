#!/usr/bin/env python
import torch
import sys
from vxmplusplus_utils import get_vxmpp_models,return_crops
sys.path.insert(0,'corrfield/')
from thin_plate_spline import *
from tqdm import trange,tqdm
from vxmplusplus_utils import adam_mind
import argparse
import numpy as np
import time
from data_utils import read_image_folder

def main(args):
    
    img_insp_all,img_exp_all,keypts_insp_all,mind_insp_all,mind_exp_all,orig_shapes_all,case_list = read_image_folder(args.imgfolder,args.maskfolder,do_MIND=False)

    unet_model,heatmap,mesh = get_vxmpp_models()
    
    #read corrfield supervision
    corrfield_all = []
    for ii in trange(len(case_list)):
        i = int(case_list[ii].split('case_')[1].split('.csv')[0])
        cf = torch.from_numpy(np.loadtxt('corrfieldTr/case_'+str(i).zfill(3)+'.csv',delimiter=',')).float()

        corrfield_all.append(cf)

    for repeat in range(2):
        num_iterations = 4*4900
        optimizer = torch.optim.Adam(list(unet_model.parameters())+list(heatmap.parameters()),lr=0.001)#0.001
        scaler = torch.cuda.amp.GradScaler()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,4*700,0.5)
        t0 = time.time()
        run_tre = torch.empty(0,1); run_tre_test = torch.empty(0,1); 
        run_loss = torch.zeros(num_iterations)

        with tqdm(total=num_iterations, file=sys.stdout) as pbar:


            for i in range(num_iterations):

                with torch.no_grad():
                    ii = torch.randperm(len(img_insp_all))[0]#
                    
                    fixed_img = img_insp_all[ii]

                    moving_img = img_exp_all[ii]

                    H,W,D = fixed_img.shape[-3:]

                    cf = corrfield_all[ii]

                    #halfres keypts
                    keypts_fix = torch.flip((cf[:,:3]-torch.tensor([H,W,D]))/torch.tensor([H,W,D]),(1,)).cuda()
                    keypts_mov = torch.flip((cf[:,3:]-torch.tensor([H,W,D]))/torch.tensor([H,W,D]),(1,)).cuda()

                    #Affine augmentation of images *and* keypoints 
                    if(i%2==0):
                        A = (torch.randn(3,4)*.035+torch.eye(3,4)).cuda()
                        affine = F.affine_grid(A.unsqueeze(0),(1,1,H,W,D))
                        keypts_fix = torch.linalg.solve(torch.cat((A,torch.tensor([0,0,0,1]).cuda().view(1,-1)),0),\
                                        torch.cat((keypts_fix,torch.ones(keypts_fix.shape[0],1).cuda()),1).t()).t()[:,:3]
                        fixed_img = F.grid_sample(fixed_img.view(1,1,H,W,D).cuda(),affine)
                    else:
                        fixed_img = fixed_img.view(1,1,H,W,D).cuda()

                    if(i%2==1):
                        A = (torch.randn(3,4)*.035+torch.eye(3,4)).cuda()
                        affine = F.affine_grid(A.unsqueeze(0),(1,1,H,W,D))
                        keypts_mov = torch.linalg.solve(torch.cat((A,torch.tensor([0,0,0,1]).cuda().view(1,-1)),0),\
                                        torch.cat((keypts_mov,torch.ones(keypts_mov.shape[0],1).cuda()),1).t()).t()[:,:3]
                        moving_img = F.grid_sample(moving_img.view(1,1,H,W,D).cuda(),affine)
                    else:
                        moving_img = moving_img.view(1,1,H,W,D).cuda()
                    disp_gt = keypts_mov-keypts_fix

                    scheduler.step()
                    optimizer.zero_grad()
                    idx = torch.randperm(keypts_fix.shape[0])[:1024]

                    with torch.cuda.amp.autocast():
                        #VoxelMorph requires some padding
                        input,x_start,y_start,z_start,x_end,y_end,z_end = return_crops(torch.cat((fixed_img,moving_img),1).cuda())
                        #input = F.interpolate(input,scale_factor=0.5,mode='trilinear')
                #end of no grad
                with torch.cuda.amp.autocast():

                    output = F.pad(F.interpolate(unet_model(input),scale_factor=2,mode='trilinear'),(z_start,(-z_end+D),y_start,(-y_end+W),x_start,(-x_end+H)))
                    #output = unet_model(input)
                    sample_xyz = keypts_fix[idx]#*torch.tensor([D,W,H]).cuda()/torch.tensor([320,256,320]).cuda()#keypts_all_fix[int(ii)][idx]#fix
                    #todo nearest vs bilinear
                    #sampled = F.grid_sample(output,sample_xyz.cuda().view(1,-1,1,1,3)+patch.view(1,1,-1,1,3),mode='bilinear')
                    sampled = F.grid_sample(output,sample_xyz.cuda().view(1,-1,1,1,3),mode='bilinear')
                    #disp_pred = heatmap(sampled.permute(2,1,0,3,4).view(512,-1,3,3,3))
                    disp_pred = heatmap(sampled.permute(2,1,0,3,4))


                    pred_xyz = torch.sum(torch.softmax(disp_pred.view(-1,11**3,1),1)*mesh.view(1,11**3,3),1)
                    loss = (pred_xyz-disp_gt[idx]).mul(torch.tensor([D,W,H]).float().cuda()).pow(2).sum(-1).sqrt().mean()


                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                run_loss[i] = loss.item()


                str1 = f"iter: {i}, loss: {'%0.3f'%(run_loss[i-28:i-1].mean())}, runtime: {'%0.3f'%(time.time()-t0)} sec, GPU max/memory: {'%0.2f'%(torch.cuda.max_memory_allocated()*1e-9)} GByte"
                pbar.set_description(str1)
                pbar.update(1)

        if(repeat==0):
            torch.save([heatmap.state_dict(),unet_model.state_dict(),run_loss],'registration_models/voxelmorphplusplus0.pth')
        else:
            torch.save([heatmap.state_dict(),unet_model.state_dict(),run_loss],args.outfile)        



        
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'training of VoxelMorph++ on Lung250M-4B')

    parser.add_argument('-m',  '--maskfolder',   default='masksTr', help="mask folder containing (/case_???_{1,2}.nii.gz)")
    parser.add_argument('-I',  '--imgfolder',    default='imagesTr', help="image folder containing (/case_???_{1,2}.nii.gz)")
    parser.add_argument('-O',  '--outfile',      default='registration_models/voxelmorphplusplus.pth', help="output file for trained model")

    #args = parser.parse_args(args=[])
    args = parser.parse_args()
    print(args)
    main(args)






