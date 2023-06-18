#!/usr/bin/env python
import torch
import argparse

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys
from glob import glob
import copy

sys.path.insert(0, 'corrfield/')

from thin_plate_spline import thin_plate_dense

#torch.backends.cuda.matmul.allow_tf32 = False
#torch.backends.cudnn.allow_tf32 = False

def main(args):
    base_img = args.imgfolder
    img_list = [i.split('/')[-1] for i in sorted(glob(base_img + '/*.nii.gz'))]

    img_list1 = [i.replace('_1.nii.gz', '').replace('_2.nii.gz', '') for i in img_list]
    if (len(img_list1) != 2 * len(set(img_list1))):
        raise Exception("Not all images seem to have both _1.nii.gz and _2.nii.gz")
    case_list = sorted(list(set(img_list1)))

    aggregate_val = []
    aggregate_test = []
    
    for i in range(len(case_list)):
        case = str(case_list[i])

        img_fixed = torch.from_numpy(nib.load(base_img + '/' + case + '_1.nii.gz').get_fdata()).float()
        H, W, D = img_fixed.shape
        cf = torch.from_numpy(np.loadtxt(args.outfolder + '/' + case + '.csv', delimiter=',')).float()

        kpts_fixed = torch.flip((cf[:, :3] - torch.tensor([H / 2, W / 2, D / 2]).view(1, -1)).div(
            torch.tensor([H / 2, W / 2, D / 2]).view(1, -1)), (-1,))
        kpts_moving = torch.flip((cf[:, 3:] - torch.tensor([H / 2, W / 2, D / 2]).view(1, -1)).div(
            torch.tensor([H / 2, W / 2, D / 2]).view(1, -1)), (-1,))

        with torch.no_grad():
            dense_flow = thin_plate_dense(kpts_fixed.unsqueeze(0).cuda(),
                                          (kpts_moving - kpts_fixed).unsqueeze(0).cuda(), (H, W, D), 4, 0.01).cpu()

        cf_mean = cf.mean(0, keepdim=True)
        cf_std = cf.std(0, keepdim=True)

        cf_aff = (cf[:, :3] - cf_mean[:, :3]) * cf_std[:, 3:] / cf_std[:, :3] + cf_mean[:, 3:]
        tre_aff = (cf_aff - cf[:, 3:]).pow(2).sum(-1).sqrt()

        ii = int(case.split('case_')[1])

        lms_validation = torch.load(args.validation)
        lms1 = torch.flip((lms_validation[str(ii)][:, :3] - torch.tensor([H / 2, W / 2, D / 2])) / torch.tensor(
            [H / 2, W / 2, D / 2]), (1,))
        lms_disp1 = F.grid_sample(dense_flow.permute(0, 4, 1, 2, 3).cpu(), lms1.view(1, -1, 1, 1, 3)).squeeze().t()
        lms_disp = torch.flip(lms_disp1, (1,)) * torch.tensor([H / 2, W / 2, D / 2])

        tre0 = (lms_validation[str(ii)][:, :3] - lms_validation[str(ii)][:, 3:]).pow(2).sum(-1).sqrt()

        tre2 = (lms_validation[str(ii)][:, :3] + lms_disp - lms_validation[str(ii)][:, 3:]).pow(2).sum(-1).sqrt()

        print('tre0', '%0.3f' % tre0.mean(), 'tre_aff', '%0.3f' % tre_aff.mean(), 'tre2', '%0.3f' % tre2.mean())
        if((i>=104)&(i<=113)):
            aggregate_test.append(tre2)
        else:
            aggregate_val.append(tre2)
    agg_test = torch.cat(aggregate_test)
    agg_val = torch.cat(aggregate_val)

    print('======= SUMMARY RESULTS ======')
    print("Validation (see Supplement Tab. 3)")
    print('mean: '+str('%0.2f'%agg_val.mean()),' 25: '+str('%0.2f'%agg_val.quantile(.25)),' 50%: '+str('%0.2f'%agg_val.median()),' 25%: '+str('%0.2f'%agg_val.quantile(.75)))
    print("Test (see main paper Tab. 2)")
    print('mean: '+str('%0.2f'%agg_test.mean()),' 25%: '+str('%0.2f'%agg_test.quantile(.25)),' 50%: '+str('%0.2f'%agg_test.median()),' 25%: '+str('%0.2f'%agg_test.quantile(.75)))

                                                                                                                                                                            
                                                                                                            
              
              

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='evaluate Lung250M-4B registrations')
    parser.add_argument('-I', '--imgfolder', default='imagesTs',
                        help="image folder containing (/case_???_{1,2}.nii.gz)")
    parser.add_argument('-o', '--outfolder', default='None',
                        help="output folder for individual keypoint displacement predictions")
    parser.add_argument('-v', '--validation', default='None', help="pth file with validation landmarks")

    args = parser.parse_args()

    main(args)