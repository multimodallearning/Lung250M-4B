import argparse
import os

from defaults import get_cfg_defaults
from ppwc import PointConvSceneFlowPWC8192


def main(args):

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg.freeze()

    # computational stuff
    use_amp = cfg.USE_AMP
    device = 'cuda'

    # model
    model = PointConvSceneFlowPWC8192(cfg)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)

    # data
    idx_8k = torch.load('all_ind_8192.pth', map_location='cpu')
    cases = [2, 8, 54, 55, 56, 94, 97, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113]
    out_dict = {'predictions': [],
                'case_list': cases,
                'keypts_src': []}

    # Inference
    model.eval()

    for i, case in enumerate(cases):
        pcd_tgt = torch.load(os.path.join(args.cloudfolder,'case_{:03d}_{}.pth'.format(case, 1)))[2].float()
        pcd_src = torch.load(os.path.join(args.cloudfolder,'case_{:03d}_{}.pth'.format(case, 2)))[2].float()
        idx_8k_tgt = idx_8k['all_ind_fix'][case]
        idx_8k_src = idx_8k['all_ind_mov'][case]
        pcd_tgt = pcd_tgt[idx_8k_tgt]
        pcd_src = pcd_src[idx_8k_src]
        pcd_tgt = pcd_tgt.unsqueeze(0).to(device)
        pcd_src = pcd_src.unsqueeze(0).to(device)

        # prealignment
        pcd_src_orig = pcd_src.clone()
        mean_tgt = torch.mean(pcd_tgt, dim=1)
        std_tgt = torch.std(pcd_tgt, dim=1)
        mean_src = torch.mean(pcd_src, dim=1)
        std_src = torch.std(pcd_src, dim=1)
        pcd_src = (pcd_src - mean_src) * std_tgt / std_src + mean_tgt
        pre_align_flow = pcd_src - pcd_src_orig

        # mean center and scale
        norm_factor = cfg.INPUT.SCALE_NORM_FACTOR
        mean = torch.mean(pcd_tgt, axis=1)
        pcd_tgt = (pcd_tgt - mean) / norm_factor
        pcd_src = (pcd_src - mean) / norm_factor

        # inference
        with torch.cuda.amp.autocast(enabled=use_amp):
            with torch.no_grad():
                pred_flows, _, _, _, _ = model(pcd_src, pcd_tgt, pcd_src, pcd_tgt)
                pred_flow = pred_flows[0].permute(0,2,1)

        pred_flow = pred_flow * cfg.INPUT.SCALE_NORM_FACTOR + pre_align_flow
        out_dict['predictions'].append(pred_flow.cpu())
        out_dict['keypts_src'].append(pcd_src_orig.cpu())

    torch.save(out_dict, args.outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='inference of PointPWC-Net on Lung250M-4B')

    parser.add_argument('-M', '--model', default='ppwc_syn.pth', help="model file (pth)")
    parser.add_argument('-C', '--cloudfolder', default='cloudsTs',
                        help="folder containing (/case_???_{1,2}.nii.gz)")
    parser.add_argument('-O', '--outfile', default='predictions.pth',
                        help="output file for keypoint displacement predictions")
    parser.add_argument('--config', default='config_ppwc_sup.yaml',
                        help="config file of the model (yaml)")
    parser.add_argument("--gpu", default="0", help="gpu to train on")

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import torch

    print(args)
    main(args)
