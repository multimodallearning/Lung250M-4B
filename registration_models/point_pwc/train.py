import argparse
import time
import os
import numpy as np

from defaults import get_cfg_defaults
from dataset import Lung250MDataset
from ppwc import PointConvSceneFlowPWC8192, multiScaleLoss


def train(cfg, args):
    root = cfg.BASE_DIRECTORY
    exp_name = cfg.EXPERIMENT_NAME
    out_folder = os.path.join(root, exp_name)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    model_path = os.path.join(out_folder, 'model.pth')
    model_path_ep = os.path.join(out_folder, 'model_ep={}.pth')


    # hyperparameters
    init_lr = cfg.SOLVER.LEARNING_RATE
    num_epochs = cfg.SOLVER.NUM_EPOCHS
    lr_steps = cfg.SOLVER.LR_MILESTONES
    lr_gamma = cfg.SOLVER.LR_LAMBDA
    batch_size = cfg.SOLVER.BATCH_SIZE

    # computational stuff
    use_amp = cfg.USE_AMP
    num_workers = 0 if args.debug else cfg.NUM_WORKERS
    device = cfg.DEVICE

    # model
    model = PointConvSceneFlowPWC8192(cfg)
    model.to(device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), init_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if cfg.SOLVER.SCHEDULER == 'multistep':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, lr_steps, lr_gamma)
    else:
        raise ValueError()


    # datasets
    train_set = Lung250MDataset(cfg, args, phase='train', split='train')
    if args.debug:
        train_set.case_list = train_set.case_list[:8]
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)
    val_set = Lung250MDataset(cfg, args, phase='test', split='val')
    if args.debug:
        val_set.case_list = val_set.case_list[:8]
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False)

    # logging
    validation_log = np.zeros([num_epochs, 3])

    for ep in range(1, num_epochs + 1):
        print('Started epoch {}/{}'.format(ep, num_epochs))
        model.train()
        loss_values = []
        start_time = time.time()

        for it, data in enumerate(train_loader, 1):
            pcd_src, pcd_tgt, gt_flow, idx = data
            pcd_src = pcd_src.to(device)
            pcd_tgt = pcd_tgt.to(device)
            gt_flow = gt_flow.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred_flows, fps_pc1_idxs, _, _, _ = model(pcd_src, pcd_tgt, pcd_src, pcd_tgt)
                loss = multiScaleLoss(pred_flows, gt_flow, fps_pc1_idxs)

                loss_values.append(loss.item())
                loss = loss * cfg.SOLVER.LOSS_FACTOR
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        train_loss = np.mean(loss_values)
        validation_log[ep - 1, 0] = train_loss
        lr_scheduler.step()

        # Validation
        model.eval()
        epe_3d = 0
        epe_initial = 0
        for it, data in enumerate(val_loader, 1):
            pcd_src, pcd_tgt, gt_flow, idx = data
            pcd_src = pcd_src.to(device)
            pcd_tgt = pcd_tgt.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                with torch.no_grad():
                    pred_flows, _, _, _, _ = model(pcd_src, pcd_tgt, pcd_src, pcd_tgt)
                    pred_flow = pred_flows[0].permute(0,2,1)

                gt_flow = gt_flow.to(device)
                err_per_sample = (pred_flow - gt_flow).square().sum(dim=2).sqrt().mean(dim=1)
                epe_3d += err_per_sample.sum().item()
                epe_initial += gt_flow.square().sum(dim=2).sqrt().mean(dim=1).sum().item()

        epe_3d = epe_3d / len(val_loader.dataset) * val_loader.dataset.norm_factor
        epe_initial = epe_initial / len(val_loader.dataset) * val_loader.dataset.norm_factor
        validation_log[ep - 1, 1:] = [epe_initial, epe_3d]

        end_time = time.time()
        print('epoch', ep, 'duration', '%0.3f' % ((end_time - start_time) / 60.), 'train_loss', '%0.6f' % train_loss,
              'initial error', epe_initial, 'EPEs', epe_3d)

        np.save(os.path.join(out_folder, "validation_history.npy"), validation_log)
        torch.save(model.state_dict(), model_path)
        if ep % cfg.SOLVER.CHECKPOINT_INTERVAL == 0:
            torch.save(model.state_dict(), model_path_ep.format(ep))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument('--config', default='config_ppwc_syn.yaml',
                        help="config file of the model (yaml)")
    parser.add_argument("--debug", default=False, help="whether to use debug mode", type=bool)
    parser.add_argument("--gpu", default="0", help="gpu to train on")
    parser.add_argument('-CTr', '--cloudfolder_train', default='cloudsTr',
                        help="folder containing (/case_???_{1,2}.pth)")
    parser.add_argument('-CVal', '--cloudfolder_val', default='cloudsTs',
                        help="folder containing (/case_???_{1,2}.pth)")

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg.freeze()

    train(cfg, args)
