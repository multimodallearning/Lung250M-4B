import numpy as np
import torch
import torch.utils.data
import open3d as o3d


class Lung250MDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, args, phase, split):
        self.is_train = True if phase == 'train' else False
        self.split = split

        self.pcd_template = '/share/data_zoe3/falta/temp/vessel_data/full/vessel_cloud_new/case_{:03d}_{}.pth'
        self.gt_template = '/share/data_supergrover3/heinrich/temp/neurips_full/case_{:03d}.pth'
        self.idx_16k = torch.load('all_ind_16384_train.pth', map_location='cpu')

        if split == 'train':
            val_cases = np.array([2, 8, 54, 55, 56, 94, 97])
            self.case_list = np.arange(104)
            self.case_list = self.case_list[~np.isin(self.case_list, val_cases)]
        elif split == 'val':
            self.case_list = np.array([2, 8, 94, 97])
        else:
            raise NotImplementedError()

        self.norm_factor = cfg.INPUT.SCALE_NORM_FACTOR

        # augmentation parameters
        self.augm_setting = cfg.AUGMENTATIONS

    def __getitem__(self, idx):
        # load input pcds
        case = self.case_list[idx]
        pcd_tgt = torch.load(self.pcd_template.format(case, 1))[2]
        pcd_src = torch.load(self.pcd_template.format(case, 2))[2]
        idx_16k_tgt = self.idx_16k['all_ind_fix'][case]
        idx_16k_src = self.idx_16k['all_ind_mov'][case]
        pcd_tgt = pcd_tgt[idx_16k_tgt].float().numpy()
        pcd_src = pcd_src[idx_16k_src].float().numpy()
        corrfield_flow = torch.load(self.gt_template.format(case))['cloud_gt_mov']
        corrfield_flow = corrfield_flow[idx_16k_src].float().numpy()
        lm_src = pcd_src.copy()
        lm_tgt = corrfield_flow + lm_src

        # prealignment
        mean_tgt = np.mean(pcd_tgt, axis=0)
        std_tgt = np.std(pcd_tgt, axis=0)
        mean_src = np.mean(pcd_src, axis=0)
        std_src = np.std(pcd_src, axis=0)
        pcd_src = (pcd_src - mean_src) * std_tgt / std_src + mean_tgt
        lm_src = (lm_src - mean_src) * std_tgt / std_src + mean_tgt

        # mean center and scale
        mean = np.mean(pcd_tgt, axis=0)
        pcd_tgt = (pcd_tgt - mean) / self.norm_factor
        pcd_src = (pcd_src - mean) / self.norm_factor
        lm_tgt = (lm_tgt - mean) / self.norm_factor
        lm_src = (lm_src - mean) / self.norm_factor
        gt_flow = lm_tgt - lm_src

        if self.is_train:
            if self.augm_setting.METHOD == 'multiscale_local_global':
                if np.random.uniform() < 0.5:
                    pcd = pcd_src
                else:
                    pcd = pcd_tgt

                setting = self.augm_setting
                num_control_points_local = setting.NUM_CONTROL_POINTS_LOCAL
                max_control_shift_local = setting.MAX_CONTROL_SHIFT_LOCAL
                kernel_std_local = setting.KERNEL_STD_LOCAL
                global_grid_spacing = setting.GLOBAL_GRID_SPACING
                max_control_shift_global = setting.MAX_CONTROL_SHIFT_GLOBAL
                kernel_std_global = setting.KERNEL_STD_GLOBAL

                local_control_idx = np.random.permutation(pcd.shape[0])[:num_control_points_local]
                local_control_shifts = np.random.uniform(-1., 1., (num_control_points_local, 3)) * max_control_shift_local
                local_control_pts = pcd[local_control_idx]
                sq_dist = np.sum(np.square(pcd[:, None] - local_control_pts[None]), axis=2)
                weights = np.exp(-0.5 * sq_dist / kernel_std_local ** 2)
                local_pcd_shifts = np.sum(weights[:, :, None] * local_control_shifts[None], axis=1) / np.sum(weights[:, :, None], axis=1)
                local_pcd_shifts = np.nan_to_num(local_pcd_shifts)
                pcd_augm = pcd + local_pcd_shifts

                o3d_cloud = o3d.geometry.PointCloud()
                o3d_cloud.points = o3d.utility.Vector3dVector(pcd_augm)
                o3d_cloud, _, _ = o3d_cloud.voxel_down_sample_and_trace(global_grid_spacing,
                                                                        min_bound=np.array([-10., -10., -10.]),
                                                                        max_bound=np.array([10., 10., 10.]))

                global_control_pts = np.float32(np.asarray(o3d_cloud.points))
                global_control_shifts = np.random.uniform(-1, 1., (
                global_control_pts.shape[0], 3)) * max_control_shift_global
                sq_dist = np.sum(np.square(pcd_augm[:, None] - global_control_pts[None]), axis=2)
                weights = np.exp(-0.5 * sq_dist / kernel_std_global ** 2)
                global_pcd_shifts = np.sum(weights[:, :, None] * global_control_shifts[None], axis=1) / np.sum(
                    weights[:, :, None], axis=1)

                pcd_augm = pcd_augm + global_pcd_shifts

                gt_flow = pcd - pcd_augm
                permutation = np.random.permutation(16384)
                pcd_src = pcd_augm[permutation[:8192]]
                gt_flow = gt_flow[permutation[:8192]]
                pcd_tgt = pcd[permutation[8192:]]

            elif self.augm_setting.METHOD == 'rigid_one':
                setting = self.augm_setting
                max_transl = setting.MAX_TRANSLATION
                scale_offset = setting.MAX_SCALE_OFFSET
                rot_max = setting.MAX_ROTATION_ANGLE
                transl = np.random.uniform(-1., 1., (1, 3)) * max_transl
                scale = np.random.uniform(1 - scale_offset, 1 + scale_offset, (1, 3))
                rot_angles = np.deg2rad(np.random.uniform(-rot_max, rot_max, 3))

                theta = rot_angles[0]
                rot_mat_x = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
                theta = rot_angles[1]
                rot_mat_y = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
                theta = rot_angles[2]
                rot_mat_z = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
                rot_mat = np.dot(np.dot(rot_mat_x, rot_mat_y), rot_mat_z)

                if np.random.uniform() < 0.5:
                    # augment only one cloud
                    pcd_src = np.dot(pcd_src, rot_mat) * scale + transl
                    lm_src = np.dot(lm_src, rot_mat) * scale + transl
                    gt_flow = lm_tgt - lm_src

                    permutation = np.random.permutation(16384)
                    pcd_src = pcd_src[permutation[:8192]]
                    gt_flow = gt_flow[permutation[:8192]]
                    pcd_tgt = pcd_tgt[permutation[8192:]]

                else:
                    pcd_src = np.dot(pcd_src, rot_mat) * scale + transl
                    lm_src = np.dot(lm_src, rot_mat) * scale + transl
                    pcd_tgt = np.dot(pcd_tgt, rot_mat) * scale + transl
                    lm_tgt = np.dot(lm_tgt, rot_mat) * scale + transl
                    gt_flow = lm_tgt - lm_src

                    permutation = np.random.permutation(16384)
                    pcd_src = pcd_src[permutation[:8192]]
                    gt_flow = gt_flow[permutation[:8192]]
                    pcd_tgt = pcd_tgt[permutation[8192:]]

            else:
                pcd_src = pcd_src[:8192]
                pcd_tgt = pcd_tgt[:8192]
                gt_flow = gt_flow[:8192]

        else:
            pcd_src = pcd_src[:8192]
            pcd_tgt = pcd_tgt[:8192]
            gt_flow = gt_flow[:8192]

        return np.float32(pcd_src), np.float32(pcd_tgt), np.float32(gt_flow), idx

    def __len__(self):
        return len(self.case_list)
