import torch
import numpy as np
import torch.utils.data as data
import os
import json
import copy
import torch.distributed as dist
from random import random

from .dataset_utils import augment_cloud

class TreeBench10K_upsampling(data.Dataset):
    def __init__(self, 
                 args, 
                 train, 
                 phase,
                 augmentation=False,
                 return_augmentation_params=False,
                ):
        self.category_file = args['CATEGORY_FILE_PATH']
        self.upsampling_path = args['data_dir']
        self.complete_npoints = args['npoints'] * args['R']
        self.partial_npoints = args['npoints']
        self.subset = ''
        self.augmentation = augmentation
        self.return_augmentation_params = return_augmentation_params
        self.train = train
        self.subset = phase

        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())
        
        self.set_fold()
        
        self.labels = np.full(shape=(len(self.file_list),), fill_value=args['R'] -1, dtype=np.int64)

    def set_fold(self):
        self.file_list = self._get_file_list(self.subset)
        if not dist.is_initialized() or dist.get_rank() == 0:
            print('Complete collecting files of the dataset. Total files: %d' % len(self.file_list))

    def _get_file_list(self, subset):     
        return self.dataset_categories.get(subset)
    
    def __getitem__(self, index):
        complete_pc = np.load(os.path.join(self.upsampling_path, self.file_list[index], str(self.complete_npoints) + "_processed.npy"))
        partial_pc = np.load(os.path.join(self.upsampling_path, self.file_list[index], str(self.partial_npoints) + "_processed.npy"))
        
        result = {}
        result['partial'] = partial_pc
        result['complete'] = complete_pc
        
        # augment the point clouds
        if (isinstance(self.augmentation, dict) and self.train):
            result_list = list(result.values())
            if self.return_augmentation_params:
                result_list, augmentation_params = augment_cloud(
                    result_list,
                    self.augmentation,
                    return_augmentation_params=True
                )
            else:
                result_list = augment_cloud(
                    result_list,
                    self.augmentation,
                    return_augmentation_params=False
                )
            for idx, key in enumerate(result.keys()):
                result[key] = result_list[idx]

        if self.return_augmentation_params:
            for key in augmentation_params.keys():
                result[key] = augmentation_params[key]
        for key in result.keys():
            result[key] = torch.from_numpy(result[key])
        result['label'] = self.labels[index]

        if (not self.train):
            result['name'] = copy.deepcopy(self.file_list[index])
            result['mode'] = 'upsampling'
            result['view'] = -1
            result['sample_key'] = result['name']
            # complete_pc_raw = np.load(os.path.join(self.upsampling_path, self.file_list[index], str(self.complete_npoints) + "_gt.npy"))
            with open(os.path.join(self.upsampling_path, self.file_list[index], "metadata.json"), 'r') as f:
                metadata = json.load(f)
            # result['complete_raw'] = torch.from_numpy(complete_pc_raw)
            result['metadata'] = metadata
            
        return result

    def __len__(self):
        return len(self.file_list)


class TreeBench10K_completion(data.Dataset):
    def __init__(self, 
                 args, 
                 train, 
                 phase,
                 augmentation=False,
                 return_augmentation_params=False,
                ):
        self.category_file = args['CATEGORY_FILE_PATH']
        self.path = args['data_dir']
        self.complete_npoints = args['npoints'] * args['R']
        self.partial_npoints = args['npoints']
        self.subset = ''
        self.augmentation = augmentation
        self.return_augmentation_params = return_augmentation_params
        self.train = train
        self.subset = phase

        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())
        
        self.set_fold()
        if self.train:
            self.labels = np.full(shape=(len(self.file_list),), fill_value=args['R'] -1, dtype=np.int64)
        else:
            self.labels = np.full(shape=(len(self.file_list)*8,), fill_value=args['R'] -1, dtype=np.int64)

    def set_fold(self):
        self.file_list = self._get_file_list(self.subset)
        if not dist.is_initialized() or dist.get_rank() == 0:
            print('Complete collecting files of the dataset. Total files: %d' % len(self.file_list))

    def _get_file_list(self, subset):     
        return self.dataset_categories.get(subset)
    
    def __getitem__(self, index):
        if not self.train:
            complete_pc = np.load(os.path.join(self.path, self.file_list[index // 8], str(self.complete_npoints) + "_complete.npy"))
            view_num = index % 8
            if view_num >= 4:
                partial_pc = np.load(os.path.join(self.path, self.file_list[index // 8], str(self.partial_npoints) + "_tls_" + str(view_num-4) + ".npy"))
            else:
                partial_pc = np.load(os.path.join(self.path, self.file_list[index // 8], str(self.partial_npoints) + "_uls_" + str(view_num) + ".npy"))
        else:
            complete_pc = np.load(os.path.join(self.path, self.file_list[index], str(self.complete_npoints) + "_complete.npy"))
            metadata = json.load(open(os.path.join(self.path, self.file_list[index], "metadata.json"), 'r'))
            scale, center = metadata['scale'], np.array(metadata['center'])
            complete_pc_raw = complete_pc * scale + center
            if random() < 0.5:
                partial_pc_raw = generate_tls_task(complete_pc_raw, self.partial_npoints)
            else:
                partial_pc_raw = generate_uls_task(complete_pc_raw, self.partial_npoints)
            partial_pc = (partial_pc_raw - center) / scale
            
        
        result = {}
        result['partial'] = partial_pc.astype(np.float32)
        result['complete'] = complete_pc.astype(np.float32)
        
        # augment the point clouds
        if (isinstance(self.augmentation, dict) and self.train):
            result_list = list(result.values())
            if self.return_augmentation_params:
                result_list, augmentation_params = augment_cloud(
                    result_list,
                    self.augmentation,
                    return_augmentation_params=True
                )
            else:
                result_list = augment_cloud(
                    result_list,
                    self.augmentation,
                    return_augmentation_params=False
                )
            for idx, key in enumerate(result.keys()):
                result[key] = result_list[idx]

        if self.return_augmentation_params:
            for key in augmentation_params.keys():
                result[key] = augmentation_params[key]
        for key in result.keys():
            result[key] = torch.from_numpy(result[key])
        result['label'] = self.labels[index]

        if (not self.train):
            result['name'] = copy.deepcopy(self.file_list[index // 8])
            result['mode'] = 'tls' if view_num >= 4 else 'uls'
            result['view'] = view_num - 4 if view_num >= 4 else view_num
            result['sample_key'] = '{}|{}|{}'.format(
                result['name'], result['mode'], result['view'])
            # complete_pc_raw = np.load(os.path.join(self.upsampling_path, self.file_list[index], str(self.complete_npoints) + "_gt.npy"))
            with open(os.path.join(self.path, self.file_list[index // 8], "metadata.json"), 'r') as f:
                metadata = json.load(f)
            # result['complete_raw'] = torch.from_numpy(complete_pc_raw)
            result['metadata'] = metadata
            
        return result

    def __len__(self):
        if self.train:
            return len(self.file_list)
        else:
            return len(self.file_list) * 8  # each sample will be augmented 8 times, so the length is multiplied by 8


import numpy as np
def get_tree_bbox(points):
    min_pt = np.min(points, axis=0)
    max_pt = np.max(points, axis=0)
    center = (min_pt + max_pt) / 2
    height = max_pt[2] - min_pt[2]
    return min_pt, max_pt, center, height

def project_and_cull(points, viewpoint, res, mode='spherical'):
    local_pc = points - viewpoint
    dist = np.linalg.norm(local_pc, axis=1)
    
    if mode == 'spherical':
        u = np.arctan2(local_pc[:, 1], local_pc[:, 0])
        v = np.arccos(local_pc[:, 2] / (dist + 1e-8))
    else:
        u = local_pc[:, 0]
        v = local_pc[:, 1]

    u_min, u_max = u.min(), u.max()
    v_min, v_max = v.min(), v.max()
    
    if u_max - u_min < 1e-6 or v_max - v_min < 1e-6:
        return np.arange(len(points))

    u_idx = ((u - u_min) / (u_max - u_min + 1e-8) * (res - 1)).astype(np.int32)
    v_idx = ((v - v_min) / (v_max - v_min + 1e-8) * (res - 1)).astype(np.int32)
    
    grid_keys = u_idx.astype(np.int64) * res + v_idx.astype(np.int64)
    sort_idx = np.lexsort((dist, grid_keys))
    mask = np.concatenate(([True], grid_keys[sort_idx][1:] != grid_keys[sort_idx][:-1]))
    return sort_idx[mask]

def clip_line_to_bbox(start, direction, min_xy, max_xy):
    t_min, t_max = -1e9, 1e9
    for i in range(2): 
        if abs(direction[i]) < 1e-6:
            if start[i] < min_xy[i] or start[i] > max_xy[i]:
                return None 
        else:
            t1 = (min_xy[i] - start[i]) / direction[i]
            t2 = (max_xy[i] - start[i]) / direction[i]
            
            if t1 > t2: t1, t2 = t2, t1
            
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
    
    if t_min > t_max:
        return None
    
    return t_min, t_max

def adaptive_sample_multi_view(points, viewpoints, target_n, mode='spherical'):
    res = 256
    visible_indices_set = set()
    
    for vp in viewpoints:
        idx = project_and_cull(points, vp, res, mode)
        visible_indices_set.update(idx)
        
    visible_indices = np.array(list(visible_indices_set))
    visible_points = points[visible_indices]
    
    current_n = len(visible_points)
    
    if current_n >= target_n:
        choice_idx = np.random.choice(current_n, target_n, replace=False)
        return visible_points[choice_idx]
    else:
        choice_idx = np.random.choice(current_n, target_n, replace=True)
        return visible_points[choice_idx]

def generate_tls_task(pc_gt, target_n):
    z_min = np.min(pc_gt[:, 2])
    z_max = np.max(pc_gt[:, 2])
    tree_height = z_max - z_min
    center_xy = np.mean(pc_gt[:, :2], axis=0)
    angle = np.random.uniform(0, 2 * np.pi)
    dist = np.random.uniform(0.5, 1.0) * tree_height
    scanner_x = center_xy[0] + dist * np.cos(angle)
    scanner_y = center_xy[1] + dist * np.sin(angle)
    scanner_z = z_min + np.random.uniform(1.5, 2.0)
    
    viewpoint = np.array([scanner_x, scanner_y, scanner_z])
    
    return adaptive_sample_multi_view(pc_gt, [viewpoint], target_n, mode='spherical')

def generate_uls_task(pc_gt, target_n):
    min_pt, max_pt, center, tree_height = get_tree_bbox(pc_gt)
    z_min = min_pt[2]

    flight_z = z_min + np.random.uniform(60.0, 75.0)

    offset = np.random.uniform(-0.5, 0.5, size=2)
    pivot_xy = center[:2] + offset
    
    angle = np.random.uniform(0, 2 * np.pi)
    direction = np.array([np.cos(angle), np.sin(angle)])
    t_bounds = clip_line_to_bbox(pivot_xy, direction, min_pt[:2], max_pt[:2])
    
    if t_bounds is None:
        viewpoints = [np.array([center[0], center[1], flight_z])]
    else:
        t_start, t_end = t_bounds
        step = 0.1
        t_values = np.arange(t_start, t_end, step)
        if len(t_values) > 10:
            key_t_indices = np.linspace(0, len(t_values)-1, 10).astype(int)
            selected_t = t_values[key_t_indices]
        else:
            selected_t = t_values
            
        viewpoints = []
        for t in selected_t:
            vp_x = pivot_xy[0] + t * direction[0]
            vp_y = pivot_xy[1] + t * direction[1]
            viewpoints.append(np.array([vp_x, vp_y, flight_z]))
    return adaptive_sample_multi_view(pc_gt, viewpoints, target_n, mode='ortho')
