import torch
import numpy as np
import torch.utils.data as data
import os
import json
import copy
import torch.distributed as dist
from .build import DATASETS
from .dataset_utils import augment_cloud
from random import random


@DATASETS.register_module()
class TreeBench_completion(data.Dataset):
    def __init__(self, 
                 config
                ):
        self.category_file = config.CATEGORY_FILE_PATH
        self.path = config.data_dir
        self.complete_npoints = config.npoints * config.R
        self.partial_npoints = config.npoints
        self.subset = config.subset
        self.augmentation = config.augmentation
        self.train = True if self.subset == 'train' else False

        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())
        
        self.set_fold()

    def set_fold(self):
        self.file_list = self._get_file_list(self.subset)

        is_main = True
        if dist.is_available() and dist.is_initialized():
            is_main = (dist.get_rank() == 0)

        if is_main:
            print(
                'Complete collecting files of the dataset. Total files: %d'
                % len(self.file_list)
            )

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
            result_list = augment_cloud(
                result_list,
                self.augmentation,
                return_augmentation_params=False
            )
            for idx, key in enumerate(result.keys()):
                result[key] = result_list[idx]

        for key in result.keys():
            result[key] = torch.from_numpy(result[key])

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
    """动态计算树的边界盒"""
    min_pt = np.min(points, axis=0)
    max_pt = np.max(points, axis=0)
    center = (min_pt + max_pt) / 2
    height = max_pt[2] - min_pt[2]
    return min_pt, max_pt, center, height

def project_and_cull(points, viewpoint, res, mode='spherical'):
    """
    核心投影函数 (保持不变，但能够处理任意尺度的坐标)
    """
    local_pc = points - viewpoint
    dist = np.linalg.norm(local_pc, axis=1)
    
    if mode == 'spherical':
        # TLS: 球坐标投影
        u = np.arctan2(local_pc[:, 1], local_pc[:, 0])
        v = np.arccos(local_pc[:, 2] / (dist + 1e-8))
    else:
        # ULS: 正射投影 (Top-down)
        u = local_pc[:, 0]
        v = local_pc[:, 1]
    
    # 量化与遮挡剔除
    u_min, u_max = u.min(), u.max()
    v_min, v_max = v.min(), v.max()
    
    if u_max - u_min < 1e-6 or v_max - v_min < 1e-6:
        return np.arange(len(points))

    u_idx = ((u - u_min) / (u_max - u_min + 1e-8) * (res - 1)).astype(np.int32)
    v_idx = ((v - v_min) / (v_max - v_min + 1e-8) * (res - 1)).astype(np.int32)
    
    grid_keys = u_idx.astype(np.int64) * res + v_idx.astype(np.int64)
    
    # ULS (ortho) 视点在高处，dist 是到飞机的距离，dist 越小说明离飞机越近（即树冠顶部）
    # TLS (spherical) 视点在侧面，dist 越小说明离扫描仪越近
    # 所以逻辑通用：保留 dist 最小的点
    sort_idx = np.lexsort((dist, grid_keys))
    mask = np.concatenate(([True], grid_keys[sort_idx][1:] != grid_keys[sort_idx][:-1]))
    return sort_idx[mask]

def clip_line_to_bbox(start, direction, min_xy, max_xy):
    """
    计算射线与 2D Bounding Box 的交点，截取在 box 内部的线段参数 t
    Line: P(t) = start + t * direction
    """
    t_min, t_max = -1e9, 1e9
    
    # 针对 X 轴和 Y 轴分别计算进入和离开的时间 t
    for i in range(2): 
        if abs(direction[i]) < 1e-6: # 平行于轴
            if start[i] < min_xy[i] or start[i] > max_xy[i]:
                return None # 线在 Box 外
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
    """
    改进版采样：支持多个视点 (Trajectory) 的可见性融合
    对于 ULS，我们在飞行轨迹上选取多个点，取所有可见点的并集，再做降采样
    """
    # 1. 确定分辨率 (为了效率，取中等分辨率进行融合，如果不够再细调)
    # 这里简化处理：直接使用较高分辨率 (256) 计算可见性并集
    res = 256
    visible_indices_set = set()
    
    # 遍历轨迹上的关键视点
    for vp in viewpoints:
        idx = project_and_cull(points, vp, res, mode)
        visible_indices_set.update(idx)
        
    visible_indices = np.array(list(visible_indices_set))
    visible_points = points[visible_indices]
    
    # 2. 补全或降采样 logic
    current_n = len(visible_points)
    
    if current_n >= target_n:
        # 降采样
        choice_idx = np.random.choice(current_n, target_n, replace=False)
        return visible_points[choice_idx]
    else:
        # 补全 (Resampling with replacement)
        # print(f"Not enough points ({current_n} < {target_n}). Resampling.")
        choice_idx = np.random.choice(current_n, target_n, replace=True)
        return visible_points[choice_idx]

def generate_tls_task(pc_gt, target_n):
    """
    TLS 模拟：基于绝对坐标
    Scanner Height: 1.5 - 2.0m (Absolute Z)
    Scanner Dist: 0.1 * Tree Height
    """
    z_min = np.min(pc_gt[:, 2])
    z_max = np.max(pc_gt[:, 2])
    tree_height = z_max - z_min
    center_xy = np.mean(pc_gt[:, :2], axis=0)
    
    # 1. 随机角度
    angle = np.random.uniform(0, 2 * np.pi)
    
    # 2. 距离 (0.1 倍树高)
    # dist = np.random.uniform(0.1) * tree_height
    dist = 0.1 * tree_height
    # 加上 XY 平面的中心偏移
    scanner_x = center_xy[0] + dist * np.cos(angle)
    scanner_y = center_xy[1] + dist * np.sin(angle)
    
    # 3. 高度 (1.5m - 2.0m 绝对高度)
    scanner_z = z_min + np.random.uniform(1.5, 2.0)
    
    viewpoint = np.array([scanner_x, scanner_y, scanner_z])
    
    # TLS 只有一个视点
    return adaptive_sample_multi_view(pc_gt, [viewpoint], target_n, mode='spherical')

def generate_uls_task(pc_gt, target_n):
    """
    ULS 模拟：飞行轨迹扫描
    Height: 60-75m (Absolute)
    Trajectory: Line segment inside bbox, offset from center
    Sampling: Continuous points along line
    """
    min_pt, max_pt, center, tree_height = get_tree_bbox(pc_gt)
    z_min = min_pt[2]
    
    # 1. 确定飞行高度 (绝对高度 60-75m)
    flight_z = z_min + np.random.uniform(60.0, 75.0)
    
    # 2. 确定轨迹穿过的枢轴点 (Pivot)
    # "经过一个采样点[(-0.5,0.5),(-0.5,0.5)]+中心点x,y"
    offset = np.random.uniform(-0.5, 0.5, size=2)
    pivot_xy = center[:2] + offset
    
    # 3. 确定飞行方向 (0-360度)
    angle = np.random.uniform(0, 2 * np.pi)
    direction = np.array([np.cos(angle), np.sin(angle)])
    
    # 4. 计算被树的 Bounding Box 截断的线段
    # 我们需要找到线段在 [min_x, max_x] x [min_y, max_y] 内的部分
    # Pivot 是线上的点，所以 Line = Pivot + t * Direction
    t_bounds = clip_line_to_bbox(pivot_xy, direction, min_pt[:2], max_pt[:2])
    
    if t_bounds is None:
        # 极端情况：Pivot 就在边界外且方向背离 (极少发生)，回退到简单的中心点上方
        viewpoints = [np.array([center[0], center[1], flight_z])]
    else:
        t_start, t_end = t_bounds
        
        # 5. 沿线采样 (Sampling along the line)
        # 采样间隔 0.1m
        step = 0.1
        
        # 生成所有采样点的 t 值
        t_values = np.arange(t_start, t_end, step)
        
        # 为了计算效率：
        # 如果我们真的生成几百个视点去投影，速度会极慢。
        # 考虑到 60m 高空对 5m 宽的树，视差变化很小。
        # 我们选取轨迹上的 Key Frames (比如 5-10 个关键点) 来代表整条线的可见性
        # 这在工程上是极佳的近似
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
            
    # 执行多视点融合采样
    return adaptive_sample_multi_view(pc_gt, viewpoints, target_n, mode='ortho')
