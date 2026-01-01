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
class TreeBench(data.Dataset):
    def __init__(self, 
                 config
                ):
        self.category_file = config.CATEGORY_FILE_PATH
        self.upsampling_path = config.data_dir
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
        # check if the file exists
        if os.path.exists(os.path.join(self.upsampling_path, self.file_list[index], str(self.complete_npoints) + "_processed.npy")):
            complete_pc = np.load(os.path.join(self.upsampling_path, self.file_list[index], str(self.complete_npoints) + "_processed.npy"))
            partial_pc = np.load(os.path.join(self.upsampling_path, self.file_list[index], str(self.partial_npoints) + "_processed.npy"))
        else:
            complete_pc = np.load(os.path.join(self.upsampling_path, self.file_list[index], str(self.complete_npoints) + "_complete.npy"))
            partial_pc = np.load(os.path.join(self.upsampling_path, self.file_list[index], str(self.partial_npoints) + "_partial.npy"))
        
        result = {}
        result['partial'] = partial_pc
        result['complete'] = complete_pc
        
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
