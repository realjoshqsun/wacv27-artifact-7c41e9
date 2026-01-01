# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-08-08 14:31:30
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-05-25 09:13:32

import logging
import open3d
import torch
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2, ChamferFunction
import os
from extensions.emd import emd_module as emd

class Metrics(object):
    ITEMS = [{
        'name': 'F-Score',
        'enabled': True,
        'eval_func': 'cls._get_f_score',
        'is_greater_better': True,
        'init_value': 0
    }, {
        'name': 'CDL1',
        'enabled': True,
        'eval_func': 'cls._get_chamfer_distancel1',
        'eval_object': ChamferDistanceL1(ignore_zeros=True),
        'is_greater_better': False,
        'init_value': 32767
    }, {
        'name': 'CDL2',
        'enabled': True,
        'eval_func': 'cls._get_chamfer_distancel2',
        'eval_object': ChamferDistanceL2(ignore_zeros=True),
        'is_greater_better': False,
        'init_value': 32767
    }, {
        'name': 'EMDistance',
        'enabled': True,
        'eval_func': 'cls._get_emd_distance',
        'eval_object': emd.emdModule(),
        'is_greater_better': False,
        'init_value': 32767
    }]

    @classmethod
    def get(cls, pred, gt, require_emd=True):
        _items = cls.items()
        _values = [0] * len(_items)
        for i, item in enumerate(_items):
            if not require_emd and 'emd' in item['eval_func']:
                _values[i] = torch.tensor(0.).to(gt.device)
            else:
                eval_func = eval(item['eval_func'])
                # print(f'Metrics.get eval_func:{eval_func}, pred.shape:{pred.shape}, gt.shape:{gt.shape}')
                _values[i] = eval_func(pred, gt)

        return _values

    @classmethod
    def items(cls):
        return [i for i in cls.ITEMS if i['enabled']]

    @classmethod
    def names(cls):
        _items = cls.items()
        return [i['name'] for i in _items]

    @classmethod
    def _get_f_score(cls, pred, gt, th=0.01):

        """References: https://github.com/lmb-freiburg/what3d/blob/master/util.py"""
        b = pred.size(0)
        device = pred.device
        assert pred.size(0) == gt.size(0)
        if b != 1:
            f_score_list = []
            for idx in range(b):
                f_score_list.append(cls._get_f_score(pred[idx:idx+1], gt[idx:idx+1]))
            return sum(f_score_list)/len(f_score_list)
        else:
            pred = cls._get_open3d_ptcloud(pred)
            gt = cls._get_open3d_ptcloud(gt)

            dist1 = pred.compute_point_cloud_distance(gt)
            dist2 = gt.compute_point_cloud_distance(pred)

            recall = float(sum(d < th for d in dist2)) / float(len(dist2))
            precision = float(sum(d < th for d in dist1)) / float(len(dist1))
            result = 2 * recall * precision / (recall + precision) if recall + precision else 0.
            result_tensor = torch.tensor(result).to(device)
            return result_tensor

    @classmethod
    def _get_open3d_ptcloud(cls, tensor):
        """pred and gt bs is 1"""
        tensor = tensor.squeeze().cpu().numpy()
        ptcloud = open3d.geometry.PointCloud()
        ptcloud.points = open3d.utility.Vector3dVector(tensor)

        return ptcloud

    @classmethod
    def _get_chamfer_distancel1(cls, pred, gt):
        chamfer_distance = cls.ITEMS[1]['eval_object']
        return chamfer_distance(pred, gt) * 1000

    @classmethod
    def _get_chamfer_distancel2(cls, pred, gt):
        chamfer_distance = cls.ITEMS[2]['eval_object']
        return chamfer_distance(pred, gt) * 1000

    @classmethod
    def _get_emd_distance(cls, pred, gt, eps=0.005, iterations=100):
        emd_loss = cls.ITEMS[3]['eval_object']
        dist, _ = emd_loss(pred, gt, eps, iterations)
        emd_out = torch.mean(torch.sqrt(dist))
        return emd_out * 1000

    def __init__(self, metric_name, values):
        self._items = Metrics.items()
        self._values = [item['init_value'] for item in self._items]
        self.metric_name = metric_name

        if type(values).__name__ == 'list':
            self._values = values
        elif type(values).__name__ == 'dict':
            metric_indexes = {}
            for idx, item in enumerate(self._items):
                item_name = item['name']
                metric_indexes[item_name] = idx
            for k, v in values.items():
                if k not in metric_indexes:
                    logging.warn('Ignore Metric[Name=%s] due to disability.' % k)
                    continue
                self._values[metric_indexes[k]] = v
        else:
            raise Exception('Unsupported value type: %s' % type(values))

    def state_dict(self):
        _dict = dict()
        for i in range(len(self._items)):
            item = self._items[i]['name']
            value = self._values[i]
            _dict[item] = value

        return _dict

    def __repr__(self):
        return str(self.state_dict())

    def better_than(self, other):
        if other is None:
            return True

        _index = -1
        for i, _item in enumerate(self._items):
            if _item['name'] == self.metric_name:
                _index = i
                break
        if _index == -1:
            raise Exception('Invalid metric name to compare.')

        _metric = self._items[i]
        _value = self._values[_index]
        other_value = other._values[_index]
        return _value > other_value if _metric['is_greater_better'] else _value < other_value

chamfer_dist = ChamferFunction.apply
def calc_dcd(x, gt, alpha=1000, n_lambda=1):
    _, n_x, _ = x.shape
    _, n_gt, _ = gt.shape

    frac_12 = n_x / n_gt
    frac_21 = n_gt / n_x

    dist1, dist2, idx1, idx2 = chamfer_dist(x, gt)

    exp_dist1 = torch.exp(-dist1 * alpha)
    exp_dist2 = torch.exp(-dist2 * alpha)

    count1 = torch.zeros_like(idx2)
    count1.scatter_add_(1, idx1.long(), torch.ones_like(idx1))
    weight1 = count1.gather(1, idx1.long()).float().detach() ** n_lambda
    weight1 = (weight1 + 1e-6) ** (-1) * frac_21
    loss1 = (1 - exp_dist1 * weight1).mean(dim=1)

    count2 = torch.zeros_like(idx1)
    count2.scatter_add_(1, idx2.long(), torch.ones_like(idx2))
    weight2 = count2.gather(1, idx2.long()).float().detach() ** n_lambda
    weight2 = (weight2 + 1e-6) ** (-1) * frac_12
    loss2 = (1 - exp_dist2 * weight2).mean(dim=1)

    return (loss1 + loss2) / 2


def fscore(dist1, dist2, threshold=1e-4):
    p1 = torch.mean((dist1 < threshold).float(), dim=1)
    p2 = torch.mean((dist2 < threshold).float(), dim=1)
    f = 2 * p1 * p2 / (p1 + p2)
    f[torch.isnan(f)] = 0
    return f, p1, p2


def hausdorff_distance(X,Y):
    B,N,C = X.shape
    dist1, dist2 ,_ ,_ = chamfer_dist(X, Y)
    h1 = torch.amax(dist1, dim=1).view(B,1)
    h2 = torch.amax(dist2, dim=1).view(B,1)
    hd_loss = torch.cat([h1,h2],dim=-1)
    return hd_loss


def compute_metrics(generated, gt):
    dist1, dist2, _, _ = chamfer_dist(generated, gt)
    cd_l2 = (dist1.mean(1) + dist2.mean(1))
    cd_l1 = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2

    f1, _, _ = fscore(dist1, dist2)
    hd = hausdorff_distance(generated, gt)
    hd = hd.mean(dim=1)
    dcd = calc_dcd(generated, gt)
    
    return cd_l1, cd_l2, f1, hd, dcd
