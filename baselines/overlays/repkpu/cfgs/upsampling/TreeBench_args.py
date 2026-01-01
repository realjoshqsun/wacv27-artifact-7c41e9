import argparse
from cfgs.utils import str2bool
import json


def parse_TreeBench_args():
    parser = argparse.ArgumentParser(description='Training Arguments')
    parser.add_argument('--config', default='cfgs/upsampling/TreeBench_upsampling_4x_genus_ood.json',
                        type=str, help='TreeBench experiment JSON configuration')
    # training
    parser.add_argument('--seed', default=21, type=float, help='seed')
    parser.add_argument('--optim', default='adam', type=str, help='optimizer, adam or sgd')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0, type=float, help='weight decay')
    parser.add_argument('--lr_decay_step', default=20, type=int, help='learning rate decay step size')
    parser.add_argument('--gamma', default=0.5, type=float, help='gamma for scheduler_steplr')
    parser.add_argument('--epochs', default=50, type=int, help='training epochs')
    parser.add_argument('--save_rate', default=1, type=int, help='validation/checkpoint frequency in epochs')
    parser.add_argument('--batch_size', default=24, type=int, help='batch size')
    parser.add_argument('--num_workers', default=8, type=int, help='workers number')
    parser.add_argument('--print_rate', default=200, type=int, help='loss print frequency in each epoch')
    parser.add_argument('--use_smooth_loss', default=False, type=str2bool, help='whether use smooth L1 loss')
    parser.add_argument('--beta', default=0.01, type=float, help='beta for smooth L1 loss')
    
    # encoder
    parser.add_argument('--k', default=16, type=int, help='neighbor number in encoder')
    parser.add_argument('--encoder_dim', default=32, type=int, help='input(output) feature dimension in each dense block')
    parser.add_argument('--out_dim', default=64, type=int, help='input(output) feature dimension in each dense block')
    parser.add_argument('--encoder_bn', default=False, type=str2bool, help='whether use batch normalization in encoder')
    parser.add_argument('--global_mlp', default=True, type=str2bool, help='whether use global_mlp in encoder')
    #  kernel points
    # kernel_radius = kernel_point_receptive_radius * 1.5
    # conv_radius  = kernel_point_receptive_radius * 4
    parser.add_argument('--conv_radius', default=0.8, type=float, help='radius of kernel point convolution')
    parser.add_argument('--neighbor_limits', default=30, type=int, help='maximum number of points')
    parser.add_argument('--kernel_radius', default=0.3, type=float, help='radius of kernel point sphere')
    parser.add_argument('--kernel_point_receptive_radius', default=0.2, type=float, help='receptive field of kernel point convolution')
    parser.add_argument('--num_kernel_points', default=15, type=int, help='number of kernel points in kernel point representation')
    parser.add_argument('--in_dim', default=64, type=int, help='input feature dimension of REM & KGM')
    parser.add_argument('--kp_dim', default=64, type=int, help='feature dimension in REM & KGM')
    parser.add_argument('--is_kp_bn', default=False, type=str2bool, help='whether use batch normalization in kpconv')
    parser.add_argument('--is_kp_bias', default=False, type=str2bool, help='whether use bias in kpconv')
    parser.add_argument('--rigid_scale', default=0.625, type=float, help='radius of kernel point sphere')
    parser.add_argument('--query_scale', default=1.0, type=float, help='radius of kernel point sphere')
    parser.add_argument('--up_rate', default=8, type=int, help='upsampling rate')
    parser.add_argument('--simple', default=False, type=str2bool, help='whether use simple decoder')
    # cross-attention
    parser.add_argument('--head_num', default=4, type=int, help='head number of attention')
    parser.add_argument('--trans_num', default=3, type=int, help='number of attention blocks')
    parser.add_argument('--trans_dim', default=128, type=int, help='dim of attention')
    parser.add_argument('--is_attn_bn', default=False, type=str2bool, help='whether use batch normalization in attention')
    
    parser.add_argument('--num_points', default=2048, type=int, help='the points number of each input patch')
    
    # ouput
    parser.add_argument('--out_path', default='./output/8-1-1_genus-OOD_upsampling_N>=8192.json', type=str, help='the checkpoint and log save path')
    
    # test
    parser.add_argument('--ckpt', default='./pretrain/8-1-1_genus-OOD_upsampling_N>=8192.json_best.pth', type=str, help='checkpoints')
    parser.add_argument('--patch_rate', default=3, type=int, help='used for patch generation')

    args = parser.parse_args()
    
    # Load the selected experiment JSON and let it override shared defaults.
    with open(args.config, 'r') as f:
        treebench_cfg = json.load(f)
    for key, value in treebench_cfg.items():
        setattr(args, key, value)
    

    return args
