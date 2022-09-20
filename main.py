import argparse
import os
import math
import time
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.autograd import Variable
from utils import *
from datetime import datetime
import dataset
import models
from zero_shot import zero_shot_eval


def main():
    global args, input_resolution, test_datasets
    #* set random seed
    if args.seed > 0:
        set_seed(args.seed)
    else:
        cudnn.benchmark = True
    test_datasets = args.test_dataset
    args.global_rank = int(os.environ['RANK'])

    #* distribute init
    dist.init_process_group(backend='nccl',
                            world_size=int(os.environ['WORLD_SIZE']),
                            rank=int(os.environ['RANK']))
    torch.cuda.set_device(args.local_rank)
    world_size = dist.get_world_size()
    if args.global_rank == 0:
        print(f'world_size: {world_size}')
    device = torch.device('cuda', args.local_rank)

    #* create model
    model = models.build_model(args.visual_model).cuda()
    input_resolution = model.visual.input_resolution 

    
    #* sync bn
    if args.global_rank != -1 and args.use_bn_sync:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    #* set up criterion 
    criterion = nn.CrossEntropyLoss()

    #* evaluate
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            print('invalid checkpoint: {}'.format(args.evaluate))
            return
        else:
            checkpoint = torch.load(args.evaluate, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
            if args.local_rank == 0:
                print("loaded checkpoint '{}' (epoch {})".format(
                            args.evaluate, checkpoint.get('epoch', -1)))

    #* DDP
    if args.global_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            output_device=args.local_rank,
            find_unused_parameters=True
        )

    if args.evaluate:
        test_dataset = list(args.test_dataset.split('+'))
        for idx in range(len(test_dataset)):
            args.test_dataset = test_dataset[idx]
            data_loaders = dataset.load_data(args, input_resolution=input_resolution, _type='test')
            test_loader = data_loaders['test_loader'].dataloader
            with torch.no_grad():
                test_prec1, test_prec5 = test(
                    test_loader, model, criterion, 0)
        return


def forward_test(data_loader, model, criterion, epoch, training=False):
    device = torch.device('cuda', args.local_rank)
    if args.test_dataset == 'imagenet':
        zero_shot_metrics = zero_shot_eval(model, data_loader, epoch, args)
        top1, top5 = zero_shot_metrics['top1'], zero_shot_metrics['top5']
        if args.local_rank == 0:
            print('{phase}\t'
                'Prec@1/5 {top1:.2f}/{top5:.2f} \t'
                .format(phase='ImageNet Zeroshot', top1=top1, top5=top5))
    elif args.test_dataset in ['dtd','flowers','cifar10','cifar100','car','pet','caltech','aircraft','food','sun','sat']:
        zero_shot_metrics = zero_shot_eval(model, data_loader, epoch, args)
        top1, top5 = zero_shot_metrics['top1'], zero_shot_metrics['top5']
        if args.local_rank == 0:
            print('{phase}\t'
                'Prec@1/5 {top1:.2f}/{top5:.2f} \t'
                .format(phase='{} Zeroshot'.format(args.test_dataset), top1=top1, top5=top5))
    return top1, top5


def test(data_loader, model, criterion, epoch):
    model.eval()
    return forward_test(data_loader, model, criterion, epoch,
                   training=False)


if __name__ == '__main__':
    main()
