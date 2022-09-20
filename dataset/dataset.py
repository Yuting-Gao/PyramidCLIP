import os
import re
import sys
import math
import json
import logging
import functools
import random
import pandas as pd
import numpy as np
from PIL import Image
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, random_split
from torch.utils.data.distributed import DistributedSampler
import torchvision.datasets as datasets
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop, Lambda
try:
    from torchvision.transforms import InterpolationMode
    bicubic = InterpolationMode.BICUBIC
except ImportError:
    bicubic = Image.BICUBIC
# from webdataset.utils import identity
# import webdataset as wds
from .simple_tokenizer import tokenize

data_configs = {}
data_configs['CLIP'] = {
    'normalize': [(0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)],
}



@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler

def preprocess_txt(text):
    return tokenize([str(text)])[0]

def get_classifi_dataset(args, preprocess_fns, split):
    if args.test_dataset == 'dtd':
        dataset = datasets.DTD(root='torchvision_datasets', split='test', transform=preprocess_fns)
    elif args.test_dataset == 'flowers':
        dataset = datasets.Flowers102(root='torchvision_datasets', split='test', transform=preprocess_fns)
    elif args.test_dataset == 'cifar10':
        dataset = datasets.CIFAR10(root='torchvision_datasets', download=True, train=False, transform=preprocess_fns)
    elif args.test_dataset == 'cifar100':
        dataset = datasets.CIFAR100(root='torchvision_datasets', download=True, train=False, transform=preprocess_fns)
    elif args.test_dataset == 'car':
        dataset = datasets.StanfordCars(root='torchvision_datasets', split='test', transform=preprocess_fns)
    elif args.test_dataset == 'pet':
        dataset = datasets.OxfordIIITPet(root='torchvision_datasets', split='test', transform=preprocess_fns)
    elif args.test_dataset == 'sat':
        dataset = datasets.EuroSAT('torchvision_datasets', transform=preprocess_fns)
    elif args.test_dataset == 'caltech':
        dataset = datasets.ImageFolder(root='/dev/shm/caltech/test', transform=preprocess_fns)
    elif args.test_dataset == 'food':
        dataset = datasets.ImageFolder(root='/dev/shm/food-101/test', transform=preprocess_fns)

    sampler = DistributedSampler(dataset) if args.local_rank > -1 else None
    shuffle = sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size_test,
        num_workers=args.workers,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=False,
        pin_memory=True,
    )

    return DataInfo(dataloader, sampler)

def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val"]

    data_path  = os.path.join(args.test_data_path, 'val')
    preprocess_fn = preprocess_fns
    assert data_path

    dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)
    sampler = DistributedSampler(dataset) if args.local_rank > -1 else None
    shuffle = sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size_test,
        num_workers=args.workers,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=False,
        pin_memory=True,
    )

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type='auto'):
    if dataset_type == "webdataset":
        # return get_wds_dataset
        raise NotImplementedError("webdataset is not supported yet")
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        else: 
            return get_tfrecord_dataset
        # else:
        #     raise ValueError(
        #         f"Tried to figure out dataset type, but failed for extention {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

def preprocess_fns(n_px, dataset, is_train=True):
    def _convert_to_rgb(image):
        return image.convert('RGB')
    normalize = Normalize(*data_configs[dataset]['normalize'])
    return Compose([
        Resize(n_px, bicubic),
        CenterCrop(n_px),
        _convert_to_rgb,
        ToTensor(),
        normalize,
    ])

def load_data(args, input_resolution=224, _type='train'):
    loaders = { 
        'train_loader': None,
        'train_sampler': None,
        'val_loader': None,
        'test_loader': None,
    }

    if args.test_dataset and _type=='test':
        if args.test_dataset == 'imagenet':
            preprocess_test = preprocess_fns(input_resolution, dataset="CLIP", is_train=False)
            loaders['test_loader'] = get_imagenet(args, preprocess_test, 'val')
        elif args.test_dataset in ['dtd','flowers','cifar10','cifar100','car','pet','caltech','sat']:
            preprocess_test = preprocess_fns(input_resolution, dataset="CLIP", is_train=False)
            loaders['test_loader'] = get_classifi_dataset(args, preprocess_test, 'val')
        else:
            raise NotImplementedError('Dataset {} is not supported yet'.format(args.test_dataset))
    return loaders

