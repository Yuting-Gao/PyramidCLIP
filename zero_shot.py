import torch
import math
from torch.utils.data import Dataset
from dataset import imagenet_classnames, openai_imagenet_template, tokenize, data_classnames, data_template
import os
import logging
from utils import *
import torch.distributed as dist
import numpy as np

def get_text_features(model, classnames, templates, args):
    world_size = dist.get_world_size()
    N, n = len(classnames), world_size
    split = np.cumsum([0]+[i+1 if no<N%n else i for no,i in enumerate([N//n]*n)])
    rank = int(os.environ['RANK'])
    with torch.no_grad():
        text_features_split = []
        for classname in classnames[split[rank]:split[rank+1]]:
            texts = [template(classname) for template in templates]
            texts = tokenize(texts).to(args.local_rank)
            if args.local_rank > -1:
                class_embeddings = model.module.encode_text(texts)
                logits = model.module.logit_scale
            else:
                class_embeddings = model.encode_text(texts)
                logits = model.logit_scale
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm(dim=-1, keepdim=True)
            text_features_split.append(class_embedding)
        text_features_split = torch.stack(text_features_split, dim=0) 
        dist.barrier()
        text_features = varsize_tensor_all_gather(text_features_split)
        return text_features.to(args.local_rank) 

def get_image_features(model, dataloader, args):
    image_features = []
    for i, (image, target) in enumerate(dataloader):
        image = image.to(args.local_rank, non_blocking=True)
        target = target.to(args.local_rank, non_blocking=True)
        if args.local_rank > -1:
            image_embeddings = model.module.encode_image(image)
        else:
            image_embeddings = model.encode_image(image)
        #* logits are already multiplied on text features
        image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
        image_features.append(torch.cat([image_embeddings,target.unsqueeze(1)],dim=1))
    image_features = torch.cat(image_features, dim=0)
    dist.barrier()
    image_features = all_gather(image_features).cpu()
    return image_features.to(args.local_rank)

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True)) for k in topk]

def run(model, text_features, image_features, dataloader, args):
    with torch.no_grad():
        split = len(image_features) // dist.get_world_size()
        rank = int(os.environ['RANK'])
        logits = image_features[rank*split:(rank+1)*split,:-1] @ text_features.to(image_features.device).t()
        target = image_features[rank*split:(rank+1)*split, -1]

        # measure accuracy
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        acc1 = torch.tensor(acc1).to(args.local_rank)
        acc5 = torch.tensor(acc5).to(args.local_rank)
        dist.barrier()
        dist.all_reduce(acc1, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc5, op=dist.ReduceOp.SUM)

    n = len(image_features)
    acc1 = (acc1 / n) * 100.
    acc5 = (acc5 / n) * 100.
    return acc1, acc5

class loader(Dataset):
    def __init__(self, path):
        data = torch.load(path)


def zero_shot_eval(model, dataloader, epoch, args):
    if args.test_dataset == 'imagenet':
        text_features = get_text_features(model, imagenet_classnames, openai_imagenet_template, args)
    else:
        text_features = get_text_features(model, data_classnames[args.test_dataset], openai_imagenet_template, args)
    image_features = get_image_features(model, dataloader, args)

    results = {}
    top1, top5 = run(model, text_features, image_features, dataloader, args)
    results['top1'] = top1
    results['top5'] = top5
    return results

