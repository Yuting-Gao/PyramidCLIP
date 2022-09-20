import argparse
import os
"""
args
"""

parser = argparse.ArgumentParser(description='CLIP')

parser.add_argument(
    '--results_dir',
    metavar='RESULTS_DIR',
    default='./results',
    help='results dir')

parser.add_argument(
    '-e',
    '--evaluate',
    type=str,
    metavar='FILE',
    help='evaluate model FILE on validation set')

parser.add_argument(
    '--seed',
    default=0,
    type=int,
    help='random seed, set to 0 to disable')

parser.add_argument(
    '--visual_model',
    type=str,
    default='RN50',
    help='visual model (RN50|RN50x4|RN50x16|RN50x64|RN101|ViT-L/14-336px|ViT-L/14|ViT-B/16|ViT-B/32)')

parser.add_argument(
    '--textual_model',
    type=str,
    default='transformer',
    help='textual model (transformer)')

parser.add_argument(
    '--precision',
    choices=['amp', 'fp16', 'fp32'],
    default='amp',
    help='choose value precision')

parser.add_argument(
    '--test_dataset',
    default='',
    type=str,
    help='Testing dataset')

parser.add_argument(
    '--test_data_path',
    type=str,
    default='',
    help='The dictionary where the testing dataset is stored.')

parser.add_argument(
    '--workers',
    default=16,
    type=int,
    help='number of data loading workers (default: 8)')

parser.add_argument(
    '--batch_size',
    default=256,
    type=int,
    help='mini-batch size for training (default: 256)')

parser.add_argument(
    '--mini_batch_size',
    default=1,
    type=int,
    help='mini-mini-batch size for gradient accumulation')

parser.add_argument(
    '--batch_size_val',
    default=128,
    type=int,
    help='mini-batch size for validation (default: 128)')

parser.add_argument(
    '--batch_size_test',
    default=128,
    type=int,
    help='mini-batch size for testing (default: 128)')

parser.add_argument(
    '--local_rank',
    default=-1,
    type=int,
    help='node rank for distributed training')

parser.add_argument(
    '--global_rank',
    default=-1,
    type=int,
    help='global node rank for distributed training')

parser.add_argument(
    '--use_bn_sync',
    default=False,
    action="store_true",
    help='whether to use batch norm sync.'
)

parser.add_argument(
    '--zeroshot_type',
    default='en',
    type=str,
    help='template language in zeershot'
)

args = parser.parse_args()