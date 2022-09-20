from utils import *
from .dataset import load_data
if args.zeroshot_type == 'en':
    from .imagenet_zeroshot_data_en import imagenet_classnames, openai_imagenet_template, data_classnames, data_template
elif args.zeroshot_type == 'zh':
    from .imagenet_zeroshot_data_zh import imagenet_classnames, openai_imagenet_template
from .simple_tokenizer import tokenize
