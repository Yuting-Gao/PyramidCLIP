from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import yaml


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.proj_pre = nn.Parameter(torch.randn(2052, embed_dim) / embed_dim**0.5)
        self.ln_pre = LayerNorm(embed_dim)
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x, isROI=False):
        if not isROI:
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
            x = x + self.positional_embedding[:, None, :].to(x.dtype)
        else:
            x = x @ self.proj_pre
            x = x.permute(1, 0, 2)
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
            x = self.ln_pre(x)
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, isROI=False):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        if isROI:
            x = self.attnpool(x, isROI)
            return x
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LEFF(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.ac1 = nn.Sequential(OrderedDict([
            ("bn", nn.BatchNorm1d(d_model*4)),
            ("gelu", QuickGELU())
        ]))
        self.depConv = nn.Sequential(OrderedDict([
            ("depconv", nn.Conv2d(
            in_channels=d_model*4,
            out_channels=d_model*4,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=d_model*4,
            bias=False)),
            ("bn", nn.BatchNorm2d(d_model*4)),
            ("gelu", QuickGELU())
        ]))
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.ac2 = nn.Sequential(OrderedDict([
            ("bn", nn.BatchNorm1d(d_model)),
            ("gelu", QuickGELU())
        ]))

    def forward(self, x):
        clsToken = x[0:1, :, :]
        patchToken = x[1:, :, :]
        x = self.linear1(patchToken)
        x = self.ac1(x.permute(1,2,0))
        sq_L = int(x.shape[2]**0.5)
        x = x.reshape(x.shape[0], x.shape[1], sq_L, sq_L)
        x = self.depConv(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.linear2(x.permute(0,2,1))
        x = self.ac2(x.permute(0,2,1))
        x = x.permute(2,0,1)
        x = torch.cat([clsToken, x], dim=0)
        
        return x

        
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, isLEFF = False):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        if isLEFF:
            self.mlp = LEFF(d_model)
        else:
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, d_model * 4)),
                ("gelu", QuickGELU()),
                ("c_proj", nn.Linear(d_model * 4, d_model))
            ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, isROI=False):
        if isROI:
            length = x.shape[0]
            mask = torch.empty(length, length)
            mask.fill_(float("-inf"))
            for i in range(11):
                for j in range(11):
                    mask[i][j] = 0
            mask = mask.to(dtype=x.dtype, device=x.device)
            return self.attn(x, x, x, need_weights=False, attn_mask=mask)[0]

        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, isROI=False):
        x = x + self.attention(self.ln_1(x), isROI)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, isLEFF = False):
        super().__init__()
        self.width = width
        self.layers = layers
        self.leffs = 9
        self.resblocks1 = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, isLEFF) for _ in range(self.leffs)])
        self.resblocks2 = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers - self.leffs)])

    def forward(self, x: torch.Tensor):
        for resblock in self.resblocks1:
            x = resblock(x)
        for resblock in self.resblocks2:
            x = resblock(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.freeze_conv1 = True

        scale = width ** -0.5
        self.proj_pre = nn.Parameter(scale * torch.randn(2052, width))
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads, isLEFF=True)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def train(self, mode=True):
        self.trianing = mode
        for module in self.children():
            module.train(mode)
        
        if self.freeze_conv1:
            for layer in [self.conv1]:
                layer.eval()
                for param in layer.parameters():
                    param.requires_grad = False
        return self        

    def forward(self, x: torch.Tensor, isROI = False):
        if isROI:
            x = x @ self.proj_pre
            x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  
            x = self.ln_pre(x)
            x = x.permute(1, 0, 2)
            for resblock in self.transformer.resblocks2:
                x = resblock(x)
            x = x.permute(1, 0, 2)
            x = self.ln_post(x[:, 0, :])
            x = x @ self.proj
            return x

        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])
        x = x @ self.proj

        return x


class MODEL(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks1:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        for block in self.transformer.resblocks2:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask


    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, isROI=False):
        return self.visual(image.type(self.dtype), isROI)

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = torch.max(x, dim=1)[0]
        x = x @ self.text_projection 
        return x

    def forward(self, images_small, texts, images_large, summarizes, roi_feats, attr_tags, training=False):
        image_small_features = self.encode_image(images_small)
        image_large_features = self.encode_image(images_large)
        roi_features = self.encode_image(roi_feats, isROI=True)
        text_features = self.encode_text(texts)
        summarize_features = self.encode_text(summarizes)
        tag_features = self.encode_text(attr_tags)

        image_small_features = image_small_features / image_small_features.norm(dim=-1, keepdim=True)
        image_large_features = image_large_features / image_large_features.norm(dim=-1, keepdim=True)
        roi_features = roi_features / roi_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)    
        summarize_features = summarize_features / summarize_features.norm(dim=-1, keepdim=True)        
        tag_features = tag_features / tag_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp() 
        if training:
            return image_small_features, text_features*logit_scale, image_large_features, summarize_features*logit_scale, roi_features, tag_features*logit_scale


def build_model(name: str):
    with open('./models/model_config.yml', 'r') as f:
        model_configs = yaml.load(f, Loader=yaml.FullLoader)
    model_config = model_configs.get(name, '')
    if not model_config:
        raise NotImplementedError(
            f'{name} model is not implemented yet! Please use (RN50|RN50x4|RN50x16|RN50x64|RN101|ViT-B/32|ViT-B/16|ViT-L/14|ViT-L/14-336px) instead.')

    embed_dim = model_config['embed_dim']
    image_resolution = model_config['image_resolution']
    vision_layers = model_config['vision_layers']
    vision_width = model_config['vision_width']
    vision_patch_size = model_config['vision_patch_size']
    context_length = model_config['context_length']
    vocab_size = model_config['vocab_size']
    transformer_width = model_config['transformer_width']
    transformer_heads = model_config['transformer_heads']
    transformer_layers = model_config['transformer_layers']

    model = MODEL(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )
    return model



if __name__ == '__main__':
    model = build_model('RN50')
    batch_size = 5
    img = torch.randn(batch_size, 3, 224, 224)
    txt = torch.randint(0, 100, [batch_size, 77])
    image_features = model.encode_image(img)
    text_features = model.encode_text(txt)
    print(image_features.shape, text_features.shape)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    logits_per_image = image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    print(logits_per_image.shape)
    print(logits_per_text.shape)