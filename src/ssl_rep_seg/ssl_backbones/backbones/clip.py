from ..archs import clip_arch
from ..registry import register
import torch.nn as nn
from .backbone_core import  CLIPBackbone

                
@register("clip")
def build_clip(model_size, patch_size, **kwargs):
    stride = kwargs.get("stride", None)
    model = clip_arch.CLIPFeaturizer(model_size, patch_size)
    return CLIPBackbone(model, stride)
