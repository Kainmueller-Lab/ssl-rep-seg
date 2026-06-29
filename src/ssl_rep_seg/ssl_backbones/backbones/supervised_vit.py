from ..archs import vit_arch
from ..registry import register
import torch.nn as nn
import timm
from .backbone_core import GeneralViTBackbone


                
@register("supervised_vit")
def build_supervised_vit(model_size, patch_size, **kwargs):
    
    stride = kwargs.get("stride", None)
    # 1. architecture (shared)
    model = vit_arch.__dict__[f"vit_{model_size}_patch{patch_size}"]()
    model.head = nn.Identity()

    # 2. pretrained weights
    timm_name = f"vit_{model_size}_patch{patch_size}_224"
    temp_model = timm.create_model(timm_name, pretrained=True)

    state_dict = temp_model.state_dict()
    state_dict.pop("head.weight", None)
    state_dict.pop("head.bias", None)

    model.load_state_dict(state_dict, strict=False)

    # # TODO test
    # missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # print("Missing keys:", missing)
    # print("Unexpected keys:", unexpected)

    return GeneralViTBackbone(model, stride)