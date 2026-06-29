
from ..registry import register
import torch
import torch.nn as nn
import os
import re
from .backbone_core import GeneralViTBackbone
from termcolor import colored

class DINOv3Backbone(GeneralViTBackbone):
    def __init__(self, model, stride=None):
        super().__init__(model, stride)
        self.model = model
        self.model_type = "dinov3"
        self.feature_dim = model.embed_dim

        self.kqv_map ={"query": 0, "key": 1 , "value":2}

    def _make_qkv_hook(self,  facet, storage):
        facet_idx = self.kqv_map[facet]
        def _inner_hook(module, input, output):
            input = self._unwrap_tensor(input)
            B, N, C = input.shape
            qkv = module.qkv(input).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            qkv = qkv.detach()
            qkv = self.remove_extra_tokens(qkv, dim=3)  # remove extra tokens from the sequence dimension
            storage.append(qkv[facet_idx])
        return _inner_hook


    def _make_token_hook(self, block_idx, is_last, storage):
        norm_func = self.get_norm_layer(block_idx) if is_last else nn.Identity()

        def _inner_hook(module, input, output):
            ## hardcode to get the norm layer after the last block, and feature after norm layer is used in this case.
            out = self._unwrap_tensor(output)
            out = norm_func(out)
            out = self.remove_extra_tokens(out, dim=1)
            storage.append(out.detach())
        return _inner_hook


    def _make_attn_hook(self, num_heads, storage):
        def _inner_hook( model, input, output):
            B, N, C = input[0].shape
            qkv = output.reshape(B, N, 3, num_heads, C // num_heads)
            q, k, v = torch.unbind(qkv, 2)
            q, k, v = [t.transpose(1, 2) for t in [q, k, v]]

            scale = (q.shape[-1]) ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale

            attn = attn.softmax(dim=-1).detach()   #   N x num_head x N x N
            attn = self.remove_extra_tokens(attn, dim=2) # remove extra tokens from the sequence dimension
            attn = self.remove_extra_tokens(attn, dim=3) # remove extra tokens from the sequence dimension
           
        
            storage.append(attn) # shape B x num_head x N x N
        return _inner_hook


    def build_hooks(self, layers, facet, storage):
        hooks = []
        blocks = self.model.blocks

        if isinstance(layers, int):
            layers = [layers]

        for block_idx in layers:
            block = blocks[block_idx]
            if facet in ["key", "query", "value"]:
                # must return module and hook_fn (e.g. block.attn and _inner_hook )to be registered in the extractor!
                hooks.append( (block.attn,
                                self._make_qkv_hook( facet, storage) ) )

            elif facet =="token":
                is_last = (block_idx == len(blocks) - 1)
                hooks.append( (block, 
                                self._make_token_hook(block_idx, is_last, storage) ) )
                

            elif facet == "attn": #attn_from_qkv
                num_heads = block.attn.num_heads
                hooks.append( (block.attn.qkv, 
                                self._make_attn_hook( num_heads, storage) ) )
            else:
                raise NotImplementedError(f"Facet {facet} not implemented for {self.model_type} backbone.")
                
        return hooks
    



@register("dinov3")
def build_dinov3(model_size, patch_size, **kwargs):

    stride = kwargs.get("stride", None)
    root_dir = kwargs.get("root_dir", None)
    pretrained_path = kwargs.get("pretrained_path", None)


    # 2. checkpoints:
    # root_dir must be provided
    if root_dir is None:
        root_dir = os.getcwd()

    if pretrained_path is None:
        pattern = rf"^dinov3_vit{model_size[0]}{patch_size}_pretrain_.*\.pth$"
        matching_files = [f for f in os.listdir(os.path.join(root_dir,"pretrained/")) if re.match(pattern, f)]
        assert len(matching_files)==1, f"Expected exactly one matching file, found {len(matching_files)}"
        pretrained_path =os.path.join(root_dir, "pretrained/"+ matching_files[0])
    
    print(colored(f"please make sure that dinov3 folder is under the root_dir {root_dir} for correct loading", "yellow"))
    print(colored(f"Loading DINOv3 pretrained weight from {pretrained_path}", "green"))
    model = torch.hub.load(os.path.join(root_dir,"dinov3/"), f'dinov3_vit{model_size[0]}{patch_size}', source='local', weights = pretrained_path)
    
    return DINOv3Backbone(model, stride)
        