from ..archs import mugs_arch
from ..registry import register
import torch
import os
from .backbone_core import GeneralViTBackbone
import torch.nn as nn
from termcolor import colored

class MugsBackbone(GeneralViTBackbone):
    def __init__(self, model, stride=None):
        super().__init__(model, stride)
        self.model = model
        self.model_type = "mugs"
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
    
    def _make_attn_hook(self, storage):
        def _inner_hook(module, input, output):
            output = self._unwrap_tensor(output)
            storage.append(output.detach())

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
                

            elif facet == "attn":  #attn
                assert isinstance(block.attn.attn_drop, nn.Dropout), "the attn_drop layer is expected to be a dropout layer"
                assert block.attn.attn_drop.p==0, "the dropout layer for attention should not drop any units,  \
                            otherwise the extracted features will be different in each forward pass"
                hooks.append( (block.attn.attn_drop,
                                   self._make_attn_hook(storage)) )
            
            else:
                raise NotImplementedError(f"Facet {facet} not implemented for {self.model_type} backbone.")
            
        return hooks

                
@register("mugs")
def build_mugs(model_size, patch_size, **kwargs):

    stride = kwargs.get("stride", None)
    root_dir = kwargs.get("root_dir", None)
    pretrained_path = kwargs.get("pretrained_path", None)

    
    # 1. architecture 
    model = mugs_arch.__dict__[f"vit_{model_size}"](patch_size=patch_size, num_relation_block=1) 

    # 2. checkpoints: defining the pretrained path and then load it
    if pretrained_path is None:
        if root_dir is None:
            root_dir = os.getcwd()
        pretrained_path = os.path.join(root_dir, "pretrained" ,f"mugs_pretrain_vit_{model_size}.pth")
    
    print(colored(f"Loading Mugs pretrained weight from {pretrained_path}", "green"))
    checkpoint = torch.load(pretrained_path, map_location="cpu")['state_dict'] 
        
    # 3. load weights 
    model.load_state_dict(checkpoint, strict=False)

    # # TODO test
    # missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    # print("Missing keys:", missing)
    # print("Unexpected keys:", unexpected)
    
    return MugsBackbone(model, stride)