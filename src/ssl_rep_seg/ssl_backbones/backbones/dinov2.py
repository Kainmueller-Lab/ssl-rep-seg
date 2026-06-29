from ..registry import register
import torch
from .backbone_core import GeneralViTBackbone
import torch.nn as nn
#########################
# DINOv2
#########################
class DINOv2Backbone(GeneralViTBackbone):
    def __init__(self, model, stride=None):
        super().__init__(model, stride)
        self.model = model
        self.model_type = "dinov2"
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
                                self._make_attn_hook(storage) ) )
            else:
                raise NotImplementedError(f"Facet {facet} not implemented for {self.model_type} backbone.")
                
        return hooks
    


                
@register("dinov2")
def build_dinov2(model_size, patch_size,**kwargs):
    assert patch_size == 14, "Only patch size 14 is supported for DINOv2"
    stride = kwargs.get("stride", None)

    model = torch.hub.load(
        "facebookresearch/dinov2",
        f"dinov2_vit{model_size[0]}14",
        trust_repo=True
    )
    return DINOv2Backbone(model, stride)


#########################
# DINOv2_reg
#########################
class DINOv2regBackbone(GeneralViTBackbone):
    def __init__(self, model, stride=None):
        super().__init__(model, stride)
        self.model = model
        self.model_type = "dinov2_reg"

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
            attn = self._unwrap_tensor(output).detach()
            attn = self.remove_extra_tokens(attn, dim=2) # remove extra tokens from the sequence dimension
            attn = self.remove_extra_tokens(attn, dim=3) # remove extra tokens from the sequence dimension
            storage.append(attn)

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
                                self._make_attn_hook(storage) ) )
            else:
                raise NotImplementedError(f"Facet {facet} not implemented for {self.model_type} backbone.")
                
        return hooks 

   
                
@register("dinov2_reg")
def build_dinov2_reg(model_size, patch_size,**kwargs):
    assert patch_size == 14, "Only patch size 14 is supported for DINOv2"
    stride = kwargs.get("stride", None)

    model = torch.hub.load(
        "facebookresearch/dinov2",
        f"dinov2_vit{model_size[0]}14_reg",
        trust_repo=True
    )
    return DINOv2regBackbone(model, stride)