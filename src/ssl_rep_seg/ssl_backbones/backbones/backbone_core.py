import torch.nn as nn
from abc import ABC, abstractmethod
import torch.nn.modules.utils as nn_utils
import types
import math
import torch
import torch.nn.functional as F
##########################################
# Base backbone class that all backbones should inherit from. Defines the interface and some common utilities.
##########################################

###### BaseBackbone
class BaseBackbone(nn.Module, ABC):
    def __init__(self):
        super().__init__()

        # must be defined by child classes
        self.patch_size = None
        self.stride = None
        self.preprocess_config = None
       

    def get_preprocess_config(self):
        if self.preprocess_config is None:
            raise ValueError(
                f"{self.__class__.__name__} must define preprocess_config"
            )
        return self.preprocess_config
    
    def _unwrap_tensor(self, x):
        if isinstance(x, (list, tuple)):
            assert len(x) == 1, (
                f"Expected single tensor, got {len(x)} elements: "
                f"{[type(e) for e in x]}"
            )
            return x[0]

        assert isinstance(x, torch.Tensor), (
            f"Expected tensor or single-element container, got {type(x)}"
        )
        return x

    
    @abstractmethod
    def forward(self, x):
        pass

    def forward_features(self, x):
        """Optional override"""
        return self.forward(x)

    @abstractmethod
    def get_norm_layer(self, layer: int):
        pass

    @abstractmethod
    def build_hooks(self, layers, facet, storage):
        pass

    
###### ViT backbone
    
class GeneralViTBackbone(BaseBackbone):
    def __init__(self, model, stride=None):
        super().__init__()

        # if stride is not None and stride is not equal to model.patch_embed.patch_size, 
        # adjust the patch embedding stride to match the desired stride (for different stride than patch size)
        # don't use ViTUtils for CLIP since it has a different architecture 
        if stride is not None:
            ViTUtils.patch_vit_resolution(model, stride)

        self.model = model

        self.patch_size = model.patch_embed.patch_size
        self.stride = model.patch_embed.proj.stride
        self.feature_dim = model.embed_dim

        self.preprocess_config = {
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225)
        }

    def forward(self, x):
        return self.model(x)

    def get_norm_layer(self, layer: int):
        if layer == len(self.model.blocks) - 1:
            return self.model.norm
        return self.model.blocks[layer].norm1
    
     # only dinov2_reg and dinov3 have this functions
    def get_num_extra_tokens(self):
        if hasattr(self.model, "n_storage_tokens"):
            return self.model.n_storage_tokens
        if hasattr(self.model, "num_register_tokens"):
            return self.model.num_register_tokens
        return 0
    
    def remove_extra_tokens(self, x, dim):
        n = self.get_num_extra_tokens()
        if n == 0:
            return x

        #torch.narrow(input, dim, start, length)
        return torch.cat(
            (x.narrow(dim, 0, 1), x.narrow(dim, 1 + n, x.size(dim) - (1 + n))),
            dim=dim
        )
    

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
        def _inner_hook(module,input, output):
            B, N, C = input[0].shape
            qkv = output.reshape(B, N, 3, num_heads, C // num_heads)
            q, k, v = torch.unbind(qkv, 2)
            q, k, v = [t.transpose(1, 2) for t in [q, k, v]]

            scale = (q.shape[-1]) ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1).detach()   #   N x num_head x N x N
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
                hooks.append((block.attn, 
                              self._make_qkv_hook(facet, storage)))

            elif facet =="token":
                ## hardcode to get the norm layer after the last block, and feature after norm layer is used in this case.
                is_last = (block_idx == len(blocks) - 1)
                hooks.append((block, 
                              self._make_token_hook(block_idx, is_last, storage)))

            elif facet == "attn": #attn_from_qkv
                hooks.append((block.attn.kqv, 
                              self._make_attn_hook(block.attn.num_heads, storage)))

            else:
                raise NotImplementedError(f"Facet {facet} not implemented for Supervised ViT backbone.")
            
        return hooks
                

###### CLIP backbone
class CLIPBackbone(BaseBackbone):
    def __init__(self, model, stride=None):
        super().__init__()

        # positional encoding adjustment for CLIP has been implemented in clip_arch.py
        # no need to call ViTUtils.patch_vit_resolution here
        if stride is not None:
            raise ValueError("Stride adjustment is not supported for CLIP backbone since it has a different architecture than standard ViTs.")
            # self.stride = nn_utils._pair(stride) # 15.06.2026

        self.model = model
        self.feature_dim = self.model.clip_model.transformer.width
        self.preprocess_config = {
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225)
        }

    def forward(self, x):
        return self.model(x)

    def get_norm_layer(self, layer: int):
        if layer == len(self.model.clip_model.transformer.resblocks) -1:
            return self.model.clip_model.ln_post
        return self.model.clip_model.transformer.resblocks[layer].ln_1
    
    def _make_qkv_hook(self,  facet, storage):
        facet_idx = self.kqv_map[facet]
        def _inner_hook(module, input, output):
            input = input[0]
            if module.batch_first:
                input = input.transpose(0, 1)

            N, B, C = input.shape
            q, k, v = F.linear(input, module.in_proj_weight, module.in_proj_bias).chunk(3, dim=-1)
            q = q.reshape(N, B, module.num_heads, C//module.num_heads).permute(1, 2, 0, 3)
            k = k.reshape(N, B, module.num_heads, C//module.num_heads).permute(1, 2, 0, 3)
            v = v.reshape(N, B, module.num_heads, C//module.num_heads).permute(1, 2, 0, 3)
            qkv = [q.detach(), k.detach(), v.detach()]
            storage.append(qkv[facet_idx])
        return _inner_hook
    
    def _make_token_hook(self, block_idx, is_last, storage):
        norm_func = self.get_norm_layer(block_idx) if is_last else nn.Identity()
        def _inner_hook(module, input, output):
            ## hardcode to get the norm layer after the last block, and feature after norm layer is used in this case.
            out = self._unwrap_tensor(output)
            out = norm_func(out)
            storage.append(out.detach())
        return _inner_hook
    
    def _make_attn_hook(self, num_heads, storage):
        def _inner_hook(model,input, output):
            x = input[0]
            B, N, C = input[0].shape
            qkv =  torch.nn.functional.linear(x, model.in_proj_weight, model.in_proj_bias) # B, N, 3 * C
            qkv = qkv.reshape(B, N , 3, num_heads, C // num_heads)
            q, k, v = torch.unbind(qkv, 2)
        
            q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
            scale = (q.shape[-1]) ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1).detach()   #   N x num_head x N x N
            
            storage.append(attn)
        return _inner_hook

    
    def build_hooks(self, layers, facet, storage):
        hooks = []
        blocks = self.model.clip_model.transformer.resblocks

        if isinstance(layers, int):
            layers = [layers]

        for block_idx in layers:
            block = blocks[block_idx]
            if facet in ["key", "query", "value"]:
                hooks.append((block.attn,
                                self._make_qkv_hook(facet, storage)))
            
            elif facet =="token":
                is_last = (block_idx == len(blocks) - 1)
                hooks.append( (block, 
                                self._make_token_hook(block_idx, is_last, storage) ) )


            elif facet == "attn": #mhattn_from_qkv
                num_heads = block.attn.num_heads
                hooks.append((block.attn,
                                self._make_attn_hook(num_heads, storage)))

            else:
                raise NotImplementedError(f"Facet {facet} not implemented for CLIP backbone.")
            
        return hooks


##########################################
# Utility class for ViT-specific operations, such as patch embedding and positional encoding adjustments.
# Not part of the inheritance tree, just a collection of static methods.
##########################################
class ViTUtils:
    """
    Pure helper class, NOT part of inheritance tree.
    """

    @staticmethod
    def patch_vit_resolution(model, stride):
        patch_size = model.patch_embed.patch_size

        if stride == patch_size:
            return model

        if isinstance(patch_size, (list, tuple)):
            patch_size = patch_size[0]

        stride = nn_utils._pair(stride)

        model.patch_embed.proj.stride = stride

        model.interpolate_pos_encoding = types.MethodType(
            ViTUtils.fix_pos_enc(patch_size, stride),
            model
        )

        return model


    @staticmethod
    def fix_pos_enc(patch_size, stride_hw):
        '''
        patch_size: int
        stride_hw: Tuple[int, int]
        '''
        def interpolate_pos_encoding(self, x, w, h):
            npatch = x.shape[1] - 1
            N = self.pos_embed.shape[1] - 1

            if npatch == N and w == h:
                return self.pos_embed

            cls_token = self.pos_embed[:, 0]
            patch_tokens = self.pos_embed[:, 1:]
            dim = x.shape[-1]

            w0 = 1 + (w - patch_size) // stride_hw[1]
            h0 = 1 + (h - patch_size) // stride_hw[0]
            assert (w0 * h0 == npatch), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and 
                                            stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""

            w0, h0 = w0 + 0.1, h0 + 0.1

            patch_tokens = nn.functional.interpolate(
                patch_tokens.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim)
                .permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode='bicubic',
                align_corners=False,
                recompute_scale_factor=False
            )

            assert int(w0) == patch_tokens.shape[-2] and int(h0) == patch_tokens.shape[-1]
            patch_tokens = patch_tokens.permute(0, 2, 3, 1).view(1, -1, dim)

            return torch.cat((cls_token.unsqueeze(0), patch_tokens), dim=1)

        return interpolate_pos_encoding
    
