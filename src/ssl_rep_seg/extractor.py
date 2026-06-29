import torch.nn as nn
from .ssl_backbones.registry import get_backbone
from .ssl_backbones.backbones import *
import torchvision.transforms as T
from PIL import Image
import torch

class Preprocessor:
    def __init__(self, resize, mean, std):
        self.resize = resize

        self.transform = T.Compose([
            T.Resize(resize, interpolation=3),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

    def __call__(self, image_path):
        img = Image.open(image_path).convert("RGB")
        pil_img = T.Resize(self.resize, interpolation=3)(img)
        tensor = self.transform(pil_img)[None, ...]
        return tensor, pil_img
    
class ViTExtractor:
    def __init__(
        self,
        model_type: str,
        model_size: str,
        patch_size: int,
        device: str = "cuda",
        model: nn.Module = None,
        **kwargs
    ):
        super().__init__()

        self.model_type = model_type
        self.device = device

        if model is not None:
            self.model = model
        else:
            self.model = get_backbone(
                model_type,
                model_size=model_size,
                patch_size=patch_size,
                **kwargs
            )

        self.model.eval()
        self.model.to(device)

        # TODO if we need this, we actually don't need this for unsup segmentation, but could be easiy to used in semantic correspondence task
        # get the preprocess config from the model and initialize the preprocessor
        self.preprocess_config = self.model.get_preprocess_config()
        self.preprocessor = Preprocessor( kwargs["resize"],
            mean=self.preprocess_config["mean"],
            std=self.preprocess_config["std"]
        )

        ### TODO specify self.patch_size and self.stride
        self.patch_size = self.model.patch_size
        self.stride = self.model.stride
        assert isinstance(self.stride, (tuple, list)), "Stride must be a tuple/list of ints"

        # self.hook_handlers and self._feats must live inside the extractor, 
        # not the backbone, to avoid conflicts when multiple extractors are used with the same backbone

        self.feature_dim= self.model.feature_dim

    def _register_hooks(self, layer, facet):
        # Clear any existing hooks before registering new ones to avoid duplicates
        self.hook_handlers = []
        hooks  = self.model.build_hooks(layer, facet, storage = self._feats)

        for module, hook_fn in hooks:
            handler = module.register_forward_hook(hook_fn)
            self.hook_handlers.append(handler)


    def _unregister_hooks(self):
        for handler in self.hook_handlers:
            handler.remove()
        self.hook_handlers = []
       
    def _extract_features(self, batch, layer=None, facet=None):
        B, C, H, W = batch.shape
        self._feats = []
        self.load_size = (H, W)
        if isinstance(self.patch_size, (list, tuple)):
            self.num_patches = (1 + (H - self.patch_size[0]) // self.stride[0], 1 + (W - self.patch_size[1]) // self.stride[1])
        else:
            self.num_patches = (1 + (H - self.patch_size) // self.stride[0], 1 + (W - self.patch_size) // self.stride[1])
       

        self._register_hooks(layer, facet)
        try:
            with torch.no_grad():
                _ = self.model(batch)
        finally:
            self._unregister_hooks()
        assert self._feats!=[], "Features were not extracted. Make sure the hook was registered correctly."

        
        return self._feats[0]
    
    def extract_descriptors(self, batch, layer=None, facet=None, include_cls= False):
        features = self._extract_features(batch, [layer], facet)

        # case for attn
        if facet == "attn":
            if not include_cls:
                desc = features[:, :, 1:, 1:]
            else:
                desc = features
        
        # case for kqvt
        else:
            if facet == 'token':
                features.unsqueeze_(dim=1) #Bx1xtxd
            if not include_cls:
                features =  features[:, :, 1:, :]  # remove cls token
            desc = features.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1).unsqueeze(dim=1)  # Bx1xtx(dxh)
        return desc
    
    
    
   