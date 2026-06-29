from PIL import Image
import torch
import numpy as np
import torchvision.transforms.functional as F
import torchvision.transforms as T
from .local_utils import ToTargetTensor, normalize
import random

def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)

# ── augmentation ──────────────────────────────────────────────────────────────
class ParamRandomResizedCrop:
    def __init__(self, size, scale=(0.8, 1.0), ratio=(3/4, 4/3)):
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def sample(self, img):
        return T.RandomResizedCrop.get_params(
            img,
            scale=self.scale,
            ratio=self.ratio,
        )

    def apply(self, img,  label, params):
        i, j, h, w = params

        img = F.resized_crop(img, i, j, h, w, self.size, interpolation=F.InterpolationMode.BILINEAR)
        label = F.resized_crop(label, i, j, h, w, self.size, interpolation=F.InterpolationMode.NEAREST)

        return img, label

class ParamRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def sample(self, img):
        return torch.rand(1).item() < self.p

    def apply(self, img, label, params):
        if params:
            img = F.hflip(img)
            label = F.hflip(label)
        return img, label


class ParamCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def sample(self, img):
        return [t.sample(img) for t in self.transforms]

    def apply(self, img, label, params_list):
        for t, p in zip(self.transforms, params_list):
            img, label = t.apply(img, label, p)
        return img, label
    
# ── Preprocessor ──────────────────────────────────────────────────────────────    
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
    
# ── colormap ──────────────────────────────────────────────────────────────
def create_pascalpart_colormap():
    colors =[(0, 0, 0),
            (220, 20, 60), 
            (51, 255, 153),
            (220, 220, 0), 
            (70, 130, 180), 
            (255, 51, 255)]
    return np.array(colors)

def bit_get(val, idx):
    """Gets the bit value.
    Args:
      val: Input value, int or numpy int array.
      idx: Which bit of the input val.
    Returns:
      The "idx"-th bit of input val.
    """
    return (val >> idx) & 1

def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.
    Returns:
      A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((512, 3), dtype=int)
    ind = np.arange(512, dtype=int)

    for shift in reversed(list(range(8))):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3

    return colormap

def create_cityscapes_colormap():
    colors = [(128, 64, 128),
              (244, 35, 232),
              (250, 170, 160),
              (230, 150, 140),
              (70, 70, 70),
              (102, 102, 156),
              (190, 153, 153),
              (180, 165, 180),
              (150, 100, 100),
              (150, 120, 90),
              (153, 153, 153),
              (153, 153, 153),
              (250, 170, 30),
              (220, 220, 0),
              (107, 142, 35),
              (152, 251, 152),
              (70, 130, 180),
              (220, 20, 60),
              (255, 0, 0),
              (0, 0, 142),
              (0, 0, 70),
              (0, 60, 100),
              (0, 0, 90),
              (0, 0, 110),
              (0, 80, 100),
              (0, 0, 230),
              (119, 11, 32),
              (0, 0, 0)]
    return np.array(colors)


def get_class_labels(dataset_name):
    if dataset_name.startswith("cityscapes"):
        return [
            'road', 'sidewalk', 'parking', 'rail track', 'building',
            'wall', 'fence', 'guard rail', 'bridge', 'tunnel',
            'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation',
            'terrain', 'sky', 'person', 'rider', 'car',
            'truck', 'bus', 'caravan', 'trailer', 'train',
            'motorcycle', 'bicycle']
    elif dataset_name == "cocostuff27":
        return [
            "electronic", "appliance", "food", "furniture", "indoor",
            "kitchen", "accessory", "animal", "outdoor", "person",
            "sports", "vehicle", "ceiling", "floor", "food",
            "furniture", "rawmaterial", "textile", "wall", "window",
            "building", "ground", "plant", "sky", "solid",
            "structural", "water"]
    elif dataset_name == "pascalpart":
        return[
            'background',
            'ear',
            'head',
            'torso',
            'leg',
            'tail'
        ]
    else:
        raise ValueError("Unknown Dataset {}".format(dataset_name))
    

def get_dataset_colormap(dataset_name, n_classes):
    """Return (n_classes, 3) uint8 colormap array for the given dataset."""
    if dataset_name == "cocostuff27":
        # cocostuff27 uses the same colormap as Pascal VOC
        return create_pascal_label_colormap()[:n_classes]
    elif dataset_name == "pascalpart":
        assert n_classes <= 6, "Pascal Part dataset has only 6 classes, please adjust code accordingly if n_classes > 6 is needed."
        return create_pascalpart_colormap()[:n_classes]
    elif dataset_name == "cityscapes":
        return create_cityscapes_colormap()[:n_classes]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    

def get_transform(res, is_label, crop_type):
    if crop_type == "center":
        cropper = T.CenterCrop(res)
    elif crop_type == "random":
        cropper = T.RandomCrop(res)
    elif crop_type is None:
        cropper = T.Lambda(lambda x: x)
        res = (res, res)
    else:
        raise ValueError("Unknown Cropper {}".format(crop_type))
    if is_label:
        return T.Compose([T.Resize(res, Image.NEAREST),
                          cropper,
                          ToTargetTensor()])
    else:
        return T.Compose([T.Resize(res, Image.BILINEAR),
                          cropper,
                          T.ToTensor(),
                          normalize])