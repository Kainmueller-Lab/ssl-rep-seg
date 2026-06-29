
from torchvision.datasets.cityscapes import Cityscapes
from torch.utils.data import Dataset
from os.path import join
import torch
import numpy as np
import random

class CityscapesSeg(Dataset):
    def __init__(self, root, image_set, transform, target_transform):
        super(CityscapesSeg, self).__init__()
        self.split = image_set
        self.root = join(root, "cityscapes")
        if image_set == "train":
            # our_image_set = "train_extra"
            # mode = "coarse"
            our_image_set = "train"
            mode = "fine"
        else:
            our_image_set = image_set
            mode = "fine"
        self.inner_loader = Cityscapes(self.root, our_image_set,
                                       mode=mode,
                                       target_type="semantic",
                                       transform=None,
                                       target_transform=None)
        self.transform = transform
        self.target_transform = target_transform
        self.first_nonvoid = 7

    def __getitem__(self, index):
        if self.transform is not None:
            image, target = self.inner_loader[index]

            # --- Save RNG states ---
            torch_state = torch.get_rng_state()
            random_state = random.getstate()
            numpy_state = np.random.get_state()

            # --- Apply transform to image ---
            image = self.transform(image)
           
             # --- Restore RNG states ---
            torch.set_rng_state(torch_state)
            random.setstate(random_state)
            np.random.set_state(numpy_state)

            # --- Apply transform to label ---
            target = self.target_transform(target)

            target = target - self.first_nonvoid
            target[target < 0] = -1
            mask = target == -1
            return image, target.squeeze(0), mask
        else:
            return self.inner_loader[index]

    def __len__(self):
        return len(self.inner_loader)