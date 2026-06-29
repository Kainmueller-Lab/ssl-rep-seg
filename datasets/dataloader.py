
from .cityscapes import CityscapesSeg
from .cocostuff import Coco
from .pascalpart import PascalPartDataset
from torch.utils.data import DataLoader,Dataset
from .agumentation import get_transform, ParamCompose, ParamRandomHorizontalFlip, ParamRandomResizedCrop, worker_init_fn
from .local_utils import flexible_collate
import torchvision.transforms as T

class ContrastiveSegDataset(Dataset):
    def __init__(self,
                 dataset_root,
                 dataset_name,
                 image_set,
                 transform,
                 target_transform,
                 aug_geometric_transform=None,
                 aug_photometric_transform=None,
                 apply_geometric_aug =False,
                 apply_photometric_aug =False
                 ):
        super(ContrastiveSegDataset).__init__()
        self.image_set = image_set
        self.dataset_name = dataset_name

        self.apply_geometric_aug = apply_geometric_aug
        self.apply_photometric_aug = apply_photometric_aug


        if dataset_name == "cityscapes":
            self.n_classes = 27
            dataset_class = CityscapesSeg
            extra_args = dict()
        # elif dataset_name == "cocostuff3":
        #     self.n_classes = 3
        #     dataset_class = Coco
        #     extra_args = dict(coarse_labels=True, subset=6, exclude_things=True)
        # elif dataset_name == "cocostuff15":
        #     self.n_classes = 15
        #     dataset_class = Coco
        #     extra_args = dict(coarse_labels=False, subset=7, exclude_things=True)
        elif dataset_name == "cocostuff27":
            print(f"Chosing cocostuff27  the full images")
            self.n_classes = 27
            dataset_class = Coco
            extra_args = dict(coarse_labels=False, subset=None, exclude_things=False)
            if image_set == "val":
                extra_args["subset"] = 7

        elif dataset_name == "pascalpart": #pascalpart
            self.n_classes = 6 #until now it is 5 classes, could be 6
            dataset_class = PascalPartDataset
            extra_args = dict()
        else:
            raise ValueError("Unknown dataset: {}".format(dataset_name))

        self.aug_geometric_transform = aug_geometric_transform
        self.aug_photometric_transform = aug_photometric_transform

        self.dataset = dataset_class(
            root=dataset_root,
            image_set=self.image_set,
            transform=transform,
            target_transform=target_transform, **extra_args)


    def __len__(self):
        return len(self.dataset)



    def __getitem__(self, ind):
        pack = self.dataset[ind]
       
        img, label = pack[0], pack[1]
        if self.apply_geometric_aug and self.aug_geometric_transform is not None:
            params = self.aug_geometric_transform.sample(img)
            img, label =self.aug_geometric_transform.apply(img, label.unsqueeze(0), params) # must unsqueeze label to have a channel dimension
            label = label.squeeze(0) # remove the channel dimension after transformation

        if self.apply_photometric_aug and self.aug_photometric_transform is not None:
            img = self.aug_photometric_transform(img)

        ret = {
            "ind": ind,
            "img": img,
            "label": label,
        }
        return ret

def build_testdataloader(
    dataset_root,
    dataset_name,
    resize,
    batch_size,
    num_workers
):

    test_dataset = ContrastiveSegDataset(
        dataset_root=dataset_root,
        dataset_name=dataset_name,
        image_set="val",
        transform=get_transform(resize[0], False, "center"),
        target_transform=get_transform(resize[0], True, "center")
    )

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=flexible_collate,
    )





def build_train_val_dataloaders(
    dataset_root,
    dataset_name,
    resize,
    batch_size,
    num_workers,
    scale_ratio,
    train_loader_crop_type,
    val_loader_crop_type,
    train_image_set,
    apply_geometric_aug,
    apply_photometric_aug,
 
):
    geometric_transforms= ParamCompose([
    ParamRandomHorizontalFlip(),
    ParamRandomResizedCrop(size=resize, scale=(scale_ratio, 1.0))
    ])

    photometric_transforms = T.Compose([
        T.ColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=.1),
        T.RandomGrayscale(.2),
        T.RandomApply([T.GaussianBlur((5, 5))])
    ])

    train_dataset = ContrastiveSegDataset(
    dataset_root=dataset_root,
    dataset_name=dataset_name,
    image_set=train_image_set,
    transform=get_transform(resize[0], False, train_loader_crop_type),
    target_transform=get_transform(resize[0], True, train_loader_crop_type),
    aug_geometric_transform=geometric_transforms,
    aug_photometric_transform=photometric_transforms,
    apply_geometric_aug= apply_geometric_aug,
    apply_photometric_aug= apply_photometric_aug
    )

    val_dataset = ContrastiveSegDataset(
    dataset_root=dataset_root,
    dataset_name=dataset_name,
    image_set= "val",
    transform=get_transform(resize[0], False,  val_loader_crop_type),  # originally cfg.resize=320
    target_transform=get_transform(resize[0], True,  val_loader_crop_type), # originally cfg.resize=320
    )

   
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers,worker_init_fn=worker_init_fn, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_init_fn,pin_memory=True)

    return train_loader, val_loader