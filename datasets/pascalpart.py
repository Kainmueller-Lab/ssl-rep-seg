'''
Code reference page: https://github.com/CSAILVision/unifiedparsing/blob/master/broden_dataset_utils/pascalseg.py

second reference:
https://github.com/facebookresearch/VLPart/tree/main/datasets#pascal-part

https://github.com/YuqiYang213/MLoRE/blob/main/data/pascal_context.py


https://github.com/ANDYZAQ/GF-SAM/blob/master/datasets/README.md

PASCAL-Part and PACO-Part


Prerequisites:
1.donwload the pascal parts dataset from https://roozbehm.info/pascal-parts/pascal-parts.html
2. pascal context label https://cs.stanford.edu/~roozbeh/pascal-context/
'''

import os
import re
import warnings
import random
import numpy
from scipy.io import loadmat
import torch
from PIL import Image
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset

class AbstractSegmentation:
    def all_names(self, category, j):
        raise NotImplementedError

    def size(self, split=None):
        return 0

    def filename(self, i):
        raise NotImplementedError

    def metadata(self, i):
        return self.filename(i)

    @classmethod
    def resolve_segmentation(cls, m):
        return {}

    def name(self, category, i):
        """
        Default implementation for segmentation_data,
        utilizing all_names.
        """
        all_names = self.all_names(category, i)
        return all_names[0] if len(all_names) else ''

    def segmentation_data(self, category, i, c=0, full=False):
        """
        Default implementation for segmentation_data,
        utilizing metadata and resolve_segmentation.
        """
        segs, segs_shape = self.resolve_segmentation(
            self.metadata(i), categories=[category])
        if category not in segs:
            return 0
        data = numpy.asarray(segs[category])
        if not full and len(data.shape) >= 3:
            return data[0]
        return data

    def image_data(self, i):
        return Image.open(self.filename(i)).convert('RGB')
    

class PascalSegmentation(AbstractSegmentation):
    """
    Implements AbstractSegmentation for the pascal PARTS dataset.
    """

    def __init__(self, directory, image_set, collapse_adjectives=None, version='VOC2010'):
        assert image_set in ["train", "val"], "image_set must be 'train' or 'val'"
        directory = os.path.expanduser(directory)
        # Default to the latest version present in the directory
        if version is None:
            contents = os.listdir(directory)
            if not list(c for c in contents if re.match('^JPEGImages$', c)):
                version = sorted(c for c in contents if c.startswith('VOC') and
                                 os.path.isdir(os.path.join(directory, c)))[-1]
            else:
                version = ''
        print(f"detect pascal version: {version}")
        self.directory = directory
        # self.version = version
        self.version = ''
        # Default to using the 'part' directory for part annotations
        self.partdir = 'part'
        if os.path.isfile(os.path.join(directory, 'part2ind.m')):
            self.partdir = ''
        # Default to using the 'part' directory for part annotations
        # self.contextdir = 'context'
        self.contextdir = 'Annotations_Part'
        # Default to removing left/right/front/back adjectives
        self.collapse_adjectives = collapse_adjectives
        # Load the parts coding metadata from part2ind.m
        # codes = load_part2ind(
        #     os.path.join('./meta_file/pascal/', 'part2ind.m'))
        codes = load_part2ind(
            os.path.join(directory, 'part2ind.m'))
        # Normalized names
        self.codes = normalize_all_readable(codes, collapse_adjectives)
   
      
        merge_rules = {
            "eye": "head",
            # "ear": "head", # TODO check if we let the ear out
            "horn": "head",
            "nose": "head",
            "muzzle": "head",
            "neck": "torso",
            "paw": "leg",
            'hoof': 'leg',
        }


        animal_classes = [ "cat", "cow", "dog", "horse", "sheep"]
        invalid_imgs =["2008_006690"] # the image contains persons, we exclude it since we want to focus on animal-only images.
        # filter image with animals
        ainimal_segs = self.filter_animials(animal_classes, invalid_imgs_list=invalid_imgs) 

        print("Merging parts according to rules:")
        self.codes = self.merge_part_labels(self.codes, merge_rules)
        # for k,v in self.codes.items():
        #     if k[0] in [8,10,12,13,17]: #print the cow case
        #         print(k, v)
        

        self.part_object_names, self.part_names, self.part_key = normalize_part_key(self.codes)
        # use to check the part names and their corresponding indexes
        # for i, name in enumerate(self.part_names):
        #     print(i, name)
       
        
        # Load the PASCAL context segmentation labels 
        # self.object_names = load_context_labels(
        #     os.path.join('./meta_file/pascal/', 'context_labels.txt'))
        self.object_names = load_context_labels(
            os.path.join(self.directory,'pascalpart', 'labels.txt')) # TODO hard coded, change here 
        self.unknown_label = self.object_names.index('unknown')
        self.object_names[self.unknown_label] = '-'  # normalize unknown
        
        split_ids = set()
        for cls in animal_classes:
            path = os.path.join(directory, 'ImageSets', 'Main', f'{cls}_{image_set}.txt')
            ids = {img_id + ".mat"
                for line in open(path )
                for img_id, label in [line.split()]
                if label == "1"
                }
            split_ids |= ids   # UNION

        # Assume every mat file in the relevant directory is a segmentation.
        try:
            ann_dir = os.path.join(directory, self.partdir, 'Annotations_Part')
            self.segs = sorted(
                n for n in os.listdir(ann_dir)
                if n in split_ids
            )
        except OSError:
            message = 'Error when searching for pascal part annotations, please check your dataset.' \
                        + ' With this error you may only use testing scripts. Training will fail unless you resolve this warning.'
            warnings.warn(message)


        print(f"{image_set} set: Before filtering animal class, {len(self.segs)} images.")
        self.segs = [x for x in ainimal_segs if x in self.segs]
        print(f"{image_set} set: After filtering animal class, {len(self.segs)} images.")

    def merge_part_labels(self, codes, merge_rules):
        collapsed = {
            k: (obj, merge_rules.get(part, part))
            for k, (obj, part) in codes.items()
        }
        return collapsed


    def all_names(self, category, j):
        if category == 'object':
            return [self.object_names[j]]
        elif category == 'part':
            # TODO: include left/right synonyms that have been collapsed
            return [self.part_names[j]]
        else:
            print('Could not find', category, j)
            raise ValueError

    def size(self):
        return len(self.segs)
    

    def filter_animials(self, animal_classes, area_threshold= None, invalid_imgs_list = None):
        img_list=[]
        annotations_dir = os.path.join(self.directory, "Annotations")
        for filename in os.listdir(annotations_dir):
            if not filename.endswith(".xml"):
                continue  
            filepath = os.path.join(annotations_dir, filename)
            tree = ET.parse(filepath)
            root = tree.getroot()

            size = root.find("size")
            width = int(size.find("width").text)
            height = int(size.find("height").text)
            image_area = width * height

            # Check objects inside annotation
            has_animal = False
            has_other = False
            for obj in root.findall("object"):
                cls = obj.find("name").text.lower()
                if cls in animal_classes:
                    bbox = obj.find("bndbox")
                    xmin = int(bbox.find("xmin").text)
                    ymin = int(bbox.find("ymin").text)
                    xmax = int(bbox.find("xmax").text)
                    ymax = int(bbox.find("ymax").text)
                    box_area = (xmax - xmin) * (ymax - ymin)
                    if area_threshold is not None:
                        if box_area> image_area*area_threshold:
                            has_animal = True
                    else:
                        has_animal = True
                else:
                    # exclude if any other object is present
                    has_other = True
                    break  
                    
            # Save only animal-related annotations
            if has_animal and not has_other:
               img_list.append(filename.replace(".xml", ".mat"))

        if invalid_imgs_list is not None:
            for item in invalid_imgs_list:
                if item+".mat" in img_list:
                    img_list.remove(item+".mat")

        print(f"Find {len(img_list)} images containing {animal_classes}.")
        return img_list


    def filename(self, i):
        # Translate the segmentation mat file name to the .jpg filename
        return os.path.join(self.directory, self.version, 'JPEGImages',
                            self.segs[i].replace('.mat', '.jpg'))

    def metadata(self, i):
        return (self.directory, self.partdir, self.contextdir, \
                self.part_key, self.segs[i])

    @classmethod
    def resolve_segmentation(cls, m, categories=None):
        directory, partdir, contextdir, part_key, seg_fn = m
        result = {}
        if wants('part', categories):
            objs, parts = load_parts_segmentation(
                os.path.join(directory, partdir, 'Annotations_Part', seg_fn),
                part_key)  # (ignore object layer the from parts segmentation)
            result['part'] = parts
        if wants('object', categories):
            result['object'] = loadmat(os.path.join(directory, contextdir, 'trainval', seg_fn))['LabelMap']
        arrs = [a for a in list(result.values()) if len(numpy.shape(a)) >= 2]
        shape = arrs[0].shape[-2:] if arrs else (1, 1)
        return result, shape


def load_parts_segmentation(filename, part_key):
    """
    Processes a single Annotations_Part annotation into two arrays:
    an object class segmentation, and a part segmentation.  We discard
    instance information for now.  If no objects are present, returns (0, 0).
    """
    d = loadmat(filename)
    instance_count = d['anno'][0, 0]['objects'].shape[1]
    # We need at least one instance annotated
    if not instance_count:
        return (0, 0)
    mask_shape = d['anno'][0, 0]['objects'][0, 0]['mask'].shape
    # We will merge all objects and parts into these two layers
    object_seg = numpy.zeros(mask_shape, dtype=numpy.int16)
    part_seg = numpy.zeros(mask_shape, dtype=numpy.int16)
    for i in range(instance_count):
        obj = d['anno'][0, 0]['objects'][0, i]
        object_ind = obj['class_ind'][0, 0]
        object_seg[obj['mask'].astype(numpy.bool)] = object_ind
        part_count = obj['parts'].shape[1]
        for i in range(part_count):
            part = obj['parts'][0, i]
            part_name = part['part_name'][0]
            part_code = part_key[(object_ind, part_name)]
            part_seg[part['mask'].astype(numpy.bool)] = part_code
    return object_seg, part_seg


def normalize_part_key(raw_names):
    """
    Enforces a coding policy: all the multiple part names such as 'engine 3'
    are just called 'engine', and they are aliased down to the same part code.
    Returns names of object classes, parts, and a part_key mapping raw
    (object, part) number pair tuples to the canonical integer part code.
    """
    # Identify the maximum object and part indexes used.
    object_class_count = max(c for c, p in raw_names) + 1
    # Create a list of object names, a list of part names, and a part key
    object_class_names = ['-'] * object_class_count
    part_class_names = ['-']
    # alias_key map (collapsed obj name, collapsed part name) -> assigned part index. 
    alias_key = {}
    part_key = {}

    for (c, p) in sorted(list(raw_names.keys())):
        n = raw_names[(c, p)]
        object_class_names[c] = n[0]
        # Alias codes that have the same name such as 'airplane engine' (1,2,3)
        if n in alias_key:
            code = alias_key[n]
        else:
            if n[1] in part_class_names:
                code = part_class_names.index(n[1])
            else:
                code = len(part_class_names)
                part_class_names.append(n[1])
            alias_key[n] = code
        part_key[(c, p)] = code
    return object_class_names, part_class_names, part_key


def normalize_all_readable(raw_keys, collapse_adjectives):
    return dict((k, (normalize_readable(c, collapse_adjectives),
                     normalize_readable(p, collapse_adjectives)))
                for k, (c, p) in list(raw_keys.items()))


def normalize_readable(name, collapse_adjectives):
    # Long names for short part names that are unexplained in the file.
    decoded_names = dict(
        lfho='left front hoof',
        rfho='right front hoof',
        lbho='left back hoof',
        rbho='right back hoof',
        fwheel='front wheel',
        bwheel='back wheel',
        frontside='front side',
        leftside='left side',
        rightside='right side',
        backside='back side',
        roofside='roof side',
        leftmirror='left mirror',
        rightmirror='right mirror'
    )
    if name in decoded_names:
        name = decoded_names[name]
    # Other datasets use 'airplane'.
    name = name.replace('aeroplane', 'airplane')
    # If we need to remove adjectives like 'left' and 'right', do so now.h
    if collapse_adjectives is not None:
        name = ' '.join(n for n in name.split() if n not in collapse_adjectives)
    return name


def load_context_labels(labels_filename):
    """
    Parses labels.txt from the PASCAL Context distribution;
    this contains an "index: name" mapping.
    """
    with open(labels_filename, 'r') as labels_file:
        lines = [s.strip().split(': ') for s in labels_file.readlines()]
    pairs = [(int(i), n) for i, n in lines]
    object_names = ['-'] * (max(i for i, n in pairs) + 1)
    for i, n in pairs:
        object_names[i] = n
    return object_names


def load_part2ind(part2ind_filename):
    """
    Parses the part2ind.m file from the PASCAL Parts distribution
    just as a series of line patterns, returning a map from (object, part)
    number paris to (objectname, partname) names.
    """
    import re
    from collections import OrderedDict
    with open(part2ind_filename, 'r') as part2ind_file:
        lines = part2ind_file.readlines()
    result = OrderedDict()
    for line in lines:
        # % [aeroplane]
        m = re.match('^% \[([^\]]*)\]', line)
        if m:
            object_name = m.group(1)
            continue
        # pimap{1}('lwing')       = 3;                % left wing
        m = re.match(r'''(?x)^
            pimap\{(\d+)\}            # group 1: the object index
            \('([^']*)'\)             # group 2: the part short name
            \s*=\s*                   # equals
            (\d+);                    # group 3: the part number
            (?:\s*%\s*(\w[\w ]*\w))?  # group 4: the part long name
            ''', line)
        if m:
            part_name = m.group(2)
            readable_name = m.group(4) or part_name
            object_index = int(m.group(1))
            part_index = int(m.group(3))
            # ('aeroplane', 'left wing')
            result[(object_index, part_name)] = (
                object_name, readable_name)
            continue
        # for ii = 1:10
        m = re.match(r'for ii = 1:(\d+)*', line)
        if m:
            iteration_count = int(m.group(1))
            continue

        # pimap{1}(sprintf('engine_%d', ii)) = 10+ii; % multiple engines
        m = re.match(r'''(?x)^\s*
            pimap\{ (\d+) \}          # group 1: the object index
            \(sprintf\('
                ([^']*)_%d            # group 2: the short part name
            ',\s*ii\)\)
            \s*=\s*                   # equals
            (\d+)                     # group 3: the part number
            \s*\+\s*ii\s*;            # plus indexing
            (?:\s*%\s*(\w[\w ]*\w))?  # group 4: the multiple-part name
            ''', line)
        if m:
            # Deal with ranges
            part_name = m.group(2)
            # Take a multiple name if it does not say 'multiple' in it.
            readable_name = part_name
            if m.group(4) and 'multiple' not in m.group(4):
                readable_name = m.group(4)
            object_index = int(m.group(1))
            first = int(m.group(3)) + 1  # one-based indexing
            for part_index in range(1, 1 + iteration_count):
                result[(object_index, part_name + '_%d' % part_index)] = (
                    object_name, readable_name)
            continue

        # % only has sihouette mask
        m = re.match(r'% only has si', line)  # (misspelled in file)
        if m:
            object_index += 1
            result[(object_index, 'silhouette')] = (
                object_name, 'silhouette')

        # keySet = keys(pimap{8});
        # valueSet = values(pimap{8});
        m = re.match(r'''(?x)^
            (key|value)Set\s*=\s*\1s\(pimap{ # group 1: key or value
            (\d+)                            # group 2: old object number
            }\);
            ''', line)
        if m:
            object_to_copy = int(m.group(2))
            continue

        # pimap{12} = containers.Map(keySet, valueSet);
        m = re.match(r'''(?x)^
            pimap{
            (\d+)                           # group 1: new object number
            }\s*=\s*containers.Map\(keySet,\s*valueSet\);
            ''', line)
        if m:
            object_index = int(m.group(1))
            for (other_obj, part_index), val in list(result.items()):
                if other_obj == object_to_copy:
                    result[(object_index, part_index)] = (object_name, val[1])
            continue
        # recognize other lines that can be ignored
        m = re.match(r'''(?x)^(?:
                function|
                pimap\ =|
                \s*pimap\{ii\}|
                end|
                \s*%|
                remove|
                $)''', line)
        if m:
            continue
        print('unrecognized line', line)
        import sys
        sys.exit(1)
    return result


def wants(what, option):
    if option is None:
        return True
    return what in option
###############################################################
# pascal part dataset class
###############################################################

class PascalPartDataset(Dataset):
    def __init__(
        self,
        root,
        image_set,
        transform, 
        target_transform,
    ):
        """
        Pascal-Part dataset wrapper mimicking CocoDataset structure.
        Uses PascalSegmentation to resolve object/part segmentations.
        """
        self.root = os.path.join(root,"VOC2010")
        self.split = image_set
        self.transform = transform
        self.label_transform = target_transform

        # Load PascalSegmentation object
        self.segmentation = PascalSegmentation(self.root, self.split,
                            collapse_adjectives=['left','right','front', 'back','upper','lower'], version='VOC2010')

        # Build dataset entries (images + metadata)
        self.dataset = self.load_data()

        print(f"Found {len(self.dataset)} images in the {self.split} split.")
        

        # Number of object classes in Pascal-Part (20 VOC + background)
        self.n_classes = len(self.segmentation.object_names)

    def load_data(self):
        """
        Build a dataset list of (image_path, segmentation_metadata).
        Similar to CocoDataset.load_data.
        """
        data = []
        for i in range(self.segmentation.size()):
            img_path = self.segmentation.filename(i)
            metadata = self.segmentation.metadata(i)
            data.append((img_path, metadata))
        return data

    def __len__(self):
        return self.segmentation.size()

    def apply_mapping(self, t: torch.Tensor) -> torch.Tensor:
        mapping = {
            0: 0,   # background
            5: 1,
            12: 2,
            14: 3,
            15: 4,
            24 : 5,
        }
        lut = torch.full((max(mapping.keys()) + 1,), -1, dtype=torch.long)
        for k, v in mapping.items():
            lut[k] = v
        return lut[t]

    def __getitem__(self, index):

        image = self.segmentation.image_data(index)

        # --- Save RNG states ---
        torch_state = torch.get_rng_state()
        random_state = random.getstate()
        numpy_state = numpy.random.get_state()

        # --- Apply transform to image ---
        img = self.transform(image)

        # --- Restore RNG states ---
        torch.set_rng_state(torch_state)
        random.setstate(random_state)
        numpy.random.set_state(numpy_state)

        # --- Apply transform to label ---
        img_path, metadata = self.dataset[index]
        masks, _ = PascalSegmentation.resolve_segmentation(metadata, categories=["part"]) # mask nunpy int16
        part_seg = torch.from_numpy(masks["part"]).long() # np.array->torch.tensor HxW
        # print(part_seg.unique(), "part_seg.unique") # check the part labels
        part_seg = self.apply_mapping(part_seg) # map to 0~4, contiguous labels
        label = self.label_transform(part_seg.unsqueeze(0)).squeeze(0).squeeze(0) # resize the label to 224x224
  
        mask = (label > 0).to(torch.float32)
        

        return img, label, mask