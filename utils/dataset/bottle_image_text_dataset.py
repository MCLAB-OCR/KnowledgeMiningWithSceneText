import json
import os
import os.path as path
import pickle
import random
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.datasets.vision import VisionDataset

from utils.lazy_list import LazyList
from utils.log import log, logger


# from .utils import check_integrity, download_and_extract_archive
class StrList(object):

    def __init__(self, str_):
        self.str_ = str_

    def __getitem__(self, item):
        return self


class BottleImageText(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(self,
                 root: str,
                 text_root: str,
                 category_path: str = None,
                 google_ocr_path: str = None,
                 use_google_ocr=False,
                 use_paddle_ocr=False,
                 use_category=False,
                 split=0,
                 set_name: str = 'train',
                 text_random_shuffle: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 using_cache=False,
                 dataset_type='bottle') -> None:

        super(BottleImageText,
              self).__init__(root,
                             transform=transform,
                             target_transform=target_transform)
        self.dataset_type = dataset_type
        self.train = set_name == 'train'  # training set or test set
        self.text_root = text_root
        self.category_path = category_path
        self.google_ocr_path = google_ocr_path
        self.use_google_ocr = use_google_ocr
        self.use_paddle_ocr = use_paddle_ocr
        self.use_category = use_category
        self.split = split
        self.text_random_shuffle = True if self.train and text_random_shuffle else False
        self.datasets = []
        with open(path.join(root, f'split_{split}.json'), 'r') as f:
            j = json.load(f)
        for k, v in j[set_name].items():
            self.datasets.append([os.path.join(root, k), int(v) - 1])

        self.imgs = LazyList(
            lambda index: Image.open(self.datasets[index][0]).convert('RGB'),
            using_cache=using_cache)

        if self.use_category:
            self.category_path = Path(self.category_path)
            with open(
                    self.category_path / f"split_{split}" / f"{set_name}.pkl",
                    'rb') as f:
                self.categories_in_imgs = pickle.load(f)

    def load_text(self, path, mode='txt'):
        max_word_length = 100
        if mode == 'google_ocr':
            lines = np.load(path, allow_pickle=True)
            words = lines.item()['word']
            bboxes = torch.empty(0)
        elif mode == 'txt':
            lines = open(path).readlines()
            words = [v.split(',')[-1].rstrip('\n') for v in lines]
            bboxes = torch.empty(0)
        elif mode == 'paddle_ocr':
            lines = np.load(path, allow_pickle=True)
            words = [x[1][0] for x in lines]
            bboxes = [x[0] for x in lines]
            if len(bboxes) == 0:
                bboxes = torch.zeros([max_word_length, 4, 2],
                                     dtype=torch.float32)
            else:
                bboxes = torch.tensor(bboxes, dtype=torch.float32)

                if bboxes.shape[0] >= max_word_length:
                    bboxes = bboxes[:max_word_length, ...]
                else:
                    z = torch.zeros([max_word_length - bboxes.shape[0], 4, 2],
                                    dtype=torch.float32)
                    bboxes = torch.cat([bboxes, z], dim=0)

        filter_english = False
        if filter_english:
            import re
            words = ' '.join(words)
            fil = re.compile(r'''[^0-9a-zA-Z_.,;:"'?!]+''', re.UNICODE)
            words = fil.sub(' ', words).strip().split(' ')

        # if len(lines) == 0:
        #     return self.dataset_type
        if self.text_random_shuffle:
            random.shuffle(words)
        if len(words) > max_word_length:
            logger.info(
                f"len(words) of {path} is too long({len(words)}). truncated to {max_word_length}."
            )
            words = words[:max_word_length]
        res = ' '.join(words)
        res = res.strip()
        if len(res) == 0:
            # if self.dataset_type == "bottle":
            #     res = "bottle"
            # elif self.dataset_type == 'context':
            #     res = 'building'
            # elif self.dataset_type == 'activity':
            #     res = 'word'
            res = "[empty]"

        return res, bboxes

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        dataset_type = self.dataset_type
        img_path, target = self.datasets[index]
        img_path = Path(img_path)
        text_root = Path(self.text_root)
        if self.google_ocr_path is not None:
            google_ocr_path = Path(self.google_ocr_path)

        if self.use_google_ocr:
            root = google_ocr_path
            ext = '.npy'
        elif self.use_paddle_ocr:
            root = text_root
            ext = '.pkl'
        else:
            root = text_root
            ext = '.txt'

        if dataset_type == 'bottle':
            text_path = root.joinpath(*img_path.parts[-2:-1],
                                      img_path.stem + ext)
        elif dataset_type == 'activity':
            text_path = root.joinpath(*img_path.parts[-4:-1],
                                      img_path.stem + ext)
        elif dataset_type == 'context':
            text_path = root / (img_path.stem + ext)

        if self.use_google_ocr:
            texts, text_bboxes = self.load_text(str(text_path),
                                                mode='google_ocr')
        elif self.use_paddle_ocr:
            texts, text_bboxes = self.load_text(str(text_path),
                                                mode='paddle_ocr')
        else:
            texts, text_bboxes = self.load_text(str(text_path), mode='txt')

        # print(texts)
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = self.imgs[index]

        if self.use_paddle_ocr:
            text_bboxes /= torch.tensor(img.size)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.use_category:
            texts = " ".join(self.categories_in_imgs[index]) + " " + texts

        # max_len = 10000
        # if len(texts) > max_len:
        #     logger.info(
        #         f"len(texts) of {img_path} is too long({len(texts)}). truncated to {max_len}."
        #     )
        #     texts = texts[:max_len]

        return img, texts, target, str(img_path), text_bboxes

    def __len__(self) -> int:
        return len(self.datasets)

    def extra_repr(self) -> str:
        return 'Split: {}'.format('Train' if self.train is True else 'Test')
