#  ==============================================================================
#  Description: Helper functions for the model.
#  Copyright (C) 2024 Xin Li
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================
import os

import numpy as np
import torch
from PIL import Image
from torchvision.io import read_image
from torch.utils.data import Dataset
import albumentations
from albumentations.pytorch import ToTensorV2

class FFHQDataset(Dataset):
    '''Get training data_paths from the FFHQ dataset.'''

    def __init__(self, root, size=512, crop_size=256, transform=None):
        self.root = root
        self.size = size
        self.crop_size = crop_size
        self.transform = transform
        self.data_paths = os.listdir(root)
        self.data_paths.sort()

        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        self.random_crop = albumentations.RandomCrop(height=self.crop_size, width=self.crop_size)
        self.preprocess = albumentations.Compose([self.rescaler,
                                                  self.random_crop])

    def __len__(self):
        return len(self.data_paths)

    def preprocess_image(self, image_path):
        # image = read_image(os.path.join(self.root, image_path)).detach().numpy()
        #
        image = Image.open(os.path.join(self.root, image_path))
        if not image.mode == 'RGB':
            image = image.convert('RGB')
        image = np.array(image).astype(np.uint8)
        image = self.preprocess(image=image)['image']
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)

        return image

    def __getitem__(self, index):
        image_path = self.data_paths[index]
        image = self.preprocess_image(image_path)
        if self.transform:
            image = self.transform(image)

        return image
