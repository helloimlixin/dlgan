#  ==============================================================================
#  Description: training utils for the vqvae.
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
import albumentations
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

"""
Data utils.
"""
class ImageDataset(Dataset):
    def __init__(self, image_dir, size=None):
        self.size = size

        self.image_paths = [
            os.path.join(image_dir, file) for file in os.listdir(image_dir)
        ]

        self._length = len(self.image_paths)

        self.rescaler = albumentations.SmallestMaxSize(
            max_size=self.size
        )  # resize to square
        self.cropper = albumentations.CenterCrop(
            height=self.size, width=self.size
        )  # center crop to square
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        self.images = []
        for img_path in self.image_paths:
            self.images.append(self.preprocess_image(img_path))

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        """Preprocess an image.

        Args:
            image_path (str): path to th image file.

        Returns:
            np.array(np.float32): C X H X W float32 numpy array.
        """
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)[
            "image"
        ]  # apply rescaling and center cropping
        image = (image / 127.5 - 1.0).astype(np.float32)  # normalize to [-1, 1]
        image = image.transpose(2, 0, 1)  # convert to channels first

        return image

    def __getitem__(self, i):
        sample = self.images[i]

        return sample


def load_data(args):
    """Load the data_paths.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        torch.utils.data.DataLoader: DataLoader for the dataset.
    """
    train_dataset = ImageDataset(args.dataset_path, size=args.image_size)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )

    return train_dataloader


"""
Module utils.
"""


def init_weights(m):
    """Initialize the weights of the module.

    Args:
        m (nn.Module): Module to initialize the weights of.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def plot_images(images):
    """Plot a batch of images.

    Args:
        images (torch.Tensor): Batch of images to plot.
    """
    x = images["input"]
    reconstruction = images["reconstruction"]
    half_sample = images["half_sample"]
    full_sample = images["full_sample"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(x.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axes[1].imshow(reconstruction.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axes[2].imshow(half_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axes[3].imshow(full_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    plt.show()