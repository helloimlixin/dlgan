#  ==============================================================================
#  Description: LPIPS loss module of the vqvae.
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
import torch
import torch.nn as nn
from torchvision.models import vgg16
from collections import namedtuple  # to create a named tuple
import requests  # to download the pretrained vqvae
from tqdm import tqdm  # to show the download progress

URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
}
CKPT_MAP = {
    "vgg_lpips": "vgg.pth",
}


def download(url, local_path, chunk_size=1024):
    """Download a file from a URL.

    Args:
        url (str): URL to download the file from.
        local_path (str): Local path to save the downloaded file.
        chunk_size (int, optional): Size of the chunks to download the file in.
            Defaults to 1024.

    Raises:
        RuntimeError: If the download fails.
    """
    os.makedirs(
        os.path.dirname(local_path), exist_ok=True
    )  # create the directory if it doesn't exist

    with requests.get(url, stream=True) as r:
        r.raise_for_status()  # raise an exception if the status code is not 200
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(
                total=total_size, unit="B", unit_scale=True, unit_divisor=1024
        ) as progress_bar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        progress_bar.update(chunk_size)


def get_ckpt_path(ckpt_name, ckpt_dir=None):
    """Get the path to the checkpoint file.

    Args:
        ckpt_name (str): Name of the checkpoint.
        ckpt_dir (str, optional): Directory to save the checkpoint in.
            Defaults to None.

    Returns:
        str: Path to the checkpoint file.
    """
    assert (
            ckpt_name in CKPT_MAP
    ), f"Checkpoint {ckpt_name} not found."  # check if the checkpoint exists
    path = os.path.join(ckpt_dir, CKPT_MAP[ckpt_name])  # get the path to the checkpoint
    if not os.path.exists(path):  # download the checkpoint if it doesn't exist
        print(f"Downloading {ckpt_name} model from {URL_MAP[ckpt_name]} to {path}...")
        download(URL_MAP[ckpt_name], path)

    return path


class LPIPS(nn.Module):
    def __init__(self):
        super(LPIPS, self).__init__()
        self.scaling_layer = (
            ScalingLayer()
        )  # scaling layer to shift and scale the image to the range [-1, 1]
        self.channels = [64, 128, 256, 512, 512]  # number of channels in each layer
        self.vgg = VGG16()  # VGG16 network
        self.linear_layers = nn.ModuleList(
            [
                NetLinLayer(self.channels[0]),
                NetLinLayer(self.channels[1]),
                NetLinLayer(self.channels[2]),
                NetLinLayer(self.channels[3]),
                NetLinLayer(self.channels[4]),
            ]
        )

        self.load_from_pretrained()  # load the pretrained weights

        # Fix the weights of the LPIPS network
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, ckpt_name="vgg_lpips"):
        ckpt_path = get_ckpt_path(
            ckpt_name, "vgg_lpips"
        )  # get the path to the checkpoint
        self.load_state_dict(
            torch.load(ckpt_path, map_location=torch.device("cpu")), strict=False
        )  # load the checkpoint

    def forward(self, real, fake):
        features_real = self.vgg(
            self.scaling_layer(real)
        )  # get the features of the real image
        features_fake = self.vgg(
            self.scaling_layer(fake)
        )  # get the features of the fake image
        diffs = {}  # dictionary to store the differences between the features

        for i in range(len(self.channels)):
            diffs[i] = (norm_tensor(features_real[i]) - norm_tensor(features_fake[i])) ** 2  # calculate the Euclidean distance between the features layer by layer

        return sum(
            [
                spatial_average(self.linear_layers[i](diffs[i]))
                for i in range(len(self.channels))
            ]
        )  # calculate the average of the differences


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer(
            "shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None]
        )  # shift the image to the range [-1, 1]
        self.register_buffer(
            "scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None]
        )  # scale the image to the range [-1, 1]

    def forward(self, x):
        return (x - self.shift) / self.scale


class VGG16(nn.Module):
    """Customized VGG16 network with layer outputs as named tuples."""

    def __init__(self):
        super(VGG16, self).__init__()
        vgg_pretrained_features = vgg16(weights="VGG16_Weights.DEFAULT").features
        slices = [
            vgg_pretrained_features[i] for i in range(30)
        ]  # get the first 30 layers of the VGG16 network
        # Slice the network into 5 parts
        self.slice1 = nn.Sequential(*slices[:4])  # first 5 layers
        self.slice2 = nn.Sequential(*slices[4:9])  # next 5 layers
        self.slice3 = nn.Sequential(*slices[9:16])  # next 7 layers
        self.slice4 = nn.Sequential(*slices[16:23])  # next 7 layers
        self.slice5 = nn.Sequential(*slices[23:30])  # last 7 layers

        # fix the weights of the VGG16 network
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        vgg_outputs = namedtuple(
            "VggOutputs", ["relu1", "relu2", "relu3", "relu4", "relu5"]
        )

        return vgg_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)


class NetLinLayer(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        """A single linear layer of the LPIPS network which consists of a dropout layer and a 1 x 1 convolutional layer."""
        super(NetLinLayer, self).__init__()
        self.model = nn.Sequential(
            nn.Dropout(),  # dropout layer
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),  # convolutional layer
        )

    def forward(self, x):
        return self.model(x)


def norm_tensor(x):
    """Normalize a tensor by their L2 norm.

    Args:
        x (torch.Tensor): Tensor to normalize.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    l2norm = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))  # calculate the L2 norm
    return x / (
            l2norm + 1e-10
    )  # normalize with a small epsilon to avoid division by zero (numerical stability)


def spatial_average(x):
    """Calculate the spatial average of a tensor.
    The image tensors have the shape (batch_size, channels, height, width), so the spatial average is calculated along
    the height and width dimensions.

    Args:
        x (torch.Tensor): Tensor to calculate the spatial average of.

    Returns:
        torch.Tensor: Spatial average of the tensor.
    """
    return x.mean(
        dim=[2, 3], keepdim=True
    )  # calculate the mean of the tensor along the spatial dimensions


"""
A simple test to check if the LPIPS network is working properly.
"""


def test():
    l = LPIPS()
    real = torch.randn(16, 3, 256, 256)
    fake = torch.randn(16, 3, 256, 256)
    print(l(real, fake).shape)


if __name__ == "__main__":
    test()  # [16, 1, 1, 1]

