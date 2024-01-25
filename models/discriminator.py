#  ==============================================================================
#  Description: Discriminator module of the vqvae.
#
# References:
#     - Esser, P., Rombach, R., & Ommer, B. (2021).
#         Taming Transformers for High-Resolution Image Synthesis. arXiv preprint arXiv:2103.17239.
#     - PatchGAN discriminator: https://arxiv.org/pdf/1611.07004.pdf.
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
import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    """CNN block for the discriminator."""

    def __init__(self, in_channels, out_channels, stride):
        """Constructor for the class.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride of the convolution.
        """
        super(CNNBlock, self).__init__()
        self.block = nn.Sequential(
            # see https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=False,
                padding_mode="reflect",
            ),
            # see https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
            nn.BatchNorm2d(num_features=out_channels),
            # see https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html
            # also see https://stackoverflow.com/questions/69913781/is-it-true-that-inplace-true-activations-in-pytorch-make-sense-only-for-infere
            # for using inplace=True in inference
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        """Forward pass of the block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.block(x)


class Discriminator(nn.Module):
    def __init__(self, args, feature_dim_last=64, num_layers=3) -> None:
        """constructor for the class.

        Args:
            in_channels (int, optional): number of input channels. Defaults to 3.
            feature_dim_last (int, optional): number of features on the last layer. Defaults to 64.
            num_layers (int, optional): number of layers. Defaults to 3.
        """
        super(Discriminator, self).__init__()
        # create a list of feature dimensions for each layer
        feature_dims = [feature_dim_last * min(2 ** i, 8) for i in range(num_layers + 1)]

        layers = [
            nn.Conv2d(
                in_channels=args.num_channels,
                out_channels=feature_dims[0],
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        ]  # layer initialization

        in_channels = feature_dims[0]  # update the number of input channels

        for feature_dim in feature_dims[1:]:  # skip the first layer
            layers.append(
                CNNBlock(
                    in_channels=in_channels,
                    out_channels=feature_dim,
                    stride=1 if feature_dim == feature_dims[-1] else 2,
                )
            )
            in_channels = feature_dim  # update the number of input channels

        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            )
        )  # add the last layer

        self.model = nn.Sequential(*layers)  # create the model

    def forward(self, x):
        """forward pass of the discriminator.

        Args:
            x (torch.Tensor): input tensor.
        """
        return self.model(x)


'''
Some basic tests that should always be there when creating a new model
'''


def test():
    x = torch.randn((1, 3, 256, 256))
    model = Discriminator(in_channels=3)
    predictions = model(x)
    print(model)
    print(predictions.shape)


if __name__ == "__main__":
    test()