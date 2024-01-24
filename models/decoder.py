#  ==============================================================================
#  Description: This file defines the decoder module of the model.
#  Copyright (C) 2024 Xin Li
#
# References:
#   - Van Den Oord, A., & Vinyals, O. (2017). Neural discrete representation learning.
#       Advances in neural information processing systems, 30.
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

import torch.nn as nn
import torch.nn.functional as F
from .helpers import ResidualStack, ResidualBlock, NonLocalBlock, UpSampleBlock, GroupNorm, Swish


class VQVAEDecoder(nn.Module):
    '''Decoder module of the vanilla VQ-VAE model.'''

    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(VQVAEDecoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                               out_channels=num_hiddens//2,
                                               kernel_size=4,
                                               stride=2,
                                               padding=1)

        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2,
                                               out_channels=3,
                                               kernel_size=4,
                                               stride=2,
                                               padding=1)

    def forward(self, x):
        x = self._conv_1(x)
        x = self._residual_stack(x)
        x = self._conv_trans_1(x)
        x = F.relu(x)

        return self._conv_trans_2(x)


class VQGANDecoder(nn.Module):
    '''Decoder module of the VQ-GAN model.

    References:
        - Esser, P., Rombach, R., & Ommer, B. (2021).
            Taming Transformers for High-Resolution Image Synthesis. arXiv preprint arXiv:2103.17239.
    '''
    def __init__(self, args):
        super(VQGANDecoder, self).__init__()
        channels = [512, 256, 256, 128, 128]
        attention_resolutions = [16]
        num_res_blocks = 3
        resolution = 16

        in_channels = channels[0]
        layers = [nn.Conv2d(args.latent_dim, in_channels, 3, 1, 1),
                  ResidualBlock(in_channels, in_channels),
                  NonLocalBlock(in_channels),
                  ResidualBlock(in_channels, in_channels)]

        for i in range(len(channels) - 1):
            out_channels = channels[i + 1]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attention_resolutions:
                    layers.append(NonLocalBlock(out_channels))
            if i != 0:
                layers.append(UpSampleBlock(in_channels, out_channels))
                resolution *= 2

        layers.append(GroupNorm(in_channels))
        layers.append(Swish())
        layers.append(nn.Conv2d(in_channels, args.image_channels, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

