#  ==============================================================================
#  Description: This file defines the decoder modules of the vqvae.
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
    '''Decoder module of the vanilla-10 VQ-VAE vqvae.'''

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
