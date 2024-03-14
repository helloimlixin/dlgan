#==============================================================================
# Description: This file defines the encoder modules of the vqvae.
#
# References:
#   - Van Den Oord, A., & Vinyals, O. (2017). Neural discrete representation learning.
#       Advances in neural information processing systems, 30.
#
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
#==============================================================================

import torch.nn as nn
import torch.nn.functional as F
from .helpers import ResidualBlock, ResidualStack, NonLocalBlock, DownSampleBlock, UpSampleBlock, GroupNorm, Swish

class VQVAEEncoder(nn.Module):
    '''Encoder module of the original VQ-VAE vqvae.'''

    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(VQVAEEncoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2,
                                 padding=1)

        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2,
                                 padding=1)

        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, x):
        x = self._conv_1(x)
        x = F.relu(x)
        x = self._conv_2(x)
        x = F.relu(x)
        x = self._conv_3(x)
        return self._residual_stack(x)
