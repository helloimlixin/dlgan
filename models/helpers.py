#==============================================================================
# Description: Helper functions for the model.
#
# References:
#   - He, K., Zhang, X., Ren, S., & Sun, J. (2016).
#       Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and
#           pattern recognition (pp. 770-778).
#
# Code Reference:
#   - https://github.com/google-deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb.
#   - https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb.
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

class ResidualBlock(nn.Module):
    """Residual Block to build up the encoder and the decoder networks."""
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1,
                      stride=1,
                      bias=False)
        )

    def forward(self, x):
        return x + self.main(x) # skip connection

class ResidualStack(nn.Module):
    """Residual Stack for the encoder and decoder networks."""
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()

        self._num_residual_layers = num_residual_layers

        self._layers = nn.ModuleList([ResidualBlock(in_channels=in_channels, num_hiddens=num_hiddens,
                                                    num_residual_hiddens=num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)
