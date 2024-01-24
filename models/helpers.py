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

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupNorm(nn.Module):
    '''Group normalization module.

    References:
      - Wu, Y., & He, K. (2018). Group normalization. In Proceedings of the European Conference on Computer Vision
          (ECCV) (pp. 3-19).
    '''

    def __init__(self, num_channels, num_groups=32, eps=1e-5, affine=True):
        super(GroupNorm, self).__init__()

        self._num_channels = num_channels
        self._num_groups = num_groups
        self._eps = eps

        self._group_norm = nn.GroupNorm(num_groups=self._num_groups,
                                        num_channels=self._num_channels,
                                        eps=self._eps,
                                        affine=affine)

    def forward(self, x):
        return self._group_norm(x)


class Swish(nn.Module):
    '''Swish activation function.

    References:
      - Ramachandran, P., Zoph, B., & Le, Q. V. (2017). Swish: a self-gated activation function. arXiv preprint
          arXiv:1710.05941.
    '''

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    '''Residual Block to build up the encoder and the decoder networks.

    References:
      - He, K., Zhang, X., Ren, S., & Sun, J. (2016).
          Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and
              pattern recognition (pp. 770-778).
    '''

    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(ResidualBlock, self).__init__()

        self.in_channels = in_channels
        self.num_hiddens = num_hiddens
        self.num_residual_hiddens = num_residual_hiddens

        self.block = nn.Sequential(
            GroupNorm(num_channels=in_channels),
            Swish(),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            GroupNorm(num_channels=num_residual_hiddens),
            Swish(),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1,
                      stride=1,
                      bias=False)
        )

        if in_channels != num_residual_hiddens: # if the number of channels is not the same, then use 1x1 conv.
            self.shortcut = nn.Conv2d(in_channels=in_channels,
                                      out_channels=num_residual_hiddens,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)

    def forward(self, x):
        if self.in_channels != self.num_residual_hiddens:
            return self.block(x) + self.shortcut(x)
        else:
            return x + self.block(x) # skip connection

class ResidualStack(nn.Module):
    '''Residual Stack for the encoder and decoder networks.

    References:
        - He, K., Zhang, X., Ren, S., & Sun, J. (2016).
            Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and
                pattern recognition (pp. 770-778).
    '''
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()

        self._num_residual_layers = num_residual_layers

        self._layers = nn.ModuleList([ResidualBlock(in_channels=in_channels, num_hiddens=num_hiddens,
                                                    num_residual_hiddens=num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)] + [Swish()])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)

        return x


class UpSampleBlock(nn.Module):
    '''UpSample Block to build up the decoder network.

    References:
        - Odena, A., Dumoulin, V., & Olah, C. (2016). Deconvolution and checkerboard artifacts. Distill, 1(10), e3.
        '''
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(UpSampleBlock, self).__init__()

        self._block = nn.Sequential(
            GroupNorm(num_channels=in_channels),
            Swish(),
            nn.ConvTranspose2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               bias=False)
        )

    def forward(self, x):
        return self._block(x)


class DownSampleBlock(nn.Module):
    '''DownSample Block to build up the encoder network.

    References:
        - Odena, A., Dumoulin, V., & Olah, C. (2016). Deconvolution and checkerboard artifacts. Distill, 1(10), e3.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DownSampleBlock, self).__init__()

        self._block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=False),
            GroupNorm(num_channels=out_channels),
            Swish()
        )

    def forward(self, x):
        return self._block(x)


class NonLocalBlockOriginal(nn.Module):
    '''Non-local Block to build up the encoder network based on the original paper.

    References:
        - Wang, X., Girshick, R., Gupta, A., & He, K. (2018). Non-local neural networks. In Proceedings of the IEEE
            conference on computer vision and pattern recognition (pp. 7794-7803).
    '''
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NonLocalBlockOriginal, self).__init__()

        self._sub_sample = sub_sample
        self._in_channels = in_channels
        self._inter_channels = inter_channels

        if self._inter_channels is None:
            self._inter_channels = in_channels // 2

        self._g = nn.Conv2d(in_channels=self._in_channels,
                            out_channels=self._inter_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0)

        if bn_layer:
            self._W = nn.Sequential(
                nn.Conv2d(in_channels=self._inter_channels,
                          out_channels=self._in_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0),
                GroupNorm(num_channels=self._in_channels)
            )
            nn.init.constant_(self._W[1].weight, 0) # initialize the weight of the GroupNorm layer to 0
            nn.init.constant_(self._W[1].bias, 0) # initialize the bias of the GroupNorm layer to 0
        else:
            self._W = nn.Conv2d(in_channels=self._inter_channels,
                                out_channels=self._in_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0)
            nn.init.constant_(self._W.weight, 0)
            nn.init.constant_(self._W.bias, 0)

        if self._sub_sample:
            self._g = nn.Sequential(self._g, nn.MaxPool2d(kernel_size=(2, 2)))
            self._phi = nn.Sequential(self._phi, nn.MaxPool2d(kernel_size=(2, 2)))

    def forward(self, x):
        batch_size = x.size(0)

        g_x = self._g(x).view(batch_size, self._inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self._theta(x).view(batch_size, self._inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1) # transpose

        phi_x = self._phi(x).view(batch_size, self._inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self._inter_channels, *x.size()[2:])
        W_y = self._W(y)
        z = W_y + x # residual connection

        return z


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NonLocalBlock, self).__init__()
        self._in_channels = in_channels

        self.group_norm = GroupNorm(num_channels=self._in_channels)
        self.q = nn.Conv2d(in_channels=self._in_channels,
                            out_channels=out_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0)
        self.k = nn.Conv2d(in_channels=self._in_channels,
                            out_channels=out_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0)
        self.v = nn.Conv2d(in_channels=self._in_channels,
                            out_channels=out_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0)
        self.out = nn.Conv2d(in_channels=out_channels,
                            out_channels=self._in_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0)

    def forward(self, x):
        x_h = self.group_norm(x)
        q = self.q(x_h)
        k = self.k(x_h)
        v = self.v(x_h)

        b, c, h, w = q.shape

        q = q.reshape(b, c, h*w)
        q.permute(0, 2, 1)
        k = k.reshape(b, c, h*w)
        v = v.reshape(b, c, h*w)

        attention = torch.bmm(q, k)
        attention = attention * (int(c) ** (-0.5))
        attention = F.softmax(attention, dim=2)
        attention = attention.permute(0, 2, 1)

        out = torch.bmm(v, attention)
        out = out.reshape(b, c, h, w)

        return x + out