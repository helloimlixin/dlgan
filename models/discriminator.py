#  ==============================================================================
#  Description: Discriminator module of the model.
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
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, args, num_filters_last=64, num_layers=3):
        super(Discriminator, self).__init__()

        layers = [nn.Conv2d(args.num_channels, num_filters_last, 4, 2, 1), nn.LeakyReLU(0.2)]
        num_filters_mult = 1

        for i in range(1, num_layers + 1):
            num_filters_mult_prev = num_filters_mult
            num_filters_mult = min(2**i, 8)
            layers += [
                nn.Conv2d(num_filters_last * num_filters_mult_prev,
                          num_filters_last * num_filters_mult, 4,
                          2 if i < num_layers else 1, 1, bias=False),
                nn.BatchNorm2d(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        layers.append(nn.Conv2d(num_filters_last * num_filters_mult, 1, 4, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)