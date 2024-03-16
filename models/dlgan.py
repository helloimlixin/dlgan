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
import torch
import torch.nn as nn
from .encoder import VQVAEEncoder
from .decoder import VQVAEDecoder
from .dictlearn import DictionaryLearningSimple, DictionaryLearningMatchingPursuit
from .discriminator import Discriminator
from .utils import init_weights

class DLGAN(nn.Module):
    '''DL-VAE model.'''

    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings,
                 embedding_dim, commitment_cost, sparsity_level, epsilon=1e-10):
        super(DLGAN, self).__init__()

        self._encoder = VQVAEEncoder(in_channels=in_channels,
                                num_hiddens=num_hiddens,
                                num_residual_layers=num_residual_layers,
                                num_residual_hiddens=num_residual_hiddens)

        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=embedding_dim,
                                      kernel_size=1,
                                      stride=1)


        # self._dl_bottleneck = DictionaryLearningSimple(dim=embedding_dim,
        #                                                num_atoms=num_embeddings,
        #                                                commitment_cost=commitment_cost,
        #                                                epsilon=epsilon)

        self._dl_bottleneck = DictionaryLearningMatchingPursuit(dim=embedding_dim,
                                                   num_atoms=num_embeddings,
                                                   commitment_cost=commitment_cost,
                                                   sparsity_level=sparsity_level,
                                                   epsilon=epsilon)

        self._decoder = VQVAEDecoder(in_channels=embedding_dim,
                                num_hiddens=num_hiddens,
                                num_residual_layers=num_residual_layers,
                                num_residual_hiddens=num_residual_hiddens)

        self._discriminator = Discriminator()
        self._discriminator.apply(init_weights)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        representation = self._dl_bottleneck.matching_pursuit(z)
        dlloss, z_recon, perplexity, representation = self._dl_bottleneck(z, representation)
        x_recon = self._decoder(z_recon)

        return dlloss, x_recon, perplexity, z

    def calculate_lambda(self, perceptual_loss, gan_loss, epsilon=1e-6, max_lambda=1e4, scale=0.8):
        '''Calculate the lambda value for the loss function.
        '''
        ell = list(self._decoder.children())[-1] # the last layer of the decoder
        ell_weight = ell.weight
        perceptual_loss_gradients = torch.autograd.grad(perceptual_loss, ell_weight, retain_graph=True)[0]
        gan_loss_gradients = torch.autograd.grad(gan_loss, ell_weight, retain_graph=True)[0]

        lambda_factor = torch.norm(perceptual_loss_gradients) / torch.norm(gan_loss_gradients + epsilon)
        lambda_factor = torch.clamp(lambda_factor, min=0.0, max=max_lambda).detach()

        return scale * lambda_factor

    @staticmethod
    def adopt_weight(disc_factor, epoch, threshold, value=0.):
        '''Adopt the weight of the discriminator.
        '''
        if epoch < threshold:
            disc_factor = value

        return disc_factor
