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
import torch.nn as nn
from .encoder import VQVAEEncoder
from .decoder import VQVAEDecoder
from .quantize import VectorQuantizer, VectorQuantizerEMA

class VQVAE(nn.Module):
    '''VQ-VAE vqvae.'''

    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings,
                 embedding_dim, commitment_cost, epsilon=1e-10, decay=0):
        super(VQVAE, self).__init__()

        self._encoder = VQVAEEncoder(in_channels=in_channels,
                                num_hiddens=num_hiddens,
                                num_residual_layers=num_residual_layers,
                                num_residual_hiddens=num_residual_hiddens)

        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=embedding_dim,
                                      kernel_size=1,
                                      stride=1)

        if decay > 0.0:
            print("Using EMA")
            self._vq_bottleneck = VectorQuantizerEMA(num_embeddings=num_embeddings,
                                              embedding_dim=embedding_dim,
                                              commitment_cost=commitment_cost,
                                              decay=decay)
        else:
            self._vq_bottleneck = VectorQuantizer(num_embeddings=num_embeddings,
                                           embedding_dim=embedding_dim,
                                           commitment_cost=commitment_cost)

        self._decoder = VQVAEDecoder(in_channels=embedding_dim,
                                num_hiddens=num_hiddens,
                                num_residual_layers=num_residual_layers,
                                num_residual_hiddens=num_residual_hiddens)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, encodings = self._vq_bottleneck(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity, quantized
