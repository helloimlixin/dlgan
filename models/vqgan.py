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
from .encoder import VQGANEncoder
from .decoder import VQGANDecoder
from .codebook import VQGANCodebook

class VQGAN(nn.Module):
    '''VQ-GAN model.

    Reference:
     - Taming Transformers for High-Resolution Image Synthesis
       https://arxiv.org/abs/2012.09841
    '''
    def __init__(self, args):
        super(VQGAN).__init__()

        self._encoder = VQGANEncoder(args).to(device=args.device)
        self._codebook = VQGANCodebook(args).to(device=args.device)
        self._decoder = VQGANDecoder(args).to(device=args.device)
        self._prequant_conv = nn.Conv2d(in_channels=args.embedding_dim,
                                        out_channels=args.embedding_dim,
                                        kernel_size=1,
                                        stride=1).to(device=args.device)
        self._postquant_conv = nn.Conv2d(in_channels=args.embedding_dim,
                                            out_channels=args.embedding_dim,
                                            kernel_size=1,
                                            stride=1).to(device=args.device)

    def forward(self, x):
        x_encoded = self._encoder(x)
        x_encoded_prequant = self._prequant_conv(x_encoded)
        vq_loss, z_q, perplexity, z = self._codebook(x_encoded_prequant)
        z_q_postquant = self._postquant_conv(z_q)
        x_hat = self._decoder(z_q_postquant)

        return x_hat, vq_loss, z # z is the latent codebook indices

    def encode(self, x):
        '''Encode an image into its latent representation, will later be used for the transformers.
        '''
        x_encoded = self._encoder(x)
        x_encoded_prequant = self._prequant_conv(x_encoded)
        vq_loss, z_q, perplexity, z = self._codebook(x_encoded_prequant)

        return z_q, z, vq_loss # z is the latent codebook indices

    def decode(self, z):
        '''Decode the latent representation into an image.
        '''
        z_q_postquant = self._postquant_conv(z)
        x_hat = self._decoder(z_q_postquant)

        return x_hat

    def calculate_lambda(self, perceptual_loss, gan_loss, epsilon=1e-4, max_lambda=1e4, scale=0.8):
        '''Calculate the lambda value for the loss function.
        '''
        ell = self.decoder.model[-1] # the last layer of the decoder
        ell_weight = ell.weight
        perceptual_loss_gradients = torch.autograd.grad(perceptual_loss, ell_weight, retain_graph=True)[0]
        gan_loss_gradients = torch.autograd.grad(gan_loss, ell_weight, retain_graph=True)[0]

        lambda_factor = torch.norm(perceptual_loss_gradients) / torch.norm(gan_loss_gradients + epsilon)
        lambda_factor = torch.clamp(lambda_factor, min=0.0, max=max_lambda).detach()

        return scale * lambda_factor

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        '''Adopt the weight of the discriminator.
        '''
        if i < threshold:
            disc_factor = value

        return disc_factor

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))


