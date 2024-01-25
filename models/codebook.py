#  ==============================================================================
#  Description: This file defines the vector quantizer module of the vqvae.
#  Copyright (C) 2024 Xin Li
#
# References:
#   - Van Den Oord, A., & Vinyals, O. (2017). Neural discrete representation learning.
#       Advances in neural information processing systems, 30.
#
# Code Reference:
#   - https://github.com/google-deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb.
#   - https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb.
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
import torch.nn.functional as F

class VQVAECodebookVanilla(nn.Module):
    '''Vector Quantizer module (codebook) of the vqvae.'''

    def __init__(self, num_embeddings, embedding_dim, commitment_cost, epsilon=1e-10):
        super(VQVAECodebookVanilla, self).__init__()

        self._embedding_dim = embedding_dim # dimension of the _embedding vectors in the codebook
        self._num_embeddings = num_embeddings # number of _embedding vectors in the codebook

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim) # codebook
        # initialize the _embedding vectors uniformly
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        # the _beta term in the paper, controlling the weighting of the commitment loss
        self._commitment_cost = commitment_cost

        # numerical stability
        self._epsilon = epsilon

    def forward(self, z_e):
        # Convert z_e from B x C x H x W to B x H x W x C, which is required by the _embedding layer. Here the
        # contiguous() method is used to make sure that the memory is laid out contiguously (in correct order), which is
        # required by most vectorized operations, such as view().
        # See https://pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html.
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        input_shape = z_e.shape # B x H x W x C

        # convert z_e from B x H x W x C to BHW x C, which is required by the quantization bottleneck.
        ze_flattened = z_e.view(-1, self._embedding_dim)

        # compute the distances between the input vectors and the _embedding vectors
        # distances: BHW x _num_embeddings
        flat_input_l2_squared = torch.sum(ze_flattened**2, dim=1, keepdim=True) # BHW x 1
        embedding_l2_squared = torch.sum(self._embedding.weight**2, dim=1) # _num_embeddings
        inner_product = torch.matmul(ze_flattened, self._embedding.weight.t()) # BHW x _num_embeddings
        distances = flat_input_l2_squared + embedding_l2_squared - 2 * inner_product

        # encoding_indices: BHW x 1
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        # encoding placeholder: BHW x _num_embeddings
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=z_e.device)
        # encodings: BHW x _num_embeddings
        encodings.scatter_(1, encoding_indices, 1) # one-hot encoding

        # z_q: BHW x C
        z_q = torch.matmul(encodings, self._embedding.weight).view(input_shape) # B x H x W x C

        # compute the commitment loss
        # commitment_loss: B x 1 x H x W
        commitment_loss = self._commitment_cost * F.mse_loss(z_q.detach(), z_e)
        # compute the z_q latent loss
        e2z_loss = F.mse_loss(z_q, z_e.detach())
        # compute the total loss - vq_loss: 1
        vq_loss = commitment_loss + e2z_loss

        # save z_q outputs
        z_q = z_e + (z_q - z_e).detach() # B x C x H x W

        # average pooling over the spatial dimensions
        # avg_probs: B x _num_embeddings
        avg_probs = torch.mean(encodings, dim=0)
        # codebook perplexity / usage: 1
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self._epsilon)))

        return vq_loss, z_q.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

class VQGANCodebook(nn.Module):
    '''Vector Quantizer module (codebook) of the vqvae.

    References:
        - Esser, P., Rombach, R., & Ommer, B. (2021).
            Taming Transformers for High-Resolution Image Synthesis. arXiv preprint arXiv:2103.17239.
    '''
    def __init__(self, args):
        super(VQGANCodebook, self).__init__()

        self._num_embeddings = args.num_embeddings
        self._embedding_dim = args.embedding_dim
        self._beta = args.beta

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1.0 / self._num_embeddings, 1.0 / self._num_embeddings)

    def forward(self, z_e):
        # z_e: B x C x H x W
        z_e = z_e.permute(0, 2, 3, 1).contiguous() # B x H x W x C
        # flatten the _embedding vectors
        ze_flattened = z_e.view(-1, self._embedding_dim)

        # compute the distances between the input vectors and the _embedding vectors
        # distances: BHW x _num_embeddings
        z_l2_squared = torch.sum(ze_flattened**2, dim=1, keepdim=True)
        embedding_l2_squared = torch.sum(self._embedding.weight ** 2, dim=1)
        inner_product = torch.matmul(ze_flattened, self._embedding.weight.t())
        distances = z_l2_squared + embedding_l2_squared - 2 * inner_product

        # encoding_indices: BHW x 1
        z = torch.argmin(distances, dim=1)
        # create the z_q with the _embedding vectors
        z_q = self._embedding(z).view(z_e.shape) # B x H x W x C

        # compute the commitment loss
        # commitment_loss: B x 1 x H x W
        commitment_loss = self._beta * F.mse_loss(z_e, z_q.detach())
        # loss term to move the embedding vectors closer to the input vectors
        e2z_loss = F.mse_loss(z_e.detach(), z_q)
        # compute the total loss - vq_loss: 1
        vq_loss = commitment_loss + e2z_loss

        # save quantized outputs
        z_q = z_e + (z_q - z_e).detach()

        # average pooling over the spatial dimensions
        # avg_probs: B x _num_embeddings
        avg_probs = torch.mean(F.one_hot(z, self._num_embeddings).float(), dim=0)

        # codebook perplexity / usage: 1
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # the last output is the encoding indices, which is used for the transformer training
        return vq_loss, z_q.permute(0, 3, 1, 2).contiguous(), perplexity, z



