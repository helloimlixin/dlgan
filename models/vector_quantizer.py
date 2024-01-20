#  ==============================================================================
#  Description: This file defines the vector quantizer module of the model.
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

class VectorQuantizer(nn.Module):
    '''Vector Quantizer module of the model.'''

    def __init__(self, num_embeddings, embedding_dim, commitment_cost, epsilon=1e-10):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim # dimension of the embedding vectors in the codebook
        self._num_embeddings = num_embeddings # number of embedding vectors in the codebook

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim) # codebook
        # initialize the embedding vectors uniformly
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        # the beta term in the paper, controlling the weighting of the commitment loss
        self._commitment_cost = commitment_cost

        # numerical stability
        self._epsilon = epsilon

    def forward(self, inputs):
        # Convert inputs from B x C x H x W to B x H x W x C, which is required by the embedding layer. Here the
        # contiguous() method is used to make sure that the memory is laid out contiguously (in correct order), which is
        # required by most vectorized operations, such as view().
        # See https://pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html.
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape # B x H x W x C

        # convert inputs from B x H x W x C to BHW x C, which is required by the quantization bottleneck.
        flat_input = inputs.view(-1, self._embedding_dim)

        # compute the distances between the input vectors and the embedding vectors
        # distances: BHW x num_embeddings
        flat_input_l2_squared = torch.sum(flat_input**2, dim=1, keepdim=True) # BHW x 1
        embedding_l2_squared = torch.sum(self._embedding.weight**2, dim=1) # num_embeddings
        inner_product = torch.matmul(flat_input, self._embedding.weight.t()) # BHW x num_embeddings
        distances = flat_input_l2_squared + embedding_l2_squared - 2 * inner_product

        # encoding_indices: BHW x 1
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        # encoding placeholder: BHW x num_embeddings
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        # encodings: BHW x num_embeddings
        encodings.scatter_(1, encoding_indices, 1) # one-hot encoding

        # quantized: BHW x C
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape) # B x H x W x C
        # quantized: B x C x H x W
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        # compute the commitment loss
        # commitment_loss: B x 1 x H x W
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        commitment_loss = self._commitment_cost * e_latent_loss
        # compute the quantized latent loss
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        # compute the total loss - vq_loss: 1
        vq_loss = commitment_loss + q_latent_loss

        # save quantized outputs
        quantized = inputs + (quantized - inputs).detach() # B x C x H x W

        # average pooling over the spatial dimensions
        # avg_probs: B x num_embeddings
        avg_probs = torch.mean(encodings, dim=0)
        # codebook perplexity / usage: 1
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self._epsilon)))

        return quantized, vq_loss, perplexity, encodings
