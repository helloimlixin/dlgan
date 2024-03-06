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
import torch.nn.functional as F

class DictionaryLearningSimple(nn.Module):
    """A simple dictionary learning algorithm.

  The algorithm is L1 regularized and optimized with SGD.

  Attributes:
    dim: data dimension D.
    num_atoms: number of dictionary atoms K.
    dictionary: dictionary matrix A with dimension K z_e D.
    representation: representation matrix R with dimension N z_e K,
      N as the number of data samples.
    commitment_cost: commitment cost beta.
    sparsity_level: sparsity level L.
  """

    def __init__(self, dim, num_atoms, commitment_cost, sparsity_level, epsilon=1e-10) -> None:
        super(DictionaryLearningSimple, self).__init__()
        self.dim = dim
        self.num_atoms = num_atoms
        self.dictionary = nn.Embedding(self.num_atoms, self.dim)

        self.commitment_cost = commitment_cost
        self.sparsity_level = sparsity_level

        self.representation = self.representation_builder()

        self._epsilon = epsilon  # a small number to avoid the numerical issues
    def representation_builder(self):
        layers = nn.ModuleList()
        layers.append(nn.Linear(self.dim, self.num_atoms))  # output dim for one sample: K
        layers.append(nn.Softmax(dim=1))  # convert the output to [0, 1] range

        return nn.Sequential(*layers)


    def forward(self, z_e):
        """Forward pass.

    Args:
      z_e: input data, for image data, the dimension is:
          batch_size z_e num_channels z_e height z_e width
    """
        z_e = z_e.permute(0, 2, 3, 1).contiguous()  # permute the input
        ze_shape = z_e.shape  # save the shape

        ze_flattened = z_e.view(-1, self.dim)  # data dimension: N z_e D

        representation = self.representation(ze_flattened)  # representation matrix R with dimension N z_e K

        # sparsity representation
        # compute the distances between the input vectors and the _embedding vectors
        # distances: BHW z_e _num_embeddings
        distances = torch.sum(ze_flattened ** 2, dim=1, keepdim=True) + torch.sum(
            self.dictionary.weight ** 2, dim=1) - 2 * torch.matmul(ze_flattened, self.dictionary.weight.t())

        min_dists, encoding_indices = distances.topk(self.sparsity_level, dim=1, largest=False)

        encodings = torch.zeros(encoding_indices.shape[0], self.num_atoms, device=z_e.device)
        encodings.scatter_(1, encoding_indices, 1)

        # representation_sparse = encodings.mul(representation).to_sparse_csr() # sparsity representation
        representation_sparse = encodings * representation  # sparsity representation
        
        # compute the reconstruction from the representation
        z_dl = torch.matmul(representation_sparse, self.dictionary.weight)  # reconstruction: B z_e D
        z_dl = z_dl.view(ze_shape).contiguous()

        # compute the commitment loss
        # commitment_loss: B z_e 1 z_e H z_e W
        commitment_loss = self.commitment_cost * F.mse_loss(z_dl.detach(), z_e)
        # compute the z_dl latent loss
        e2z_loss = F.mse_loss(z_dl, z_e.detach())

        recon_loss = commitment_loss + e2z_loss

        z_dl = z_e + (z_dl - z_e).detach() # straight-through gradient

        # average pooling over the spatial dimensions
        # avg_probs: B z_e _num_embeddings
        avg_probs = torch.mean(encodings, dim=0)
        avg_probs = avg_probs / torch.sum(avg_probs)  # normalize the representation

        # codebook perplexity / usage: 1
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self._epsilon)))

        # return representation, reconstruction, perplexity, regularization
        return recon_loss, z_dl.permute(0, 3, 1, 2).contiguous(), perplexity, representation


class DictionaryLearningEMA(nn.Module):
    """A dictionary learning algorithm based on exponential moving average (EMA).

  The algorithm is L1 regularized and optimized with SGD.

  Attributes:
    dim: data dimension D.
    num_atoms: number of dictionary atoms K.
    dictionary: dictionary matrix A with dimension K z_e D.
    representation: representation matrix R with dimension N z_e K,
      N as the number of data samples.
    commitment_cost: commitment cost beta.
    sparsity_level: sparsity level L.
  """

    def __init__(self, dim, num_atoms, commitment_cost, sparsity_level, decay=0.99, epsilon=1e-10) -> None:
        super(DictionaryLearningEMA, self).__init__()
        self.dim = dim
        self.num_atoms = num_atoms
        self.dictionary = nn.Embedding(self.num_atoms, self.dim)

        self.commitment_cost = commitment_cost
        self.sparsity_level = sparsity_level
        self.decay = decay

        self.representation = self.representation_builder()

        self.register_buffer('_ema_cluster_size', torch.zeros(num_atoms))
        self._ema_w = nn.Parameter(torch.Tensor(num_atoms, self.dim))
        self._ema_w.data.normal_()

        self._epsilon = epsilon  # a small number to avoid the numerical issues

    def representation_builder(self):
        layers = nn.ModuleList()
        layers.append(nn.Linear(self.dim, self.num_atoms))  # output dim for one sample: K
        layers.append(nn.Softmax(dim=1))  # convert the output to [0, 1] range

        return nn.Sequential(*layers)

    def forward(self, z_e):
        """Forward pass.

    Args:
      z_e: input data, for image data, the dimension is:
          batch_size z_e num_channels z_e height z_e width
    """
        z_e = z_e.permute(0, 2, 3, 1).contiguous()  # permute the input
        ze_shape = z_e.shape  # save the shape

        ze_flattened = z_e.view(-1, self.dim)  # data dimension: N z_e D

        representation = self.representation(ze_flattened)  # representation matrix R with dimension N z_e K

        # sparsity representation
        # compute the distances between the input vectors and the _embedding vectors
        # distances: BHW z_e _num_embeddings
        distances = torch.sum(ze_flattened ** 2, dim=1, keepdim=True) + torch.sum(
            self.dictionary.weight ** 2, dim=1) - 2 * torch.matmul(ze_flattened, self.dictionary.weight.t())

        min_dists, encoding_indices = distances.topk(self.sparsity_level, dim=1, largest=False)

        encodings = torch.zeros(encoding_indices.shape[0], self.num_atoms, device=z_e.device)
        encodings.scatter_(1, encoding_indices, 1)

        representation_sparse = encodings * representation / self.sparsity_level # sparsity representation

        # compute the reconstruction from the representation
        z_dl = torch.matmul(representation_sparse, self.dictionary.weight)  # reconstruction: B z_e D
        z_dl = z_dl.view(ze_shape).contiguous()

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self.decay + \
                                     (1 - self.decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self.num_atoms * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), ze_flattened)
            self._ema_w = nn.Parameter(self._ema_w * self.decay + (1 - self.decay) * dw)

            self.dictionary.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
            self.dictionary.weight = nn.Parameter(F.normalize(self.dictionary.weight, dim=1))

        # Loss
        e_latent_loss = F.mse_loss(z_dl.detach(), z_e)
        loss = self.commitment_cost * e_latent_loss

        z_dl = z_e + (z_dl - z_e).detach() # straight-through gradient

        # average pooling over the spatial dimensions
        # avg_probs: B z_e _num_embeddings
        avg_probs = torch.mean(encodings, dim=0)
        avg_probs = avg_probs / torch.sum(avg_probs)  # normalize the representation

        # codebook perplexity / usage: 1
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self._epsilon)))

        # return representation, reconstruction, perplexity, regularization
        return loss, z_dl.permute(0, 3, 1, 2).contiguous(), perplexity, representation_sparse

