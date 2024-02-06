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


class DictLearn(nn.Module):
    """A simple dictionary learning algorithm.

  The algorithm is L1 regularized and optimized with SGD.

  Attributes:
    dim: data dimension D.
    num_atoms: number of dictionary atoms K.
    dictionary: dictionary matrix A with dimension K z_e D.
    representation: representation matrix R with dimension N z_e K,
      N as the number of data samples
    sparsity: sparsity control parameter, i.e., the lambda.
  """

    def __init__(self, dim, num_atoms, sparsity_level) -> None:
        super(DictLearn, self).__init__()
        self.dim = dim
        self.num_atoms = num_atoms
        self.dictionary = nn.Parameter(
            torch.rand((num_atoms, dim), dtype=torch.float, requires_grad=True))
        self.representation = self.representation_builder()
        self.sparsity_level = sparsity_level

    def representation_builder(self):
        layers = nn.ModuleList()
        layers.append(nn.Linear(self.dim, self.num_atoms))  # output dim for one sample: K
        layers.append(nn.Softmax(dim=1))  # convert the output to [0, 1] range

        return nn.Sequential(*layers)

    def loss(self, x, representation):
        """Calculate the loss.

    Args:
      x: input data, for image data, the dimension is: N z_e C z_e H z_e W
      representation: representation matrix R with dimension N z_e K,
        N as the number of data samples

    Returns:
      combined loss, reconstruction, representation
    """
        batch_size, num_channels, height, width = x.shape
        reconstruction = torch.matmul(representation, self.dictionary)
        reconstruction = reconstruction.view(batch_size, self.dim, height, width).contiguous()

        recon_loss = nn.MSELoss()(x, reconstruction.detach()) * 0.25 + nn.MSELoss()(x.detach(), reconstruction)
        regularization = torch.sum(torch.abs(representation))

        reconstruction = reconstruction + (reconstruction - x).detach()  # straight-through gradient

        return recon_loss, reconstruction, representation

    def forward(self, x):
        """Forward pass.

    Args:
      x: input data, for image data, the dimension is:
          batch_size z_e num_channels z_e height z_e width
    """
        batch_size, num_channels, height, width = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x_flattened = x.view(-1, self.dim)  # data dimension: N z_e D
        representation = self.representation(x_flattened)
        encodings = torch.zeros(representation.shape, requires_grad=True).cuda()

        # compute the l2 distances (N z_e K) between x_flattened (N z_e D) and the dictionary (K z_e D)
        distances = torch.sum(x_flattened ** 2, dim=1, keepdim=True) + torch.sum(
            self.dictionary ** 2, dim=1) - 2 * torch.matmul(x_flattened,
                                                            self.dictionary.t())  # taking a negation for topk operation

        elements, indices = distances.topk(self.sparsity_level, dim=1, largest=False)  # topk operation

        encodings.scatter_(1, indices, 1)
        representation = representation * encodings

        reconstruction = torch.matmul(representation, self.dictionary)
        reconstruction = reconstruction.view(batch_size, self.dim, height, width).contiguous()

        x = x.permute(0, 3, 1, 2).contiguous()  # permute back
        x = x.view(batch_size, self.dim, height, width).contiguous()

        recon_loss = nn.MSELoss()(x, reconstruction.detach()) + nn.MSELoss()(x.detach(), reconstruction)
        regularization = torch.sum(torch.abs(representation))

        reconstruction = x + (reconstruction - x).detach()  # straight-through gradient

        # average pooling over the spatial dimensions
        # avg_probs: B z_e _num_embeddings
        avg_probs = torch.mean(encodings, dim=0)
        avg_probs = avg_probs / torch.sum(avg_probs)  # normalize the representation
        # codebook perplexity / usage: 1
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # return representation
        return recon_loss + regularization, reconstruction, perplexity, representation


class DictionaryLearningSimple(nn.Module):
    """A simple dictionary learning algorithm.

  The algorithm is L1 regularized and optimized with SGD.

  Attributes:
    dim: data dimension D.
    num_atoms: number of dictionary atoms K.
    dictionary: dictionary matrix A with dimension K z_e D.
    representation: representation matrix R with dimension N z_e K,
      N as the number of data samples
    sparsity: sparsity control parameter, i.e., the lambda.
  """

    def __init__(self, dim, num_atoms, commitment_cost, epsilon=1e-8) -> None:
        super(DictionaryLearningSimple, self).__init__()
        self.dim = dim
        self.num_atoms = num_atoms
        # self.dictionary = nn.Parameter(torch.rand((self.num_atoms, self.dim), dtype=torch.float, requires_grad=True))
        self.dictionary = nn.Embedding(self.num_atoms, self.dim)
        # self.dictionary.weight.data.uniform_(-1 / self.num_atoms, 1 / self.num_atoms)
        self.commitment_cost = commitment_cost
        self.representation = self.representation_builder()

        self._epsilon = epsilon  # a small number to avoid the numerical issues

    def representation_builder(self):
        layers = nn.ModuleList()
        layers.append(nn.Linear(self.dim, self.num_atoms))  # output dim for one sample: K
        layers.append(nn.Softmax(dim=1))  # convert the output to [0, 1] range

        return nn.Sequential(*layers)

    def loss(self, z_e, representation):
        """Forward pass.

    Args:
      z_e: input data, for image data, the dimension is:
          batch_size z_e num_channels z_e height z_e width
    """
        z_e = z_e.permute(0, 2, 3, 1).contiguous()  # permute the input
        ze_shape = z_e.shape  # save the shape

        # compute the reconstruction from the representation
        alpha = 1.0
        reconstruction = (1 - alpha) * z_e + alpha * torch.matmul(representation, self.dictionary.weight).view(ze_shape)

        recon_loss = F.mse_loss(reconstruction, z_e.detach()) + F.mse_loss(reconstruction.detach(), z_e) * self.commitment_cost  # L2 loss

        # average pooling over the spatial dimensions
        # avg_probs: B z_e _num_embeddings
        avg_probs = torch.mean(representation, dim=0)
        avg_probs = avg_probs / torch.sum(avg_probs)  # normalize the representation

        # codebook perplexity / usage: 1
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self._epsilon)))

        # return representation, reconstruction, perplexity, regularization
        return recon_loss, reconstruction.permute(0, 3, 1, 2).contiguous(), perplexity, representation

    def forward(self, z_e):
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        ze_flattened = z_e.view(-1, self.dim)
        return self.representation(ze_flattened)
