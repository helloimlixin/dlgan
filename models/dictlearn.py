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
import numpy as np
from sklearn.linear_model import orthogonal_mp_gram


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

    def __init__(self, dim, num_atoms, commitment_cost, epsilon=1e-10) -> None:
        super(DictionaryLearningSimple, self).__init__()
        self.dim = dim
        self.num_atoms = num_atoms
        self.dictionary = nn.Embedding(self.num_atoms, self.dim)

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

        ze_flattened = z_e.view(-1, self.dim)  # data dimension: N z_e D

        representation = self.representation(ze_flattened)  # representation matrix R with dimension N z_e K
        
        # compute the reconstruction from the representation
        z_dl = torch.matmul(representation, self.dictionary.weight)  # reconstruction: B z_e D
        z_dl = z_dl.view(ze_shape).contiguous()

        # compute the commitment loss
        # commitment_loss: B z_e 1 z_e H z_e W
        commitment_loss = self.commitment_cost * F.mse_loss(z_dl.detach(), z_e)
        # compute the z_dl latent loss
        e2z_loss = F.mse_loss(z_dl, z_e.detach())

        recon_loss = commitment_loss + e2z_loss + self.commitment_cost * torch.abs(representation).mean()

        # average pooling over the spatial dimensions
        # avg_probs: B z_e _num_embeddings
        avg_probs = torch.mean(representation, dim=0)
        avg_probs = avg_probs / torch.sum(avg_probs)  # normalize the representation

        # codebook perplexity / usage: 1
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self._epsilon)))

        # return representation, reconstruction, perplexity, regularization
        return recon_loss, z_dl.permute(0, 3, 1, 2).contiguous(), perplexity, representation

    def forward(self, z_e):
        """Forward pass.

        Args:
            z_e: input data, for image data, the dimension is:
                batch_size z_e num_channels z_e height z_e width
            """
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        ze_shape = z_e.shape

        ze_flattened = z_e.view(-1, self.dim)
        representation = self.representation(ze_flattened)

        return representation


class DictionaryLearningBatchOMP(nn.Module):
    """Dictionary learning algorithm with Batch Orthogonal Matching Pursuit.

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
        super(DictionaryLearningBatchOMP, self).__init__()
        self.dim = dim
        self.num_atoms = num_atoms
        self.dictionary = nn.Embedding(self.num_atoms, self.dim) # dictionary matrix A with dimension D x K

        self.commitment_cost = commitment_cost

        self.sparsity_level = sparsity_level

        self._epsilon = epsilon  # a small number to avoid the numerical issues


    def representation_builder(self):
        layers = nn.ModuleList()
        layers.append(nn.Linear(self.dim, self.num_atoms))
        layers.append(nn.Softmax(dim=1))

        return nn.Sequential(*layers)

    def forward(self, z_e, representation):
        """Forward pass.

    Args:
      z_e: input data, for image data, the dimension is:
          batch_size z_e num_channels z_e height z_e width
    """
        z_e = z_e.permute(0, 2, 3, 1).contiguous()  # permute the input
        # normalize the input
        z_e = z_e / torch.norm(z_e, p=2, dim=3, keepdim=True)
        ze_shape = z_e.shape  # save the shape

        # compute the reconstruction from the representation
        z_dl = torch.matmul(representation, self.dictionary.weight)  # reconstruction: B z_e D
        z_dl = z_dl.view(ze_shape).contiguous()

        # compute the commitment loss
        # commitment_loss: B z_e 1 z_e H z_e W
        commitment_loss = self.commitment_cost * F.mse_loss(z_dl.detach(), z_e)
        # compute the z_dl latent loss
        e2z_loss = F.mse_loss(z_dl, z_e.detach())

        recon_loss = commitment_loss + e2z_loss

        # average pooling over the spatial dimensions
        # avg_probs: B z_e _num_embeddings
        avg_probs = torch.mean(representation, dim=0)
        avg_probs = avg_probs / torch.sum(avg_probs)  # normalize the representation

        # codebook perplexity / usage: 1
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self._epsilon)))

        # return representation, reconstruction, perplexity, regularization
        return recon_loss, z_dl.permute(0, 3, 1, 2).contiguous(), perplexity, representation

    def batch_omp(self, z_e):
        """Compute the representation with Matching Pursuit."""
        ze_flattened = z_e.view(-1, self.dim)  # data dimension: N x D

        # representation = torch.zeros(ze_flattened.shape[0], self.num_atoms).to(z_e.device)  # representation matrix R with dimension N x K

        # # check for the furthest representation
        # for i in range(ze_flattened.shape[0]):
        #     # solve for the representation
        #     measurement = ze_flattened[i, :] # measurement vector for the i-th sample with dimension N x D = m
        #     gamma = representation[i, :] # representation vector for the i-th sample with dimension N x K = n
        #     residual = measurement # residual vector for the i-th sample with dimension N x D = m
        #     supp = [] # support set
        #     # dictionary dimension: K x D = n x m
        #
        #     for k in range(self.sparsity_level):
        #         # compute the inner product
        #         inner_product = torch.matmul(self.dictionary.weight / torch.norm(self.dictionary.weight.T, p=2, dim=1), residual) # inner product with dimension K x N
        #         candidates = torch.abs(inner_product)
        #         # exclude the columns that have been selected
        #         candidates[torch.tensor(supp, dtype=torch.int)] = -torch.inf
        #         # find the index of the maximum value
        #         max_index = torch.argmax(candidates)
        #         # update the support set
        #         supp.append(max_index)
        #         # update the representation
        #         gamma[max_index] = torch.matmul(self.dictionary.weight[max_index, :], residual) / torch.norm(self.dictionary.weight[max_index, :], p=2) ** 2
        #         # update the residual
        #         reconstruction = torch.matmul(self.dictionary.weight[max_index, :], measurement) * self.dictionary.weight[max_index, :] / torch.norm(self.dictionary.weight[max_index, :], p=2) ** 2
        #         residual = residual - reconstruction
        #
        #     representation[i, :] = nn.Parameter(gamma)

        D = self.dictionary.weight.detach().cpu().numpy()
        # normalize the dictionary
        D = D / np.linalg.norm(D, axis=0, keepdims=True, ord=2)
        X = ze_flattened.detach().cpu().numpy()
        # normalize the input
        X = X / np.linalg.norm(X, axis=1, keepdims=True, ord=2)

        gram = D.dot(D.T)
        Xy = D.dot(X.T)

        representation_np = orthogonal_mp_gram(gram, Xy, n_nonzero_coefs=self.sparsity_level).T
        representation = torch.from_numpy(representation_np).to(z_e.device)

        # straight-through estimator
        est = torch.matmul(ze_flattened, self.dictionary.weight.T)
        representation = est + (representation - est).detach()

        return representation

