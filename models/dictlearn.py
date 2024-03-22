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
from ksvd import ApproximateKSVD

class DictionaryLearningKNN(nn.Module):
    """A simple dictionary learning algorithm with k-Nearest Neighbors.

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
        super(DictionaryLearningKNN, self).__init__()
        self.dim = dim
        self.num_atoms = num_atoms
        self.dictionary = nn.Embedding(self.num_atoms, self.dim)
        self.commitment_cost = commitment_cost
        self.sparsity_level = sparsity_level

        self.representation = self.representation_builder()

        self._epsilon = epsilon  # a small number to avoid the numerical issues

    def representation_builder(self):
        '''
        Build the representation layer.

        :return: weight matrix is of dimension N x K
        '''
        layers = nn.ModuleList()
        layers.append(nn.Linear(self.dim, self.num_atoms))
        layers.append(nn.Softmax(dim=1))

        return nn.Sequential(*layers)


    def loss(self, z_e, representation):
        """Forward pass.

    Args:
      z_e: input data, for image data, the dimension is:
          batch_size z_e num_channels z_e height z_e width
    """
        z_e = z_e.permute(0, 2, 3, 1).contiguous()  # permute the input
        ze_shape = z_e.shape  # save the shape
        ze_flattened = z_e.view(-1, self.dim)  # data dimension: N x D

        distances = (torch.sum(ze_flattened ** 2, dim=1, keepdim=True) + torch.sum(self.dictionary.weight ** 2, dim=1)
                     - 2 * torch.matmul(ze_flattened, self.dictionary.weight.T))

        # find the nearest neighbors
        _, indices = torch.topk(distances, self.sparsity_level, dim=1, largest=False)

        # construct the representation
        encodings = torch.zeros(ze_flattened.shape[0], self.num_atoms).to(z_e.device)
        encodings.scatter_(1, indices, 1)

        representation = representation * encodings

        # compute the reconstruction from the sparse representation
        z_dl = torch.matmul(representation, self.dictionary.weight)

        z_dl = z_dl.view(ze_shape).contiguous()

        # compute the commitment loss
        # commitment_loss: B z_e 1 z_e H z_e W
        commitment_loss = self.commitment_cost * F.mse_loss(z_dl.detach(), z_e)
        # compute the z_dl latent loss
        e2z_loss = F.mse_loss(z_dl, z_e.detach())

        recon_loss = commitment_loss + e2z_loss

        # straight-through gradient
        z_dl = z_e + (z_dl - z_e).detach()  # B z_e C z_e H z_e W

        # average pooling over the spatial dimensions
        # avg_probs: B z_e _num_embeddings
        avg_probs = torch.mean(representation, dim=0)
        avg_probs = avg_probs / torch.sum(avg_probs)  # normalize the representation

        # codebook perplexity / usage: 1
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self._epsilon)))

        # return representation, reconstruction, perplexity, regularization
        return recon_loss, z_dl.permute(0, 3, 1, 2).contiguous(), perplexity, representation

    def forward(self, z_e):
        """Forward pass to compute the sparse representation.

        Args:
            z_e: input data, for image data, the dimension is:
                batch_size z_e num_channels z_e height z_e width
        #     """
        z_e = z_e.permute(0, 2, 3, 1).contiguous()

        ze_flattened = z_e.view(-1, self.dim).contiguous()  # data dimension: N x D
        D = self.dictionary.weight  # dictionary subset with dimension D x K
        representation = self.representation(ze_flattened)  # representation with dimension N x K

        return representation


def reconstruction_distance(D, cur_Z, last_Z):
    distance = torch.norm(D.mm(last_Z - cur_Z), p=2, dim=0) / torch.norm(D.mm(last_Z), p=2, dim=0)
    max_distance = distance.max()
    return distance, max_distance


class DictionaryLearningOMP(nn.Module):
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
        super(DictionaryLearningOMP, self).__init__()
        self.dim = dim
        self.num_atoms = num_atoms
        self.dictionary = nn.Parameter(torch.randn(self.num_atoms, self.dim)) # dictionary matrix A with dimension D x K

        self.commitment_cost = commitment_cost

        self.sparsity_level = sparsity_level
        self.dl = ApproximateKSVD(n_components=self.num_atoms,
                                    transform_n_nonzero_coefs=self.sparsity_level)

        self._epsilon = epsilon  # a small number to avoid the numerical issues

    def loss(self, z_e, representation):
        """Forward pass.

    Args:
      z_e: input data, for image data, the dimension is:
          batch_size z_e num_channels z_e height z_e width
    """
        z_e = z_e.permute(0, 2, 3, 1).contiguous()  # permute the input
        # normalize the input
        z_e = z_e / torch.norm(z_e, p=2, dim=3, keepdim=True)
        ze_shape = z_e.shape  # save the shape

        X = z_e.view(-1, self.dim)  # data dimension: N x D
        D = self.dictionary
        Gamma = representation
        self.dictionary.data, representation = self.update_dict(X, D, Gamma)

        # compute the reconstruction from the representation
        z_dl = torch.matmul(representation, self.dictionary)  # reconstruction: B z_e D
        z_dl = z_dl.view(ze_shape).contiguous()

        # compute the commitment loss
        # commitment_loss: B z_e 1 z_e H z_e W
        commitment_loss = self.commitment_cost * F.mse_loss(z_dl.detach(), z_e)
        # compute the z_dl latent loss
        e2z_loss = F.mse_loss(z_dl, z_e.detach())

        recon_loss = commitment_loss + e2z_loss

        z_dl = z_e + (z_dl - z_e).detach()  # B z_e C z_e H z_e W, straight-through gradient

        # average pooling over the spatial dimensions
        # avg_probs: B z_e _num_embeddings
        avg_probs = torch.mean(representation, dim=0)
        avg_probs = avg_probs / torch.sum(avg_probs)  # normalize the representation

        # codebook perplexity / usage: 1
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self._epsilon)))

        # return representation, reconstruction, perplexity, regularization
        return recon_loss, z_dl.permute(0, 3, 1, 2).contiguous(), perplexity, representation

    def forward(self, z_e):
        """Compute the representation with Matching Pursuit."""
        X = z_e.view(self.dim, -1)  # data dimension: N x D

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

        D = self.dictionary.t()
        k = self.sparsity_level
        Dt = D.t()
        Dpinv = torch.pinverse(D)
        r = X
        I = []
        stopping = False
        last_sparse_code = torch.zeros((D.size()[1], X.size()[1]), dtype=X.dtype).cuda()
        sparse_code = torch.zeros((D.size()[1], X.size()[1]), dtype=X.dtype).cuda()

        step = 0
        while not stopping:
            k_hat = torch.argmax(Dt.mm(r), 0)
            I.append(k_hat)
            sparse_code = Dpinv.mm(X)  # Should be: (torch.pinverse(D[:,I])*X).sum(0)
            r = X - D.mm(sparse_code)

            distance, max_distance = reconstruction_distance(D, sparse_code, last_sparse_code)
            stopping = len(I) >= k or max_distance < self._epsilon
            last_sparse_code = sparse_code

            step += 1

        return sparse_code.t()

    def update_dict(self, X, D, Gamma):
        for j in range(self.sparsity_level):
            I = Gamma[:, j] > 0
            if torch.sum(I) == 0:
                continue

            D[j, :].data = torch.tensor(0, dtype=torch.float).to(X.device)
            g = Gamma[I, j].t()
            r = X[I, :] - Gamma[I, :] @ D
            d = r.t() @ g
            d /= torch.linalg.norm(d)
            g = r @ d
            D[j, :].data = d
            Gamma[I, j].data = g.t()
        return D, Gamma

class DictionaryLearningkSVD(nn.Module):
    """Online dictionary learning.

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
        super(DictionaryLearningkSVD, self).__init__()
        self.dim = dim
        self.num_atoms = num_atoms
        self.dictionary = nn.Parameter(torch.randn(self.num_atoms, self.dim)) # dictionary matrix A with dimension D x K

        self.commitment_cost = commitment_cost

        self.sparsity_level = sparsity_level
        self.dl = ApproximateKSVD(n_components=self.num_atoms,
                                    transform_n_nonzero_coefs=self.sparsity_level)

        self._epsilon = epsilon  # a small number to avoid the numerical issues

    def forward(self, z_e, update_dictionary=True):
        """Forward pass.

    Args:
      z_e: input data, for image data, the dimension is:
          batch_size z_e num_channels z_e height z_e width
    """
        z_e = z_e.permute(0, 2, 3, 1).contiguous()  # permute the input
        # normalize the input
        z_e = z_e / torch.norm(z_e, p=2, dim=3, keepdim=True)
        ze_flattened = z_e.view(-1, self.dim)  # data dimension: N x D
        ze_shape = z_e.shape  # save the shape

        # put ze_flattened to the CPU
        ze_flattened = ze_flattened.cpu().detach().numpy()
        if update_dictionary:
            self.dictionary.data = torch.tensor(self.dl.fit(ze_flattened).components_, dtype=torch.float).to(z_e.device)
        representation = torch.tensor(self.dl._transform(self.dictionary.cpu().detach().numpy(), ze_flattened), dtype=torch.float).to(z_e.device)

        # compute the reconstruction from the representation
        z_dl = torch.matmul(representation, self.dictionary)  # reconstruction: B z_e D
        z_dl = z_dl.view(ze_shape).contiguous()

        # compute the commitment loss
        # commitment_loss: B z_e 1 z_e H z_e W
        commitment_loss = self.commitment_cost * F.mse_loss(z_dl.detach(), z_e)
        # compute the z_dl latent loss
        e2z_loss = F.mse_loss(z_dl, z_e.detach())

        recon_loss = commitment_loss + e2z_loss

        z_dl = z_e + (z_dl - z_e).detach()  # B z_e C z_e H z_e W, straight-through gradient

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

        D = self.dictionary.data.detach().cpu().numpy()
        # normalize the dictionary
        D = D / np.linalg.norm(D, axis=0, keepdims=True, ord=2)
        X = ze_flattened.detach().cpu().numpy()
        # normalize the input
        X = X / np.linalg.norm(X, axis=1, keepdims=True, ord=2)

        gram = D.dot(D.T)
        Xy = D.dot(X.T)

        representation_np = orthogonal_mp_gram(gram, Xy, n_nonzero_coefs=self.sparsity_level).T
        representation = torch.from_numpy(representation_np).to(z_e.device)

        # normalize the representation
        representation = representation / torch.norm(representation, p=1, dim=1, keepdim=True)

        return representation

