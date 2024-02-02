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
    layers.append(nn.Linear(self.dim, self.num_atoms)) # output dim for one sample: K
    layers.append(nn.Softmax(dim=1)) # convert the output to [0, 1] range

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

    reconstruction = reconstruction + (reconstruction - x).detach() # straight-through gradient

    return recon_loss, reconstruction, representation

  def forward(self, x):
    """Forward pass.

    Args:
      x: input data, for image data, the dimension is:
          batch_size z_e num_channels z_e height z_e width
    """
    batch_size, num_channels, height, width = x.shape
    x = x.permute(0, 2, 3, 1).contiguous()
    x_flattened = x.view(-1, self.dim) # data dimension: N z_e D
    representation = self.representation(x_flattened)
    encodings = torch.zeros(representation.shape, requires_grad=True).cuda()

    # compute the l2 distances (N z_e K) between x_flattened (N z_e D) and the dictionary (K z_e D)
    distances = torch.sum(x_flattened**2, dim=1, keepdim=True) + torch.sum(
        self.dictionary**2, dim=1) - 2 * torch.matmul(x_flattened, self.dictionary.t()) # taking a negation for topk operation

    elements, indices = distances.topk(self.sparsity_level, dim=1, largest=False) # topk operation

    encodings.scatter_(1, indices, 1)
    representation = representation * encodings

    reconstruction = torch.matmul(representation, self.dictionary)
    reconstruction = reconstruction.view(batch_size, self.dim, height, width).contiguous()

    x = x.permute(0, 3, 1, 2).contiguous() # permute back
    x = x.view(batch_size, self.dim, height, width).contiguous()

    recon_loss = nn.MSELoss()(x, reconstruction.detach()) + nn.MSELoss()(x.detach(), reconstruction)
    regularization = torch.sum(torch.abs(representation))

    reconstruction = x + (reconstruction - x).detach() # straight-through gradient

    # average pooling over the spatial dimensions
    # avg_probs: B z_e _num_embeddings
    avg_probs = torch.mean(encodings, dim=0)
    avg_probs = avg_probs / torch.sum(avg_probs)  # normalize the representation
    # codebook perplexity / usage: 1
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

    # return representation
    return recon_loss + regularization, reconstruction, perplexity, representation


class DictLearnEMA(nn.Module):
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
  def __init__(self, dim, num_atoms, sparsity_level, decay, epsilon=1e-8) -> None:
    super(DictLearnEMA, self).__init__()
    self.dim = dim
    self.num_atoms = num_atoms
    self.dictionary = nn.Parameter(torch.rand((self.num_atoms, self.dim), dtype=torch.float, requires_grad=True))
    self.dictionary.data.normal_()
    self.commitment_cost = nn.Parameter(torch.tensor(0.25)) # make commitment cost learnable
    self.representation = self.representation_builder()
    self.sparsity_level = sparsity_level

    '''
    Here the register_buffer() method is typically used to register a buffer
    that should not to be considered a model parameter. For example,
    BatchNorm’s running_mean is not a parameter, but is part of the module’s
    state. Buffers, by default, are persistent and will be saved alongside
    parameters. This behavior can be changed by setting persistent to False.
    The only difference between a persistent buffer and a non-persistent
    buffer is that the latter will not be a part of this module’s state_dict.
    '''
    self.register_buffer('_ema_cluster_size', torch.zeros(num_atoms))  # N
    self._ema_w = nn.Parameter(torch.Tensor(num_atoms, self.dim), requires_grad=True)  # m_i
    self._ema_w.data.normal_()

    self._decay = decay # decay for the moving averages
    self._epsilon = epsilon # a small number to avoid the numerical issues

  def representation_builder(self):
    layers = nn.ModuleList()
    layers.append(nn.Linear(self.dim, self.num_atoms)) # output dim for one sample: K
    layers.append(nn.Softmax(dim=1)) # convert the output to [0, 1] range

    return nn.Sequential(*layers)

  def loss(self, z_e, representation):
    """Forward pass.

    Args:
      z_e: input data, for image data, the dimension is:
          batch_size z_e num_channels z_e height z_e width
    """
    z_e = z_e.permute(0, 2, 3, 1).contiguous() # permute the input
    ze_shape = z_e.shape # save the shape

    # flatten the input
    ze_flattened = z_e.view(-1, self.dim) # data dimension: N z_e D

    # compute the l2 distances (N z_e K) between x_flattened (N z_e D) and the dictionary (K z_e D)
    distances = -torch.sum(ze_flattened**2, dim=1, keepdim=True) + torch.sum(
        self.dictionary**2, dim=1) - 2 * torch.matmul(ze_flattened, self.dictionary.t()) # taking a negation for topk operation

    # encodings: N z_e K
    elements, encoding_indices = distances.topk(self.sparsity_level, dim=1) # topk operation
    encodings = torch.zeros(encoding_indices.shape[0], self.num_atoms, device=ze_flattened.device)
    encodings.scatter_(1, encoding_indices, 1)

    # compute the representation
    representation = representation * encodings

    # compute the reconstruction from the representation
    reconstruction = torch.matmul(representation, self.dictionary).view(ze_shape)

    # Use EMA to update the dictionary
    if self.training:
      self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                               (1 - self._decay) * torch.sum(encodings, 0)

      # Laplace smoothing of the cluster size
      n = torch.sum(self._ema_cluster_size.data)
      self._ema_cluster_size = (
              (self._ema_cluster_size + self._epsilon)
              / (n + self.num_atoms * self._epsilon) * n)  # N_i

      dw = torch.matmul(encodings.t(), ze_flattened)  # z_ij
      self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)  # m_i

      self.dictionary = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))  # e_i
      self.dictionary = nn.Parameter(self.dictionary / torch.norm(self.dictionary, dim=1, keepdim=True)) # e_i

    recon_loss = F.mse_loss(reconstruction.detach(), z_e) * 0.25
    regularization = torch.sum(torch.abs(representation))


    reconstruction = z_e + (reconstruction - z_e).detach() * 0.25 # straight-through gradient

    # average pooling over the spatial dimensions
    # avg_probs: B z_e _num_embeddings
    avg_probs = torch.mean(encodings, dim=0)
    avg_probs = avg_probs / torch.sum(avg_probs)  # normalize the representation

    # codebook perplexity / usage: 1
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self._epsilon)))

    # return representation
    return recon_loss, reconstruction.permute(0, 3, 1, 2).contiguous(), perplexity, representation

  def forward(self, z_e):
    z_e = z_e.permute(0, 2, 3, 1).contiguous()
    ze_flattened = z_e.view(-1, self.dim)
    return self.representation(ze_flattened)