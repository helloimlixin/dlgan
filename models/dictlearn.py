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
    """
    Online Dictionary Learning with Batch Orthogonal Matching Pursuit (Batch-OMP) for Sparse Coding.

    References:
    - Rubinstein, R., Zibulevsky, M. and Elad, M., "Efficient Implementation of the K-SVD Algorithm using Batch Orthogonal Matching Pursuit," CS Technion, 2008.
    - Mairal J, Bach F, Ponce J, Sapiro G. Online dictionary learning for sparse coding.
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost, sparsity_level):
        """
        class constructor for DictLearn
        :param num_embeddings: number of dictionary atoms
        :param sparsity_level: maximum sparsity (number of non-zero coefficients) of the representation, reduces to K-Means (Vector Quantization) when set to 1
        :param initial_dict: initial dictionary if given, otherwise random rows from the data matrix are used
        :param max_iter: maximum number of iterations
        """
        super(DictLearn, self).__init__()

        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._commitment_cost = commitment_cost
        self._sparsity_level = sparsity_level
        self._dictionary = nn.Parameter(torch.randn(self._embedding_dim, self._num_embeddings, device='cuda'))
        # normalize the dictionary
        self._dictionary.data /= torch.linalg.norm(self._dictionary, dim=0)

        self._col_update = nn.Parameter(torch.zeros(self._embedding_dim, device='cuda'))

        self._gamma = None
        self._A = None
        self._B = None

        self._beta = 0.75 # parameter for dictionary update

    def forward(self, z_e):
        # permute
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        ze_shape = z_e.shape

        # Flatten input
        z_e = z_e.view(self._embedding_dim, -1)  # convert to column-major order, i.e., each column is a data point

        """
        Sparse Coding Stage
        """

        # normalize the dictionary
        # self._dictionary.data = self._dictionary / torch.linalg.norm(self._dictionary, dim=0)

        if self._gamma is None:
            self._gamma = nn.Parameter(torch.zeros((self._num_embeddings, z_e.shape[1]), device='cuda'))
        else:
            self._gamma.data.copy_(nn.Parameter(self.update_gamma(z_e.detach(), self._dictionary.detach(), debug=True)))
            # self._gamma.data.copy_(nn.Parameter(Batch_OMP(z_e.detach(), self._dictionary.detach(), self._sparsity_level, debug=True)))

        encodings = self._gamma

        # compute reconstruction
        recon = self._dictionary @ self._gamma

        e_latent_loss = F.mse_loss(recon.detach(), z_e)  # latent loss from encoder
        loss = e_latent_loss * self._commitment_cost

        # straight-through gradient estimator
        recon = z_e + (recon - z_e).detach()

        # compute perplexity
        perplexity = torch.exp(
            -torch.sum(F.softmax(self._gamma, dim=0) * torch.log(F.softmax(self._gamma, dim=0) + 1e-10), dim=0).mean())

        return loss, recon.reshape(ze_shape).permute(0, 3, 1, 2).contiguous(), z_e.detach(), perplexity, encodings

    def update_dictionary(self, z_e, t):
        """online dictionary update via Block Coordinate Descent.

        References:
        - Mairal J, Bach F, Ponce J, Sapiro G. Online dictionary learning for sparse coding.
          In Proceedings of the 26th annual international conference on machine learning 2009 Jun 14 (pp. 689-696).
        """
        '''
        Online Dictionary Update Stage (Block Coordinate Descent)
        '''

        # compute beta
        eta = z_e.shape[1]

        if t < eta:
            theta = t * eta
        else:
            theta = eta ** 2 + t - eta

        beta = (theta + 1 - eta) / (theta + 1)

        # precomputations
        if self._A is None:
            self._A = nn.Parameter(torch.diag(torch.ones(self._num_embeddings, device='cuda')) * 1e-10)
        else:
            self._A.data.copy_(nn.Parameter(torch.diag(torch.ones(self._num_embeddings, device='cuda')) * 1e-10 + beta * self._A + self._gamma.detach().mm(self._gamma.detach().t())))
        if self._B is None:
            self._B = nn.Parameter(torch.zeros((self._embedding_dim, self._num_embeddings), device='cuda'))
        else:
            self._B.data.copy_(nn.Parameter(beta * self._B.data + z_e.mm(self._gamma.detach().t())))

        # choose the column with the largest gradient
        # j = torch.argmax(-(self._B - self._dictionary @ self._A - self._dictionary * self._A.diag()), dim=1)

        # Block-Coordinate Descent
        self._dictionary.data.copy_(nn.Parameter(self._B - self._dictionary @ self._A) / self._A.diag() + self._dictionary)
        self._dictionary.data.copy_(nn.Parameter(self._dictionary / (torch.linalg.norm(self._dictionary, dim=0) + 1e-10)))

    def update_gamma(self, signals, dictionary, debug=False):
        """sparse coding stage

        Implemented using the Batch Orthogonal Matching Pursuit (OMP) algorithm.

        Reference:
        - Rubinstein, R., Zibulevsky, M. and Elad, M., "Efficient Implementation of the K-SVD Algorithm using Batch Orthogonal Matching Pursuit," CS Technion, 2008.

        :param signals: input signals to be sparsely coded
        """
        embedding_dim, num_signals = signals.size()
        dictionary_t = dictionary.t()  # save the transpose of the dictionary for faster computation
        gram_matrix = dictionary_t.mm(dictionary)  # the Gram matrix, dimension: num_atoms x num_atoms
        corr_init = dictionary_t.mm(signals).t()  # initial correlation vector, transposed to make num_signals the first dimension
        gamma = torch.zeros_like(corr_init)  # placeholder for the sparse coefficients
        delta = torch.zeros(num_signals, device=signals.device)
        eps = torch.norm(signals, dim=0)  # the residual, initialized as the L2 norm of the signal

        corr = corr_init
        L = torch.ones(num_signals, 1, 1, device=signals.device)  # contains the progressive Cholesky of the Gram matrix in the selected indices
        I = torch.zeros(num_signals, 0, dtype=torch.long, device=signals.device)  # placeholder for the index set
        omega = torch.ones_like(corr_init, dtype=torch.bool)  # used to zero out elements in corr before argmax
        signal_idx = torch.arange(num_signals, device=signals.device)

        k = 0
        while k < self._sparsity_level:
            k += 1
            k_hats = torch.argmax(torch.abs(corr * omega), dim=1)  # select the index of the maximum correlation
            # update omega to make sure we do not select the same index twice
            omega[torch.arange(k_hats.shape[0], device=signals.device), k_hats] = 0
            expanded_signal_idx = signal_idx.unsqueeze(0).expand(k, num_signals).t()  # expand is more efficient than repeat

            if k > 1:  # Cholesky update
                G_ = gram_matrix[I[signal_idx, :], k_hats[expanded_signal_idx[...,:-1]]].view(num_signals, k - 1, 1)  # compute for all signals in a vectorized manner
                w = torch.linalg.solve_triangular(L, G_, upper=False).view(-1, 1, k - 1)
                w_br = torch.sqrt(1 - (w ** 2).sum(dim=2, keepdim=True))  # L bottom-right corner element: sqrt(1 - w.t().mm(w))

                # concatenate into the new Cholesky: L <- [[L, 0], [w, w_br]]
                k_zeros = torch.zeros(num_signals, k - 1, 1, device=signals.device)
                L = torch.cat((
                    torch.cat((L, k_zeros), dim=2),
                    torch.cat((w, w_br), dim=2),
                ), dim=1)

            # update non-zero indices
            I = torch.cat([I, k_hats.unsqueeze(1)], dim=1)

            # solve L
            corr_ = corr_init[expanded_signal_idx, I[signal_idx, :]].view(num_signals, k, 1)
            gamma_ = torch.cholesky_solve(corr_, L)

            # de-stack gamma into the non-zero elements
            gamma[signal_idx.unsqueeze(1), I[signal_idx]] = gamma_[signal_idx].squeeze(-1)

            # beta = G_I * gamma_I
            beta = gamma[signal_idx.unsqueeze(1), I[signal_idx]].unsqueeze(1).bmm(
                gram_matrix[I[signal_idx], :]).squeeze(1)

            corr = corr_init - beta

            # update residual
            new_delta = (gamma * beta).sum(dim=1)
            eps += delta - new_delta
            delta = new_delta

            if debug and k % 1 == 0:
                print('Step {}, residual: {:.4f}, below tolerance: {:.4f}'.format(k, eps.max(), (eps < 1e-7).float().mean().item()))

        return gamma.t()  # transpose the sparse coefficients to make num_signals the first dimension
