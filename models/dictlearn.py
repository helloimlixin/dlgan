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

        self._beta = nn.Parameter(torch.tensor(1 - commitment_cost, device='cuda')) # learnable parameter for dictionary update

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
        self._dictionary.data = self._dictionary / torch.linalg.norm(self._dictionary, dim=0)

        if self._gamma is None:
            self._gamma = nn.Parameter(torch.zeros((self._num_embeddings, z_e.shape[1]), device='cuda'))
        else:
            self._gamma.data.copy_(nn.Parameter(Batch_OMP(z_e, self._dictionary, self._sparsity_level, debug=False)))

        encodings = self._gamma

        # compute reconstruction
        recon = self._dictionary @ self._gamma

        e_latent_loss = F.mse_loss(recon.detach(), z_e)  # latent loss from encoder
        loss = e_latent_loss * self._commitment_cost + self._beta * F.mse_loss(recon, z_e.detach())

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
            self._A = nn.Parameter(torch.diag(torch.ones(self._num_embeddings, device='cuda')) * 0.01)
        else:
            self._A.data.copy_(nn.Parameter(beta * self._A + self._gamma.mm(self._gamma.t())))
        if self._B is None:
            self._B = nn.Parameter(torch.zeros((self._embedding_dim, self._num_embeddings), device='cuda'))
        else:
            self._B.data.copy_(nn.Parameter(beta * self._B.data + z_e.mm(self._gamma.t())))

        # choose the column with the largest gradient
        # j = torch.argmax(-(self._B - self._dictionary @ self._A - self._dictionary * self._A.diag()), dim=1)

        # Block-Coordinate Descent
        for j in range(self._num_embeddings):
            self._dictionary[:, j].data.copy_((nn.Parameter(self._B[:, j] - self._dictionary @ self._A[:, j]) / self._A[j, j] + self._dictionary[:, j]))
            self._dictionary[:, j].data.copy_(nn.Parameter(self._dictionary[:, j] / torch.clamp(torch.linalg.norm(self._dictionary[:, j]), min=1.)))
        # self._dictionary[:, j].data = (self._B[:, j] - self._dictionary @ self._A[:, j] + self._dictionary[:, j] * self._A[j, j]) / self._A[j, j]
        # self._dictionary[:, j].data = self._dictionary[:, j] / torch.linalg.norm(self._dictionary[:, j])

        print(self._dictionary)

    def update_gamma(self, signals):
        """sparse coding stage

        Implemented using the Batch Orthogonal Matching Pursuit (OMP) algorithm.

        Reference:
        - Rubinstein, R., Zibulevsky, M. and Elad, M., "Efficient Implementation of the K-SVD Algorithm using Batch Orthogonal Matching Pursuit," CS Technion, 2008.

        :param signals: input signals to be sparsely coded
        """
        embedding_dim, num_signals = signals.shape
        dictionary_t = self._dictionary.t()  # save the transpose of the dictionary for faster computation
        gram_matrix = dictionary_t.mm(self._dictionary)  # the Gram matrix, dimension: num_atoms x num_atoms
        corr_init = dictionary_t.mm(
            signals).t()  # initial correlation vector, transposed to make num_signals the first dimension
        gamma = torch.zeros_like(corr_init)  # placeholder for the sparse coefficients

        corr = corr_init
        L = torch.ones(num_signals, 1, 1,
                       device=signals.device)  # contains the progressive Cholesky of the Gram matrix in the selected indices
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
                G_ = gram_matrix[I[signal_idx, :], k_hats[expanded_signal_idx[..., :-1]]].view(num_signals, k - 1, 1)  # compute for all signals in a vectorized manner
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
            corr_ = corr[expanded_signal_idx, I[signal_idx, :]].view(num_signals, k, 1)
            gamma_ = torch.cholesky_solve(corr_, L)

            # de-stack gamma into the non-zero elements
            gamma[signal_idx.unsqueeze(1), I[signal_idx]] = gamma_[signal_idx].squeeze(-1)

            # beta = G_I * gamma_I
            beta = gamma[signal_idx.unsqueeze(1), I[signal_idx]].unsqueeze(1).bmm(
                gram_matrix[I[signal_idx], :]).squeeze(1)

            corr = corr_init - beta

            # # update residual
            # new_delta = (gamma * beta).sum(dim=1)
            # eps += delta - new_delta
            # delta = new_delta

        return gamma.t()  # transpose the sparse coefficients to make num_signals the first dimension

import torch
from sklearn.decomposition import DictionaryLearning


def get_largest_eigenvalue(X):
    eigs = torch.eig(X, eigenvectors=False).eigenvalues
    max_eign = eigs.max(dim=0)
    return max_eign.values[0]


def shrink_function(Z, cutoff):
    cutted = shrink1(Z, cutoff)
    maxed = shrink2(Z, cutted)
    signed = shrink3(Z, maxed)
    return signed


def shrink3(Z, maxed):
    signed = maxed * torch.sign(Z)
    return signed


def shrink2(Z, cutted):
    maxed = torch.max(cutted, torch.zeros(Z.size(), dtype=Z.dtype).cuda(3))
    return maxed


def shrink1(Z, cutoff):
    cutted = torch.abs(Z) - cutoff
    return cutted


def reconstruction_distance(D, cur_Z, last_Z):
    distance = torch.norm(D.mm(last_Z - cur_Z), p=2, dim=0) / torch.norm(D.mm(last_Z), p=2, dim=0)
    max_distance = distance.max()
    return distance, max_distance


def OMP(X, D, K, tolerance, debug=False):
    Dt = D.t()
    Dpinv = torch.pinverse(D)
    r = X
    I = []
    stopping = False
    last_sparse_code = torch.zeros((D.size()[1], X.size()[1]), dtype=X.dtype)#.cuda(3)
    sparse_code = torch.zeros((D.size()[1], X.size()[1]), dtype=X.dtype)#.cuda(3)

    step = 0
    while not stopping:
        k_hat = torch.argmax(Dt.mm(r), 0)
        I.append(k_hat)
        sparse_code = Dpinv.mm(X) # Should be: (torch.pinverse(D[:,I])*X).sum(0)
        r = X - D.mm(sparse_code)

        distance, max_distance = reconstruction_distance(D, sparse_code, last_sparse_code)
        stopping = len(I) >= K or max_distance < tolerance
        last_sparse_code = sparse_code

        if debug and step % 1 == 0:
            print('Step {}, code improvement: {}, below tolerance: {}'.format(step, max_distance, (distance < tolerance).float().mean().item()))

        step += 1

    return sparse_code


def _update_logical(logical, to_add):
    running_idx = torch.arange(to_add.shape[0], device=to_add.device)
    logical[running_idx, to_add] = 1


def Batch_OMP(data, dictionary, max_nonzero, tolerance=1e-7, debug=False):
    """
    for details on variable names, see
    https://sparse-plex.readthedocs.io/en/latest/book/pursuit/omp/batch_omp.html
    or the original paper
    http://www.cs.technion.ac.il/~ronrubin/Publications/KSVD-OMP-v2.pdf

    NOTE - the implementation below works on transposed versions of the input signal to make the batch size the first
           coordinate, which is how pytorch expects the data..
    """
    vector_dim, batch_size = data.size()
    dictionary_t = dictionary.t()
    G = dictionary_t.mm(dictionary)  # this is a Gram matrix
    eps = torch.norm(data, dim=0)  # the residual, initalized as the L2 norm of the signal
    h_bar = dictionary_t.mm(data).t()  # initial correlation vector, transposed to make batch_size the first dimension

    # note - below this line we no longer use "data" or "dictionary"

    h = h_bar
    x = torch.zeros_like(h_bar)  # the resulting sparse code
    L = torch.ones(batch_size, 1, 1, device=h.device)  # Contains the progressive Cholesky of G in selected indices
    I = torch.ones(batch_size, 0, device=h.device).long()
    I_logic = torch.zeros_like(h_bar).bool()  # used to zero our elements is h before argmax
    delta = torch.zeros(batch_size, device=h.device)  # to track errors

    k = 0
    while k < max_nonzero and eps.max() > tolerance:
        k += 1
        # use "I_logic" to make sure we do not select same index twice
        index = (h*(~I_logic).float()).abs().argmax(dim=1)  # todo - can we use "I" rather than "I_logic"
        _update_logical(I_logic, index)
        batch_idx = torch.arange(batch_size, device=G.device)
        expanded_batch_idx = batch_idx.unsqueeze(0).expand(k, batch_size).t()

        if k > 1:  # Cholesky update
            # Following line is equivalent to:
            #   G_stack = torch.stack([G[I[i, :], index[i]] for i in range(batch_size)], dim=0).view(batch_size, k-1, 1)
            G_stack = G[I[batch_idx, :], index[expanded_batch_idx[...,:-1]]].view(batch_size, k-1, 1)
            w = torch.linalg.solve_triangular(L, G_stack, upper=False, ).view(-1, 1, k-1)
            w_corner = torch.sqrt(1-(w**2).sum(dim=2, keepdim=True))  # <- L corner element: sqrt(1- w.t().mm(w))

            # do concatenation into the new Cholesky: L <- [[L, 0], [w, w_corner]]
            k_zeros = torch.zeros(batch_size, k-1, 1, device=h.device)
            L = torch.cat((
                torch.cat((L, k_zeros), dim=2),
                torch.cat((w, w_corner), dim=2),
            ), dim=1)

        # update non-zero indices
        I = torch.cat([I, index.unsqueeze(1)], dim=1)

        # x = solve L
        # The following line is equivalent to:
        #   h_stack = torch.stack([h_bar[i, I[i, :]] for i in range(batch_size)]).unsqueeze(2)
        h_stack = h_bar[expanded_batch_idx, I[batch_idx, :]].view(batch_size, k, 1)
        x_stack = torch.cholesky_solve(h_stack, L)

        # de-stack x into the non-zero elements
        # The following line is equivalent to:
        #   for i in range(batch_size):
        #       x[i:i+1, I[i, :]] = x_stack[i, :].t()
        x[batch_idx.unsqueeze(1), I[batch_idx]] = x_stack[batch_idx].squeeze(-1)

        # beta = G_I * x_I
        # The following line is equivalent to:
        # beta = torch.cat([x[i:i+1, I[i, :]].mm(G[I[i, :], :]) for i in range(batch_size)], dim=0)
        beta = x[batch_idx.unsqueeze(1), I[batch_idx]].unsqueeze(1).bmm(G[I[batch_idx], :]).squeeze(1)

        h = h_bar - beta

        # update residual
        new_delta = (x * beta).sum(dim=1)
        eps += delta-new_delta
        delta = new_delta

        if debug and k % 1 == 0:
            print('Step {}, residual: {:.4f}, below tolerance: {:.4f}'.format(k, eps.max(), (eps < tolerance).float().mean().item()))

    return x.t()  # transpose since sparse codes should be used as D * x


if __name__ == '__main__':
    import time
    from tqdm import tqdm
    torch.manual_seed(0)
    use_gpu = torch.cuda.device_count() > 0
    device = 'cuda' if use_gpu else 'cpu'

    num_nonzeros = 4
    num_samples = int(1e4)
    num_atoms = 512
    embedding_size = 64

    Wd = torch.randn(embedding_size, num_atoms)
    Wd = torch.nn.functional.normalize(Wd, dim=0).to(device)

    codes = []
    for i in tqdm(range(num_samples), desc='generating codes... '):
        tmp = torch.zeros(num_atoms).to(device)
        tmp[torch.randperm(num_atoms)[:num_nonzeros]] = 0.5 * torch.rand(num_nonzeros).to(device) + 0.5
        codes.append(tmp)
    codes = torch.stack(codes, dim=1)
    X = Wd.mm(codes)
    # X += torch.randn(X.size()) / 100  # add noise
    # X = torch.nn.functional.normalize(X, dim=0)  # normalize signal

    if use_gpu:  # warm start?
        print('doing warm start...')
        Batch_OMP(X[:, :min(num_nonzeros, 1000)], Wd, num_nonzeros)

    tic = time.time()
    Z2 = Batch_OMP(X, Wd, num_nonzeros, debug=True)
    Z2_time = time.time() - tic
    print(f'Z2, {torch.isclose(codes, Z2, rtol=1e-03, atol=1e-05).float().mean()}, time/sample={1e6*Z2_time/num_samples/num_nonzeros:.4f}usec')
    pass

