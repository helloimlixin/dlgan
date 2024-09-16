#  ==============================================================================
#  Description: Helper functions for the model.
#  im shady.
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
from torch import nn
from torch.nn import functional as F

class SepDictLearn(nn.Module):
    '''Separable dictionary learning.'''

    def __init__(self, in_channels, out_channels, num_atoms, atom_size, sparsity, num_iters):
        super(SepDictLearn, self).__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._num_atoms = num_atoms
        self._atom_size = atom_size
        self._sparsity = sparsity
        self._num_iters = num_iters

        self._atoms = nn.Parameter(torch.randn(out_channels, num_atoms, atom_size, atom_size))
        self._alphas = nn.Parameter(torch.randn(in_channels, num_atoms, atom_size, atom_size))

    def forward(self, x):
        '''Forward pass.'''

        # initialize activations
        alphas = self._alphas.repeat(x.size(0), 1, 1, 1, 1)

        # initialize atoms
        atoms = self._atoms.repeat(x.size(0), 1, 1, 1, 1)

        for _ in range(self._num_iters):
            # update activations
            alphas = self.update_activations(x, atoms, alphas)

            # update atoms
            atoms = self.update_atoms(x, atoms, alphas)

        return atoms

    def update_activations(self, x, atoms, alphas):
        '''Update activations.'''

        # compute residuals
        residuals = x - F.conv2d(alphas, atoms, stride=1, padding=1)

        # compute activations
        alphas = F.conv_transpose2d(residuals, atoms, stride=1, padding=1)

        # apply sparsity
        alphas = F.softshrink(alphas, lambd=self._sparsity)

        return alphas

    def update_atoms(self, x, atoms, alphas):
        '''Update atoms.'''

        # compute residuals
        residuals = x - F.conv2d(alphas, atoms, stride=1, padding=1)

        # compute atoms
        atoms = F.conv2d(alphas, residuals, stride=1, padding=1)

        return atoms