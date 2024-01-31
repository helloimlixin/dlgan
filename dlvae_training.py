#  ==============================================================================
#  Description: Helper functions for the vqvae.
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
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import utils as torchvisionutils
from torchvision import transforms

from dataloaders.cifar10 import get_cifar10_train_loader
from dataloaders.flowers import FlowersDataset

from models.dlvae import DLVAE
import numpy as np
import time
import os

from utils import init_weights, load_data

# fix the bug of "OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized."
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# hyperparameters
train_batch_size = 4
test_batch_size = 32
num_training_updates = 100000

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512

commitment_cost = 0.25

decay = 0.99

learning_rate = 1e-5

epsilon = 1e-10 # a small number to avoid the numerical issues

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

l2_loss_factor = 0.9
lpips_loss_factor = 1 - l2_loss_factor

sparsity_level = 200 # number of atoms selected

epsilon = 1e-10 # a small number to avoid the numerical issues

# data_paths loaders
flowers_dataset = FlowersDataset(root='./data/flowers')
train_loader = DataLoader(flowers_dataset, batch_size=train_batch_size, shuffle=True)
# train_loader, data_variance = get_cifar10_train_loader(batch_size=train_batch_size)()

# dlvae
dlvae = DLVAE(in_channels=3,
            num_hiddens=num_hiddens,
            num_residual_hiddens=num_residual_hiddens,
            num_residual_layers=num_residual_layers,
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            sparsity_level=sparsity_level,
            decay=decay).to(device)

# dlvae_optimizer
optimizer = torch.optim.Adam(dlvae.parameters(), lr=learning_rate, amsgrad=False)

# loss function
def loss_function(recon_x, x):
    recon_error = F.mse_loss(recon_x, x)
    return recon_error

def train_dlvae():
    from lpips import LPIPS
    '''Train the vqvae.'''
    train_res_recon_error = []
    train_res_perplexity = []

    perceptual_loss_criterion = LPIPS(net='vgg').to(device)

    dlvae.train() # set the vqvae to training mode
    writer = SummaryWriter('./runs/dictlearn') # create a writer object for TensorBoard

    for i in range(num_training_updates):
        # sample the mini-batch
        x = next(iter(train_loader))
        x = x.to(device)

        optimizer.zero_grad() # clear the gradients

        # forward pass
        dl_loss, data_recon, perplexity, representation = dlvae(x)

        recon_error = l2_loss_factor * loss_function(data_recon, x) + lpips_loss_factor * perceptual_loss_criterion(data_recon, x).mean()
        # recon_error = loss_function(data_recon, x)

        loss = recon_error # total loss

        # backward pass
        loss.backward(retain_graph=True)

        if (i + 1) % 200 == 0:
            dl_loss.backward()  # update the dictionary learning bottleneck

        # update parameters
        optimizer.step()

        # print training information
        if (i + 1) % 100 == 0:
            print('%d iterations' % (i + 1))
            print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
            print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
            print()

        originals = x + 0.5 # add 0.5 to match the range of the original images [0, 1]
        reconstructions = data_recon + 0.5 # add 0.5 to match the range of the original images [0, 1]

        # save training information for TensorBoard
        writer.add_scalar('Train Recon Error', recon_error.item(), i+1)
        writer.add_scalar('Train Dictionary Learning Loss', dl_loss.item(), i+1)
        writer.add_scalar('Train Perplexity', perplexity.item(), i+1)
        writer.add_scalar('Train Loss', loss.item(), i+1)

        # save training information for plotting
        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())

        # save the vqvae graph
        if i == 0:
            writer.add_graph(dlvae, x)

        # save the reconstructed images
        if (i + 1) % 100 == 0:
            writer.add_images('Train Original Images', originals, i+1)
            writer.add_images('Train Reconstructed Images', reconstructions, i+1)

        # save the codebook
        if (i + 1) % 100 == 0:
            writer.add_embedding(representation.view(train_batch_size, -1), label_img=originals, global_step=i+1)

        # save the gradient visualization
        # if (i + 1) % 100 == 0:
        #     for name, param in vqvae.named_parameters():
        #         writer.add_histogram(name, param.clone().cpu().data_paths.numpy(), i+1)
        #         writer.add_histogram(name+'/grad', param.grad.clone().cpu().data_paths.numpy(), i+1)

        # save the training information
        if (i + 1) % 100 == 0:
            np.save('train_res_recon_error.npy', train_res_recon_error)
            np.save('train_res_perplexity.npy', train_res_perplexity)

        # save the vqvae
        if (i + 1) % 1000 == 0:
            torch.save(dlvae.state_dict(), './checkpoints/dictlearn/vqvae_%d.pt' % (i + 1))

        # save the images
        if (i + 1) % 1000 == 0:
            torchvisionutils.save_image(originals, './dlvae_results/vanilla/originals_%d.png' % (i + 1))
            torchvisionutils.save_image(reconstructions, './dlvae_results/vanilla/reconstructions_%d.png' % (i + 1))

    writer.close()


if __name__ == '__main__':
    start = time.time()
    train_dlvae()
    end = time.time()
    print('Training time: %f seconds.' % (end-start))

