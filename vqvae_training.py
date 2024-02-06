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
from dataloaders.flowers import FlowersDataset

from lpips import LPIPS
from models.vqvae import VQVAE
import numpy as np
import time
import os

from pathlib import Path
import shutil

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

learning_rate = 1e-4

epsilon = 1e-10 # a small number to avoid the numerical issues

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

l2_loss_factor = 0.5
lpips_loss_factor = 1 - l2_loss_factor

# data_paths loaders
# train_loader, data_variance = get_cifar10_train_loader(batch_size=train_batch_size)()

flowers_dataset = FlowersDataset(root='./data/flowers')
train_loader = DataLoader(flowers_dataset, batch_size=train_batch_size, shuffle=True)
# test_loader = get_cifar10_test_loader(batch_size=test_batch_size)()

# vqvae
vqvae = VQVAE(in_channels=3,
              num_hiddens=num_hiddens,
              num_residual_layers=num_residual_layers,
              num_residual_hiddens=num_residual_hiddens,
              num_embeddings=num_embeddings,
              embedding_dim=embedding_dim,
              commitment_cost=commitment_cost,
              decay=decay).to(device)

vqvae.load_state_dict(torch.load('./checkpoints/vqvae/vqvae_ema_100000.pt'))

# vqvae_optimizer
vqvae_optimizer = torch.optim.Adam(vqvae.parameters(), lr=learning_rate, amsgrad=False)

# loss function
def loss_function(recon_x, x):
    recon_error = F.mse_loss(recon_x, x)
    return recon_error

def train_vqvae():
    '''Train the vqvae.'''
    train_res_recon_error = []
    train_res_perplexity = []

    perceptual_loss_criterion = LPIPS(net='vgg').to(device)

    vqvae.train() # set the vqvae to training mode

    dirpath = Path(f'./runs/vqvae/ema/ffhq')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)

    writer = SummaryWriter(dirpath) # create a writer object for TensorBoard

    for i in range(num_training_updates):
        # sample the mini-batch
        x = next(iter(train_loader))
        x = x.to(device)

        vqvae_optimizer.zero_grad() # clear the gradients

        # forward pass
        vq_loss, data_recon, perplexity, quantized = vqvae(x)

        recon_error = l2_loss_factor * loss_function(data_recon, x) + lpips_loss_factor * perceptual_loss_criterion(data_recon, x).mean()

        loss = recon_error + vq_loss # total loss

        # backward pass
        loss.backward()

        # update parameters
        vqvae_optimizer.step()

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
        writer.add_scalar('Train VQ Loss', vq_loss.item(), i+1)
        writer.add_scalar('Train Perplexity', perplexity.item(), i+1)
        writer.add_scalar('Train Loss', loss.item(), i+1)

        # save training information for plotting
        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())

        # save the vqvae graph
        if i == 0:
            writer.add_graph(vqvae, x)

        # save the reconstructed images
        if (i + 1) % 100 == 0:
            writer.add_images('Train Original Images', originals, i+1)
            writer.add_images('Train Reconstructed Images', reconstructions, i+1)

        # save the codebook
        if (i + 1) % 100 == 0:
            writer.add_embedding(quantized.view(train_batch_size, -1), label_img=originals, global_step=i+1)

        # save the gradient visualization
        if (i + 1) % 100 == 0:
            for name, param in vqvae.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), i + 1)
                if param.grad is not None:
                    writer.add_histogram(name+'/grad', param.grad.clone().cpu().data.numpy(), i+1)

        # save the training information
        if (i + 1) % 100 == 0:
            np.save('train_res_recon_error.npy', train_res_recon_error)
            np.save('train_res_perplexity.npy', train_res_perplexity)

        # save the vqvae
        if (i + 1) % 1000 == 0:
            torch.save(vqvae.state_dict(), './checkpoints/vqvae/vqvae_ema_%d.pt' % (i + 1))

        # save the images
        if (i + 1) % 1000 == 0:
            torchvisionutils.save_image(originals, './results/vqvae/ema/originals_%d.png' % (i + 1))
            torchvisionutils.save_image(reconstructions, './results/vqvae/ema/reconstructions_%d.png' % (i + 1))

    writer.close()


if __name__ == '__main__':
    start = time.time()
    train_vqvae()
    end = time.time()
    print('Training time: %f seconds.' % (end-start))

