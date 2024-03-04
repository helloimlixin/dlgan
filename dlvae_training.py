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
from dataloaders.ffhq import FFHQDataset

from models.dlvae import DLVAE
from skimage.metrics import structural_similarity as ssim
import numpy as np
import time
import os
from pathlib import Path
import shutil

# fix the bug of "OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized."
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# hyperparameters
train_batch_size = 4
test_batch_size = 4
num_training_updates = 200000

num_hiddens = 128
num_residual_hiddens = 4
num_residual_layers = 2

embedding_dim = 16
num_embeddings = 64

commitment_cost = 0.25

decay = 0.99

learning_rate = 1e-4

lr_schedule = [200000]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

l2_loss_factor = 0.5
lpips_loss_factor = 1 - l2_loss_factor

sparsity_level = 5 # number of atoms selected

epsilon = 1e-10 # a small number to avoid the numerical issues

# data_paths loaders
# flowers_dataset = FlowersDataset(root='./data/flowers')
# train_loader = DataLoader(flowers_dataset, batch_size=train_batch_size, shuffle=True)
# train_loader, data_variance = get_cifar10_train_loader(batch_size=train_batch_size)()
ffhq_dataset = FFHQDataset(root='./data/ffhq')
train_loader = DataLoader(ffhq_dataset,
                          batch_size=train_batch_size,
                          shuffle=True,
                          pin_memory=True,
                          num_workers=0)

# dlvae
dlvae = DLVAE(in_channels=3,
            num_hiddens=num_hiddens,
            num_residual_hiddens=num_residual_hiddens,
            num_residual_layers=num_residual_layers,
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            commitment_cost=commitment_cost,
            sparsity_level=sparsity_level,
            decay=decay,
            epsilon=epsilon).to(device)

dlvae.load_state_dict(torch.load(f'./checkpoints/dlvae/vanilla/sparsity-{sparsity_level}/ffhq/iter_200000.pt'))
# dlvae.eval()

# dlvae_optimizer
# optimizer = torch.optim.SparseAdam(dlvae.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(dlvae.parameters(), lr=learning_rate, amsgrad=False)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_schedule, gamma=0.1)
opt_vae = torch.optim.Adam(list(dlvae._encoder.parameters()) +
                           list(dlvae._decoder.parameters()) +
                           list(dlvae._pre_vq_conv.parameters()), lr=learning_rate, amsgrad=False)
opt_dl = torch.optim.Adam(list(dlvae._dl_bottleneck.parameters()), lr=learning_rate, amsgrad=False)

# loss function
def loss_function(recon_x, x):
    recon_error = F.mse_loss(recon_x, x)
    return recon_error

def train_dlvae():
    from lpips import LPIPS
    '''Train the vqvae.'''
    train_res_recon_error = []
    train_res_recon_psnr = []
    train_res_recon_ssim = []
    train_res_recon_lpips = []
    train_res_perplexity = []

    perceptual_loss_criterion = LPIPS(net='vgg').to(device)

    dlvae.train() # set the vqvae to training mode
    dirpath = Path(f'./runs/dlvae/vanilla/sparsity-{sparsity_level}/ffhq')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)

    writer = SummaryWriter(dirpath) # create a writer object for TensorBoard

    start = time.time()

    for i in range(num_training_updates):
        # sample the mini-batch
        x = next(iter(train_loader))
        x = x.to(device)

        # for cifar10 loader
        # (x, _) = next(iter(train_loader))
        # x = x.to(device)

        # opt_vae.zero_grad() # clear the gradients
        # opt_dl.zero_grad() # clear the gradients
        optimizer.zero_grad()  # clear the gradients

        # forward pass
        # x_lr = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)
        # x_hr = F.interpolate(x_lr, scale_factor=4, mode='bilinear', align_corners=False)
        dl_loss, data_recon, perplexity, representation = dlvae(x)

        recon_error = l2_loss_factor * loss_function(data_recon, x) + lpips_loss_factor * perceptual_loss_criterion(data_recon, x).mean()

        loss = recon_error + dl_loss # total loss

        loss.backward()

        optimizer.step() # update the parameters
        scheduler.step()

        # print training information
        if (i + 1) % 100 == 0:
            print('%d iterations ' % (i + 1), end='')
            print('%.2f iterations/s ' % ((i + 1) / (time.time() - start)), end='')
            print('ETA %.2f seconds' % ((num_training_updates - (i + 1)) / ((i + 1) / (time.time() - start))))
            print('recon_error: %.3f | ' % np.mean(train_res_recon_error[-100:]), end='')
            print('psnr: %.3f | ' % np.mean(train_res_recon_psnr[-100:]), end='')
            print('lpips: %.3f | ' % np.mean(train_res_recon_lpips[-100:]), end='')
            print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
            print()

        originals = x + 0.5 # add 0.5 to match the range of the original images [0, 1]
        # low_res = x_lr + 0.5 # add 0.5 to match the range of the original images [0, 1]
        # inputs = x_hr + 0.5 # add 0.5 to match the range of the original images [0, 1]
        reconstructions = data_recon + 0.5 # add 0.5 to match the range of the original images [0, 1]

        psnr = 10 * torch.log10(1 / loss_function(data_recon, x))
        lpips = perceptual_loss_criterion(data_recon, x).mean()

        # save training information for TensorBoard
        writer.add_scalar('Train Recon Error', recon_error.item(), i+1)
        writer.add_scalar('Train PSNR', psnr.item(), i+1)
        writer.add_scalar('Train LPIPS', lpips.item(), i+1)
        writer.add_scalar('Train Dictionary Learning Loss', dl_loss.item(), i+1)
        writer.add_scalar('Train Perplexity', perplexity.item(), i+1)
        writer.add_scalar('Train Loss', loss.item(), i+1)

        # save training information for plotting
        train_res_recon_error.append(recon_error.item())
        train_res_recon_psnr.append(psnr.item())
        # train_res_recon_ssim.append(torch.mean(torch.Tensor([torch.Tensor(ssim(x[i], reconstructions[i], data_range=1, size_average=True)) for i in range(x.size(0))])).item())
        train_res_recon_lpips.append(lpips.item())
        train_res_perplexity.append(perplexity.item())

        # save the reconstructed images
        if (i + 1) % 100 == 0:
            # writer.add_images('Train Low Resolution Images', low_res, i+1)
            writer.add_images('Train Target Images', originals, i+1)
            # writer.add_images('Train Input Images', inputs, i+1)
            writer.add_images('Train Reconstructed Images', reconstructions, i+1)

        # save the codebook
        if (i + 1) % 100 == 0:
            writer.add_embedding(representation.view(train_batch_size, -1), label_img=originals, global_step=i+1)

        # save the gradient visualization
        if (i + 1) % 100 == 0:
            for name, param in dlvae.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), i+1)
                if param.grad is not None:
                    writer.add_histogram(name+'/grad', param.grad.clone().cpu().data.numpy(), i+1)

        # save the training information
        if (i + 1) % 100 == 0:
            np.save('train_res_recon_error.npy', train_res_recon_error)
            np.save('train_res_perplexity.npy', train_res_perplexity)

        # save the dlvae
        if (i + 1) % 1000 == 0:
            torch.save(dlvae.state_dict(), f'./checkpoints/dlvae/vanilla/sparsity-{sparsity_level}/ffhq/iter_{(i + 1)}.pt')

        # save the images
        if (i + 1) % 100 == 0:
            # torchvisionutils.save_image(low_res, f'./dlvae_results/ffhq/sr/ema-{sparsity_level}/low_res_{(i + 1)}.png')
            # torchvisionutils.save_image(inputs, f'./dlvae_results/ffhq/sr/ema-{sparsity_level}/input_{(i + 1)}.png')
            torchvisionutils.save_image(originals, f'./results/dlvae/vanilla/sparsity-{sparsity_level}/ffhq/target_{(i + 1)}.png')
            torchvisionutils.save_image(reconstructions, f'./results/dlvae/vanilla/sparsity-{sparsity_level}/ffhq/reconstruction_{(i + 1)}.png')

    writer.close()


if __name__ == '__main__':
    start = time.time()
    train_dlvae()
    end = time.time()
    print('Training time: %f seconds.' % (end-start))

