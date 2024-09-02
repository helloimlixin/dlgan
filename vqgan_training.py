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
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from dataloaders.flowers import FlowersDataset
from dataloaders.ffhq import FFHQDataset
from flip.pytorch.flip_loss import LDRFLIPLoss
from models.lpips import LPIPS
from skimage.metrics import structural_similarity as ssim
from models.vqgan import VQGAN
import numpy as np
import time
import os
from pathlib import Path
import shutil

import sys

# fix the bug of "OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized."
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# hyperparameters
train_batch_size = 8
test_batch_size = 4
num_epochs = 10

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512

commitment_cost = 0.25

decay = 0.

model_tag = 'vanilla'

if decay > 0.:
    model_tag = 'ema'

learning_rate = 1e-4

lr_schedule = [200000]

epsilon = 1e-10 # a small number to avoid the numerical issues

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

l2_loss_factor = 0.5
lpips_loss_factor = 1 - l2_loss_factor

discriminator_factor = 0.01
disc_start = 80000000000

validation_on = False

validation_interval = 1000 if validation_on else sys.maxsize

load_pretrained = False

# data_paths loaders
# train_loader, data_variance = get_cifar10_train_loader(batch_size=train_batch_size)()

# flowers_dataset = FlowersDataset(root='./data/flowers')
# train_loader = DataLoader(flowers_dataset, batch_size=train_batch_size, shuffle=True)
# test_loader = get_cifar10_test_loader(batch_size=test_batch_size)()

ffhq_dataset = FFHQDataset(root='./data/ffhq')

# train, val, test split
train_size = int(0.999 * len(ffhq_dataset))
val_size = int(0.0008 * len(ffhq_dataset))
test_size = len(ffhq_dataset) - train_size - val_size
ffhq_dataset_train, ffhq_dataset_val, ffhq_dataset_test = torch.utils.data.random_split(ffhq_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(ffhq_dataset_train,
                          batch_size=train_batch_size,
                          shuffle=True,
                          pin_memory=False,
                          num_workers=0)

val_loader = DataLoader(ffhq_dataset_val,
                        batch_size=test_batch_size,
                        shuffle=False,
                        pin_memory=True,
                        num_workers=0)

test_loader = DataLoader(ffhq_dataset_test,
                            batch_size=test_batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=0)

# vqgan
vqgan = VQGAN(in_channels=3,
              num_hiddens=num_hiddens,
              num_residual_layers=num_residual_layers,
              num_residual_hiddens=num_residual_hiddens,
              num_embeddings=num_embeddings,
              embedding_dim=embedding_dim,
              commitment_cost=commitment_cost,
              decay=decay).to(device)

global global_step
global_step = 0
if load_pretrained:
    checkpoint = torch.load(f'./checkpoints/vqgan-{model_tag}/epoch_2.pt')
    vqgan.load_state_dict(checkpoint['model'])
    global_step = checkpoint['global_step']

# vqgam_optimizer
opt_vae = torch.optim.Adam(list(vqgan._encoder.parameters()) +
                           list(vqgan._decoder.parameters()) +
                           list(vqgan._vq_bottleneck.parameters()) +
                           list(vqgan._pre_vq_conv.parameters()), lr=learning_rate, amsgrad=False)
opt_disc = torch.optim.Adam(vqgan._discriminator.parameters(), lr=1e-4, amsgrad=False)

scheduler = torch.optim.lr_scheduler.MultiStepLR(opt_vae, milestones=lr_schedule, gamma=0.1)

# loss function
def loss_function(recon_x, x):
    recon_error = F.mse_loss(recon_x, x)
    return recon_error

def train_vqgan(global_step=0):
    '''Train the vqgan.'''
    train_res_recon_error = []
    train_res_vq_loss = []
    train_res_recon_psnr = []
    train_res_recon_ssim = []
    train_res_recon_flip = []
    train_res_recon_lpips = []
    train_res_perplexity = []

    perceptual_loss_criterion = LPIPS().to(device)
    flip_loss_criterion = LDRFLIPLoss().to(device)

    vqgan.train() # set the vqgan to training mode

    # set up tensorboard directory
    dirpath = Path(f'./runs/vqgan-{model_tag}')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)

    writer = SummaryWriter(dirpath) # create a writer object for TensorBoard

    for epoch in range(num_epochs):
        with tqdm(range(len(train_loader)), colour='green') as pbar:
            for i, x in zip(pbar, train_loader):
                global_step = epoch * len(train_loader) + i + 1

                # sample the mini-batch
                x = x.to(device)

                # forward pass
                vq_loss, x_recon, perplexity, quantized = vqgan(x)
                perceptual_loss = perceptual_loss_criterion(x_recon, x).mean()

                recon_error = l2_loss_factor * loss_function(x_recon, x) + lpips_loss_factor * perceptual_loss

                flip_loss = flip_loss_criterion(x_recon, x)

                disc_real = vqgan._discriminator(x)
                disc_fake = vqgan._discriminator(x_recon)

                g_loss = -torch.mean(disc_fake)

                disc_factor = vqgan.adopt_weight(discriminator_factor, global_step, threshold=disc_start)

                d_loss_real = torch.mean(F.relu(1. - disc_real))
                d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)

                lambda_factor = vqgan.calculate_lambda(perceptual_loss, gan_loss)

                loss = recon_error + vq_loss + flip_loss + disc_factor * lambda_factor * g_loss # total loss

                opt_vae.zero_grad()  # clear the gradients
                loss.backward(retain_graph=True)  # compute the gradients

                opt_disc.zero_grad()  # clear the gradients
                gan_loss.backward()  # compute the gradients

                opt_vae.step()  # update the parameters
                opt_disc.step()  # update the parameters
                scheduler.step()

                originals = x + 0.5  # add 0.5 to match the range of the original images [0, 1]
                # low_res = x_lr + 0.5 # add 0.5 to match the range of the original images [0, 1]
                # inputs = x_hr + 0.5 # add 0.5 to match the range of the original images [0, 1]
                reconstructions = x_recon + 0.5  # add 0.5 to match the range of the original images [0, 1]

                psnr = 10 * torch.log10(1 / loss_function(x_recon, x))

                x_np = x.cpu().detach().numpy()
                reconstructions_np = reconstructions.cpu().detach().numpy()

                ssim_val = torch.mean(torch.Tensor([ssim(x_np[i], reconstructions_np[i],
                                                         data_range=1,
                                                         size_average=True,
                                                         channel_axis=0) for i in range(x_np.shape[0])]))

                # save training information for TensorBoard
                writer.add_scalar('Train Recon Error', recon_error.item(), global_step)
                writer.add_scalar('Train PSNR', psnr.item(), global_step)
                writer.add_scalar('Train LPIPS', perceptual_loss.item(), global_step)
                writer.add_scalar('Train SSIM', ssim_val.item(), global_step)
                writer.add_scalar('Train FLIP', flip_loss.item(), global_step)
                writer.add_scalar('Train Vector Quantization Loss', vq_loss.item(), global_step)
                writer.add_scalar('Train Perplexity', perplexity.item(), global_step)
                writer.add_scalar('Train Loss', loss.item(), global_step)
                writer.add_scalar('Train GAN Loss', gan_loss.item(), global_step)

                # save training information for plotting
                train_res_recon_error.append(recon_error.item())
                train_res_vq_loss.append(vq_loss.item())
                train_res_recon_psnr.append(psnr.item())
                train_res_recon_ssim.append(ssim_val.item())
                train_res_recon_lpips.append(perceptual_loss.item())
                train_res_recon_flip.append(flip_loss.item())
                train_res_perplexity.append(perplexity.item())

                # save the reconstructed images
                if global_step % 1000 == 0:
                    # writer.add_images('Train Low Resolution Images', low_res, i+1)
                    writer.add_images('Train Target Images', originals, global_step)
                    # writer.add_images('Train Input Images', inputs, i+1)
                    writer.add_images('Train Reconstructed Images', reconstructions, global_step)

                # save the codebook
                if global_step % 1000 == 0:
                    writer.add_embedding(quantized.view(train_batch_size, -1),
                                         label_img=originals,
                                         global_step=global_step)

                # save the gradient visualization
                if global_step % 1000 == 0:
                    for name, param in vqgan.named_parameters():
                        writer.add_histogram(name, param.clone().cpu().data.numpy(), global_step)
                        if param.grad is not None:
                            writer.add_histogram(name + '/grad', param.grad.clone().cpu().data.numpy(), global_step)

                # save the training information
                if global_step % 10000 == 0:
                    np.save('train_res_recon_error.npy', train_res_recon_error)
                    np.save('train_res_perplexity.npy', train_res_perplexity)

                # save the images
                # create the results directory if it does not exist
                if not os.path.exists('./results/vqgan-{model_tag}'):
                    os.makedirs('./results/vqgan-{model_tag}')

                if global_step % 1000 == 0:
                    torchvisionutils.save_image(originals, f'./results/vqgan-{model_tag}/target_{global_step}.png')
                    torchvisionutils.save_image(reconstructions,
                                                f'./results/vqgan-{model_tag}/reconstruction_{global_step}.png')

                # perform the validation
                if global_step % validation_interval == 0:
                    vqgan.eval()
                    with torch.no_grad():
                        val_res_recon_error = []
                        val_res_recon_psnr = []
                        val_res_recon_ssim = []
                        val_res_recon_flip = []
                        val_res_recon_lpips = []
                        val_res_perplexity = []

                        x_val = next(iter(val_loader))
                        x_val = x_val.to(device)

                        # forward pass
                        dl_loss_val, data_recon_val, perplexity_val, quantized_val = vqgan(x_val)
                        perceptual_loss_val = perceptual_loss_criterion(data_recon_val, x_val).mean()

                        recon_error_val = l2_loss_factor * loss_function(data_recon_val,
                                                                         x_val) + lpips_loss_factor * perceptual_loss_val

                        # compute the NVIDIA FLIP metric
                        flip_val = flip_loss_criterion(data_recon_val, x_val)

                        originals_val = x_val + 0.5
                        reconstructions_val = data_recon_val + 0.5

                        psnr_val = 10 * torch.log10(1 / loss_function(data_recon_val, x_val))

                        x_val_np = x_val.cpu().detach().numpy()
                        reconstructions_val_np = reconstructions_val.cpu().detach().numpy()

                        ssim_val_val = torch.mean(torch.Tensor([ssim(x_val_np[i], reconstructions_val_np[i],
                                                                     data_range=1,
                                                                     size_average=True,
                                                                     channel_axis=0) for i in
                                                                range(x_val_np.shape[0])]))

                        # save validation information for plotting
                        val_res_recon_error.append(recon_error_val.item())
                        val_res_recon_psnr.append(psnr_val.item())
                        val_res_recon_ssim.append(ssim_val_val.item())
                        val_res_recon_lpips.append(perceptual_loss_val.item())
                        val_res_recon_flip.append(flip_val.item())
                        val_res_perplexity.append(perplexity_val.item())

                        # save the reconstructed images
                        writer.add_images('Val Target Images', originals_val, global_step)
                        writer.add_images('Val Reconstructed Images', reconstructions_val, global_step)

                        # save the real and fake images
                        real_fake_images = torch.cat((originals_val, reconstructions_val))
                        writer.add_images('Val Real and Fake Images', real_fake_images, global_step)

                        # save the validation information
                        writer.add_scalar('Val Recon Error', recon_error_val.item(), global_step)
                        writer.add_scalar('Val PSNR', psnr_val.item(), global_step)
                        writer.add_scalar('Val LPIPS', perceptual_loss_val.item(), global_step)
                        writer.add_scalar('Val SSIM', ssim_val_val.item(), global_step)
                        writer.add_scalar('Val FLIP', flip_val.item(), global_step)
                        writer.add_scalar('Val Vector Quantization Loss', dl_loss_val.item(), global_step)
                        writer.add_scalar('Val Perplexity', perplexity_val.item(), global_step)

                        # save the images
                        torchvisionutils.save_image(originals_val,
                                                    f'./results/vqgan-{model_tag}/val_target_{global_step}.png')
                        torchvisionutils.save_image(reconstructions_val,
                                                    f'./results/vqgan-{model_tag}/val_reconstruction_{global_step}.png')

                    vqgan.train()

                pbar.set_description(f'Epoch {epoch + 1} / {num_epochs}: ')

                pbar.set_postfix(
                    PSNR=np.mean(train_res_recon_psnr[-100:]),
                    VQ_Loss=np.mean(train_res_vq_loss[-100:]),
                    LPIPS=np.mean(train_res_recon_lpips[-100:]),
                    SSIM=np.mean(train_res_recon_ssim[-100:]),
                    FLIP=np.mean(train_res_recon_flip[-100:]),
                    Perplexity=np.mean(train_res_perplexity[-100:]),
                    global_step=global_step
                )
                pbar.update(0)

            # save the model
            # create the checkpoints directory if it does not exist
            if not os.path.exists('./checkpoints/vqgan-{model_tag}'):
                os.makedirs('./checkpoints/vqgan-{model_tag}')

            torch.save({"model": vqgan.state_dict(),
                        "global_step": global_step}, f'./checkpoints/vqgan-{model_tag}/epoch_{(epoch + 1)}.pt')

    writer.close()


if __name__ == '__main__':
    start = time.time()
    print(f'Training the VQ-GAN from global step {global_step}...')
    train_vqgan(global_step=global_step)
    end = time.time()
    print('Training time: %f seconds.' % (end-start))

