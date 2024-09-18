#  ==============================================================================
#  Description: Helper functions for the vqvae.
#  Codebase for 2025 WACV Submission
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

from dataloaders.celebahq import CelebA
from dataloaders.cifar10 import *
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
train_batch_size = 16
test_batch_size = 8
num_epochs = 100

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 16
num_embeddings = 512

commitment_cost = 0.25

decay = 0.99

model_tag = 'vanilla'

if decay > 0.:
    model_tag = 'emad-inpainting-ffhq'

learning_rate = 1e-4

lr_schedule = [200000]

epsilon = 1e-10 # a small number to avoid the numerical issues

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

l2_loss_factor = 0.5
lpips_loss_factor = 1 - l2_loss_factor

discriminator_factor = 0.01
disc_start = 100000000

validation_on = True

validation_interval = 1000 if validation_on else sys.maxsize

load_pretrained = False

ckpt = 0

ckpt_start = 8

if load_pretrained:
    ckpt = ckpt_start
else:
    ckpt = 0

log_interval = 100

# data_paths loaders
# train_loader, data_variance = get_cifar10_train_loader(batch_size=train_batch_size)()
# val_loader = get_cifar10_test_loader(batch_size=test_batch_size)()

# flowers_dataset = FlowersDataset(root='./data/flowers')
# train_loader = DataLoader(flowers_dataset, batch_size=train_batch_size, shuffle=True)

# define the training, validation, and test datasets
ffhq_dataset_train = FFHQDataset(root='./data/ffhq-512x512/train')
ffhq_dataset_val = FFHQDataset(root='./data/ffhq-512x512/val', size=128, crop_size=128)
ffhq_dataset_test = FFHQDataset(root='./data/ffhq-512x512/test', crop_size=512)

train_loader = DataLoader(ffhq_dataset_train,
                          batch_size=train_batch_size,
                          shuffle=False,
                          pin_memory=False,
                          drop_last=True,
                          num_workers=0)

val_loader = DataLoader(ffhq_dataset_val,
                        batch_size=test_batch_size,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=True,
                        num_workers=0)

test_loader = DataLoader(ffhq_dataset_test,
                         batch_size=test_batch_size,
                         shuffle=False,
                         pin_memory=True,
                         drop_last=True,
                         num_workers=0)

# celebahq_dataset_train = CelebA(root='./data/CelebAMask-HQ/CelebA-HQ-img/train', size=128, crop_size=128)
# celebahq_dataset_val = CelebA(root='./data/CelebAMask-HQ/CelebA-HQ-img/val', size=256, crop_size=256)
# celebahq_dataset_test = CelebA(root='./data/CelebAMask-HQ/CelebA-HQ-img/test', size=256, crop_size=256)

# train_loader = DataLoader(celebahq_dataset_train,
#                           batch_size=train_batch_size,
#                           shuffle=False,
#                           pin_memory=False,
#                           drop_last=True,
#                           num_workers=0)
#
# val_loader = DataLoader(celebahq_dataset_val,
#                         batch_size=test_batch_size,
#                         shuffle=False,
#                         pin_memory=True,
#                         drop_last=True,
#                         num_workers=0)
#
# test_loader = DataLoader(celebahq_dataset_test,
#                         batch_size=test_batch_size,
#                         shuffle=False,
#                         pin_memory=True,
#                         drop_last=True,
#                         num_workers=0)

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
    checkpoint = torch.load(f'./checkpoints/vqgan-{model_tag}/epoch_{ckpt}.pt')
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


def create_mask(x, mask_ratio):
    '''Create a bitmask that masks out center rectangular regions of the images.'''
    mask = torch.full_like(x, 1.0)
    mask_size = int(mask_ratio * x.size(-1))
    x1 = mask.size(-1) // 2 - mask_size // 2
    y1 = mask.size(-2) // 2 - mask_size // 2
    mask[:, :, x1:x1 + mask_size, y1:y1 + mask_size] = 0.0
    return mask


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

    for epoch in range(ckpt, num_epochs):
        with tqdm(range(len(train_loader)), colour='green') as pbar:
            for i, x in zip(pbar, train_loader):
                global_step = epoch * len(train_loader) + i + 1

                # sample the mini-batch
                # x = x.to(device)

                x = x.to(device)

                # create a bitmask that masks out center rectangular regions of the images
                x_masked = x * create_mask(x, mask_ratio=0.25)

                # forward pass
                vq_loss, x_recon, perplexity, quantized = vqgan(x_masked)
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
                masked = x_masked + 0.5  # add 0.5 to match the range of the original images [0, 1]
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
                if global_step % log_interval == 0:
                    # writer.add_images('Train Low Resolution Images', low_res, i+1)
                    writer.add_images('Train Target Images', originals, global_step)
                    writer.add_images('Train Masked Images', masked, global_step)
                    # writer.add_images('Train Input Images', inputs, i+1)
                    writer.add_images('Train Reconstructed Images', reconstructions, global_step)

                # save the codebook
                if global_step % log_interval == 0:
                    writer.add_embedding(vqgan._vq_bottleneck._embedding.weight.data,
                                         global_step=global_step)

                # save the gradient visualization
                if global_step % log_interval == 0:
                    for name, param in vqgan.named_parameters():
                        writer.add_histogram(name, param.clone().cpu().data.numpy(), global_step)
                        if param.grad is not None:
                            writer.add_histogram(name + '/grad', param.grad.clone().cpu().data.numpy(), global_step)

                # save the images
                # create the results directory if it does not exist
                if not os.path.exists(f'./results/vqgan-{model_tag}'):
                    os.makedirs(f'./results/vqgan-{model_tag}')

                if global_step % log_interval == 0:
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

                        # create a random mask
                        x_masked_val = x_val * create_mask(x_val, mask_ratio=0.25)

                        # forward pass
                        dl_loss_val, data_recon_val, perplexity_val, quantized_val = vqgan(x_masked_val)
                        perceptual_loss_val = perceptual_loss_criterion(data_recon_val, x_val).mean()

                        recon_error_val = l2_loss_factor * loss_function(data_recon_val,
                                                                         x_val) + lpips_loss_factor * perceptual_loss_val

                        # compute the NVIDIA FLIP metric
                        flip_val = flip_loss_criterion(data_recon_val, x_val)

                        originals_val = x_val + 0.5
                        masked_val = x_masked_val + 0.5
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
                        writer.add_images('Val Masked Images', masked_val, global_step)
                        writer.add_images('Val Reconstructed Images', reconstructions_val, global_step)

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
            if not os.path.exists(f'./checkpoints/vqgan-{model_tag}'):
                os.makedirs(f'./checkpoints/vqgan-{model_tag}')

            torch.save({"model": vqgan.state_dict(),
                        "global_step": global_step}, f'./checkpoints/vqgan-{model_tag}/epoch_{(epoch + 1)}.pt')

    writer.close()


if __name__ == '__main__':
    start = time.time()
    print(f'Training the VQ-GAN from global step {global_step}...')
    train_vqgan(global_step=global_step)
    end = time.time()
    print('Training time: %f seconds.' % (end-start))

