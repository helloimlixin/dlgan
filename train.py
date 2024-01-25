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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision import utils as torchvisionutils
from dataloaders.cifar10 import get_cifar10_train_loader, get_cifar10_test_loader
from models import discriminator
from models.discriminator import Discriminator
from models.lpips import LPIPS
from models.vqgan import VQGAN
from models.vqvae import VQVAE
import numpy as np
import time
import os

from utils import init_weights, load_data

# fix the bug of "OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized."
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# hyperparameters
train_batch_size = 256
test_batch_size = 32
num_training_updates = 100000

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512

commitment_cost = 0.25

decay = 0

learning_rate = 1e-4

epsilon = 1e-10 # a small number to avoid the numerical issues

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data loaders
train_loader, data_variance = get_cifar10_train_loader(batch_size=train_batch_size)()
test_loader = get_cifar10_test_loader(batch_size=test_batch_size)()

# vqvae
vqvae = VQVAE(in_channels=3,
              num_hiddens=num_hiddens,
              num_residual_layers=num_residual_layers,
              num_residual_hiddens=num_residual_hiddens,
              num_embeddings=num_embeddings,
              embedding_dim=embedding_dim,
              commitment_cost=commitment_cost,
              decay=decay).to(device)

# vqvae_optimizer
vqvae_optimizer = torch.optim.Adam(vqvae.parameters(), lr=learning_rate, amsgrad=False)

# loss function
def loss_function(recon_x, x):
    recon_error = F.mse_loss(recon_x, x) / data_variance
    return recon_error

def train_vqvae():
    '''Train the vqvae.'''
    train_res_recon_error = []
    train_res_perplexity = []

    vqvae.train() # set the vqvae to training mode
    writer = SummaryWriter('./runs/vanilla') # create a writer object for TensorBoard

    for i in range(num_training_updates):
        # sample the mini-batch
        (x,_) = next(iter(train_loader))
        x = x.to(device)

        vqvae_optimizer.zero_grad() # clear the gradients

        # forward pass
        vq_loss, data_recon, perplexity, quantized = vqvae(x)
        recon_error = loss_function(data_recon, x)

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
                writer.add_histogram(name, param.clone().cpu().data.numpy(), i+1)
                writer.add_histogram(name+'/grad', param.grad.clone().cpu().data.numpy(), i+1)

        # save the training information
        if (i + 1) % 100 == 0:
            np.save('train_res_recon_error.npy', train_res_recon_error)
            np.save('train_res_perplexity.npy', train_res_perplexity)

        # save the vqvae
        if (i + 1) % 1000 == 0:
            torch.save(vqvae.state_dict(), './checkpoints/vanilla/vqvae_%d.pt' % (i + 1))

    writer.close()

def configure_optimizers(opt_args):
    optimizer_quantization = torch.optim.Adam(
        params=list(vqgan._encoder.parameters())
        + list(vqgan._decoder.parameters())
        + list(vqgan._codebook.parameters())
        + list(vqgan._prequant_conv.parameters())
        + list(vqgan._postquant_conv.parameters()),
        lr=opt_args.learning_rate,
        eps=1e-8,
        betas=(opt_args.beta1, opt_args.beta2),
    )
    optimizer_discriminator = torch.optim.Adam(
        params=discriminator.parameters(),
        lr=opt_args.learning_rate,
        eps=1e-8,
        betas=(opt_args.beta1, opt_args.beta2),
    )
    return optimizer_quantization, optimizer_discriminator


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--embedding-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--num-embeddings', type=int, default=1024,
                        help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--num-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='data', help='Path to data (default: data)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=8, help='Input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=1e-5, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=400000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=1., help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1.,
                        help='Weighting factor for perceptual loss.')
    start = time.time()
    # train_vqvae()

    args = parser.parse_args()
    args.dataset_path = './data/flowers'

    os.makedirs("vqgan_results", exist_ok=True)
    os.makedirs("vqgan_ckpts", exist_ok=True)

    vqgan = VQGAN(args)
    vqgan.load_checkpoint('./vqgan_ckpts/vqgan_20.pt')

    vqgan.eval()
    vqgan.to(device=args.device)

    discriminator = Discriminator(args).to(device=args.device)
    discriminator.apply(init_weights)
    perceptual_loss = LPIPS().eval().to(device=args.device)

    opt_vq, opt_disc = configure_optimizers(args)

    train_dataloader = load_data(args)

    steps_per_epoch = len(train_loader)

    global_step = 0

    for epoch in range(args.epochs):
        with tqdm(range(len(train_dataloader))) as progress_bar:
            for i, imgs in zip(progress_bar, train_dataloader):
                imgs = imgs.to(device=args.device)

                decoded_imgs, vq_loss, z = vqgan(imgs)

                disc_real = discriminator(imgs)
                disc_fake = discriminator(decoded_imgs)

                disc_factor = vqgan.adopt_weight(
                    args.disc_factor,
                    global_step,
                    threshold=args.disc_start,
                )

                lpips_loss = perceptual_loss(imgs, decoded_imgs)  # LPIPS(real, fake)
                reconstruction_loss = torch.abs(imgs - decoded_imgs)  # L1(real, fake)
                perceptual_reconstruction_loss = (
                        args.perceptual_loss_factor * lpips_loss
                        + args.rec_loss_factor * reconstruction_loss
                )
                perceptual_reconstruction_loss = (
                    perceptual_reconstruction_loss.mean()
                )  # mean over batch
                generator_loss = - torch.mean(disc_fake)

                # compute lambda
                lamda = vqgan.calculate_lambda(perceptual_reconstruction_loss, generator_loss)

                # compute vector quantization loss
                vq_loss = perceptual_reconstruction_loss + vq_loss + disc_factor * lamda * generator_loss

                disc_loss_real = torch.mean(F.relu(1. - disc_real))
                disc_loss_fake = torch.mean(F.relu(1. + disc_fake))
                gan_loss = disc_factor * 0.5 * (disc_loss_real + disc_loss_fake)

                opt_vq.zero_grad()
                vq_loss.backward(retain_graph=True)

                opt_disc.zero_grad()
                gan_loss.backward()

                opt_vq.step()
                opt_disc.step()

                if i % 1000 == 0:
                    with torch.no_grad():
                        real_fake_images = torch.cat((imgs[:4], decoded_imgs.add(1).mul(0.5)[:4]))
                        torchvisionutils.save_image(real_fake_images, os.path.join("vqgan_results", f"{epoch}_{i}.png"),
                                                    nrow=4)

                progress_bar.set_postfix(
                    Perceptual_Loss=np.round(perceptual_reconstruction_loss.cpu().detach().numpy().item(), 5),
                    VQ_Loss=np.round(vq_loss.cpu().detach().numpy().item(), 5),
                    GAN_Loss=np.round(gan_loss.cpu().detach().numpy().item(), 3)
                )
                progress_bar.update(0)

                global_step += 1

            torch.save(vqgan.state_dict(), os.path.join("vqgan_ckpts", f"vqgan_{epoch}.pt"))
    end = time.time()
    print('Training time: %f seconds.' % (end-start))

