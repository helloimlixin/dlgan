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
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from dataloaders.cifar10 import get_cifar10_train_loader, get_cifar10_test_loader
from models.vqvae import VQVAE
import numpy as np
import time

# hyperparameters
train_batch_size = 256
test_batch_size = 32
num_training_updates = 50000

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512

commitment_cost = 0.25

decay = 0

learning_rate = 1e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data loaders
train_loader, data_variance = get_cifar10_train_loader(batch_size=train_batch_size)()
test_loader = get_cifar10_test_loader(batch_size=test_batch_size)()

# model
model = VQVAE(in_channels=3,
              num_hiddens=num_hiddens,
              num_residual_layers=num_residual_layers,
              num_residual_hiddens=num_residual_hiddens,
              num_embeddings=num_embeddings,
              embedding_dim=embedding_dim,
              commitment_cost=commitment_cost,
              decay=decay).to(device)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

# loss function
def loss_function(recon_x, x):
    recon_error = F.mse_loss(recon_x, x) / data_variance
    return recon_error

def train():
    '''Train the model.'''
    train_res_recon_error = []
    train_res_perplexity = []

    model.train() # set the model to training mode
    writer = SummaryWriter() # create a writer object for TensorBoard

    for i in range(num_training_updates):
        # sample the mini-batch
        (x,_) = next(iter(train_loader))
        x = x.to(device)

        optimizer.zero_grad() # clear the gradients

        # forward pass
        vq_loss, data_recon, perplexity, quantized = model(x)
        recon_error = loss_function(data_recon, x)
        loss = recon_error + vq_loss # total loss

        # backward pass
        loss.backward()

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
        writer.add_scalar('Train Perplexity', perplexity.item(), i+1)
        writer.add_scalar('Train Loss', loss.item(), i+1)

        # save training information for plotting
        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())

        # save the model graph
        if i == 0:
            writer.add_graph(model, x)

        # save the model
        if (i + 1) % 1000 == 0:
            torch.save(model.state_dict(), 'vqvae_%d.pt' % (i + 1))

        # save the reconstructed images
        if (i + 1) % 100 == 0:

            writer.add_images('Train Original Images', originals, i+1)
            writer.add_images('Train Reconstructed Images', reconstructions, i+1)

        # save the codebook
        if (i + 1) % 100 == 0:
            writer.add_embedding(quantized.view(train_batch_size, -1), label_img=originals, global_step=i+1)

        # save the training information
        if (i + 1) % 100 == 0:
            np.save('train_res_recon_error.npy', train_res_recon_error)
            np.save('train_res_perplexity.npy', train_res_perplexity)

        # save the model
        if (i + 1) % 1000 == 0:
            torch.save(model.state_dict(), 'vqvae_%d.pt' % (i + 1))

    writer.close()

if __name__ == '__main__':
    start = time.time()
    train()
    end = time.time()
    print('Training time: %f seconds.' % (end-start))

