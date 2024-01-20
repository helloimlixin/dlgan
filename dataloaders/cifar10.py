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

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

class get_cifar10_train_loader():
    '''Get training data loader.'''

    def __init__(self, batch_size):
        self._batch_size = batch_size

    def __call__(self):
        dataset = datasets.CIFAR10(root='./data',
                                   train=True,
                                   download=False,
                                   transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5),
                                                             (1.0, 1.0, 1.0))
                                   ]))
        loader = DataLoader(dataset,
                            batch_size=self._batch_size,
                            shuffle=True,
                            num_workers=0,
                            pin_memory=True) # CUDA only, pin_memory=True enables faster data transfer to CUDA-enabled GPUs.

        # calculate data variance
        data_variance = np.var(dataset.data / 255.0)

        return loader, data_variance

class get_cifar10_test_loader():
    '''Get test data loader.'''

    def __init__(self, batch_size):
        self._batch_size = batch_size

    def __call__(self):
        dataset = datasets.CIFAR10(root='./data',
                                   train=False,
                                   download=False,
                                   transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5),
                                                             (1.0, 1.0, 1.0))
                                   ]))
        loader = DataLoader(dataset,
                            batch_size=self._batch_size,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True) # CUDA only, pin_memory=True enables faster data transfer to CUDA-enabled GPUs.

        return loader
