# MIT License

# Copyright (c) 2023 Jérémy Fix

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
# Example using a complex valued Vision transformer to classify MNIST. 



Requires dependencies :
    python3 -m pip install torchvision tqdm
"""

# Standard imports
import random
import sys
from typing import List, Tuple, Union

# External imports
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as v2_transforms

import torchcvnn.nn as c_nn
import torchcvnn.models as c_models

import torchinfo

# Local imports
import utils
import vit_huy
import vit_tcvnn


class PseudoNorm(nn.Module):

    def __init__(self, dim, dtype=None, device=None):
        super(PseudoNorm, self).__init__()

    def forward(self, x):
        return x


def train():
    """
    Train function

    Sample output :
        ```.bash
        (venv) me@host:~$ python mnist.py
        Logging to ./logs/CMNIST_0
        >> Training
        100%|████| 844/844 [00:15<00:00, 53.13it/s]
        >> Testing
        [Step 0] Train : CE  1.21 Acc  0.64 | Valid : CE  0.57 Acc  0.84 | Test : CE  0.56 Acc  0.84[>> BETTER <<]
        >> Training
        100%|████| 844/844 [00:15<00:00, 53.64it/s]
        >> Testing
        [Step 1] Train : CE  0.39 Acc  0.89 | Valid : CE  0.31 Acc  0.91 | Test : CE  0.29 Acc  0.92[>> BETTER <<]
        >> Training
        100%|████| 844/844 [00:15<00:00, 54.35it/s]
        >> Testing
        [Step 2] Train : CE  0.24 Acc  0.93 | Valid : CE  0.20 Acc  0.94 | Test : CE  0.21 Acc  0.94[>> BETTER <<]
        [...]
        ```

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid_ratio = 0.1
    batch_size = 512
    epochs = 10
    cdtype = torch.complex64

    # Dataloading
    train_valid_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=v2_transforms.Compose([v2_transforms.PILToTensor(), torch.fft.fft]),
    )
    train_valid_dataset = torch.utils.data.Subset(train_valid_dataset, indices=np.arange(5000))

    test_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=v2_transforms.Compose([v2_transforms.PILToTensor(), torch.fft.fft]),
    )

    all_indices = list(range(len(train_valid_dataset)))
    random.shuffle(all_indices)
    split_idx = int(valid_ratio * len(train_valid_dataset))
    valid_indices, train_indices = all_indices[:split_idx], all_indices[split_idx:]

    # Train dataloader
    train_dataset = torch.utils.data.Subset(train_valid_dataset, train_indices)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # Valid dataloader
    valid_dataset = torch.utils.data.Subset(train_valid_dataset, valid_indices)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False
    )

    # Test dataloader
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    # Model
    num_classes = 10
    opt = {
        "patch_size": 4,
        "input_size": 28,
        "hidden_dim": 32,
        "num_layers": 3,
        "num_heads": 8,
        "num_channels": 1,
        "dropout": 0.3,
        "attention_dropout": 0.1,
        # "norm_layer": "rms_norm",
        "model_type": "vit"
    }

    ## Our implementation
    # model = vit_tcvnn.Model(opt, num_classes)

    ## Huy implementation
    model = vit_huy.VisionTransformer(opt, num_classes)

    model = nn.Sequential(
        model,
        c_nn.Mod(),
    )
    model = model.to(device)

    torchinfo.summary(model)


    # Loss, optimizer, callbacks
    f_loss = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=3e-4)
    logpath = utils.generate_unique_logpath("./logs", "CMNIST")
    print(f"Logging to {logpath}")
    checkpoint = utils.ModelCheckpoint(model, logpath, 4, min_is_best=True)

    # Training loop
    for e in range(epochs):
        print(">> Training")
        train_loss, train_acc = utils.train_epoch(
            model, train_loader, f_loss, optim, device
        )

        print(">> Testing")
        valid_loss, valid_acc = utils.test_epoch(model, valid_loader, f_loss, device)
        test_loss, test_acc = utils.test_epoch(model, test_loader, f_loss, device)
        updated = checkpoint.update(valid_loss)
        better_str = "[>> BETTER <<]" if updated else ""

        print(
            f"[Step {e}] Train : CE {train_loss:5.2f} Acc {train_acc:5.2f} | Valid : CE {valid_loss:5.2f} Acc {valid_acc:5.2f} | Test : CE {test_loss:5.2f} Acc {test_acc:5.2f}"
            + better_str
        )


if __name__ == "__main__":
    train()
