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
# Example using complex valued neural networks to classify MNIST from the Fourier Transform of the digits.



Requires dependencies :
    python3 -m pip install torchvision tqdm
"""

# Standard imports
import random
import sys
from typing import List

# External imports
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as v2_transforms

import torchcvnn.nn as c_nn

# Local imports
import utils


def conv_block(in_c: int, out_c: int, cdtype: torch.dtype) -> List[nn.Module]:
    """
    Builds a basic building block of
    `Conv2d`-`Cardioid`-`Conv2d`-`Cardioid`-`AvgPool2d`

    Arguments:
        in_c : the number of input channels
        out_c : the number of output channels
        cdtype : the dtype of complex values (expected to be torch.complex64 or torch.complex32)
    """
    return [
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, dtype=cdtype),
        c_nn.BatchNorm2d(out_c),
        c_nn.Cardioid(),
        nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, dtype=cdtype),
        c_nn.BatchNorm2d(out_c),
        c_nn.Cardioid(),
        c_nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
    ]


def train():
    """
    Train function

    Sample output :
        ```.bash
        (venv) me@host:~$ python mnist.py
        Logging to ./logs/CMNIST_0
        >> Training
        100%|██████| 844/844 [00:17<00:00, 48.61it/s]
        >> Testing
        [Step 0] Train : CE  0.20 Acc  0.94 | Valid : CE  0.08 Acc  0.97 | Test : CE 0.06 Acc  0.98[>> BETTER <<]

        >> Training
        100%|██████| 844/844 [00:16<00:00, 51.69it/s]
        >> Testing
        [Step 1] Train : CE  0.06 Acc  0.98 | Valid : CE  0.06 Acc  0.98 | Test : CE 0.05 Acc  0.98[>> BETTER <<]

        >> Training
        100%|██████| 844/844 [00:15<00:00, 53.47it/s]
        >> Testing
        [Step 2] Train : CE  0.04 Acc  0.99 | Valid : CE  0.04 Acc  0.99 | Test : CE 0.04 Acc  0.99[>> BETTER <<]

        [...]
        ```

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid_ratio = 0.1
    batch_size = 64
    epochs = 10
    cdtype = torch.complex64

    # Dataloading
    train_valid_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=v2_transforms.Compose([v2_transforms.PILToTensor(), torch.fft.fft]),
    )
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
    conv_model = nn.Sequential(
        *conv_block(1, 16, cdtype),
        *conv_block(16, 16, cdtype),
        *conv_block(16, 32, cdtype),
        *conv_block(32, 32, cdtype),
        nn.Flatten(),
    )

    with torch.no_grad():
        conv_model.eval()
        dummy_input = torch.zeros((64, 1, 28, 28), dtype=cdtype, requires_grad=False)
        out_conv = conv_model(dummy_input).view(64, -1)
    lin_model = nn.Sequential(
        nn.Linear(out_conv.shape[-1], 124, dtype=cdtype),
        c_nn.Cardioid(),
        nn.Linear(124, 10, dtype=cdtype),
        c_nn.Mod(),
    )
    model = nn.Sequential(conv_model, lin_model)
    model.to(device)

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
