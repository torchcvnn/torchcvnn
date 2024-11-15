# coding: utf-8
# MIT License

# Copyright (c) 2023 Jeremy Fix

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

# Standard imports
import os
from typing import Tuple

# External imports
import torch
import torch.nn as nn
import tqdm

# import torch.onnx


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    f_loss: nn.Module,
    optim: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Run the training loop for nsteps minibatches of the dataloader

    Arguments:
        model: the model to train
        loader: an iterable dataloader
        f_loss (nn.Module): the loss
        optim : an optimizing algorithm
        device: the device on which to run the code

    Returns:
        The averaged training loss
        The averaged training accuracy
    """
    model.train()

    loss_avg = 0
    acc_avg = 0
    num_samples = 0
    for inputs, outputs in tqdm.tqdm(loader):
        inputs = inputs.to(device)
        outputs = outputs.to(device)

        # Forward propagate through the model
        pred_outputs = model(inputs)

        # Forward propagate through the loss
        loss = f_loss(pred_outputs, outputs)

        # Backward pass and update
        optim.zero_grad()
        loss.backward()
        optim.step()

        num_samples += inputs.shape[0]

        # Denormalize the loss that is supposed to be averaged over the
        # minibatch
        loss_avg += inputs.shape[0] * loss.item()
        pred_cls = pred_outputs.argmax(dim=-1)
        acc_avg += (pred_cls == outputs).sum().item()

    return loss_avg / num_samples, acc_avg / num_samples


def test_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    f_loss: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Run the test loop for n_test_batches minibatches of the dataloader

    Arguments:
        model: the model to evaluate
        loader: an iterable dataloader
        f_loss: the loss
        device: the device on which to run the code

    Returns:
        The averaged test loss
        The averaged test accuracy

    """
    model.eval()

    loss_avg = 0
    acc_avg = 0
    num_samples = 0
    with torch.no_grad():
        for inputs, outputs in loader:
            inputs = inputs.to(device)
            outputs = outputs.to(device)

            # Forward propagate through the model
            pred_outputs = model(inputs)

            # Forward propagate through the loss
            loss = f_loss(pred_outputs, outputs)

            loss_avg += inputs.shape[0] * loss.item()
            pred_cls = pred_outputs.argmax(dim=-1)
            acc_avg += (pred_cls == outputs).sum().item()
            num_samples += inputs.shape[0]

    return loss_avg / num_samples, acc_avg / num_samples


class ModelCheckpoint(object):
    def __init__(
        self,
        model: torch.nn.Module,
        savepath: str,
        num_input_dims: int,
        min_is_best: bool = True,
    ) -> None:
        """
        Early stopping callback

        Arguments:
            model: the model to save
            savepath: the location where to save the model's parameters
            num_input_dims: the number of dimensions for the input tensor (required for onnx export)
            min_is_best: whether the min metric or the max metric as the best
        """
        self.model = model
        self.savepath = savepath
        self.num_input_dims = num_input_dims
        self.best_score = None
        if min_is_best:
            self.is_better = self.lower_is_better
        else:
            self.is_better = self.higher_is_better

    def lower_is_better(self, score: float) -> bool:
        """
        Test if the provided score is lower than the best score found so far

        Arguments:
            score: the score to test

        Returns:
            res : is the provided score lower than the best score so far ?
        """
        return self.best_score is None or score < self.best_score

    def higher_is_better(self, score: float) -> bool:
        """
        Test if the provided score is higher than the best score found so far

        Arguments:
            score: the score to test

        Returns:
            res : is the provided score higher than the best score so far ?
        """
        return self.best_score is None or score > self.best_score

    def update(self, score: float) -> bool:
        """
        If the provided score is better than the best score registered so far,
        saves the model's parameters on disk as a pytorch tensor

        Arguments:
            score: the new score to consider

        Returns:
            res: whether or not the provided score is better than the best score
                 registered so far
        """
        if self.is_better(score):
            self.model.eval()

            torch.save(
                self.model.state_dict(), os.path.join(self.savepath, "best_model.pt")
            )

            # torch.onnx.export(
            #     self.model,
            #     dummy_input,
            #     os.path.join(self.savepath, "best_model.onnx"),
            #     verbose=False,
            #     input_names=["input"],
            #     output_names=["output"],
            #     dynamic_axes={
            #         "input": {0: "batch"},
            #         "output": {0: "batch"},
            #     },
            # )

            self.best_score = score
            return True
        return False


def generate_unique_logpath(logdir: str, raw_run_name: str) -> str:
    """
    Generate a unique directory name and create it if necessary

    Arguments:
        logdir: the prefix directory
        raw_run_name: the base name

    Returns:
        log_path: a non-existent path like logdir/raw_run_name_xxxx
                  where xxxx is an int
    """
    i = 0
    while True:
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            os.makedirs(log_path)
            return log_path
        i = i + 1
