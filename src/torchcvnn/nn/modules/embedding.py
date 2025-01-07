# MIT License

# Copyright (c) 2025 Jeremy Fix

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

# External imports
import torch
import torch.nn as nn


class Embedding(nn.Module):
    """
    An Embedding layer is just a look up table of a fixed size dictionnary. The embeddings
    by themselves are trained.

    Args:
        num_embeddings (int): The number of embeddings
        embedding_dim (int): The dimension of each embedding
        device (torch.device): The device to use for the embeddings
        dtype (torch.dtype): The data type to use for the embeddings Default: torch.complex64
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device = None,
        dtype: torch.dtype = torch.complex64,
    ) -> None:
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), **factory_kwargs)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize the embeddings with a normal distribution
        """
        nn.init.normal_(self.weight)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the embedding layer, returning the embeddings for the given indices.

        The input indices must be in :math:`[0, \text{num\_embeddings} - 1]`.
        """
        return self.weight[idx]
