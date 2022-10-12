
import torch
import torch.nn as nn

import MinkowskiEngine as Me
from .resnet_sparse import BasicBlock as SparseBasicBlock

class GeometryHeadSparse(nn.Module):
    def __init__(self, channel_in: int, channel_out: int, num_blocks: int) -> None:
        super().__init__()


        self.network = [
            Me.MinkowskiInstanceNorm(channel_in),
            Me.MinkowskiReLU()
        ]

        for _ in range(num_blocks):
            self.network.append(SparseBasicBlock(channel_in, channel_in, dimension=3))

        self.network.extend([
            Me.MinkowskiConvolution(channel_in, channel_out, kernel_size=3, stride=1, bias=True, dimension=3)
        ])

        self.network = nn.Sequential(*self.network)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.network(x)

        return output

class ClassificationHeadSparse(nn.Module):
    def __init__(self, channel_in: int, channel_out: int, num_blocks: int) -> None:
        super().__init__()

        self.network = [
            Me.MinkowskiInstanceNorm(channel_in),
            Me.MinkowskiReLU()
        ]

        for _ in range(num_blocks):
            self.network.append(SparseBasicBlock(channel_in, channel_in, dimension=3))

        self.network.extend([
            Me.MinkowskiConvolution(channel_in, channel_out, kernel_size=3, stride=1, bias=True, dimension=3)
        ])

        self.network = nn.Sequential(*self.network)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)