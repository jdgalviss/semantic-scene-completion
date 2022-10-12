import MinkowskiEngine as Me
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union,Tuple

from .resnet3d import ResNetBlock3d
from .resnet_sparse import BasicBlock as SparseBasicBlock
import utils

class BlockContent:
    def __init__(self, data: Union[torch.Tensor, Me.SparseTensor], encoding: Optional[torch.Tensor]):
        self.data = data
        self.encoding = encoding

class UNetSparse(nn.Module):
    def __init__(self, num_output_features: int, num_features: int = 64) -> None:
        super().__init__()
        block = UNetBlockInner(num_features * 8, num_features * 8)
        block = UNetBlock(num_features * 4, num_features * 8, num_features * 16, num_features * 4, block)
        block = UNetBlock(num_features * 2, num_features * 4, num_features * 8, num_features * 2, block)
        block = UNetBlockHybridSparse(num_features, num_features * 2, num_features * 6, num_features * 2, block)
        block = UNetBlockOuterSparse(None, num_features, num_features * 2, num_output_features, block)
        self.model = block

    def forward(self, x: torch.Tensor, batch_size, frustum_mask) -> BlockContent:
        # print("\n\nUNetSparse input shape:", x.shape)
        output = self.model(BlockContent(x, frustum_mask), batch_size)
        return output

class UNetBlock(nn.Module):
    def __init__(self,
                 num_input_features: Optional[int], num_inner_features: int,
                 num_outer_features: Optional[int], num_output_features: int,
                 submodule: Optional[nn.Module]) -> None:
        super().__init__()

        num_input_features = num_output_features if num_input_features is None else num_input_features
        num_outer_features = num_inner_features * 2 if num_outer_features is None else num_outer_features

        downsample = nn.Conv3d(num_input_features, num_inner_features, kernel_size=4, stride=2, padding=1, bias=False)

        self.encoder = nn.Sequential(
            ResNetBlock3d(num_input_features, num_inner_features, stride=2, downsample=downsample),
            ResNetBlock3d(num_inner_features, num_inner_features, stride=1)
        )

        self.submodule = submodule
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(num_outer_features, num_output_features, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(num_output_features),
            nn.ReLU(inplace=True)
        )

        self.verbose = True

    def forward(self, x: BlockContent) -> BlockContent:
        content = x.data

        encoded = self.encoder(content) if not self.verbose else self.forward_verbose(content, self.encoder)
        processed = self.submodule(BlockContent(encoded, None))
        decoded = self.decoder(processed.data) if not self.verbose else self.forward_verbose(processed.data,
                                                                                             self.decoder)

        output = torch.cat([content, decoded], dim=1)

        if self.verbose:
            # self.print_summary(content, decoded)
            self.verbose = False

        return BlockContent(output, processed.encoding)

    def print_summary(self, x: torch.Tensor, output: torch.Tensor):
        shape_before = list(x.shape)
        shape_after = list(output.shape)
        self.logger.info(
            f"{shape_before} --> {shape_after}\t[{type(self).__name__}]\tInput: {type(x).__name__}\tDecoded: {type(output).__name__}")

    def forward_verbose(self, x: torch.Tensor, modules: torch.nn.Module) -> torch.Tensor:
        for module in modules.children():
            shape_before = list(x.shape)
            x = module(x)
            shape_after = list(x.shape)
            self.logger.info(f"[{type(self).__name__} - {type(module).__name__}]\t {shape_before} --> {shape_after}")

        return x


class UNetBlockOuterSparse(UNetBlock):
    def __init__(self,
                 num_input_features: Optional[int], num_inner_features: int,
                 num_outer_features: Optional[int], num_output_features: int,
                 submodule: Optional[nn.Module]):
        super().__init__(num_input_features, num_inner_features, num_outer_features, num_output_features, submodule)

        # define encoders
        num_encoders = 1

        # define depth feature encoder
        self.num_input_features = 1
        depth_downsample = nn.Sequential(
            Me.MinkowskiConvolution(self.num_input_features, num_inner_features, kernel_size=1, stride=1, bias=True, dimension=3),
            Me.MinkowskiInstanceNorm(num_inner_features)
        )
        self.encoder_depth = nn.Sequential(
            SparseBasicBlock(self.num_input_features, num_inner_features, dimension=3, downsample=depth_downsample)
        )
        
        num_combined_encoder_features = num_encoders * num_inner_features
        encoder_downsample = nn.Sequential(
            Me.MinkowskiConvolution(num_combined_encoder_features, num_inner_features, kernel_size=4, stride=2, bias=True, dimension=3),
            Me.MinkowskiInstanceNorm(num_inner_features)
        )
        self.encoder = nn.Sequential(
            SparseBasicBlock(num_combined_encoder_features, num_inner_features, stride=2, dimension=3, downsample=encoder_downsample),
            SparseBasicBlock(num_inner_features, num_inner_features, dimension=3)
        )

        # define proxy outputs
        self.proxy_occupancy_128_head = nn.Sequential(Me.MinkowskiLinear(num_outer_features, 1))

        # define decoder
        num_conv_input_features = num_outer_features + num_inner_features

        self.decoder = nn.Sequential(
            Me.MinkowskiConvolutionTranspose(num_conv_input_features, num_output_features, kernel_size=4, stride=2, bias=False, dimension=3, expand_coordinates=True),
            Me.MinkowskiInstanceNorm(num_output_features),
            Me.MinkowskiReLU()
        )

    def forward(self, x: BlockContent, batch_size: int):
        content = x.data
        cm = content.coordinate_manager
        key = content.coordinate_map_key

        # process each input feature type individually
        # concat all processed features and process with another encoder
        # process depth features
        start_features = 0
        end_features = self.num_input_features

        input_features = Me.SparseTensor(content.F[:, start_features:end_features], coordinate_manager=cm, coordinate_map_key=key)
        encoded_input = self.encoder_depth(input_features) if not self.verbose else self.forward_verbose(input_features, self.encoder_depth)

        # encoded_input = Me.cat(encoded_input, encoded_features) unused!

        # process input features
        encoded = self.encoder(encoded_input) if not self.verbose else self.forward_verbose(encoded_input, self.encoder)

        # forward to next hierarchy
        processed: BlockContent = self.submodule(BlockContent(encoded, x.encoding), batch_size)

        if processed is None:
            return None

        sparse, dense = processed.data

        if sparse is not None:
            sparse = Me.SparseTensor(sparse.F, sparse.C, coordinate_manager=cm, tensor_stride=sparse.tensor_stride)
        # proxy occupancy output
        if sparse is not None:
            proxy_output = self.proxy_occupancy_128_head(sparse)
        else:
            proxy_output = None

        should_concat = proxy_output is not None

        if should_concat:
            
            output = self.decoder(encoded) if not self.verbose else self.forward_verbose(encoded, self.decoder)
            output = Me.SparseTensor(output.F, output.C, coordinate_manager=cm)  # Fix

        else:
            output = None

        if self.verbose:
            self.verbose = False
        return BlockContent([output, [proxy_output], dense], processed.encoding)

class UNetBlockInner(UNetBlock):
    def __init__(self, num_inner_features: int, num_output_features: int):
        super().__init__(num_inner_features, num_inner_features, num_inner_features, num_output_features, None)

    def forward(self, x: BlockContent) -> BlockContent:
        content = x.data
        encoded = self.encoder(content) if not self.verbose else self.forward_verbose(content, self.encoder)
        decoded = self.decoder(encoded) if not self.verbose else self.forward_verbose(encoded, self.decoder)
        output = torch.cat([content, decoded], dim=1)

        if self.verbose:
            self.verbose = False
        return BlockContent(output, encoded)



class UNetBlockOuter(UNetBlock):
    def __init__(self,
                 num_input_features: int, num_inner_features: int,
                 num_outer_features: int, num_output_features,
                 submodule: nn.Module):
        super().__init__(num_input_features, num_inner_features, num_outer_features, num_outer_features, submodule)

        self.encoder = nn.Sequential(
            nn.Conv3d(num_input_features, num_inner_features, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(num_inner_features),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_inner_features, num_inner_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(num_inner_features),
            nn.ReLU(inplace=True)
        )

        modules = list(self.decoder.children())[:-2]
        modules += [
            nn.ConvTranspose3d(num_outer_features, num_outer_features, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(num_outer_features), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(num_outer_features, num_output_features, 1, 1, 0),
        ]

        self.decoder = nn.Sequential(*modules)

    def forward(self, x: BlockContent) -> BlockContent:
        content = x.data
        encoded = self.encoder(x.data) if not self.verbose else self.forward_verbose(content, self.encoder)
        processed = self.submodule(BlockContent(encoded, None))
        output = self.decoder(processed.data) if not self.verbose else self.forward_verbose(processed.data, self.decoder)

        if self.verbose:
            self.verbose = False
        return BlockContent(output, processed.encoding)

class UNetBlockHybridSparse(UNetBlockOuter):
    """
    UNetBlock with a sparse encoder and sparse decoder.
    The encoder output is densified.
    The decoder input is sparsified.
    """

    def __init__(self,
                 num_input_features: int, num_inner_features: int,
                 num_outer_features: int, num_output_features: int,
                 submodule: nn.Module):
        super().__init__(num_input_features, num_inner_features, num_outer_features, num_output_features, submodule)

        downsample = Me.MinkowskiConvolution(num_input_features, num_inner_features, kernel_size=4, stride=2, bias=False, dimension=3)
        self.encoder = nn.Sequential(
            SparseBasicBlock(num_input_features, num_inner_features, stride=2, dimension=3, downsample=downsample),
            SparseBasicBlock(num_inner_features, num_inner_features, dimension=3)
        )

        self.num_inner_features = num_inner_features
        self.proxy_occupancy_64_head = nn.Sequential(nn.Linear(num_inner_features * 2, 1))
        num_conv_input_features = num_outer_features

        self.decoder = nn.Sequential(
            Me.MinkowskiConvolutionTranspose(num_conv_input_features, num_output_features, kernel_size=4, stride=2, bias=False, dimension=3, expand_coordinates=True), Me.MinkowskiInstanceNorm(num_output_features),
            Me.MinkowskiReLU(),
            SparseBasicBlock(num_output_features, num_output_features, dimension=3)
        )

    def forward(self, x: BlockContent, batch_size: int) -> BlockContent:
        # encode
        content = x.data
        encoded = self.encoder(content) if not self.verbose else self.forward_verbose(content, self.encoder)

        # to dense at 64x64x64 with min_coordinate at 0,0,0
        shape = torch.Size([batch_size, self.num_inner_features, 64, 64, 64])
        min_coordinate = torch.IntTensor([0, 0, 0]).to(encoded.device)

        # mask out all voxels that are outside (<0 & >256)
        mask = (encoded.C[:, 1] < 256) & (encoded.C[:, 2] < 256) & (encoded.C[:, 3] < 32)
        mask = mask & (encoded.C[:, 1] >= 0) & (encoded.C[:, 2] >= 0) & (encoded.C[:, 3] >= 0)

        encoded = Me.MinkowskiPruning()(encoded, mask)

        if len(encoded.C) == 0:
            return BlockContent([None, None], content)

        dense, _, _ = encoded.dense(shape, min_coordinate=min_coordinate)

        # next hierarchy
        processed: BlockContent = self.submodule(BlockContent(dense, None))

        # decode
        # occupancy proxy -> mask
        proxy_flat = processed.data.view(batch_size, processed.data.shape[1], -1).permute(0, 2, 1)
        proxy_output = self.proxy_occupancy_64_head(proxy_flat)
        proxy_output = proxy_output.view(batch_size, 64, 64, 8)
        dense_to_sparse_mask: torch.Tensor = torch.sigmoid(proxy_output) > 0.5

        # mask dense occupancy
        frustum_mask = x.encoding
        dense_to_sparse_mask = torch.masked_fill(dense_to_sparse_mask, frustum_mask.squeeze() == False, False)
        proxy_output = torch.masked_fill(proxy_output, frustum_mask.squeeze() == False, 0.0).unsqueeze(1)

        

        proxy_output = proxy_output

        if True:
            coordinates, _, _ = Sparsify()(dense_to_sparse_mask, features=processed.data)
            locations = coordinates.long()

            dense_features = processed.data

            sparse_features = dense_features[locations[:, 0], :, locations[:, 1], locations[:, 2], locations[:, 3]]

            if coordinates.shape[0] == 0:
                return None

            coords_next = coordinates
            stride = encoded.tensor_stride[0]
            coords_next[:, 1:] *= stride  # "upsample coordinates"
            cm = encoded.coordinate_manager
            key, _ = cm.insert_and_map(coords_next, encoded.tensor_stride, string_id="decoded")
            sparse_features = Me.SparseTensor(sparse_features, coordinates=coords_next.int(), tensor_stride=4, coordinate_manager=cm)
            concat = utils.sparse_cat_union(encoded, sparse_features)

            output = self.decoder(concat) if not self.verbose else self.forward_verbose(concat, self.decoder)
        else:
            output = None
        if self.verbose:
            self.verbose = False
        return BlockContent([output, proxy_output], processed.encoding)

class Sparsify:
    def __call__(self, occupancy: torch.Tensor, features=None, *args, **kwargs) -> Tuple[np.array, np.array, np.array]:
        device = occupancy.device
        ones = torch.ones([1], device=device)
        zeros = torch.zeros([1], device=device)
        coords = torch.stack(torch.where(occupancy.squeeze(1) == 1.0, ones, zeros).nonzero(as_tuple=True), dim=1).int()

        if features is not None and len(features.shape) == len(occupancy.shape):
            num_dimensions = coords.shape[1]
            locations = coords.long()
            if num_dimensions == 4:  # BxCx Volume
                feats = features[locations[:, 0], :, locations[:, 1], locations[:, 2], locations[:, 3]]
            elif num_dimensions == 3:
                feats = features[0, :, locations[:, 0], locations[:, 1], locations[:, 2]]
                feats = feats.permute(1, 0)
            else:
                feats = torch.ones_like(coords[:, :1], dtype=torch.float)
        else:
            feats = torch.ones_like(coords[:, :1], dtype=torch.float)

        labels = torch.ones_like(coords[:, :1], dtype=torch.int32)

        return coords, feats, labels
