import MinkowskiEngine as Me
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union,Tuple, List
from configs import config

from .resnet3d import ResNetBlock3d
from .resnet_sparse import BasicBlock as SparseBasicBlock
from .sparse_unet_utils import sparse_cat_union
import logging
import sys
from pathlib import Path
from math import log
from .unet_parts import DoubleConv

logger = logging.getLogger("trainer")
features2d_channels = [1,2,8,32] #TODO(jdgalviss): this is not nice, but it works for now

def get_sparse_values(tensor: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    values = tensor[coordinates[:, 0], :, coordinates[:, 1], coordinates[:, 2], coordinates[:, 3]]
    return values

def setup_logger(save_path: Path, filename: str = "log.txt") -> None:
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s: %(message)s", datefmt='%d.%m %H:%M:%S')

    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_path:
        fh = logging.FileHandler(save_path / filename)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

def sparse_cat_union(a: Me.SparseTensor, b: Me.SparseTensor):
    cm = a.coordinate_manager
    assert cm == b.coordinate_manager, "different coords_man"
    assert a.tensor_stride == b.tensor_stride, "different tensor_stride"

    zeros_cat_with_a = torch.zeros([a.F.shape[0], b.F.shape[1]], dtype=a.dtype).to(a.device)
    zeros_cat_with_b = torch.zeros([b.F.shape[0], a.F.shape[1]], dtype=a.dtype).to(a.device)

    feats_a = torch.cat([a.F, zeros_cat_with_a], dim=1)
    feats_b = torch.cat([zeros_cat_with_b, b.F], dim=1)

    new_a = Me.SparseTensor(
        features=feats_a,
        coordinates=a.C,
        coordinate_manager=cm,
        tensor_stride=a.tensor_stride,
    )

    new_b = Me.SparseTensor(
        features=feats_b,
        coordinates=b.C,
        coordinate_manager=cm,
        tensor_stride=a.tensor_stride,
    )

    return new_a + new_b

class BlockContent:
    def __init__(self, data: Union[torch.Tensor, Me.SparseTensor], encoding: Optional[torch.Tensor], features2D: Optional[List[torch.Tensor]] = None):
        self.data = data
        self.encoding = encoding
        self.features2D = features2D

class UNetSparse(nn.Module):
    def __init__(self, num_output_features: int, num_features: int = 64) -> None:
        super().__init__()
        block = UNetBlockInner(num_features * 8, num_features * 8)
        block = UNetBlock(num_features * 4, num_features * 8, num_features * 16, num_features * 4, block)
        block = UNetBlock(num_features * 2, num_features * 4, num_features * 8, num_features * 2, block)
        block = UNetBlockHybridSparse(num_features, num_features * 2, num_features * 6, num_features * 2, block)
        block = UNetBlockOuterSparse(None, num_features, num_features * 2, num_output_features, block)
        self.model = block

    def forward(self, x: torch.Tensor, batch_size=1, valid_mask=None, features2D=None) -> BlockContent:
        output = self.model(BlockContent(x, valid_mask, features2D), batch_size)
        return output

class UNetBlock(nn.Module):
    def __init__(self,
                 num_input_features: Optional[int], num_inner_features: int,
                 num_outer_features: Optional[int], num_output_features: int,
                 submodule: Optional[nn.Module]) -> None:
        super().__init__()

        num_input_features = num_output_features if num_input_features is None else num_input_features
        num_outer_features = num_inner_features * 2 if num_outer_features is None else num_outer_features
        self.current_level = int(log(num_output_features // config.MODEL.UNET_FEATURES, 2)) # TODO(jdgalviss): How robust is this?
        self.num_input_features = num_input_features
        num_input_features2d = 0
        num_outer_features2d = 0

        if config.MODEL.UNET2D and (num_input_features in [32,64,128]): #TODO(jdgalviss): this is not nice, but it works for now
            # current level used to know what feature from the 2D UNet to concatenate

            # Project the 2D features to the 3D space using ConvNet
            self.projection = nn.Sequential(
                DoubleConv(in_channels=num_output_features*2, out_channels=num_output_features),
                DoubleConv(in_channels=num_output_features, out_channels=num_output_features//2)
            )
            self.num_input_features = num_input_features
            self.num_outer_features = num_outer_features

            num_input_features2d = features2d_channels[self.current_level]
            num_outer_features2d = features2d_channels[self.current_level+1] if num_input_features != 128 else 0

            # if num_input_features != 128:
            #     num_outer_features += features2d_channels[self.current_level+1]

            # num_input_features += features2d_channels[self.current_level]


        downsample = nn.Conv3d(num_input_features+num_input_features2d, num_inner_features, kernel_size=4, stride=2, padding=1, bias=False)

        self.encoder = nn.Sequential(
            ResNetBlock3d(num_input_features+num_input_features2d, num_inner_features, stride=2, downsample=downsample),
            ResNetBlock3d(num_inner_features, num_inner_features, stride=1)
        )

        self.submodule = submodule
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(num_outer_features+num_outer_features2d, num_output_features, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(num_output_features),
            nn.ReLU(inplace=True)
        )

        
        self.verbose = True
        self.logger = logger


    def forward(self, x: BlockContent) -> BlockContent:
        # print("\n UNetBlock")
        content = x.data
        input_shape = content.shape
        
        features2D = x.features2D
        # print("\ncontent shape:", content.shape)
        # print("current_level:", self.current_level)
        # print("features2D shape:", features2D[self.current_level].shape)
        # print("num_input_features:", self.num_input_features)
        
        if config.MODEL.UNET2D and (self.num_input_features in [32,64,128]): #TODO(jdgalviss): this is not nice, but it works for now
            # project 2D features and concatenate
            projected_features = self.projection(features2D[self.current_level]).view(input_shape[0], -1, input_shape[4], input_shape[2], input_shape[3])
            projected_features = torch.permute(projected_features,(0,1,3,4,2))
            # print("projected_features shape:", projected_features.shape)
            content = torch.cat([content,projected_features], dim=1)
            encoded = self.encoder(content) if not self.verbose else self.forward_verbose(content, self.encoder)
        else:
            encoded = self.encoder(content) if not self.verbose else self.forward_verbose(content, self.encoder)
        # print("encoded shape:", encoded.shape)
        processed = self.submodule(BlockContent(encoded, None, features2D))
        # decoded = self.decoder(processed.data) if not self.verbose else self.forward_verbose(processed.data,
        #                                                                                      self.decoder)
        # print("processed shape:", processed.data.shape)
        # print("num_input_features:", self.num_input_features)

        decoded = self.decoder(processed.data)
        # print("\n... back to UnetBlock")
        # print("decoded shape:", decoded.shape)
        output = torch.cat([content, decoded], dim=1)
        # print("UNetBlock output shape:", output.shape)
        if self.verbose:
            # self.print_summary(content, decoded)
            self.verbose = False

        return BlockContent(output, processed.encoding, features2D)

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
        self.encoder_input = nn.Sequential(
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

        self.num_semantic_features = config.SEGMENTATION.NUM_CLASSES
        num_proxy_input_features = num_outer_features + num_inner_features

        self.proxy_semantic_128_head = nn.Sequential(
            SparseBasicBlock(num_proxy_input_features, num_proxy_input_features, dimension=3),
            Me.MinkowskiConvolution(num_proxy_input_features, self.num_semantic_features, kernel_size=3, stride=1, bias=True, dimension=3)
        )

        # define decoder
        num_conv_input_features = num_outer_features + num_inner_features + self.num_semantic_features
        # print("num_conv_input_features:", num_conv_input_features)

        self.decoder = nn.Sequential(
            Me.MinkowskiConvolutionTranspose(num_conv_input_features, num_output_features, kernel_size=4, stride=2, bias=False, dimension=3, expand_coordinates=True),
            Me.MinkowskiInstanceNorm(num_output_features),
            Me.MinkowskiReLU()
        )

    def forward(self, x: BlockContent, batch_size: int):
        # print("\n Outter Sparse Block: ")
        input_features = x.data
        # print("input_features shape:", input_features.shape)
        cm = input_features.coordinate_manager
        # key = input_features.coordinate_map_key

        # process each input feature type individually
        # concat all processed features and process with another encoder
        # process depth features

        encoded_input = self.encoder_input(input_features) if not self.verbose else self.forward_verbose(input_features, self.encoder_input)
        # print("encoded_input: ", encoded_input.shape)
        # encoded_input = Me.cat(encoded_input, encoded_features) unused!

        # process input features
        encoded = self.encoder(encoded_input) if not self.verbose else self.forward_verbose(encoded_input, self.encoder)
        # print("encoded: ", encoded.shape)
        # forward to next hierarchy
        processed: BlockContent = self.submodule(BlockContent(encoded, x.encoding, x.features2D), batch_size)

        if processed is None:
            return None
        # print("\n ...back to Outter Sparse Block: ")
        sparse, dense = processed.data
        # print("sparse: ", sparse.shape)

        if sparse is not None:
            sparse = Me.SparseTensor(sparse.F, sparse.C, coordinate_manager=cm, tensor_stride=sparse.tensor_stride)
            # print("sparse2: ", sparse.shape)
        # proxy occupancy output
        if sparse is not None:
            proxy_output = self.proxy_occupancy_128_head(sparse)
            # print("proxy_output: ", proxy_output.shape)
        else:
            proxy_output = None

        should_concat = proxy_output is not None

        if should_concat:
            proxy_mask = (Me.MinkowskiSigmoid()(proxy_output).F > 0.5).squeeze(1)
            # print("proxy_mask: ", proxy_mask.shape)
            # no valid voxels
            if proxy_mask.sum() == 0:
                cat = None
                proxy_semantic = None

            else:
                sparse_pruned = Me.MinkowskiPruning()(sparse, proxy_mask)  # mask out invalid voxels
                # print("sparse_pruned: ", sparse_pruned.shape)
                if len(sparse_pruned.C) == 0:
                    return BlockContent([None, [proxy_output, None], dense], processed.encoding)

                # Skip connection
                cat = sparse_cat_union(encoded, sparse_pruned)
                # print("cat: ", cat.shape)
                # proxy semantic prediction
                proxy_semantic = self.proxy_semantic_128_head(cat)
                cat = sparse_cat_union(cat, proxy_semantic)
            if  (config.GENERAL.LEVEL == "256" or config.GENERAL.LEVEL == "FULL") and proxy_output is not None:
                output = self.decoder(cat) if not self.verbose else self.forward_verbose(cat, self.decoder)
                # print("output: ", output.shape)
                output = Me.SparseTensor(output.F, output.C, coordinate_manager=cm)  # Fix
                # print("output2: ", output.shape)
            else:
                # print("CURRENT_LEVEL: ", config.GENERAL.LEVEL)
                output = None
        else:
            output = None
            proxy_semantic = None

        if self.verbose:
            self.verbose = False
        
        return BlockContent([output, [proxy_output, proxy_semantic], dense], processed.encoding)


class UNetBlockInner(UNetBlock):
    def __init__(self, num_inner_features: int, num_output_features: int):
        super().__init__(num_inner_features, num_inner_features, num_inner_features, num_output_features, None)


    def forward(self, x: BlockContent) -> BlockContent:
        # print("\n \tInner Block: ")
        content = x.data
        features2D = x.features2D
        input_shape = content.shape


       
        if config.MODEL.UNET2D and (self.num_input_features in [32,64,128]): #TODO(jdgalviss): this is not nice, but it works for now
            # project 2D features and concatenate
            projected_features = self.projection(features2D[self.current_level]).view(input_shape[0], -1, input_shape[4], input_shape[2], input_shape[3])
            projected_features = torch.permute(projected_features,(0,1,3,4,2))
            content = torch.cat([content,projected_features], dim=1)
            # print("\n\tcat_features shape:", content.shape)
            # print("\tprojected_features shape:", projected_features.shape)
            # print("\tnum_input_features:", self.num_input_features)
            # print("\tnum_outer_features:", self.num_outer_features)

            # print("\tinner content: ", content.shape)
            encoded = self.encoder(content) if not self.verbose else self.forward_verbose(content, self.encoder)
        else:
            encoded = self.encoder(content) if not self.verbose else self.forward_verbose(content, self.encoder)
        # print("\tinner encoded: ", encoded.shape)

        decoded = self.decoder(encoded) if not self.verbose else self.forward_verbose(encoded, self.decoder)
        output = torch.cat([content, decoded], dim=1)
        # print("\tinner decoded: ", decoded.shape)
        # print("\tinner decoded: ", decoded.shape)


        # print("\tcurrent_level: ", self.current_level)
        # print("\tfeature shape: ", x.features2D[self.current_level].shape)
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
        # print("\nOuter Block: ")
        content = x.data
        encoded = self.encoder(x.data) if not self.verbose else self.forward_verbose(content, self.encoder)
        processed = self.submodule(BlockContent(encoded, None))
        output = self.decoder(processed.data) if not self.verbose else self.forward_verbose(processed.data, self.decoder)
        # print("outer content: ", content.shape)
        # print("content: ", content.shape)
        # print("encoded: ", encoded.shape)
        # print("output: ", output.shape)
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
        num_input_features2d = 1 if config.MODEL.UNET2D else 0
        downsample = Me.MinkowskiConvolution(num_input_features+num_input_features2d, num_inner_features, kernel_size=4, stride=2, bias=False, dimension=3)
        self.encoder = nn.Sequential(
            SparseBasicBlock(num_input_features+num_input_features2d, num_inner_features, stride=2, dimension=3, downsample=downsample),
            SparseBasicBlock(num_inner_features, num_inner_features, dimension=3)
        )
        self.num_input_features = num_input_features+num_input_features2d
        self.num_inner_features = num_inner_features
        inner_features_2D = 2 if config.MODEL.UNET2D else 0
        self.proxy_occupancy_64_head = nn.Sequential(nn.Linear(num_inner_features * 2 + inner_features_2D, 1))
        self.proxy_semantic_64_head = nn.Sequential(
            ResNetBlock3d(num_inner_features * 2 + inner_features_2D, num_inner_features * 2 + inner_features_2D),
            ResNetBlock3d(num_inner_features * 2 + inner_features_2D, num_inner_features * 2 + inner_features_2D),
            nn.Conv3d(num_inner_features * 2 + inner_features_2D, config.SEGMENTATION.NUM_CLASSES, kernel_size=3, stride=1, padding=1, bias=True)
        )
        num_conv_input_features = num_outer_features + config.SEGMENTATION.NUM_CLASSES
        if config.MODEL.UNET2D:
            num_conv_input_features += 2

        self.decoder = nn.Sequential(
            Me.MinkowskiConvolutionTranspose(num_conv_input_features, num_output_features, kernel_size=4, stride=2, bias=False, dimension=3, expand_coordinates=True), Me.MinkowskiInstanceNorm(num_output_features),
            Me.MinkowskiReLU(),
            SparseBasicBlock(num_output_features, num_output_features, dimension=3)
        )

        # Project the 2D features to the 3D space using ConvNet
        if config.MODEL.UNET2D:
            self.projection = nn.Sequential(
                DoubleConv(in_channels=32, out_channels=32), #TODO(jdgalviss): Hardcoded
                DoubleConv(in_channels=32, out_channels=16)
            )

    def forward(self, x: BlockContent, batch_size: int) -> BlockContent:
        # encode
        content = x.data
        # Concat  2D Features
        if config.MODEL.UNET2D:
            input_shape = (1, 16, 128, 128, 16) #TODO: Hardcoded
            projected_features = self.projection(x.features2D[0]).view(input_shape[0], -1, input_shape[4], input_shape[2], input_shape[3])
            projected_features = torch.permute(projected_features,(0,1,3,4,2))
            coords = torch.nonzero(projected_features[0]) #TODO: only works for batch size 0
            projected_features = get_sparse_values(projected_features, coords)
            projected_features = Me.SparseTensor(projected_features, coordinates=coords.int().contiguous(), 
                                                 coordinate_manager = content.coordinate_manager, tensor_stride = content.tensor_stride)
            content = sparse_cat_union(content, projected_features)

        shape = torch.Size([batch_size, self.num_input_features, 128, 128, 16])
        min_coordinate = torch.IntTensor([0, 0, 0])
        dense_content, _, _ = content.dense(shape, min_coordinate=min_coordinate)
        
        encoded = self.encoder(content) if not self.verbose else self.forward_verbose(content, self.encoder)
        # coords = encoded.C
        # print("\nHybrid sparse block: ")
        ## Debugging
        # print("content: ", content.shape)
        
        # print("encoded: ", encoded.shape)

        # to dense at 64x64x64 with min_coordinate at 0,0,0
        shape = torch.Size([batch_size, self.num_inner_features, 64, 64, 8])
        min_coordinate = torch.IntTensor([0, 0, 0])

        # mask out all voxels that are outside (<0 & >256)
        mask = (encoded.C[:, 1] < 256) & (encoded.C[:, 2] < 256) & (encoded.C[:, 3] < 32)
        mask = mask & (encoded.C[:, 1] >= 0) & (encoded.C[:, 2] >= 0) & (encoded.C[:, 3] >= 0)

        encoded = Me.MinkowskiPruning()(encoded, mask)

        if len(encoded.C) == 0:
            return BlockContent([None, None], content, x.features2D)

        # print("encoded: ", encoded.shape)
        dense, _, _ = encoded.dense(shape, min_coordinate=min_coordinate)
        # print("dense: ", dense.shape)
        # next hierarchy
        processed: BlockContent = self.submodule(BlockContent(dense, None, x.features2D))
        # print("\n...back to Hybrid sparse block")
        # print("processed.data: " , processed.data.shape)
        # decode
        # occupancy proxy -> mask
        proxy_flat = processed.data.view(batch_size, processed.data.shape[1], -1).permute(0, 2, 1)
        # print("proxy_flat: ", proxy_flat.shape)
        proxy_output = self.proxy_occupancy_64_head(proxy_flat)
        # print("proxy_output: ", proxy_output.shape)
        proxy_output = proxy_output.view(batch_size, 64, 64, 8)
        # print("proxy_output: ", proxy_output.shape)
        dense_to_sparse_mask: torch.Tensor = torch.sigmoid(proxy_output) > 0.5
        # print("dense_to_sparse_mask: ", proxy_output.shape)

        
        # mask dense occupancy
        frustum_mask = x.encoding

        # print("frustum_mask shape: ",frustum_mask.shape)
        dense_to_sparse_mask = torch.masked_fill(dense_to_sparse_mask, frustum_mask.squeeze() == False, False)
        proxy_output = torch.masked_fill(proxy_output, frustum_mask.squeeze() == False, 0.0).unsqueeze(1)

        # semantic proxy
        semantic_prediction = self.proxy_semantic_64_head(processed.data)
        semantic_prediction = torch.masked_fill(semantic_prediction, frustum_mask.squeeze() == False, 0.0)
        semantic_prediction[:, 0] = torch.masked_fill(semantic_prediction[:, 0], frustum_mask.squeeze() == False, 1.0)


        proxy_output = [proxy_output, semantic_prediction]

        if config.GENERAL.LEVEL != "64":
            coordinates, _, _ = Sparsify()(dense_to_sparse_mask, features=processed.data)
            locations = coordinates.long()

            dense_features = processed.data

            if semantic_prediction is not None:
                dense_features = torch.cat([dense_features, semantic_prediction], dim=1)

            sparse_features = dense_features[locations[:, 0], :, locations[:, 1], locations[:, 2], locations[:, 3]]
            # print("dense_features: ", dense_features.shape)
            if coordinates.shape[0] == 0:
                return None

            coords_next = coordinates
            stride = encoded.tensor_stride[0]
            coords_next[:, 1:] *= stride  # "upsample coordinates"
            cm = encoded.coordinate_manager
            key, _ = cm.insert_and_map(coords_next, encoded.tensor_stride, string_id="decoded")
            sparse_features = Me.SparseTensor(sparse_features, coordinates=coords_next.int(), tensor_stride=4, coordinate_manager=cm)
            # print("sparse_features: ", sparse_features.shape)
            concat = sparse_cat_union(encoded, sparse_features)
            # print("concat: ", concat.shape)
            output = self.decoder(concat) if not self.verbose else self.forward_verbose(concat, self.decoder)
            # print("output: ", output.shape)
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
