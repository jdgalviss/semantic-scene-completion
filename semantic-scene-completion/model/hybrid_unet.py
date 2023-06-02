import MinkowskiEngine as Me
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union,Tuple, List
from configs import config

from .resnet3d import ResNetBlock3d
from .resnet_sparse import BasicBlock as SparseBasicBlock
from .model_utils import sparse_cat_union
import logging
import sys
from pathlib import Path
from math import log
from .model_utils import get_sparse_values

logger = logging.getLogger("trainer")

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

def sparse_cat_union(a: Me.SparseTensor, b: Me.SparseTensor) -> Me.SparseTensor:
    '''
    Concatenate two sparse tensors along the feature dimension.
    The coordinates of the two tensors must be the same.
    Args:
        a: SparseTensor
        b: SparseTensor
    Returns:
        SparseTensor
    '''
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
    '''Data structure for passing data through the sparse generative network.'''
    def __init__(self, data: Union[torch.Tensor, Me.SparseTensor], encoding: Optional[torch.Tensor], features2D: Optional[List[torch.Tensor]] = None, features: Optional[List[Me.SparseTensor]] = None):
        '''
        Args:
            data: List of Tensor or SparseTensor
            encoding: Tensor
            features2D: List of 2D features (unused)
            features: List of SparseTensors containing features at different stages of the UNet
        '''
        self.data = data
        self.encoding = encoding
        self.features2D = features2D
        self.features = features

class UNetHybrid(nn.Module):
    '''Sparse generative network'''
    def __init__(self, num_output_features: int, num_features: int = 64) -> None:
        super().__init__()
        # Create Network from all the different submodules (both sparse and dense)
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
    '''Dense UNet block containing downsample and upsample stages'''
    def __init__(self,
                 num_input_features: Optional[int], num_inner_features: int,
                 num_outer_features: Optional[int], num_output_features: int,
                 submodule: Optional[nn.Module]) -> None:
        super().__init__()

        num_input_features = num_output_features if num_input_features is None else num_input_features
        num_outer_features = num_inner_features * 2 if num_outer_features is None else num_outer_features
        self.current_level = int(log(num_output_features // config.MODEL.NUM_INPUT_FEATURES, 2)) # TODO(jdgalviss): How robust is this?
        self.num_input_features = num_input_features
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
        self.logger = logger


    def forward(self, x: BlockContent) -> BlockContent:
        content = x.data
        features2D = x.features2D
        encoded = self.encoder(content) if not self.verbose else self.forward_verbose(content, self.encoder)
        # Create sparse tensor out of encoded features
        # coors_factor = int(256/encoded.shape[2])
        # encoded_nz = torch.count_nonzero(encoded,dim=1)
        # coords = torch.nonzero(encoded_nz)
        # encoded_values = encoded[coords[:, 0], :, coords[:, 1], coords[:, 2], coords[:, 3]]
        # coords*=coors_factor
        # sparse_encoded = Me.SparseTensor(encoded_values, coordinates=coords.contiguous().float())
        processed = self.submodule(BlockContent(encoded, None, features2D))
        features = []
        decoded = self.decoder(processed.data)
        output = torch.cat([content, decoded], dim=1)
        if self.verbose:
            # self.print_summary(content, decoded)
            self.verbose = False
        return BlockContent(output, processed.encoding, features2D, features=features)

    def print_summary(self, x: torch.Tensor, output: torch.Tensor) -> None:
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
    '''Contains the outer most layers of the sparse generative model. Performs downsampling of sparse input
    and at the end, final upsampling to the original resolution, outputing also a sparse tensor.
    Contains proxy heads for segmentation and occupancy prediction at scale 128'''
    def __init__(self,
                 num_input_features: Optional[int], num_inner_features: int,
                 num_outer_features: Optional[int], num_output_features: int,
                 submodule: Optional[nn.Module]):
        super().__init__(num_input_features, num_inner_features, num_outer_features, num_output_features, submodule)

        # define encoders
        num_encoders = 1

        # define feature encoder
        if config.MODEL.SEG_HEAD:
            if config.SEGMENTATION.SEG_MODEL == "2DPASS":
                self.num_input_features = 256
            else:
                self.num_input_features = config.MODEL.NUM_INPUT_FEATURES
        else:
            self.num_input_features = 1

        # Encoder for input pointcloud features
        input_downsample = nn.Sequential(
            Me.MinkowskiConvolution(self.num_input_features, num_inner_features, kernel_size=1, stride=1, bias=True, dimension=3),
            Me.MinkowskiInstanceNorm(num_inner_features)
        )
        self.encoder_feat = nn.Sequential(
            SparseBasicBlock(self.num_input_features, num_inner_features, dimension=3, downsample=input_downsample)
        )
        self.num_semantic_features = config.SEGMENTATION.NUM_CLASSES 

        # Encoder for semantic logits
        if config.MODEL.SEG_HEAD:
            num_encoders+=1
            # define segmentation feature encoder
            seg_downsample = nn.Sequential(
                Me.MinkowskiConvolution(self.num_semantic_features-1, num_inner_features, kernel_size=1, stride=1, bias=True, dimension=3), # no empty class for semantic predictions
                Me.MinkowskiInstanceNorm(num_inner_features)
            )
            self.encoder_seg = nn.Sequential(
                SparseBasicBlock(self.num_semantic_features-1, num_inner_features, dimension=3, downsample=seg_downsample)
            )
        
        num_combined_encoder_features = num_encoders * num_inner_features
        
        # Encoded to further process concatenated features
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
        num_proxy_input_features = num_outer_features + num_inner_features
        self.proxy_semantic_128_head = nn.Sequential(
            SparseBasicBlock(num_proxy_input_features, num_proxy_input_features, dimension=3),
            Me.MinkowskiConvolution(num_proxy_input_features, self.num_semantic_features, kernel_size=3, stride=1, bias=True, dimension=3)
        )

        # define decoder
        num_conv_input_features = num_outer_features + num_inner_features + self.num_semantic_features
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
        # process pointcloud features
        start_features = 0
        end_features = self.num_input_features
        features_input = Me.SparseTensor(content.F[:, start_features:end_features], coordinate_manager=cm, coordinate_map_key=key)
        encoded_input = self.encoder_feat(features_input) if not self.verbose else self.forward_verbose(features_input, self.encoder_feat)

        # process seg logits if present
        if config.MODEL.SEG_HEAD:
            start_features = end_features
            end_features += self.num_semantic_features
            seg_input = Me.SparseTensor(content.F[:, start_features:end_features], coordinate_manager=cm, coordinate_map_key=key)

            encoded_seg = self.encoder_seg(seg_input) if not self.verbose else self.forward_verbose(seg_input, self.encoder_seg)
            # print("encoded_seg: ", encoded_seg.shape)
            encoded_input = Me.cat(encoded_input, encoded_seg)

        # process all input_features
        encoded = self.encoder(encoded_input) if not self.verbose else self.forward_verbose(encoded_input, self.encoder)
        # forward to next submodule
        processed: BlockContent = self.submodule(BlockContent(encoded, x.encoding, x.features2D), batch_size)
        # Features (used for distillation)
        # features = processed.features
        
        if processed is None:
            return None
        sparse, dense = processed.data
        if sparse is not None:
            sparse = Me.SparseTensor(sparse.F, sparse.C, coordinate_manager=cm, tensor_stride=sparse.tensor_stride)
        if sparse is not None:
            proxy_output = self.proxy_occupancy_128_head(sparse)
        else:
            proxy_output = None
        should_concat = proxy_output is not None
        features = []

        if should_concat:
            proxy_mask = (Me.MinkowskiSigmoid()(proxy_output).F > 0.5).squeeze(1)
            if proxy_mask.sum() == 0:
                cat = None
                proxy_semantic = None
            else:
                sparse_pruned = Me.MinkowskiPruning()(sparse, proxy_mask)  # mask out invalid voxels
                if len(sparse_pruned.C) == 0:
                    return BlockContent([None, [proxy_output, None], dense], processed.encoding)
                # Skip connection
                cat = sparse_cat_union(encoded, sparse_pruned)
                # proxy semantic prediction
                proxy_semantic = self.proxy_semantic_128_head(cat)
                cat = sparse_cat_union(cat, proxy_semantic)
                # features = [cat]
            if  (config.GENERAL.LEVEL == "256" or config.GENERAL.LEVEL == "FULL") and proxy_output is not None:
                output = self.decoder(cat) if not self.verbose else self.forward_verbose(cat, self.decoder)
                output = Me.SparseTensor(output.F, output.C, coordinate_manager=cm)  # Fix
            else:
                output = None
        else:
            output = None
            proxy_semantic = None
        features.extend(processed.features)
        if self.verbose:
            self.verbose = False
        
        return BlockContent([output, [proxy_output, proxy_semantic], dense], processed.encoding, features=features)


class UNetBlockInner(UNetBlock):
    '''Inner most dense layers'''
    def __init__(self, num_inner_features: int, num_output_features: int):
        super().__init__(num_inner_features, num_inner_features, num_inner_features, num_output_features, None)


    def forward(self, x: BlockContent) -> BlockContent:
        content = x.data
        encoded = self.encoder(content) if not self.verbose else self.forward_verbose(content, self.encoder)
        features = []
        decoded = self.decoder(encoded) if not self.verbose else self.forward_verbose(encoded, self.decoder)
        output = torch.cat([content, decoded], dim=1)
        if self.verbose:
            self.verbose = False
        return BlockContent(output, encoded, features=features)

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
        self.num_input_features = num_input_features
        self.num_inner_features = num_inner_features
        self.proxy_occupancy_64_head = nn.Sequential(nn.Linear(num_inner_features * 2, 1))
        self.proxy_semantic_64_head = nn.Sequential(
            ResNetBlock3d(num_inner_features * 2, num_inner_features * 2),
            ResNetBlock3d(num_inner_features * 2, num_inner_features * 2),
            nn.Conv3d(num_inner_features * 2, config.SEGMENTATION.NUM_CLASSES, kernel_size=3, stride=1, padding=1, bias=True)
        )
        num_conv_input_features = num_outer_features + config.SEGMENTATION.NUM_CLASSES
        self.decoder = nn.Sequential(
            Me.MinkowskiConvolutionTranspose(num_conv_input_features, num_output_features, kernel_size=4, stride=2, bias=False, dimension=3, expand_coordinates=True), Me.MinkowskiInstanceNorm(num_output_features),
            Me.MinkowskiReLU(),
            SparseBasicBlock(num_output_features, num_output_features, dimension=3)
        )

    def forward(self, x: BlockContent, batch_size: int) -> BlockContent:
        content = x.data
        # encode sparse input
        encoded = self.encoder(content) if not self.verbose else self.forward_verbose(content, self.encoder)
        shape = torch.Size([batch_size, self.num_inner_features, 64, 64, 8])
        min_coordinate = torch.IntTensor([0, 0, 0])
        # mask out all voxels that are outside (<0 & >256)
        mask = (encoded.C[:, 1] < 256) & (encoded.C[:, 2] < 256) & (encoded.C[:, 3] < 32)
        mask = mask & (encoded.C[:, 1] >= 0) & (encoded.C[:, 2] >= 0) & (encoded.C[:, 3] >= 0)
        encoded = Me.MinkowskiPruning()(encoded, mask)
        if len(encoded.C) == 0:
            # next network layers return dense feature map
            return BlockContent([None, None], content, x.features2D, features=[])
        # densify encoded feature map for residual connection
        dense, _, _ = encoded.dense(shape, min_coordinate=min_coordinate)
        processed: BlockContent = self.submodule(BlockContent(dense, None, x.features2D))
        # features = [encoded]
        # features.extend(processed.features)
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

        # semantic proxy
        # features = [processed.data]
        semantic_prediction = self.proxy_semantic_64_head(processed.data)
        semantic_prediction = torch.masked_fill(semantic_prediction, frustum_mask.squeeze() == False, 0.0)
        semantic_prediction[:, 0] = torch.masked_fill(semantic_prediction[:, 0], frustum_mask.squeeze() == False, 1.0)
        proxy_output = [proxy_output, semantic_prediction]
        features = []


        if config.GENERAL.LEVEL != "64":
            coordinates, _, _ = Sparsify()(dense_to_sparse_mask, features=processed.data)
            locations = coordinates.long()
            dense_features = processed.data
            if semantic_prediction is not None:
                dense_features = torch.cat([dense_features, semantic_prediction], dim=1)
            sparse_features = dense_features[locations[:, 0], :, locations[:, 1], locations[:, 2], locations[:, 3]]
            if coordinates.shape[0] == 0:
                return None
            coords_next = coordinates
            stride = encoded.tensor_stride[0]
            coords_next[:, 1:] *= stride  # "upsample coordinates"
            cm = encoded.coordinate_manager
            key, _ = cm.insert_and_map(coords_next, encoded.tensor_stride, string_id="decoded")
            sparse_features = Me.SparseTensor(sparse_features, coordinates=coords_next.float(), tensor_stride=4, coordinate_manager=cm)
            # print("sparse_features: ", sparse_features.shape)
            concat = sparse_cat_union(encoded, sparse_features)
            # print("concat: ", concat.shape)
            output = self.decoder(concat) if not self.verbose else self.forward_verbose(concat, self.decoder)
            # print("output: ", output.shape)
            # features = [semantic_prediction]
            features = [concat]
            # features = [processed.data]
        else:
            output = None
        if self.verbose:
            self.verbose = False
        return BlockContent([output, proxy_output], processed.encoding, features=features)

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
