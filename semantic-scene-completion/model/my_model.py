from typing import Dict, Any, Tuple, List

import torch
import torch.nn as nn
from configs import config
from torch.nn import functional as F
from model import UNetSparse, GeometryHeadSparse, ClassificationHeadSparse
import MinkowskiEngine as Me

device = torch.device("cuda:0")


class MyModel(nn.Module):
    def __init__(self,num_output_channels=16,unet_features=16,resnet_blocks=1):
        super().__init__()
        self.model = UNetSparse(num_output_channels, unet_features)
        
        # Heads
        self.occupancy_256_head = GeometryHeadSparse(num_output_channels, 1, resnet_blocks)
        self.occupancy_256_head = self.occupancy_256_head

        # Semantic head
        self.semantic_head = ClassificationHeadSparse(num_output_channels,
                                                        config.SEGMENTATION.NUM_CLASSES,
                                                        resnet_blocks)
        self.semantic_head = self.semantic_head

        # Criterions
        self.criterion_occupancy = F.binary_cross_entropy_with_logits
        self.criterion_semantics = F.cross_entropy  #nn.CrossEntropyLoss(reduction="none")



    def forward(self,complet_coords, complet_invalid,complet_labels):
        # Put coordinates in the right order
        complet_coords = complet_coords[:, [0, 3, 2, 1]]
        complet_coords[:, 3] += 3  # TODO SemanticKITTI will generate [256,256,31]

        # Transform to sparse tensor
        sparse_coords = Me.SparseTensor(features=complet_coords[:,0].unsqueeze(0).transpose(0,1).type(torch.FloatTensor).to(device),
                            coordinates=complet_coords.to(device),
                            quantization_mode=Me.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE)
        # print("sparse_coords: ",sparse_coords.shape)
        complet_valid = torch.logical_not(complet_invalid)
        complet_valid_64 = F.max_pool3d(complet_valid.float(), kernel_size=2, stride=4).bool()
        # print("complet_valid shape:", complet_valid.shape)

        # Forward pass through model
        unet_output = self.model(sparse_coords, batch_size=1, valid_mask=complet_valid_64)
        predictions = unet_output.data
        
        # Filter out invalid voxels
        invalid_locs = torch.nonzero(complet_invalid[0])
        complet_labels[0,invalid_locs[:,0], invalid_locs[:,1], invalid_locs[:,2]] = 255
        
        losses = {}
        results = {}

        # level-128
        if predictions[1] is None:
            return {}
        losses_128, results_128 = self.forward_128(predictions[1], complet_labels)
        
        losses.update(losses_128)
        results.update(results_128)

        #leve-256
        if predictions[0] is None:
            return losses
        
        # Occupancy
        losses_256, results_256, features_256 = self.forward_256(predictions[0], complet_labels)
        losses.update(losses_256)
        results.update(results_256)

        # Semantic
        if config.GENERAL.LEVEL == "FULL":
            losses_output, results_output = self.forward_output(features_256, complet_labels)
            losses.update(losses_output)
            results.update(results_output)

        # output_features = predictions[0]
        # print("output_features: ", output_features.shape)

        # occupancy_prediction = self.occupancy_256_head(output_features)
        # print("occupancy_prediction: ",occupancy_prediction.shape)

        # occupancy_prediction = mask_invalid_sparse_voxels(occupancy_prediction)
        # predicted_coordinates = occupancy_prediction.C.long()
        # predicted_coordinates[:, 1:] = predicted_coordinates[:, 1:] // occupancy_prediction.tensor_stride[0]
        # print("occupancy_prediction: ",occupancy_prediction.shape)
        # occupancy_prediction = Me.MinkowskiSigmoid()(occupancy_prediction)
        # print("occupancy_prediction: ",occupancy_prediction.shape)
        # dense_dimensions = torch.Size([1, 1] + [256, 256, 32])
        # min_coordinates = torch.IntTensor([0, 0, 0]).to(device)
        # occupancy_prediction_dense, _, _ = occupancy_prediction.dense(dense_dimensions, min_coordinates)
        # print("Occupancy prediction dense: ", occupancy_prediction_dense.shape)

        # # Use occupancy prediction to refine sparse voxels
        # sparse_threshold_256 = 0.5
        # occupancy_mask = (occupancy_prediction.F > sparse_threshold_256).squeeze()
        # print("occupancy_mask: ", occupancy_mask.shape)
        # print("occupancy_mask values: ",torch.unique(occupancy_mask))
        # output_features = Me.MinkowskiPruning()(output_features, occupancy_mask)

        # # Semantic predictions
        # semantic_prediction = self.semantic_head(output_features)
        # print("semantic_prediction: ", semantic_prediction.shape)
        # print("semantic_prediction values: ", torch.unique(semantic_prediction))


        return losses

    def forward_output(self, predictions: List[Me.SparseTensor], targets) -> Tuple[Dict, Dict]:
        hierarchy_losses = {}
        hierarchy_results = {}

        # Semantics
        semantic_prediction = self.semantic_head(predictions)
        # print("\nsemantic_prediction: ", semantic_prediction.shape)
        if semantic_prediction is not None:
            semantic_ground_truth: torch.LongTensor = targets.long()
            semantic_loss, semantic_result = self.compute_semantic_256_loss(semantic_prediction, semantic_ground_truth)
            hierarchy_losses.update(semantic_loss)
            # hierarchy_results.update(semantic_result)
        return hierarchy_losses, hierarchy_results

    def compute_semantic_256_loss(self, prediction: Me.SparseTensor, ground_truth: torch.Tensor) -> Tuple[Dict, Dict]:
        prediction = mask_invalid_sparse_voxels(prediction)
        predicted_coordinates = prediction.C.long()
        # predicted_coordinates[:, 1:] = torch.div(predicted_coordinates[:, 1:], prediction.tensor_stride[0], rounding_mode="floor")
        predicted_coordinates[:, 1:] = predicted_coordinates[:, 1:] // prediction.tensor_stride[0]

        # Get sparse GT values from dense tensor
        ground_truth_values = get_sparse_values(ground_truth.unsqueeze(0), predicted_coordinates)
        # print("semantic_ground_truth values_256: ", ground_truth_values.shape)
        # print("semantic_ground_truth values: ", torch.unique(ground_truth_values))

        # print("prediction.F values_256: ", prediction.F.shape)
        # print("prediction.F values: ", torch.unique(prediction.F))


        loss_mean = self.criterion_semantics(prediction.F, ground_truth_values.squeeze(), reduction="mean", ignore_index=255)

        # Get sparse weighting values from dense tensor

        # if len(loss) > 0:
        #     loss_mean = loss.mean()
        # else:
        #     loss_mean = 0

        semantic_softmax = Me.MinkowskiSoftmax(dim=1)(prediction)
        semantic_labels = Me.SparseTensor(torch.argmax(semantic_softmax.F, 1).unsqueeze(1),
                                          coordinate_map_key=prediction.coordinate_map_key,
                                          coordinate_manager=prediction.coordinate_manager)

        return {"256/semantic": loss_mean}, {"256/semantic_softmax": semantic_softmax, "256/semantic_labels": semantic_labels}

    def forward_256(self, predictions, labels):
        hierarchy_losses = {}
        hierarchy_results = {}

        feature_prediction: Me.SparseTensor = predictions
        feature_prediction = mask_invalid_sparse_voxels(feature_prediction)
        # For low memory: TODO(jdgalviss): Is there a memory leak?
        mask = torch.rand(feature_prediction.F.shape[0]) < 0.5
        # print("mask values: ", torch.unique(mask))
        feature_prediction = Me.MinkowskiPruning()(feature_prediction, mask.to(device))
        # print("feature_prediction: ", feature_prediction.shape)


        if feature_prediction is not None:
            # occupancy at 256
            occupancy_prediction = self.occupancy_256_head(feature_prediction)

            one = torch.ones([1], device=device)
            zero = torch.zeros([1], device=device)
            occupancy_ground_truth = torch.where(labels > 0, one, zero)
            # print("occupancy_gt: ", occupancy_ground_truth.shape)
            # print("occupancy_gt values:", torch.unique(occupancy_ground_truth))

            occupancy_loss, occupancy_result = self.compute_occupancy_256_loss(occupancy_prediction,
                                                                               occupancy_ground_truth)

            hierarchy_losses.update(occupancy_loss)
            # hierarchy_results.update(occupancy_result)

            # Use occupancy prediction to refine sparse voxels
            occupancy_masking_threshold = 0.5
            occupancy_mask = (occupancy_result["256/occupancy"].F > occupancy_masking_threshold).squeeze()
            # print("occupancy_mask: ", occupancy_mask.shape)
            # print("occupancy_mask values: ", torch.unique(occupancy_mask))
            # print("feature_prediction: ", feature_prediction.shape)
            feature_prediction = Me.MinkowskiPruning()(feature_prediction, occupancy_mask)

        return hierarchy_losses, hierarchy_results, feature_prediction

    def compute_occupancy_256_loss(self, prediction, ground_truth) -> Tuple[Dict, Dict]:
        prediction = mask_invalid_sparse_voxels(prediction)
        predicted_coordinates = prediction.C.long()
        # predicted_coordinates[:, 1:] = torch.div(predicted_coordinates[:, 1:], prediction.tensor_stride[0], rounding_mode="floor")
        predicted_coordinates[:, 1:] = predicted_coordinates[:, 1:] // prediction.tensor_stride[0]

        # Get sparse GT values from dense tensor
        ground_truth_values = get_sparse_values(ground_truth.unsqueeze(0), predicted_coordinates)
        # print("prediction.F: ", prediction.F.shape)
        # print("ground_truth_values: ", ground_truth_values.shape)
        loss = self.criterion_occupancy(prediction.F, ground_truth_values, reduction="none")


        if len(loss) > 0:
            loss_mean = loss.mean()
        else:
            loss_mean = 0

        occupancy = Me.MinkowskiSigmoid()(prediction)

        return {"256/occupancy": loss_mean}, {"256/occupancy": occupancy}

    def forward_128(self, predictions, labels ):
        hierarchy_losses = {}
        hierarchy_results = {}

        # print("labels: ", labels.shape)
        # print("label values: ", torch.unique(labels))
        labels_128 = (F.interpolate(labels.unsqueeze(0), size=(128,128,16), mode="nearest"))[0]
        # print("labels_128: ", labels_128.shape)
        # print("labels_128 values: ", torch.unique(labels_128))
        one = torch.ones([1], device=device)
        zero = torch.zeros([1], device=device)
        occupancy_ground_truth = torch.where(labels_128 > 0, one, zero)
        # print("occupancy_gt_128: ", occupancy_ground_truth.shape)
        # print("occupancy_gt_128 values:", torch.unique(occupancy_ground_truth))

        # Compute occupancy loss
        occupancy_prediction: Me.SparseTensor = predictions[0]
        # print("\noccupancy_prediction: ", occupancy_prediction.shape)
        if occupancy_prediction is not None:
            occupancy_loss, occupancy_result = self.compute_occupancy_128_loss(occupancy_prediction, occupancy_ground_truth)
            hierarchy_losses.update(occupancy_loss)
            hierarchy_results.update(occupancy_result)
        
        # Compute semantic loss
        semantic_prediction: Me.SparseTensor = predictions[1]
        # print("\nsemantic_prediction_128: ", semantic_prediction.shape)
        if semantic_prediction is not None:
            semantic_ground_truth = labels_128.long()
            semantic_loss, semantic_result = self.compute_semantic_128_loss(semantic_prediction, semantic_ground_truth)
            hierarchy_losses.update(semantic_loss)
            hierarchy_results.update(semantic_result)
        

        return hierarchy_losses, hierarchy_results

    def compute_occupancy_128_loss(self,prediction: Me.SparseTensor, ground_truth: torch.Tensor):
        prediction = mask_invalid_sparse_voxels(prediction, frustum_dim=[128,128,16])
        predicted_coordinates = prediction.C.long()
        # predicted_coordinates[:, 1:] = torch.div(predicted_coordinates[:, 1:], prediction.tensor_stride[0], rounding_mode="floor")
        predicted_coordinates[:, 1:] = predicted_coordinates[:, 1:] // prediction.tensor_stride[0]
        # Get sparse GT values from dense tensor
        # print("occupancy_ground_truth values: ", ground_truth.shape)
        ground_truth_values = get_sparse_values(ground_truth.unsqueeze(0), predicted_coordinates)
        # print("occupancy_ground_truth values: ", ground_truth_values.shape)
        loss = self.criterion_occupancy(prediction.F, ground_truth_values, reduction="none")
        if len(loss) > 0:
            loss_mean = loss.mean()
        else:
            loss_mean = 0
        occupancy = Me.MinkowskiSigmoid()(prediction)
        return {"128/occupancy": loss_mean}, {"128/occupancy": occupancy}

    def compute_semantic_128_loss(self, prediction: Me.SparseTensor, ground_truth: torch.Tensor) -> Tuple[Dict, Dict]:
        prediction = mask_invalid_sparse_voxels(prediction, frustum_dim=[128,128,16])
        predicted_coordinates = prediction.C.long()
        # predicted_coordinates[:, 1:] = torch.div(predicted_coordinates[:, 1:], prediction.tensor_stride[0], rounding_mode="floor")
        predicted_coordinates[:, 1:] = predicted_coordinates[:, 1:] // prediction.tensor_stride[0]

        # Get sparse GT values from dense tensor
        # print("semantic_ground_truth values_128: ", ground_truth.shape)
        # ground_truth = OneHot(ground_truth)
        # print("semantic_ground_truth values: ", ground_truth.shape)

        ground_truth_values = get_sparse_values(ground_truth.unsqueeze(0), predicted_coordinates)
        # print("semantic_ground_truth values_128: ", ground_truth_values.shape)
        # print("semantic_ground_truth values: ", torch.unique(ground_truth_values))
        # print("semantic_ground_truth values: ", torch.unique(ground_truth_values))

        # print("prediction.F values_128: ", prediction.F.shape)
        # print("prediction.F values: ", torch.unique(prediction.F))
        loss = self.criterion_semantics(prediction.F, ground_truth_values.squeeze(), reduction="none", ignore_index=255)

        if len(loss) > 0:
            loss_mean = loss.mean()
        else:
            loss_mean = 0

        semantic_softmax = Me.MinkowskiSoftmax(dim=1)(prediction)
        # print(semantic_softmax.shape)
        # print(torch.argmax(semantic_softmax.F, 1).unsqueeze(1).shape)
        semantic_labels = Me.SparseTensor(torch.argmax(semantic_softmax.F, 1).unsqueeze(1),
                                          coordinate_map_key=prediction.coordinate_map_key,
                                          coordinate_manager=prediction.coordinate_manager)
        semantic_softmax = None
        return {"128/semantic": loss_mean}, {"128/semantic": semantic_labels}

def get_sparse_values(tensor: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    values = tensor[coordinates[:, 0], :, coordinates[:, 1], coordinates[:, 2], coordinates[:, 3]]
    return values

def mask_invalid_sparse_voxels(grid: Me.SparseTensor, frustum_dim = [256,256,32]) -> Me.SparseTensor:
    # Mask out voxels which are outside of the grid
    valid_mask = (grid.C[:, 1] < frustum_dim[0] - 1) & (grid.C[:, 1] >= 0) & \
                    (grid.C[:, 2] < frustum_dim[1] - 1) & (grid.C[:, 2] >= 0) & \
                    (grid.C[:, 3] < frustum_dim[2] - 1) & (grid.C[:, 3] >= 0)
    num_valid_coordinates = valid_mask.sum()

    if num_valid_coordinates == 0:
        return {}, {}

    num_masked_voxels = grid.C.size(0) - num_valid_coordinates
    grids_needs_to_be_pruned = num_masked_voxels > 0

    # Fix: Only prune if there are invalid voxels
    if grids_needs_to_be_pruned:
        grid = Me.MinkowskiPruning()(grid, valid_mask)

    return grid

class OneHot:
    def __init__(self, num_classes=19):
        self.num_classes = num_classes

    def __call__(self, semantics: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        one_hot = F.one_hot(semantics.long().squeeze(0), self.num_classes).permute(3, 0, 1, 2)
        return one_hot