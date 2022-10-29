from typing import Dict, Any, Tuple, List

import torch
import torch.nn as nn
from configs import config
from torch.nn import functional as F
from model import UNetSparse, GeometryHeadSparse, ClassificationHeadSparse
import MinkowskiEngine as Me
import sys
sys.path.append("..") 
from structures import collect
device = torch.device("cuda:0")


class MyModel(nn.Module):
    def __init__(self,num_output_channels=8,unet_features=4,resnet_blocks=1):
        super().__init__()
        self.model = UNetSparse(num_output_channels, unet_features)
        
        # Heads
        self.occupancy_256_head = GeometryHeadSparse(num_output_channels, 1, resnet_blocks)
        # self.occupancy_256_head = self.occupancy_256_head

        # Semantic head
        self.semantic_head = ClassificationHeadSparse(num_output_channels,
                                                        config.SEGMENTATION.NUM_CLASSES,
                                                        resnet_blocks)
        # self.semantic_head = self.semantic_head

        # Criterions
        self.criterion_occupancy = F.binary_cross_entropy_with_logits
        self.criterion_semantics = F.cross_entropy  #nn.CrossEntropyLoss(reduction="none")



    def forward(self, targets, weights):
        # Put coordinates in the right order
        complet_coords = collect(targets, "complet_coords").squeeze()
        complet_coords = complet_coords[:, [0, 3, 2, 1]]
        complet_coords[:, 3] += 1  # TODO SemanticKITTI will generate [256,256,31]
        complet_features = collect(targets, "complet_features")
        # complet_coords[:, 0] += 1 
        # Transform to sparse tensor
        complet_coords = Me.SparseTensor(features=complet_features.transpose(0,1).type(torch.FloatTensor).to(device),
                            coordinates=complet_coords.int().to(device),
                            quantization_mode=Me.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE)
        # print("sparse_coords: ",sparse_coords.shape)
        # complet_invalid = collect(targets,"complet_invalid")
        # complet_valid = torch.logical_not(complet_invalid)
        # complet_valid = F.max_pool3d(complet_invalid.float(), kernel_size=2, stride=4).bool()
        # complet_valid = torch.ones_like(complet_valid).bool()
        # complet_valid = (torch.rand((1,64,64,8), dtype=torch.float) < 0.9).to(device).bool()
        # complet_valid = torch.rand_like(complet_valid, dtype=torch.float) > 0.5
        # print("completion_valid values: ", torch.max(complet_valid))
        # print("completion_valid values: ", torch.min(complet_valid))
        complet_valid = torch.ones(1,64,64,8).to(device).bool()
        # return {}, {}

        # print("complet_valid_64 values: ", torch.unique(complet_valid_64))
        # print("complet_valid_64: ",torch.sum(complet_valid_64))
        # print("complet_valid shape:", complet_valid.shape)
        # one = torch.ones([1], device=device)
        # zero = torch.zeros([1], device=device)
        # occupied_voxels = torch.where( (F.interpolate(complet_labels.unsqueeze(0), size=(64,64,8), mode="nearest"))[0] > 0, one, zero)
        # print("occupied_voxels: ", occupied_voxels.shape)
        # print("occupied_voxels values: ", torch.unique(occupied_voxels))
        # print("occupied_voxels: ",torch.sum(occupied_voxels))
        
        # complet_valid_64 = torch.logical_and(complet_valid_64, occupied_voxels)
        # print("\complet_valid_64: ", complet_valid_64.shape)
        # print("complet_valid_64 values: ", torch.unique(complet_valid_64))
        # print("complet_valid_64: ",torch.sum(complet_valid_64))

        # Forward pass through model
        unet_output = self.model(complet_coords, batch_size=1, valid_mask=complet_valid)
        predictions = unet_output.data
        
        losses = {}
        results = {}

        # level-64
        if predictions[2] is None:
            return {}, {}
        losses_64, results_64 = self.forward_64(predictions[2], targets, complet_valid, weights)
        
        losses.update(losses_64)
        results.update(results_64)

        # level-128
        if predictions[1] is None:
            return {}, {}
        losses_128, results_128 = self.forward_128(predictions[1], targets, weights)
        
        losses.update(losses_128)
        results.update(results_128)

        #leve-256
        if predictions[0] is None:
            return losses, results
        
        # Occupancy
        losses_256, results_256, features_256 = self.forward_256(predictions[0], targets)
        losses.update(losses_256)
        results.update(results_256)

        # Semantic
        if config.GENERAL.LEVEL == "FULL":
            losses_output, results_output = self.forward_output(features_256, targets, weights)
            losses.update(losses_output)
            results.update(results_output)


        return losses, results

    def forward_output(self, predictions: List[Me.SparseTensor], targets, weights) -> Tuple[Dict, Dict]:
        hierarchy_losses = {}
        hierarchy_results = {}

        # Semantics
        semantic_prediction = self.semantic_head(predictions)
        # print("\nsemantic_prediction: ", semantic_prediction.shape)
        if semantic_prediction is not None:
            semantic_ground_truth: torch.LongTensor = collect(targets, "complet_labels").long()
            semantic_loss, _ = self.compute_semantic_256_loss(semantic_prediction, semantic_ground_truth, weights)
            hierarchy_losses.update(semantic_loss)
            # hierarchy_results.update(semantic_result)
        return hierarchy_losses, hierarchy_results

    def compute_semantic_256_loss(self, prediction: Me.SparseTensor, ground_truth: torch.Tensor, weights: torch.Tensor) -> Tuple[Dict, Dict]:
        prediction = mask_invalid_sparse_voxels(prediction)
        predicted_coordinates = prediction.C.long()
        # predicted_coordinates[:, 1:] = torch.div(predicted_coordinates[:, 1:], prediction.tensor_stride[0], rounding_mode="floor")
        predicted_coordinates[:, 1:] = predicted_coordinates[:, 1:] // prediction.tensor_stride[0]

        # Get sparse GT values from dense tensor
        ground_truth = get_sparse_values(ground_truth.unsqueeze(0), predicted_coordinates)
        # print("semantic_ground_truth values_256: ", ground_truth_values.shape)
        # print("semantic_ground_truth values: ", torch.unique(ground_truth_values))

        # print("prediction.F values_256: ", prediction.F.shape)
        # print("prediction.F values: ", torch.unique(prediction.F))


        loss = self.criterion_semantics(prediction.F, ground_truth.squeeze(), weight=weights, reduction="none", ignore_index=255)

        # Get sparse weighting values from dense tensor

        if len(loss) > 0:
            loss_mean = loss.mean()
        else:
            loss_mean = 0

        # semantic_softmax = Me.MinkowskiSoftmax(dim=1)(prediction)
        # semantic_labels = Me.SparseTensor(torch.argmax(semantic_softmax.F, 1).unsqueeze(1),
        #                                   coordinate_map_key=prediction.coordinate_map_key,
        #                                   coordinate_manager=prediction.coordinate_manager)
        # print("semantic_labels: ", semantic_labels.shape)
        return {"semantic_256": loss_mean}, {"semantic_256": prediction}

    def forward_256(self, predictions, targets):
        hierarchy_losses = {}
        hierarchy_results = {}

        feature_prediction: Me.SparseTensor = predictions
        feature_prediction = mask_invalid_sparse_voxels(predictions)
        torch.cuda.empty_cache()
        # For low memory: TODO(jdgalviss): Is there a memory leak?
        # shape = torch.Size([1, 1, 128, 128, 16])
        # min_coordinate = torch.IntTensor([0, 0, 0]).to(device)
        # occupancy_128, _, _ = occupancy_128.dense(shape, min_coordinate=min_coordinate)
        # print("occupancy_128: ", occupancy_128.shape)
        # occupancy_128 = (F.interpolate(occupancy_128, size=(256,256,32), mode="nearest"))[0]
        # occupancy_128 = occupancy_128 > 0.5
        # print("occupancy_128: ", occupancy_128.shape)

        # mask = torch.rand(feature_prediction.F.shape[0]) < 0.5
        # # # print("mask values: ", torch.unique(mask))
        # feature_prediction = Me.MinkowskiPruning()(feature_prediction, mask.to(device))
        # print("feature_prediction: ",feature_prediction.shape)


        if feature_prediction is not None:
            # occupancy at 256
            # print("feature_prediction: ", feature_prediction.shape)

            occupancy_prediction = self.occupancy_256_head(feature_prediction)
            # print("occupancy_prediction: ", occupancy_prediction.shape)
            occupancy_ground_truth = collect(targets, "complet_occupancy")
            # print("occupancy_gt: ", occupancy_ground_truth.shape)
            # print("occupancy_gt values:", torch.unique(occupancy_ground_truth))

            occupancy_loss, occupancy_result = self.compute_occupancy_256_loss(occupancy_prediction,
                                                                               occupancy_ground_truth)

            hierarchy_losses.update(occupancy_loss)
            # hierarchy_results.update(occupancy_result)

            # Use occupancy prediction to refine sparse voxels
            occupancy_masking_threshold = 0.5
            occupancy_mask = (occupancy_result["occupancy_256"].F > occupancy_masking_threshold).squeeze()
            # print("occupancy_mask: ", occupancy_mask.shape)
            # print("occupancy_mask values: ", torch.unique(occupancy_mask))
            feature_prediction = Me.MinkowskiPruning()(feature_prediction, occupancy_mask)
            # print("predictions: ", feature_prediction.shape)

        return hierarchy_losses, hierarchy_results, feature_prediction

    def compute_occupancy_256_loss(self, prediction, ground_truth) -> Tuple[Dict, Dict]:
        prediction = mask_invalid_sparse_voxels(prediction)
        predicted_coordinates = prediction.C.long()
        # predicted_coordinates[:, 1:] = torch.div(predicted_coordinates[:, 1:], prediction.tensor_stride[0], rounding_mode="floor")
        predicted_coordinates[:, 1:] = predicted_coordinates[:, 1:] // prediction.tensor_stride[0]

        # Get sparse GT values from dense tensor
        # print("ground_truth: ", torch.sum(ground_truth))
        ground_truth = get_sparse_values(ground_truth.unsqueeze(0), predicted_coordinates)
        # print("prediction.F: ", prediction.F.shape)
        # print("ground_truth_values: ", ground_truth_values.shape)
        loss_mean = self.criterion_occupancy(prediction.F, ground_truth, reduction="mean")


        # if len(loss) > 0:
        #     loss_mean = loss.mean()
        # else:
        #     loss_mean = 0

        prediction = Me.MinkowskiSigmoid()(prediction)
        # print("occupancy_prediction: ", prediction.shape)
        return {"occupancy_256": loss_mean}, {"occupancy_256": prediction}

    def forward_128(self, predictions, targets, weights ):
        hierarchy_losses = {}
        hierarchy_results = {}

        # Compute occupancy loss
        occupancy_prediction: Me.SparseTensor = predictions[0]
        # print("\noccupancy_prediction: ", occupancy_prediction.shape)
        if occupancy_prediction is not None:
            occupancy_ground_truth = collect(targets, "complet_occupancy_128")
            occupancy_loss, _ = self.compute_occupancy_128_loss(occupancy_prediction, occupancy_ground_truth)
            hierarchy_losses.update(occupancy_loss)
            # hierarchy_results.update(occupancy_result)
        
        # Compute semantic loss
        semantic_prediction: Me.SparseTensor = predictions[1]
        # print("\nsemantic_prediction_128: ", semantic_prediction.shape)
        if semantic_prediction is not None:
            semantic_ground_truth = collect(targets, "complet_labels_128").long()
            semantic_loss, _ = self.compute_semantic_128_loss(semantic_prediction, semantic_ground_truth, weights)
            hierarchy_losses.update(semantic_loss)
            # hierarchy_results.update(semantic_result)
        

        return hierarchy_losses, hierarchy_results

    def compute_occupancy_128_loss(self,prediction: Me.SparseTensor, ground_truth: torch.Tensor):
        prediction = mask_invalid_sparse_voxels(prediction, frustum_dim=[128,128,16])
        predicted_coordinates = prediction.C.long()
        # predicted_coordinates[:, 1:] = torch.div(predicted_coordinates[:, 1:], prediction.tensor_stride[0], rounding_mode="floor")
        predicted_coordinates[:, 1:] = predicted_coordinates[:, 1:] // prediction.tensor_stride[0]
        # Get sparse GT values from dense tensor
        ground_truth_values = get_sparse_values(ground_truth.unsqueeze(0), predicted_coordinates)
        loss = self.criterion_occupancy(prediction.F, ground_truth_values, reduction="none")
        if len(loss) > 0:
            loss_mean = loss.mean()
        else:
            loss_mean = 0
        prediction = Me.MinkowskiSigmoid()(prediction)
        return {"occupancy_128": loss_mean}, {"occupancy_128": prediction}

    def compute_semantic_128_loss(self, prediction: Me.SparseTensor, ground_truth: torch.Tensor, weights: torch.Tensor) -> Tuple[Dict, Dict]:
        prediction = mask_invalid_sparse_voxels(prediction, frustum_dim=[128,128,16])
        predicted_coordinates = prediction.C.long()
        # predicted_coordinates[:, 1:] = torch.div(predicted_coordinates[:, 1:], prediction.tensor_stride[0], rounding_mode="floor")
        predicted_coordinates[:, 1:] = predicted_coordinates[:, 1:] // prediction.tensor_stride[0]

        # Get sparse GT values from dense tensor
        # print("semantic_ground_truth values_128: ", ground_truth.shape)
        # ground_truth = OneHot(ground_truth)
        # print("semantic_ground_truth values: ", ground_truth.shape)

        ground_truth = get_sparse_values(ground_truth.unsqueeze(0), predicted_coordinates)
        # print("semantic_ground_truth values_128: ", ground_truth_values.shape)
        # print("semantic_ground_truth values: ", torch.unique(ground_truth_values))
        # print("semantic_ground_truth values: ", torch.unique(ground_truth_values))

        # print("prediction.F values_128: ", prediction.F.shape)
        # print("prediction.F values: ", torch.unique(prediction.F))
        loss = self.criterion_semantics(prediction.F, ground_truth.squeeze(), weight=weights, reduction="none", ignore_index=255)

        if len(loss) > 0:
            loss_mean = loss.mean()
        else:
            loss_mean = 0

        prediction = Me.MinkowskiSoftmax(dim=1)(prediction)
        # print(semantic_softmax.shape)
        # print(torch.argmax(semantic_softmax.F, 1).unsqueeze(1).shape)
        prediction = Me.SparseTensor(torch.argmax(prediction.F, 1).unsqueeze(1),
                                          coordinate_map_key=prediction.coordinate_map_key,
                                          coordinate_manager=prediction.coordinate_manager)
        # semantic_softmax = None
        return {"semantic_128": loss_mean}, {"semantic_128": prediction}

    def forward_64(self, predictions, targets, valid_mask, weights) -> Tuple[Dict, Dict]:
        hierarchy_losses = {}
        hierarchy_results = {}

        # Occupancy 64
        occupancy_prediction = predictions[0]
        occupancy_ground_truth = collect(targets, "complet_occupancy_64")
        occupancy_loss, occupancy_result = self.compute_occupancy_64_loss(occupancy_prediction, occupancy_ground_truth,
                                                                          valid_mask)
        hierarchy_losses.update(occupancy_loss)
        hierarchy_results.update(occupancy_result)

        # Semantic 64
        semantic_prediction = predictions[1]
        semantic_ground_truth = collect(targets, "complet_labels_64")
        semantic_loss, semantic_result = self.compute_semantic_64_loss(semantic_prediction, semantic_ground_truth,
                                                                       valid_mask, weights)
        hierarchy_losses.update(semantic_loss)
        hierarchy_results.update(semantic_result)

        return hierarchy_losses, hierarchy_results

    def compute_occupancy_64_loss(self, prediction: torch.Tensor, ground_truth: torch.Tensor,
                                  mask: torch.Tensor) -> Tuple[Dict, Dict]:
        loss = self.criterion_occupancy(prediction[0], ground_truth, reduction="none")

        # Only consider loss within the camera frustum
        loss = torch.masked_select(loss, mask)

        if len(loss) > 0:
            loss_mean = loss.mean()
        else:
            loss_mean = 0

        occupancy_probability = torch.sigmoid(prediction)
        occupancy = torch.masked_fill(occupancy_probability, mask == False, 0.0)  # mask out regions outside of frustum

        return {"occupancy_64": loss_mean}, {"occupancy_64": occupancy}

    def compute_semantic_64_loss(self, prediction: torch.Tensor, ground_truth: torch.Tensor,
                                  mask: torch.Tensor, weights: torch.Tensor) -> Tuple[Dict, Dict]:

        loss = self.criterion_semantics(prediction, ground_truth.long(), weight=weights, reduction="none", ignore_index=255)
        # Only consider loss within the camera frustum
        loss = torch.masked_select(loss, mask)

        if len(loss) > 0:
            loss_mean = loss.mean()
        else:
            loss_mean = 0

        prediction = torch.argmax(prediction, dim=1)
        prediction = torch.masked_fill(prediction, mask == False, 0)

        return {"semantic_64": loss_mean}, {"semantic_64": prediction}

    def inference(self, inputs):
        # Put coordinates in the right order
        complet_coords = collect(inputs,"complet_coords").squeeze()
        complet_coords = complet_coords[:, [0, 3, 2, 1]]
        complet_coords[:, 3] += 1  # TODO SemanticKITTI will generate [256,256,31]
        complet_features = collect(inputs, "complet_features")
        # complet_coords[:, 0] += 1 
        # Transform to sparse tensor
        sparse_coords = Me.SparseTensor(features=complet_features.transpose(0,1).type(torch.FloatTensor).to(device),
                            coordinates=complet_coords.int().to(device),
                            quantization_mode=Me.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE)
        # print("sparse_coords: ",sparse_coords.shape)
        # Transform to sparse tensor
        
        # print("sparse_coords: ", sparse_coords.shape)
        # Forward pass through model
        # complet_valid = torch.logical_not(complet_invalid)
        # complet_invalid = collect(inputs,"complet_invalid")

        # complet_valid = F.max_pool3d(complet_invalid.float(), kernel_size=2, stride=4).bool()

        complet_valid = torch.ones(1,64,64,8).to(device).bool()
        unet_output = self.model(sparse_coords, batch_size=1, valid_mask=complet_valid)
        feature_prediction = unet_output.data[0]
        # print("feature_prediction: ", feature_prediction.shape)
        
        feature_prediction = mask_invalid_sparse_voxels(feature_prediction)
        occupancy_prediction = self.occupancy_256_head(feature_prediction)
        occupancy_prediction = mask_invalid_sparse_voxels(occupancy_prediction)
        occupancy_prediction = Me.MinkowskiSigmoid()(occupancy_prediction)
        occupancy_mask = (occupancy_prediction.F > 0.5).squeeze()
        prediction_pruned = Me.MinkowskiPruning()(feature_prediction, occupancy_mask)
        # print("prediction_pruned: ", prediction_pruned.shape)

        semantic_prediction = self.semantic_head(prediction_pruned)
        semantic_prediction = mask_invalid_sparse_voxels(semantic_prediction)
        semantic_softmax = Me.MinkowskiSoftmax(dim=1)(semantic_prediction)
        semantic_labels = Me.SparseTensor(torch.argmax(semantic_softmax.F, 1).unsqueeze(1),
                                          coordinate_map_key=semantic_prediction.coordinate_map_key,
                                          coordinate_manager=semantic_prediction.coordinate_manager)

        # print("occupancy_prediction: ", occupancy_prediction.shape)
        # print("occupancy_mask: ", occupancy_mask.shape)

        # print("semantic_prediction: ", semantic_prediction.shape)
        # print("semantic_labels: ", semantic_labels.shape)

        results = {"occupancy_256": occupancy_prediction, "semantic_prediction":semantic_prediction, "semantic_256": semantic_labels}
        return results

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