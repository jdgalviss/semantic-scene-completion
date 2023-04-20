# Large-scale Outdoor Semantic Scene Completion with multi-scal sparse generative networks

In this work, we formulate a method that leverages a sparse generative neural network with point-cloud segmentation priors and dense to sparse knowledge distillation for single-frame semantic scene completion. Our method employs a state-of-the-art semantic segmentation model to predict point cloud features and semantic probabilities from a LiDAR point cloud, which are subsequently fed into a sparse multi-scale generative network to predict geometry and semantics jointly. In addition, we train a multi-frame replica of our model, which takes multiple sequential point clouds as input and apply Knowledge Distillation (KD) to transfer the dense knowledge to the single-frame model.

## Table of Contents
- [Installation](#installation)
- [Prepare Dataset](#prepare-dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Test](#test)

## Installation

Follow these steps to set up the environment:

1. Install [Docker](https://docs.docker.com/engine/install/ubuntu/)
2. Install [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
3. Download the Semantic-KITTI dataset from [http://www.semantic-kitti.org/](http://www.semantic-kitti.org/) and extract it to the `semantic-scene-completion/data` folder.
4. Download the semantic segmentation pretrained model from [this Google Drive folder](https://drive.google.com/drive/folders/1VpY2MCp5i654pXjizFMw0mrmuxC_XboW) and place it into the `semantic-scene-completion/data` folder.
5. (Optional) Download our pretrained models and save them to the `semantic-scene-completion/data` folder.
6. Build the Docker container using the following command:

```bash
docker build -t ssc .
```
## Prepare Dataset

Run the `labels_downscale.py` script to create labels for the 1/2 and 1/4 scales using majority vote pooling:

```bash
python3 tools/labels_downscale.py
```

## Training

First, run the Docker container using our script (modify the --shm-size parameter depending on your systems' specs):

```bash
source run_docker.sh
```

To train the semantic scene completion model, run the following command inside the Docker container:

```bash
python3 train.py --config-file configs/ssc.yaml
```

To train the multi-frame semantic scene completion model, use the following command:

```bash
python3 train_multi.py --config-file configs/ssc_multi.yaml
```

To monitor the training progress, use TensorBoard with the following command:

```bash
tensorboard --logdir ./semantic-scene-completion/experiments
```

## Evaluation

To evaluate the trained model, run the following command inside the Docker container:

```bash
python3 eval.py --config-file configs/ssc_overfit.yaml --checkpoint data/modelFULL-19.pth
```

## Test

To generate the submission files to for the Semantic-KITTI benchmark, use the following command:

```bash
python3 test.py --config-file configs/ssc_overfit.yaml --checkpoint data/modelFULL-19.pth
```

