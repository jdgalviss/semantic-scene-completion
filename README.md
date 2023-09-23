# LiDaR-based Outdoor Semantic Scene Completion with multi-scale sparse generative networks

https://github.com/jdgalviss/semantic-scene-completion/assets/18732666/b15aba05-70bd-4426-bbd2-3f8da3fb2ba7



3D Semantic Scene Completion (SSC) predicts dense 3D geometry and semantic labels from sparse observations. Although effective indoors, existing methods falter in dynamic, sparse outdoor settings. We introduce a technique harnessing a sparse generative neural network, point-cloud segmentation priors, and dense-to-sparse knowledge distillation for single-frame SSC. Our method employs a state-of-the-art semantic segmentation model to predict point cloud features and semantic probabilities from a LiDAR point cloud, which are subsequently fed into a sparse multi-scale generative network to predict geometry and semantics jointly. In addition, we train a multi-frame replica of our model, which takes multiple sequential point clouds as input and apply Knowledge Distillation (KD) to transfer the dense knowledge to the single-frame model. On the SemanticKITTI benchmark, our model achieves a mIoU of 27.1, compared to the previous top scores of 29.5 (S3CNet) and 23.8 (JS3CNet). Furthermore, our approach achieves the highest completion score (60.6) versus 45.6 and 56.6 from S3CNet and JS3CNet respectively.

![approach](https://github.com/jdgalviss/semantic-scene-completion/assets/18732666/5f3be7cf-9ee0-4f66-b1a2-ce7244c79f7f)

## Model
![arch](https://github.com/jdgalviss/semantic-scene-completion/assets/18732666/6e81fe76-6720-428b-8650-8244df8123d8)


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
3. Download the Semantic-KITTI dataset from [http://www.semantic-kitti.org/](http://www.semantic-kitti.org/).
4. Download the semantic segmentation pretrained model from [this Google Drive folder](https://drive.google.com/drive/folders/1VpY2MCp5i654pXjizFMw0mrmuxC_XboW) and place it into the `semantic-scene-completion/data` folder.
5. (Optional) Download our pretrained models from [this Godle Drive Folder](https://drive.google.com/drive/folders/14s675nQXmDg6q4fWgSG5sBfywH7ATm22?usp=sharing) and save them to the `semantic-scene-completion/data` folder.
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

First, run the Docker container using our script (modify the --shm-size parameter depending on your systems' specs and **change the directory `/home/galvis/data/ssc_dataset/dataset/` for the path where your dataset is stored**):

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
python3 eval.py --config-file configs/ssc.yaml --checkpoint data/modelFULL-19.pth
```

## Test

To generate the submission files to for the Semantic-KITTI benchmark, use the following command:

```bash
python3 test.py --config-file configs/ssc.yaml --checkpoint data/modelFULL-19.pth
```
## Results
These are our method's results compared to the semantic-kitti ssc leaderboard:

| Method   | mIoU  | completion |
|----------|-------|------------|
| S3CNet   | **29.5** | 45.6       |
| JS3C-Net | 23.8  | 56.6       |
| LMSCNet  | 17.0  | 55.3       |
| **Ours**     | 27.1  |**60.6**      |

## Acknowledgement
Many thanks to these open-source projects:
- [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)
- [2DPASS](https://github.com/yanx27/2DPASS)
- [sgnn](https://github.com/angeladai/sgnn)
- [semantic-kitti-api](https://github.com/PRBonn/semantic-kitti-api)

<!-- ./validate_submission.py --task completion /usr/src/app/semantic-scene-completion/output/test/sequences.zip /usr/src/app/data -->

<!-- Semantic Scene Completion on the SemanticKITTI dataset
Run jupyter lab inside Docker
jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser
Evaluate Semantic-kitti ./evaluate_completion.py --dataset /usr/src/app/data --predictions /usr/src/app/semantic-scene-completion/output/valid --split valid ./evaluate_completion.py --dataset /usr/src/app/data --predictions /usr/src/app/semantic-scene-completion/output/gt --split valid -->
