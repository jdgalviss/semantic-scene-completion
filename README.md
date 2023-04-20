# Large-scale Outdoor Semantic Scene Completion with multi-scal sparse generative networks

Lidar-based Semantic Scene Completion 

## Installation
1. Install Docker
2. Install nvidia-docker
3. Download semantic-kitti dataset: http://www.semantic-kitti.org/
4. Download semantic segmentatiol pretrained model from: https://drive.google.com/drive/folders/1VpY2MCp5i654pXjizFMw0mrmuxC_XboW into the 'semantic-scene-completion/data' folder
5. (Optional) Download our pretrained models to the 'semantic-scene-completion/data' folder
6. Build docker container: docker build -t ssc .

## Prepare Dataset
Run the labels_downscale.py script to create labels for the 1/2 and 1/4 scales using majority vote pooling:
python3 tools/labels_downscale.py
## Training

Run the docker container using our script
source run_docker.sh

Train our semantic scene completion model:
python3 train.py --config-file configs/ssc.yaml

Train the multi-frame semantic scene completion model:
python3 train_multi.py --config-file configs/ssc_multi.yaml

Check Tenosrboard: 
tensorboard --logdir .semantic-scene-completion/experiments 
## Evaluation
Inside the docker container:
python3 eval




<!-- 1. Run jupyter lab inside Docker
```bash
jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser
```

2. Evaluate Semantic-kitti
./evaluate_completion.py --dataset /usr/src/app/data --predictions /usr/src/app/semantic-scene-completion/output/valid --split valid
./evaluate_completion.py --dataset /usr/src/app/data --predictions /usr/src/app/semantic-scene-completion/output/gt --split valid

4. Debug
python3 -m debugpy --listen 131.159.98.103:5678 --wait-for-client -m train
curl -sL https://deb.nodesource.com/setup_18.x | bash
# and install node 
apt-get install nodejs -->

<!-- tensorboard --logdir ./experiments --samples_per_plugin images=100 --port 6007 -->