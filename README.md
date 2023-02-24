# semantic-scene-completion
Semantic Scene Completion on the SemanticKITTI dataset


1. Run jupyter lab inside Docker
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
apt-get install nodejs

tensorboard --logdir ./experiments --samples_per_plugin images=100