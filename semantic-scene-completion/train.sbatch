#!/bin/bash
#SBATCH --job-name="tensorf_experiments12 test"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1,VRAM:16G
#SBATCH --mem=32G
#SBATCH --time=60:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
srun python test.py --config-file /usr/stud/gaj/ssc/semantic-scene-completion/semantic-scene-completion/experiments/027_segpriors-segnet128-continue08/config.yaml --checkpoint /usr/stud/gaj/ssc/semantic-scene-completion/semantic-scene-completion/experiments/027_segpriors-segnet128-continue08/modelFULL-12.pth

# sbatch --constraint="GPU_CC:8.6" train.sbatch 
# Comment (Lukas K.)
# For a normal DL task, it is useful to use multiple CPUs per GPU for example --cpus-per-task=3. This depends on how many CPUs your dataloader (pytorch num_workers) can use effectively.
# For a normal DL task, one should use about 32G of memory. Of course, this depends on the task. Try not to take too much, but if the memory is insufficient the task will get killed.