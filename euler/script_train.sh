#PBS -N train
#PBS -l ngpus=1
#PBS -l walltime=100:00:00


module load python/3.4.3
module load cuda-toolkit/9.0.176
module load cudnn/7.0

source ~/venv/bin/activate

cd ~/Deep-Activity-Recognition/src

python Trainer.py

