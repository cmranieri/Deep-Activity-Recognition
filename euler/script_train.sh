#PBS -N train_optflow
#PBS -l ngpus=1
#PBS -l walltime=300:00:00


module load python/3.5.4
module load cuda-toolkit/9.0.176
module load cudnn/7.0

source ~/venv2/bin/activate

cd ~/Deep-Activity-Recognition/src

python TemporalStack.py
