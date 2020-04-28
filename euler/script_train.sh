#PBS -N multi_tcn
#PBS -l ngpus=1
#PBS -l walltime=96:00:00


module load cuda-toolkit/9.0.176
module load cudnn/7.0

source ~/venv2/bin/activate

cd ~/Deep-Activity-Recognition/src

python train_model.py 1
python train_model.py 2
python train_model.py 3
python train_model.py 4
python train_model.py 5
python train_model.py 6
python train_model.py 7
python train_model.py 8
python train_model.py 9
python train_model.py 10
