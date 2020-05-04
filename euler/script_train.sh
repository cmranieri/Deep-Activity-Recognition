#PBS -N video_lstm
#PBS -l ngpus=1
#PBS -l walltime=128:00:00


module load cuda-toolkit/10.2.89
module load cudnn/7.6.5

source ~/venv2/bin/activate

cd ~/Deep-Activity-Recognition/src

python train_model2.py 1
python train_model2.py 2
python train_model2.py 3
python train_model2.py 4
python train_model2.py 5
python train_model2.py 6
python train_model2.py 7
python train_model2.py 8
