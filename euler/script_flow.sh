#PBS -N opt_flow
#PBS -l ngpus=1
#PBS -l walltime=100:00:00


module load python/3.4.3

source ~/venv/bin/activate

cd ~/Deep-Activity-Recognition/src

python opt_flow.py

