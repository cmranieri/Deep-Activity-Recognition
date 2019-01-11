#PBS -N opt_flow
#PBS -l ngpus=1
#PBS -l walltime=300:00:00


module load python/3.5.4

source ~/venv2/bin/activate

cd ~/Deep-Activity-Recognition/src

python opt_flow.py

