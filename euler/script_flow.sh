#PBS -N opt_flow
#PBS -l ncpus=1
#PBS -l walltime=48:00:00


source ~/venv2/bin/activate

cd ~/Deep-Activity-Recognition/src

python opt_flow.py

