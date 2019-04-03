#BSUB -q gpuqueue
#BSUB -J hyperparam_attention
#BSUB -m "ld-gpu ls-gpu lt-gpu lp-gpu lg-gpu lv-gpu lu-gpu"
#####BSUB -m "lu-gpu"
#BSUB -q gpuqueue -n 24 -gpu "num=1:j_exclusive=yes:mps=yes"
#BSUB -R "rusage[mem=4] span[ptile=24]"
####BSUB -R V100
#BSUB -W 5:59
#BSUB -o %J.stdout
#BSUB -eo %J.stderr


module add cuda/9.0
python hyperparameter_tuning_attention.py 
