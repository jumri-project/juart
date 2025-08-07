#!/bin/bash
#SBATCH --account=drecon
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --output=/p/project1/drecon/qrage-dl/logs/hankel_dual_domain_v19_2.out
#SBATCH --time=12:00:00
#SBATCH --partition=dc-gpu
#SBATCH --gres=gpu:4

# chmod +x /p/project1/drecon/qrage-dl/zs-ssl-recon/scripts/slurm_hankel_dual_domain_v19_2.sh
# salloc --account drecon --job-name "InteractiveJob" --nodes=1 --partition=dc-gpu --time 00:30:00
# srun --ntasks 4 bash -c 'echo "Rank: $PMI_RANK CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"' | sort
# bash /p/project1/drecon/qrage-dl/zs-ssl-recon/scripts/slurm_hankel_dual_domain_v19_2.sh
# sbatch /p/project1/drecon/qrage-dl/zs-ssl-recon/scripts/slurm_hankel_dual_domain_v19_2.sh
# watch squeue --account drecon
# tail -f /p/project1/drecon/qrage-dl/logs/hankel_dual_domain_v19_2.out

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12340

export PYTHONUNBUFFERED=TRUE
export ZS_SSL_RECON_NUM_THREADS=24
export ZS_SSL_RECON_NUM_CPU_PER_WORKER=24
export ZS_SSL_RECON_NUM_GPU_PER_WORKER=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

export ZS_SSL_RECON_SOFTWARE_DIR='/p/project1/drecon/qrage-dl/zs-ssl-recon'

export OMP_NUM_THREADS=$ZS_SSL_RECON_NUM_THREADS
export MKL_NUM_THREADS=$ZS_SSL_RECON_NUM_THREADS
export OPENBLAS_NUM_THREADS=$ZS_SSL_RECON_NUM_THREADS
export BLIS_NUM_THREADS=$ZS_SSL_RECON_NUM_THREADS
export FFT_NUM_THREADS=$ZS_SSL_RECON_NUM_THREADS

LOG_DIR="/p/project1/drecon/qrage-dl/logs"
mkdir -p $LOG_DIR

echo "NODELIST="$SLURM_JOB_NODELIST
echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT

module --force purge
source /p/project1/drecon/anaconda3/bin/activate drecon
echo "drecon activated"

# Set configuration file and run tasks
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
MASTER_PORT=$MASTER_PORT \
ZS_SSL_RECON_CONFIG_FILE='/p/project1/drecon/qrage-dl/zs-ssl-recon/schemes/hankel_dual_domain_v19_2_hpc.yaml' \
srun --ntasks=$SLURM_NTASKS \
     --cpus-per-task=$SLURM_CPUS_PER_TASK \
     --gpus-per-task=1 \
     --distribution=block \
     python /p/project1/drecon/qrage-dl/zs-ssl-recon/zs_ssl_recon/train/slurm.py &

# Wait for all background tasks to complete
wait

sbatch /p/project1/drecon/qrage-dl/zs-ssl-recon/scripts/slurm_hankel_dual_domain_v19_2.sh
