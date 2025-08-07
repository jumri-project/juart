#!/bin/bash
#SBATCH --account=drecon
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --partition=dc-gpu
#SBATCH --output /p/project1/drecon/qrage-dl/logs/test.out

# chmod +x slurm_super_wide_4.sh
# salloc --account drecon --job-name "InteractiveJob" --nodes=1 --partition=dc-gpu --time 00:30:00
# CUDA_VISIBLE_DEVICES=0,1,2,3 srun --ntasks 4 --cpus-per-task 32 --gpus-per-task 1 bash -c 'echo "Rank: $PMI_RANK CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"' | sort
# bash /p/project1/drecon/qrage-dl/zs-ssl-recon/scripts/slurm_super_wide_4.sh
# sbatch /p/project1/drecon/qrage-dl/zs-ssl-recon/scripts/slurm_super_wide_4.sh
# squeue --account drecon
# tail -f /p/project1/drecon/qrage-dl/logs/test.out

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12340
export WORLD_SIZE=4

export PYTHONUNBUFFERED=TRUE
export ZS_SSL_RECON_NUM_THREADS=24
export ZS_SSL_RECON_NUM_CPU_PER_WORKER=24
export ZS_SSL_RECON_NUM_GPU_PER_WORKER=1

export ZS_SSL_RECON_SOFTWARE_DIR='/p/project1/drecon/qrage-dl/zs-ssl-recon'
export ZS_SSL_RECON_CONFIG_FILE='/p/project1/drecon/qrage-dl/zs-ssl-recon/schemes/hankel_dual_domain_v15_hpc.yaml'

export OMP_NUM_THREADS=$ZS_SSL_RECON_NUM_THREADS
export MKL_NUM_THREADS=$ZS_SSL_RECON_NUM_THREADS
export OPENBLAS_NUM_THREADS=$ZS_SSL_RECON_NUM_THREADS
export BLIS_NUM_THREADS=$ZS_SSL_RECON_NUM_THREADS
export FFT_NUM_THREADS=$ZS_SSL_RECON_NUM_THREADS

echo "NODELIST="$SLURM_JOB_NODELIST
echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE

module --force purge
source /p/project1/drecon/anaconda3/bin/activate drecon
echo "drecon activated"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
srun --ntasks=$WORLD_SIZE \
    python /p/project1/drecon/qrage-dl/zs-ssl-recon/zs_ssl_recon/train/slurm.py
