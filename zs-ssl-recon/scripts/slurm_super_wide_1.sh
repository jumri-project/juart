#!/bin/bash
#SBATCH --account=drecon
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --partition=dc-cpu

# salloc --account drecon --job-name "InteractiveJob" --nodes=1 --time 00:30:00
# srun --ntasks=1 --ntasks-per-node=1 --cpus-per-task=128 /p/project/drecon/zs-ssl-recon/slurm_test.sh

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12340
export WORLD_SIZE=$SLURM_JOB_NUM_NODES

export PYTHONUNBUFFERED=TRUE
export ZS_SSL_RECON_NUM_THREADS=24
export ZS_SSL_RECON_NUM_CPU_PER_WORKER=128
export ZS_SSL_RECON_NUM_GPU_PER_WORKER=0

export ZS_SSL_RECON_FEATURES=512
export ZS_SSL_RECON_SOFTWARE_DIR='/p/project/drecon/zs-ssl-recon'
export ZS_SSL_RECON_DATA_DIR='/p/project/drecon/qrage/datasets/nfft-mc-all'
export ZS_SSL_RECON_MODEL_DIR='/p/project/drecon/qrage/models/saved_models_ddp/ZS_SSL_Model_ddp_all_super_wide_1'

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
source /p/project/drecon/anaconda3/bin/activate drecon

srun --ntasks=$WORLD_SIZE \
     --ntasks-per-node=1 \
     --cpus-per-task=$ZS_SSL_RECON_NUM_CPU_PER_WORKER \
     python /p/project/drecon/zs-ssl-recon/zs_ssl_recon/main_hpc.py
