#!/bin/bash

export PYTHONUNBUFFERED=TRUE
export SLURM_PROCID0=0
export WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12340
export OMP_NUM_THREADS=24

bash /p/project/drecon/zs-ssl-recon/main_hpc.sh
