#!/bin/bash

# Unlock ssh keys

eval `ssh-agent -s`
ssh-add ~/.ssh/id_ed25519

# Run batch script

cat slurm_super_wide_1.sh | ssh -i ~/.ssh/id_ed25519 zimmermann9@jureca.fz-juelich.de 'cat | sbatch'
cat slurm_super_wide_2.sh | ssh -i ~/.ssh/id_ed25519 zimmermann9@jureca.fz-juelich.de 'cat | sbatch'
cat slurm_super_wide_4.sh | ssh -i ~/.ssh/id_ed25519 zimmermann9@jureca.fz-juelich.de 'cat | sbatch'
cat slurm_super_wide_8.sh | ssh -i ~/.ssh/id_ed25519 zimmermann9@jureca.fz-juelich.de 'cat | sbatch'
cat slurm_super_wide_16.sh | ssh -i ~/.ssh/id_ed25519 zimmermann9@jureca.fz-juelich.de 'cat | sbatch'

ssh -i ~/.ssh/id_ed25519 zimmermann9@jureca.fz-juelich.de 'squeue -u zimmermann9'

