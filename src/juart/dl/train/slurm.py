import os
import sys

if os.getenv("ZS_SSL_RECON_SOFTWARE_DIR") is not None:
    sys.path.insert(0, os.getenv("ZS_SSL_RECON_SOFTWARE_DIR"))

import torch.distributed as dist
from zs_ssl_recon.train.train import train_loop_per_worker
from zs_ssl_recon.utils.parser import options_parser


def main():
    print(
        f"SLURM_PROCID {os.environ['SLURM_PROCID']} of WORLD_SIZE {os.environ['SLURM_NTASKS']} - Initialize process group ..."
    )

    options = options_parser()

    dist.init_process_group(
        backend="gloo",
        world_size=int(os.environ["SLURM_NTASKS"]),
        rank=int(os.environ["SLURM_PROCID"]),
    )

    global_rank = int(dist.get_rank())
    world_size = int(dist.get_world_size())
    print(f"Rank {global_rank} of {world_size} - Starting training ...")

    train_loop_per_worker(options)


if __name__ == "__main__":
    main()
