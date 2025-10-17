import os

import torch.distributed as dist

from ..utils.parser import options_parser
from .train import train_loop_per_worker


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
