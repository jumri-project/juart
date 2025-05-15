from typing import Tuple

import numpy as np


class AngularTrajectory:
    """
    Generates an angular trajectory for MRI acquisition based on the golden angle.
    """

    def __init__(
        self,
        num_sets: int,
        num_echoes: int,
        angle_0: float = 0.0,
    ) -> None:
        """
        Initialize the AngularTrajectory.

        Parameters:
        ----------
        num_sets : int
            The number of sets in the trajectory.
        num_echoes : int
            The number of echoes per set.
        angle_0 : float, optional
            The initial angle (default is 0.0).
        """
        GA = 2 * np.pi / (1 + np.sqrt(5))  # Golden angle

        self.num_sets = num_sets
        self.num_echoes = num_echoes
        self.increment_phase = GA
        self.increment_set = GA / self.num_sets
        self.increment_echo = GA / self.num_sets / self.num_echoes + np.pi
        self.angle_0 = angle_0

    def angle(
        self,
        index_phase: int,
        index_set: int,
        index_echo: int,
    ) -> float:
        """
        Computes the angle for a given phase, set, and echo index.

        Parameters:
        ----------
        index_phase : int
            The phase index.
        index_set : int
            The set index.
        index_echo : int
            The echo index.

        Returns:
        -------
        float
            The computed angle.
        """
        return (
            self.angle_0
            + index_phase * self.increment_phase
            + index_set * self.increment_set
            + index_echo * self.increment_echo
        )


class PartitionSampling:
    """
    A class to handle partition sampling with autocalibration and acceleration.

    This class is responsible for sampling partitions in a block-based manner
    with the possibility of including autocalibration lines and acceleration.
    """

    def __init__(
        self,
        num_partitions: int,
        num_partitions_per_block: int,
        num_autocalibration_lines: int,
        acceleration_factor: int,
    ) -> None:
        """
        Initialize the PartitionSampling class.

        Parameters:
        ----------
        num_partitions : int
            Total number of partitions.
        num_partitions_per_block : int
            Number of partitions to sample per block.
        num_autocalibration_lines : int
            Number of autocalibration lines used for image reconstruction.
        acceleration_factor : int
            Factor by which the sampling is accelerated.
        """
        self.acceleration_factor = acceleration_factor
        self.autocalibration_width = num_autocalibration_lines
        self.autocalibration_lines = [
            (num_partitions - num_autocalibration_lines) // 2,
            (num_partitions + num_autocalibration_lines) // 2,
        ]

        self.num_partitions = num_partitions
        self.num_actual_partitions = (
            self.num_partitions - self.autocalibration_width
        ) // self.acceleration_factor + self.autocalibration_width

        self.num_partitions_per_block = num_partitions_per_block
        self.num_blocks = self.num_actual_partitions // self.num_partitions_per_block

    def partition(
        self,
        index_block: int,
        index_partition_in_block: int,
    ) -> int:
        """
        Computes the partition index for the given block and partition within the block.

        Parameters:
        ----------
        index_block : int
            The block index.
        index_partition_in_block : int
            The partition index within the block.

        Returns:
        -------
        int
            The calculated partition index.
        """
        index = index_block + index_partition_in_block * self.num_blocks

        if index < self.autocalibration_lines[0] / self.acceleration_factor:
            partition_index = self.acceleration_factor * index
        elif (
            index
            < self.autocalibration_lines[0] / self.acceleration_factor
            + self.autocalibration_width
        ):
            partition_index = (
                index
                - self.autocalibration_lines[0] / self.acceleration_factor
                + self.autocalibration_lines[0]
            )
        else:
            partition_index = (
                self.acceleration_factor
                * (
                    index
                    - self.autocalibration_lines[0] / self.acceleration_factor
                    - self.autocalibration_width
                )
                + self.autocalibration_lines[1]
            )

        return int(partition_index)


def test_partition_sampling(
    sampling: PartitionSampling,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Test the partition sampling by creating a partition array and mask.

    Parameters:
    ----------
    sampling : PartitionSampling
        The PartitionSampling object used to generate the partition indices.

    Returns:
    -------
    tuple of ndarray
        A tuple containing the partition array and mask.
    """
    partition_array = np.zeros((sampling.num_blocks, sampling.num_partitions_per_block))
    partition_mask = np.zeros((256, 160))

    for index_block in range(sampling.num_blocks):
        for index_partition_in_block in range(sampling.num_partitions_per_block):
            partition_index = sampling.partition(index_block, index_partition_in_block)
            partition_array[index_block, index_partition_in_block] = partition_index
            partition_mask[:, int(partition_index)] = index_block + 1

    return partition_array, partition_mask
