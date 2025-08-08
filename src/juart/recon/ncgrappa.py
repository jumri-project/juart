import time
from collections import defaultdict
from typing import Optional, Union

import torch
from scipy.spatial import KDTree
from torch.nn.functional import grid_sample
from torch_geometric.utils import lexsort
from tqdm import tqdm

from juart.conopt.functional.fourier import (
    fourier_transform_adjoint,
    fourier_transform_forward,
)
from juart.utils import resize


def calibrate_group_worker(args):
    group, calib_signal, tik = args
    group.calibrate(calib_signal=calib_signal, tik=tik)
    return group


class NonCartGrappa:
    def __init__(
        self,
        ktraj: torch.Tensor,
        calib_signal: torch.Tensor,
        img_size: Union[tuple[int, ...], list[int], int],
        kernel_size: Union[tuple[int, ...], list[int], int] = 5,
        do_sift: bool = True,
        device: Optional[torch.device] = None,
        p_norm: float = torch.inf,
        threads: int = -1,
        shift_tol: float = 1e-3,
        tik: float = 1e-3,
        verbose: int = 0,
    ):
        """Non cartesian GRAPPA reconstruction.
        This class is used to perform non-cartesian GRAPPA reconstruction
        on k-space data.


        Parameters
        ----------
        ktraj : torch.Tensor, shape (D+1, N)
            Trajectory in k-space scaled between [-0.5, 0.5]
            with D spatial dimensions and N points. Last dimension of first
            axis is the sampling mask (1.0 for sampled points, 0.0 for unsampled).
        calib_signal : torch.Tensor, shape (C, R, P1, P2)
            Complex ACS.
        kernel_size : tensor-like | int, optional
            Size of GRAPPA kernel.
            If integer, the same size will be use in every dimension.
            By default 5
        device : torch.device, optional
            Device to use for the computation.
            If None, the device of the ktraj tensor will be used.
        p_norm : float, optional
            p-norm to use for the path shape limitation.
            By default torch.inf
        treads : int, optional
            Number of threads to use for the kdtree search.
            By default -1, which means using all available threads.
        verbose: int, optional
            Verbosity level for debugging.
            By default 0 (no output).
        """
        self.calib_signal = calib_signal.clone()
        self.ktraj = ktraj.clone()
        self.threads = threads
        self.num_dim = ktraj.shape[0] - 1
        self.shift_tol = shift_tol
        self.tik = tik
        self.p_norm = p_norm
        self.pad_out = 128

        if device is None:
            device = calib_signal.device
        else:
            self.ktraj = self.ktraj.to(device)
            calib_signal = calib_signal.to(device)

        # Convert kernel size to tuple
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * self.num_dim
        elif isinstance(kernel_size, (list, tuple)):
            assert len(kernel_size) == self.num_dim, (
                f"Kernel size ({kernel_size}) must have the same number of "
                f"spatial dimensions as ktraj ({self.num_dim})"
            )
            kernel_size = tuple(kernel_size)

        if isinstance(img_size, int):
            img_size = (img_size,) * self.num_dim
        elif isinstance(img_size, (list, tuple)):
            assert len(img_size) == self.num_dim, (
                f"Image size ({img_size}) must have the same number of spatial "
                f"dimensions as ktraj ({self.num_dim})"
            )
            img_size = tuple(img_size)

        self.kernel_size = torch.tensor(kernel_size, dtype=torch.int32, device=device)
        self.img_size = torch.tensor(img_size, dtype=torch.int32, device=device)

        self.ktraj[: self.num_dim, :] = (
            self.ktraj[: self.num_dim, :] * self.img_size[:, None]
        )  # scale to cycle/fov

        if verbose > 3:
            print("Scaled kspace trajectory to [cycle/fov] units.")
            if verbose > 4:
                print(
                    f"Maximum k-space coordinates [x, y]: "
                    f"{self.ktraj[: self.num_dim, :].max(dim=1).values}"
                )
                print(
                    f"Minimum k-space coordinates [x, y]: "
                    f"{self.ktraj[: self.num_dim, :].min(dim=1).values}"
                )
                print("")

        # Normalize acs kernel by sum of squares
        # self.calib_signal = calib_signal / torch.sqrt(
        #    torch.sum(calib_signal**2, dim=0, keepdim=True)
        # )
        # if verbose > 3:
        #     print("Normalised calibration signal by root sum of squares.\n")

        # Define limitation of the kernel shape
        self.p_mask = _norm_ball_mask(kernel_size, p=self.p_norm)

        # Get indices of neighbors for each unsampled point in k-space
        if verbose > 2:
            print(f"Number of total locations in k-space: {self.ktraj.shape[1]}")
            print(
                f"Number of sampled locations in k-space: {self.ktraj_sampled.shape[1]}"
            )
            print(
                "Number of unsampled locations in k-space: "
                f"{self.ktraj_unsampled.shape[1]} \n"
            )

        patches = create_patches(
            self.ktraj_sampled,
            self.ktraj_unsampled,
            self.kernel_size,
            p_norm=self.p_norm,
            threads=self.threads,
            verbose=verbose,
            do_sift=do_sift,
        )

        if verbose > 2:
            print(f"Number of patches: {len(patches)}")

        # Group patches by similar shift patterns
        unq_patch_groups = NCG_PatchGroup.create_patch_groups(
            patches=patches,
            tol=self.shift_tol,
            verbose=verbose,
        )

        # Remove empty patches
        unq_patch_groups = [group for group in unq_patch_groups if not group.is_empty]

        if verbose > 2:
            print(f"Number of unique patch groups: {len(unq_patch_groups)} \n")

        # Precompute padded coilpair images and kspace
        ksp_coilpair, coilpair_ind, resize_factor = self._calc_padded_coilpair_kspace()

        # Sequential calibration of groups (no multiprocessing)
        if verbose > 0:
            for group in tqdm(unq_patch_groups, desc="Calibrating patch groups"):
                group.calibrate(
                    ksp_coilpair=ksp_coilpair,
                    coilpair_ind=coilpair_ind,
                    padding_scale=resize_factor,
                    tik=self.tik,
                )
        else:
            for group in unq_patch_groups:
                group.calibrate(
                    ksp_coilpair=ksp_coilpair,
                    coilpair_ind=coilpair_ind,
                    padding_scale=resize_factor,
                    tik=self.tik,
                )

        # Store calibrated groups
        self.unq_patch_groups = unq_patch_groups

    def apply(
        self,
        ksp: torch.Tensor,
        verose: int = 0,
    ):
        num_cha, num_samples = ksp.shape

        if num_samples != self.ktraj.shape[1]:
            raise ValueError(
                f"Input k-space data has {num_samples} samples but the GRAPPA "
                f"object was initialized with {self.ktraj.shape[1]} samples."
            )

        # Apply GRAPPA weights to unsampled k-space points
        for group in tqdm(self.unq_patch_groups, desc="Applying GRAPPA weights"):
            if group.is_empty:  # No neightbours, no weights
                continue
            fill_ind = [self.indices_unsampled[idx] for idx in group.center_ind]

            ksp[:, fill_ind] = group.apply_weights(ksp[:, self.indices_sampled])

        return ksp

    @property
    def sample_mask(self) -> torch.Tensor:
        """Mask of sampled points in k-space."""
        return self.ktraj[-1, :] == 1.0

    @property
    def ktraj_sampled(self) -> torch.Tensor:
        """Sampled k-space trajectory."""
        return self.ktraj[: self.num_dim, self.sample_mask]

    @property
    def ktraj_unsampled(self) -> torch.Tensor:
        """Unsampled k-space trajectory."""
        return self.ktraj[: self.num_dim, ~self.sample_mask]

    @property
    def indices_sampled(self) -> list[int]:
        """Indices of sampled points in k-space."""
        return torch.where(self.sample_mask)[0].tolist()

    @property
    def indices_unsampled(self) -> list[int]:
        """Indices of unsampled points in k-space."""
        return torch.where(~self.sample_mask)[0].tolist()

    def _calc_padded_coilpair_kspace(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes padded coil pair k-space data from the calibration signal.

        This function takes the calibration signal in image space and combines each
        channel with every other channel, forming coil pairs. The indices of these
        coil pairs are returned by `coilpair_ind`. The resulting coil pair images are
        padded to a size of `self.pad_out` in each dimension, then Fourier transformed
        to k-space. Since the image size changes due to padding, a scaling factor for
        the locations in k-space is also computed and returned.

        Returns
        -------
        ksp_coilpair : torch.Tensor
            The k-space data of the padded coil pair combinations.
        coilpair_ind : torch.Tensor
            Indices indicating the channel pairs used for combination.
        resize_factor : torch.Tensor
            Scaling factor for k-space locations due to image size change.
        """
        img_coilpair, coilpair_ind = calc_coil_pair_img(self.calib_signal)
        old_size = img_coilpair.shape[1:]

        # Pad to large FOV
        new_size = (self.pad_out,) * self.num_dim
        dims = tuple(range(1, self.num_dim + 1))
        img_coilpair = resize(img_coilpair, size=new_size, dims=dims)

        # Forward FFT to kspace
        ksp_coilpair = fourier_transform_forward(img_coilpair, axes=dims, norm="ortho")

        # Return also resizing factor
        resize_factor = torch.tensor(
            [(old_size[i] - 1) / (new_size[i] - 1) for i in range(self.num_dim)]
        )

        return ksp_coilpair, coilpair_ind, resize_factor


class NCG_PatchGroup:
    """A group of patches that share the same shift pattern.
    This group holds one set of GRAPPA weights for all patches in the group."""

    def __init__(self, patches: list[Union["FilledPatch", "EmptyPatch"]]):
        if patches == []:
            raise ValueError("Cannot create a patch group from an empty list.")
        # Check that all patches have the same number of neighbors
        num_neighbors_set = {patch.num_neighbors for patch in patches}
        if len(num_neighbors_set) > 1:
            raise ValueError(
                f"All patches in a group must have the same number of neighbors, "
                f"but got counts: {num_neighbors_set}"
            )

        self.patches = patches

    def calibrate(
        self,
        ksp_coilpair: torch.Tensor,
        coilpair_ind: torch.Tensor,
        padding_scale: torch.Tensor,
        tik: float,
    ):
        """Calibrate the patch group weights using the coil pair k-space data."""
        if self.is_empty:
            return None
        else:
            self.patches[0].calibrate(
                ksp_coilpair=ksp_coilpair,
                coilpair_ind=coilpair_ind,
                padding_scale=padding_scale,
                tik=tik,
            )

    def apply_weights(
        self,
        ksp_sampled: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the GRAPPA weights to the k-space neighbors."""
        if self.is_empty:
            raise ValueError("Patch group is empty, cannot apply weights.")
        else:
            ksp_sampled = ksp_sampled[:, torch.tensor(self.neighbor_inds)]

            ksp_filled = torch.zeros(
                (ksp_sampled.shape[0], self.num_patches),
                dtype=ksp_sampled.dtype,
                device=ksp_sampled.device,
            )
            for i in range(self.num_patches):
                ksp_filled[:, i] = self.patches[0].apply_weights(
                    ksp_sampled[:, i, :],
                )
            return ksp_filled

    @property
    def num_patches(self) -> int:
        """Number of patches in the group."""
        return len(self.patches)

    @property
    def num_neighbors(self):
        return self.patches[0].num_neighbors

    @property
    def is_empty(self):
        return all(isinstance(patch, EmptyPatch) for patch in self.patches)

    @property
    def weights(self) -> Union[torch.Tensor, None]:
        """Weights of the patch group."""
        if self.is_empty:
            return None
        else:
            return self.patches[0].weights

    @property
    def center_ind(self) -> list[int]:
        """Indices of the unsampled points in the center of the patches."""
        return [patch.center_ind for patch in self.patches]

    @property
    def center_loc(self) -> torch.Tensor:
        """Locations of the unsampled points in the center of the patches."""
        return torch.stack([patch.center_loc for patch in self.patches], dim=0)

    @property
    def neighbor_inds(self) -> list[list[int]]:
        """Indices of the sampled neighbors in k-space."""
        if self.is_empty:
            return [[] for _ in self.patches]
        return [patch.neighbor_inds for patch in self.patches]

    @property
    def neighbor_locs(self) -> None | torch.Tensor:
        """Locations of the sampled neighbors in k-space."""
        if self.is_empty:
            return None
        return torch.stack([patch.neighbor_locs for patch in self.patches], dim=-1)

    @property
    def num_dim(self) -> int:
        """Number of spatial dimensions."""
        return self.patches[0].num_dim

    @classmethod
    def create_patch_groups(
        cls,
        patches: list[Union["FilledPatch", "EmptyPatch"]],
        tol: float = 1e-6,
        verbose: int = 0,
    ) -> list["NCG_PatchGroup"]:
        """
        Group patches into NCG_PatchGroup instances based on neighbor count
        and shift pattern.

        Parameters
        ----------
        patches : list of NCG_Patch
            All patches to group.
        tol : float
            Tolerance for comparing shift patterns. Default = exact match
            (rounded integers).

        Returns
        -------
        groups : list of NCG_PatchGroup
            Each group contains patches with the same shift layout.
        """
        patch_groups = []

        # Split patches into FilledPatch and EmptyPatch lists
        filled_patches = [p for p in patches if isinstance(p, FilledPatch)]
        empty_patches = [p for p in patches if isinstance(p, EmptyPatch)]

        patch_groups = patch_groups + empty_patches

        # Step 1: Group into buckets by number of neighbors
        neighbor_buckets: defaultdict[int, list[FilledPatch]] = defaultdict(list)

        for patch in filled_patches:
            neighbor_buckets[patch.num_neighbors].append(patch)

        # Sort keys (number of neighbors) for processing
        sorted_keys = sorted(neighbor_buckets.keys())

        # Print the list in verbose mode
        if verbose > 3:
            # Generate the list of [number of neighbors, number of patches]
            neighbor_patch_list_empty = [[0, len(empty_patches)]]
            neighbor_patch_list_filled = [
                [n_neighbors, len(neighbor_buckets[n_neighbors])]
                for n_neighbors in sorted_keys
            ]
            print(
                "[#Neigbors, Num Patches]",
                neighbor_patch_list_empty + neighbor_patch_list_filled,
            )

        # Step 2: For each bucket, subgroup by distance pattern of neighbors
        for n_neighbors in sorted_keys:
            patch_list = neighbor_buckets[n_neighbors]

            # Init matrix with all shifts of all patches
            shift_matrix = torch.zeros(
                (len(patch_list), patch_list[0].num_dim * patch_list[0].num_neighbors),
                dtype=torch.float32,
            )
            # Fill the matrix with shift patterns
            for i, patch in enumerate(patch_list):
                shift_matrix[i] = patch.get_shift_pattern(tol=tol).flatten()

            # Find unique patches
            _, inverse_indices = torch.unique(shift_matrix, dim=0, return_inverse=True)

            # Create groups based on unique shift patterns and the number of neighbors
            grouped_dict = defaultdict(list)
            for patch, group_idx in zip(patch_list, inverse_indices):
                grouped_dict[group_idx.item()].append(patch)

            patch_groups.extend([cls(group) for group in grouped_dict.values()])

        return patch_groups


class PatchBase:
    """Base class for patches in non-cartesian GRAPPA."""

    def __init__(
        self,
        center_ind: int,
        center_loc: torch.Tensor,
        do_sift: bool = True,
        verbose: int = 0,
        device: Optional[Union[torch.device, str]] = None,
    ):
        self.center_ind = center_ind
        self.center_loc = center_loc
        self.do_sift = do_sift
        self.verbose = verbose
        self.device = device if device is not None else center_loc.device
        self.num_dim = center_loc.shape[0]

    @property
    def num_neighbors(self) -> int:
        raise NotImplementedError

    @property
    def is_empty(self) -> bool:
        raise NotImplementedError

    def calibrate(
        self,
        ksp_coilpair: torch.Tensor,
        coilpair_ind: torch.Tensor,
        padding_scale: torch.Tensor,
        tik: float = 0,
    ) -> None:
        """Calibrate the patch weights using the coil pair k-space data."""
        raise NotImplementedError

    def apply_weights(
        self,
        ksp_neighbor: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the GRAPPA weights to the k-space neighbors."""
        raise NotImplementedError


class FilledPatch(PatchBase):
    """
    Patch of a constellation for an unsampled point in the center of the patch and
    its sampled neighbors.
    """

    def __init__(
        self,
        center_ind: int,
        center_loc: torch.Tensor,
        neighbor_locs: torch.Tensor,
        neighbor_inds: list[int],
        device: Optional[Union[torch.device, str]] = None,
        verbose: int = 0,
        do_sift: bool = True,
    ):
        """
        Parameters
        ----------
        center_ind : int
            Index of the unsampled point in the center of the patch.
        center_loc : torch.Tensor, shape (D,)
            Location of the unsampled point in k-space.
        neighbor_inds : list[int]
            Indices of the sampled neighborok bis in k-space.
        neighbor_locs : torch.Tensor, shape (D, N)
            Locations of the sampled neighbors in k-space.
        do_sift : bool, optional
            Sift neighbors to remove duplicates.
        verbose : int, optional
            Verbosity level for debugging. Default is 0 (no output). Maximum is 5.
        """
        super().__init__(
            center_ind=center_ind,
            center_loc=center_loc,
            do_sift=do_sift,
            verbose=verbose,
            device=device,
        )
        self.neighbor_locs = neighbor_locs
        self.neighbor_inds = neighbor_inds
        self.neighbor_dist = neighbor_locs - center_loc[:, None]
        self.weights = None  # To be filled during calibration

        # Perfome sifting
        self.sift_mask = self.get_sift_mask()

        self.neighbor_locs = neighbor_locs[:, self.sift_mask]
        self.neighbor_inds = [
            idx for idx, m in zip(self.neighbor_inds, self.sift_mask) if m
        ]
        self.neighbor_dist = self.neighbor_locs - self.center_loc[:, None]

        # Sort by distance
        self.sort_neighbors()

    def get_sift_mask(self) -> torch.Tensor:
        """
        Return a mask to sift neighbors basted on the distance to the center sample.
        """

        # Get unique distances and their indices
        # Remember that torch.unique is a shit function and does not work
        # as numpy unique
        # This is from https://stackoverflow.com/questions/72001505/how-to-get-unique-elements-and-their-firstly-appeared-indices-of-a-pytorch-tenso
        # and I do not know why, but it works the way it should.

        if not self.do_sift:
            sift_mask = torch.ones(
                (self.num_dim, self.num_neighbors), device=self.device, dtype=torch.bool
            )
            return sift_mask

        else:
            _, idx, counts = torch.unique(
                self.get_shift_pattern(return_int=True),
                dim=1,
                sorted=True,
                return_inverse=True,
                return_counts=True,
            )
            _, ind_sorted = torch.sort(idx, stable=True)
            cum_sum = counts.cumsum(0)
            cum_sum = torch.cat((torch.tensor([0]), cum_sum[:-1]))
            unique_indices = ind_sorted[cum_sum]

            # Create a mask for unique indices
            sift_mask = torch.zeros(
                self.neighbor_locs.shape[1],
                dtype=torch.bool,
                device=self.device,
            )
            sift_mask[unique_indices] = True

            return sift_mask

    def sort_neighbors(self) -> None:
        """Sort neighbors by distance to the center point.
        The distance is rounded to integer values and sorted lexicographically."""
        neighbor_dist_int = self.get_shift_pattern(return_int=True)

        # Sort the neighbors by rounded distance to the center point
        keys = [neighbor_dist_int[dim, :] for dim in reversed(range(self.num_dim))]

        # sort by last dimension first (that's why we reversed the order)
        sorted_indices = lexsort(keys)

        # Apply sorting
        self.neighbor_locs = self.neighbor_locs[:, sorted_indices]
        self.neighbor_inds = [self.neighbor_inds[i] for i in sorted_indices]
        self.neighbor_dist = self.neighbor_dist[:, sorted_indices]

    def get_shift_pattern(
        self,
        return_int: bool = False,
        tol: float = 1e-6,
    ) -> torch.Tensor:
        """Return the integer shift pattern (D, N)."""
        if return_int:
            return torch.round(self.neighbor_dist).to(torch.int32)
        else:
            return torch.round(self.neighbor_dist / tol) * tol

    def calibrate(
        self,
        ksp_coilpair: torch.Tensor,
        coilpair_ind: torch.Tensor,
        padding_scale: torch.Tensor,
        tik: float = 0,
    ) -> None:
        # Get combination of all neighbor shifts and themself
        neigh_shift_comb = self.get_neighbor_shift_combinations()

        # Scale as ksp_coilpair is from padded images
        neigh_shift_comb = neigh_shift_comb / padding_scale[:, None]

        # Interpolate coilpair kspace to neighbor shift combinations
        # as trick for phase shift
        ksp_neigh_shift_comb = bilinear_interpolate_matlab(
            ksp_coilpair, neigh_shift_comb
        )

        AhA, AhB = self.get_block_matrices(
            coil_pair_ind=coilpair_ind, ksp_shift_comb=ksp_neigh_shift_comb
        )

        # Solve least squares problem with Tikhonov regularization
        Eye = torch.eye(AhA.shape[0], dtype=AhA.dtype, device=AhA.device)
        Tik = (tik * self.num_neighbors) * Eye

        # self.weights = (AhA + Tik).pinverse() @ AhB
        self.weights = torch.linalg.lstsq(AhA + Tik, AhB, driver="gelsy").solution

    def get_neighbor_shift_combinations(self) -> torch.Tensor:
        """Return the neighbor shift combinations."""
        neigh_shift_comb = (
            self.neighbor_dist[..., None] - self.neighbor_dist[:, None, :]
        ).reshape(self.num_dim, -1)

        # Concatenate with negative and positive shifts
        neigh_shift_comb = torch.cat(
            [neigh_shift_comb, -self.neighbor_dist, self.neighbor_dist], dim=-1
        )
        return neigh_shift_comb

    def get_block_matrices(
        self, coil_pair_ind, ksp_shift_comb
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_cha = int(coil_pair_ind.max() + 1)

        N = num_cha * self.num_neighbors

        AhA = torch.zeros((N, N), dtype=torch.complex64, device=ksp_shift_comb.device)
        AhB = torch.zeros(
            (N, num_cha), dtype=torch.complex64, device=ksp_shift_comb.device
        )

        # Fill AhA and Ahb matrices
        for enum, (cha_i, cha_j) in enumerate(coil_pair_ind.T):
            start_1 = cha_i * self.num_neighbors
            end_1 = (cha_i + 1) * self.num_neighbors

            start_2 = cha_j * self.num_neighbors
            end_2 = (cha_j + 1) * self.num_neighbors

            block = ksp_shift_comb[enum, : -2 * self.num_neighbors]
            block = block.reshape(self.num_neighbors, self.num_neighbors).T
            vec_1 = ksp_shift_comb[enum, -2 * self.num_neighbors : -self.num_neighbors]
            vec_2 = ksp_shift_comb[enum, -self.num_neighbors :]

            AhA[start_1:end_1, start_2:end_2] = block
            AhB[start_1:end_1, cha_j] = vec_1

            if cha_i != cha_j:
                AhA[start_2:end_2, start_1:end_1] = torch.conj(block.T)
                AhB[start_2:end_2, cha_i] = torch.conj(vec_2)

        return AhA, AhB

    def apply_weights(
        self,
        ksp_neighbor: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the GRAPPA weights to the k-space neighbors."""
        if self.weights is None:
            raise ValueError(
                "Weights have not been calibrated yet. Call `calibrate` first."
            )
        # Neighbours are sifted by indices mask in patch group
        # ksp_neighbor = ksp_neighbor[:, self.sift_mask]

        return ksp_neighbor.reshape(-1) @ self.weights

    @property
    def num_neighbors(self) -> int:
        """Number of neighbors in the patch."""
        return self.neighbor_locs.shape[1]

    @property
    def is_empty(self) -> bool:
        return False

    @classmethod
    def from_indices(
        cls,
        inds_c: list[int],
        ktraj_sampled: torch.Tensor,
        ktraj_unsampled: torch.Tensor,
        do_sift: bool = True,
        verbose: int = 0,
    ) -> "FilledPatch":
        """Create patches from indices of neighbors."""

        instance = cls(
            center_ind=inds_c[0],
            center_loc=ktraj_unsampled[:, inds_c[0]],
            neighbor_inds=inds_c[1:],
            neighbor_locs=ktraj_sampled[:, inds_c[1:]],
            do_sift=do_sift,
            verbose=verbose,
        )
        return instance

    @classmethod
    def from_indices_list(
        cls,
        inds_c_list: list[list[int]],
        ktraj_sampled: torch.Tensor,
        ktraj_unsampled: torch.Tensor,
        do_sift: bool = True,
        verbose: int = 0,
    ) -> list["FilledPatch"]:
        """Create a list of NCG_Patch instances from a list of neighbor indices."""
        cls_list = []

        for inds_c in inds_c_list:
            if len(inds_c) == 1:
                raise ValueError(
                    "Cannot create a FilledPatch from a single unsampled point."
                )

            cls_list.append(
                cls.from_indices(
                    inds_c=inds_c,
                    ktraj_sampled=ktraj_sampled,
                    ktraj_unsampled=ktraj_unsampled,
                    do_sift=do_sift,
                    verbose=verbose,
                )
            )

        return cls_list


class EmptyPatch(PatchBase):
    """Patch of a constellation for an unsampled point in the center of the patch
    without any neighbors."""

    def __init__(
        self,
        center_ind: int,
        center_loc: torch.Tensor,
        device: Optional[Union[torch.device, str]] = None,
        verbose: int = 0,
    ):
        """
        Parameters
        ----------
        center_ind : int
            Index of the unsampled point in the center of the patch.
        center_loc : torch.Tensor, shape (D,)
            Location of the unsampled point in k-space.
        verbose : int, optional
            Verbosity level for debugging. Default is 0 (no output). Maximum is 5.
        """
        super().__init__(
            center_ind=center_ind,
            center_loc=center_loc,
            do_sift=False,
            verbose=verbose,
            device=device,
        )

    def calibrate(
        self,
        ksp_coilpair: torch.Tensor,
        coilpair_ind: torch.Tensor,
        padding_scale: torch.Tensor,
        tik: float = 0,
    ) -> None:
        return None

    @property
    def num_neighbors(self) -> int:
        """Number of neighbors in the patch."""
        return 0

    @property
    def is_empty(self) -> bool:
        """Check if the patch is empty."""
        return True

    @classmethod
    def from_indices(
        cls,
        inds_c: list[int],
        ktraj_unsampled: torch.Tensor,
        verbose: int = 0,
    ) -> "EmptyPatch":
        """Create patches from indices of neighbors."""

        instance = cls(
            center_ind=inds_c[0],
            center_loc=ktraj_unsampled[:, inds_c[0]],
            verbose=verbose,
        )
        return instance

    @classmethod
    def from_indices_list(
        cls,
        inds_c_list: list[list[int]],
        ktraj_unsampled: torch.Tensor,
        verbose: int = 0,
    ) -> list["EmptyPatch"]:
        """Create a list of NCG_Patch instances from a list of neighbor indices."""
        cls_list = []

        for inds_c in inds_c_list:
            if len(inds_c) != 1:
                raise ValueError(
                    "Cannot create a EmptyPatch from unsampled point with neighbors."
                )

            cls_list.append(
                cls.from_indices(
                    inds_c=inds_c,
                    ktraj_unsampled=ktraj_unsampled,
                    verbose=verbose,
                )
            )

        return cls_list


def create_patches(
    ktraj_sampled: torch.Tensor,
    ktraj_unsampled: torch.Tensor,
    kernel_size: Union[torch.Tensor, tuple[int, ...], list[int], int] = 5,
    p_norm: float = torch.inf,
    threads: int = -1,
    do_sift: bool = True,
    verbose: int = 0,
) -> list[Union[FilledPatch, EmptyPatch]]:
    """Create patches from k-space trajectory using a KDTree search."""
    if isinstance(kernel_size, (list, tuple)):
        kernel_size = torch.tensor(kernel_size, dtype=torch.int32)
        if kernel_size.ndim != ktraj_sampled.shape[0]:
            raise ValueError(
                f"Kernel size {kernel_size} must have the same number of "
                f"spatial dimensions as ktraj ({ktraj_sampled.shape[0]})"
            )
    if isinstance(kernel_size, int):
        kernel_size = torch.tensor(
            [kernel_size] * ktraj_sampled.shape[0], dtype=torch.int32
        )

    # Make KDTree search for neighbor indices
    inds_c_list = _get_neighbor_indices(
        traj_sampled=ktraj_sampled,
        traj_unsampled=ktraj_unsampled,
        kernel_size=kernel_size,
        p_norm=p_norm,
        threads=threads,
        verbose=verbose,
    )

    empty_patch_list = EmptyPatch.from_indices_list(
        inds_c_list=[ind for ind in inds_c_list if len(ind) == 1],
        ktraj_unsampled=ktraj_unsampled,
        verbose=verbose,
    )

    filled_patch_list = FilledPatch.from_indices_list(
        inds_c_list=[ind for ind in inds_c_list if len(ind) > 1],
        ktraj_sampled=ktraj_sampled,
        ktraj_unsampled=ktraj_unsampled,
        do_sift=do_sift,
        verbose=verbose,
    )

    # Combine empty and filled patches
    return empty_patch_list + filled_patch_list


def _get_neighbor_indices(
    traj_sampled: torch.Tensor,
    traj_unsampled: torch.Tensor,
    kernel_size: torch.Tensor,
    p_norm: float = torch.inf,
    threads: int = -1,
    verbose: int = 0,
) -> list[list[int]]:
    """
    Get indices of sampled neighbors for each unsampled point in k-space using
    KDTree search.
    For each list of indices, the first element is the index of the unsampled
    point while the other indices are the indices of the sampled neighbors.
    """
    # Start timer
    start_time = time.time()

    if verbose > 2:
        print(
            "------------------------------------------------\n",
            "Perform KDTree search of patch constellations.",
            sep="",
        )

    # Scale the trajectory units of kernel size
    norm_factor = kernel_size / 2.0
    traj_sampled = traj_sampled / norm_factor[:, None]
    traj_unsampled = traj_unsampled / norm_factor[:, None]

    if verbose > 3:
        print(f"Neighbor search distance: \u00b1{norm_factor.tolist()}")
        print(f"Sampled points shape: {traj_sampled.shape}")
        print(f"Unsampled points shape: {traj_unsampled.shape}")

    kdtree_sampled = KDTree(traj_sampled.cpu().numpy().T)
    kdtree_unsampled = KDTree(traj_unsampled.cpu().numpy().T)

    # radius = kernel_size[0] / 2.0  # Radius in cycle/fov units
    radius = 1.0 + 1e-6  # Buffer for floating point precision

    neighbors = kdtree_unsampled.query_ball_tree(kdtree_sampled, r=radius, p=p_norm)

    inds_c = [[i] + n for i, n in enumerate(neighbors)]

    stop_time = time.time() - start_time
    if verbose > 2:
        print(f"Found {len(inds_c)} patches.")
        print(f"KDTree search completed in {(1e3 * stop_time):.3f} ms. \n")

    return inds_c


def _norm_ball_mask(kernel_size: tuple[int, ...], p: float = 2) -> torch.Tensor:
    if p == float("inf"):
        return torch.ones(kernel_size, dtype=torch.bool)

    # Create a grid of coordinates for the kernel
    coords = [torch.arange(s, dtype=torch.float32) for s in kernel_size]
    mesh = torch.meshgrid(*coords, indexing="ij")
    center = [(s - 1) / 2 for s in kernel_size]
    norm = sum(torch.abs(m - c) ** p for m, c in zip(mesh, center)) ** (1 / p)
    radius = (min(kernel_size) - 1) / 2 + 1e-6  # small epsilon for float precision

    mask = norm <= radius
    return mask.bool()


def calc_coil_pair_img(calib_signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the coil product of the calibration signal.

    Parameters
    ----------
    calib_signal : torch.Tensor, shape (C, ...)
        Calibration signal to use for the coil product.

    Returns
    -------
    prod : torch.Tensor, shape (C*(C+1)/2, ...)
        Coil product images, where each image is the product of two channels.

    idx_ij : torch.Tensor, shape (2, C*(C+1)/2)
        Indices of the channels used for the coil product.
    """
    num_cha = calib_signal.shape[0]
    fft_axes = tuple(range(1, calib_signal.ndim))
    calib_img = fourier_transform_adjoint(calib_signal, fft_axes)

    idx_ij = torch.triu_indices(num_cha, num_cha)
    # Follow convention that first axis is changing first
    idx_ij = idx_ij.flip(0)

    prod = torch.conj(calib_img[idx_ij[0]]) * calib_img[idx_ij[1]]

    prod = prod.reshape(-1, *calib_signal.shape[1:])

    return prod, idx_ij


def bilinear_interpolate(
    ksp: torch.Tensor,
    locs: torch.Tensor,
) -> torch.Tensor:
    num_comb, *spatial_dims = ksp.shape
    # Locs must be in range [-1, 1]
    spatial_dims = torch.tensor(spatial_dims, dtype=torch.float32, device=ksp.device)
    locs = locs.clone() / (spatial_dims[:, None] / 2)

    print(locs.min(), locs.max())
    # Grid sample operates on real valued data
    ksp = torch.view_as_real(ksp)
    ksp = ksp.moveaxis(-1, 1)  # Move complex dimension to second position
    ksp = ksp.reshape(-1, *ksp.shape[2:])  # Combine complex dim with channel dim

    # Bring spatioal dim to last dim and flip xyz
    locs = locs.T
    locs[:, 1] = -locs[:, 1]  # Flip z-axis for grid_sample compatibility

    # Add columns and partitons
    # locs = locs[..., None, :] if locs.shape[-1] == 2 else locs[..., None, None, :]

    ksp_locs = grid_sample(
        ksp[None, ...],
        locs[None, None, :, :],
        mode="bilinear",
        align_corners=True,
    )

    # Get back to complex valued data
    ksp_locs = ksp_locs.squeeze(0)
    ksp_locs = ksp_locs.reshape(num_comb, 2, -1)
    ksp_locs = ksp_locs.moveaxis(1, -1)  # Move complex dim to last position
    ksp_locs = torch.view_as_complex(ksp_locs.contiguous())

    return ksp_locs


def bilinear_interpolate_matlab(
    ksp: torch.Tensor,
    locs: torch.Tensor,
):
    # Is identical to matlab when swap coulumn and row axes
    ksp = ksp.swapaxes(1, 2)

    num_cha, *spatial_dims = ksp.shape
    num_dim, num_comb = locs.shape

    multiplier = torch.cumprod(torch.tensor([1, *spatial_dims[:-1]]), dim=0)
    base = torch.zeros(num_comb, dtype=torch.float32, device=ksp.device)
    intp_coeffs = torch.ones(
        (2**num_dim, num_comb), dtype=torch.float32, device=ksp.device
    )
    digit_mask = _binary_digits_mask(num_dim, device=ksp.device)

    iofst = torch.sum(
        digit_mask
        * torch.cumprod(
            torch.tensor([1, *spatial_dims[:-1]], dtype=torch.int32, device=ksp.device),
            dim=0,
        ),
        dim=1,
    )

    for i in range(num_dim):
        base = base + multiplier[i] * torch.floor(locs[i])
        intp_coeffs = intp_coeffs * torch.abs(
            (1 - digit_mask[:, i])[:, None] - torch.remainder(locs[i], 1)
        )

    # Get the center index when flatten ksp
    spatial_dim_tensor = torch.tensor(
        spatial_dims, dtype=torch.int32, device=ksp.device
    )
    ctr_ind = _center_index(spatial_dim_tensor)

    indices = (ctr_ind + iofst + base[:, None]).to(torch.int32)
    indices_flat = indices.reshape(-1)
    ksp_flat = ksp.reshape(num_cha, -1)

    ksp_indices = ksp_flat[:, indices_flat].reshape(num_cha, -1, 4)
    ksp_indices = ksp_indices.swapaxes(1, 2)

    return torch.sum(ksp_indices * intp_coeffs[None, ...], dim=1)


def _binary_digits_mask(
    num_dim: int, device: Optional[Union[torch.device, str]] = None
) -> torch.Tensor:
    if num_dim == 2:
        digit_mask = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [1, 1],
            ],
            dtype=torch.int32,
            device=device,
        )
    elif num_dim == 3:
        digit_mask = torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ],
            dtype=torch.int32,
            device=device,
        )
    else:
        raise ValueError(
            f"Binary digit mask is only implemented for 2 or 3 dimensions, "
            f"but got {num_dim} dimensions."
        )
    return digit_mask


def _center_index(
    spatial_dims: Union[torch.Tensor, tuple[int, ...], list[int]],
) -> torch.Tensor:
    """Compute the center index of a raveled 2D or 3D matrix."""

    if isinstance(spatial_dims, tuple) or isinstance(spatial_dims, list):
        spatial_dims = torch.tensor(spatial_dims, dtype=torch.int32)

    ctr_sub = torch.ceil((spatial_dims) / 2)
    ctr_sub[spatial_dims == 0] = 0

    # Prepare the shape multipliers for each dimension
    multipliers = torch.cat(
        (
            torch.tensor([1], dtype=torch.int32, device=spatial_dims.device),
            torch.cumprod(spatial_dims[:-1], dim=0),
        ),
        dim=0,
    )

    return torch.sum(ctr_sub * multipliers)


# def calc_coil_pair_img(
#   calib_signal: torch.Tensor
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     """Compute the coil product of the calibration signal.

#     Parameters
#     ----------
#     calib_signal : torch.Tensor, shape (C, ...)
#         Calibration signal to use for the coil product.

#     Returns
#     -------
#     prod : torch.Tensor, shape (C*(C+1)/2, ...)
#         Coil product images, where each image is the product of two channels.

#     idx_ij : torch.Tensor, shape (2, C*(C+1)/2)
#         Indices of the channels used for the coil product.
#     """
#     num_cha = calib_signal.shape[0]
#     fft_axes = tuple(range(1, calib_signal.ndim))
#     calib_img = fourier_transform_adjoint(calib_signal, fft_axes)

#     idx_ij = torch.triu_indices(num_cha, num_cha)
#     # Follow convention that first axis is changing first
#     idx_ij = idx_ij.flip(0)

#     prod = torch.conj(calib_img[idx_ij[0]]) * calib_img[idx_ij[1]]

#     prod = prod.reshape(-1, *calib_signal.shape[1:])

#     return prod, idx_ij
