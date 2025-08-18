import math
import time
from collections import Counter, defaultdict
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


class NonCartesianGrappa:
    """Non-Cartesian GRAPPA reconstruction class."""

    def __init__(
        self,
        ktraj: torch.Tensor,
        calib_signal: torch.Tensor,
        kernel_size: torch.Tensor,
        do_sift: bool = True,
        tik: float = 0.0,
        p_norm: float = torch.inf,
        shift_tolerance: float = 1e-6,
        verbose: int = 0,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """_summary_

        Parameters
        ----------
        ktraj : torch.Tensor, shape (D+1, N)
            The trajectory in k-space with D spatial dimensions and N samples
            in units of cycles/FOV.
            The last dimension is a sampling mask with 1 for sampled locations.
        calib_signal : torch.Tensor, shape (C, spatial_dims)
            Autocalibration signal with C coils and D spatial dimensions.
        kernel_size : torch.Tensor
            Kernel size of shape (D,) in cycles/FOV units.
            with int in all dimensions.
        do_sift : bool, optional
            Set True to sift the patches, by default True
        tik : float, optional
            Tikhonov regularisation parameter, by default 0.0
        p_norm : float, optional
            For KDTree search, by default torch.inf
        shift_tolerance : float, optional
            Shift tolerance when looking for similar patches, by default 1e-6
        verbose : int, optional
            Verbosity level between 0 and 5, by default 0
        device : Optional[Union[torch.device, str]], optional
            _description_, by default None
        """

        self.ktraj = ktraj
        self.num_dim = ktraj.shape[0] - 1
        self.shift_tol = shift_tolerance
        self.do_sift = do_sift
        self.verbose = verbose
        self.tik = tik
        self.p_norm = p_norm
        self.pad_out = 128
        self.kernel_size = kernel_size

        if device is None:
            self.device = ktraj.device

        if calib_signal.ndim - 1 != self.num_dim:
            raise ValueError(
                "Dimension missmatch of calibration signal and k-space trajectory.",
                f"Trajectory has {self.num_dim} spatial dimensions",
                f"Calibration signal has {calib_signal.ndim - 1} spatial dimensions.",
            )

        self.calib_signal = calib_signal

        self.num_cha = self.calib_signal.shape[0]

        if kernel_size.shape[0] != self.num_dim:
            raise ValueError(
                f"Kernel size must have {self.num_dim} dimensions, "
                f"but got {kernel_size.shape[0]}."
            )

        if verbose > 2:
            print(
                "------------------------------------------------\n",
                "Non-Cartesian GRAPPA reconstruction initialized.",
                sep="",
            )
            print(f"Kernel size: {kernel_size.tolist()}")
            print(f"Number of dimensions: {self.num_dim}")
            print(f"Device: {self.device}")
            print(f"Shift tolerance: {self.shift_tol}")
            print(f"Number of locations in k-space: {ktraj.shape[1]}")
            print("Number of sampled locations in k-space:", int(ktraj[-1, :].sum()))
            print(
                "Number of unsampled locations in k-space:",
                int(ktraj.shape[1] - ktraj[-1, :].sum()),
            )

        # Create patches from ktraj
        self.patches = create_patches(
            ktraj=ktraj,
            kernel_size=kernel_size,
            p_norm=p_norm,
            verbose=verbose,
        )

        # Group similar patches into PatchGroups
        self.patch_groups = PatchGroup.create_patchgroups(
            patches=self.patches,
            shift_tolerance=self.shift_tol,
            verbose=verbose,
        )

        # Calibrate the patch weights
        self.calibrate()

    @property
    def coilpair_indices(self) -> torch.Tensor:
        """Return the indices channel pairs as a tensor of shape (2, C*(C+1)/2)."""
        ind_ij = torch.triu_indices(self.num_cha, self.num_cha)
        ind_ij = ind_ij.flip(0)
        return ind_ij

    @property
    def pad_resize_factor(self) -> torch.Tensor:
        """Returns the factor by which the k-space distances
        of the ACS region change due to the padding with self.pad_out as
        torch.Tensor, shape (num_dim,)
        """
        new_size = (self.pad_out,) * self.num_dim
        old_size = self.calib_signal.shape[1:]

        # Calculate the resize factor for each dimension
        resize_factor = torch.tensor(
            [old_size[i] - 1 / new_size[i] - 1 for i in range(self.num_dim)],
            device=self.device,
        )

        return resize_factor

    @property
    def padded_coilpair_ksp(self) -> torch.Tensor:
        """Return a oversampled version of the paired coil k-space data
        as tensor of shape (C*(C+1)/2, spatial_dims).
        """

        fft_axes = tuple(range(1, self.calib_signal.ndim))
        calib_img = fourier_transform_adjoint(self.calib_signal, axes=fft_axes)

        # Combine the channels
        indices = self.coilpair_indices
        coilpair_img = torch.conj(calib_img[indices[0]]) * calib_img[indices[1]]
        coilpair_img = coilpair_img.reshape(-1, *calib_img.shape[1:])

        # Pad the calibration signal in the spatial dimensions
        new_size = (self.pad_out,) * self.num_dim
        coilpair_img = resize(coilpair_img, size=new_size, dims=fft_axes)

        # Transform back to k-space
        pad_coilpair_ksp = fourier_transform_forward(coilpair_img, axes=fft_axes)

        return pad_coilpair_ksp

    def calibrate(self):
        """Calibrate the patch weights."""
        if self.verbose > 0:
            print("------------------------------------------------")
            print("Calibrating patches...")

        t0 = time.time()
        # Save some arrays for computation efficiency
        coilpair_indices = self.coilpair_indices
        pad_coilpair_ksp = self.padded_coilpair_ksp
        resize_factors = self.pad_resize_factor

        if self.verbose > 2:
            print(f"Time for preparation: {(time.time() - t0) * 1e3} ms")
            print(f"Found {coilpair_indices.shape[1]} coil pairs.")
            print(
                "Calculated padded coilpair ACS with",
                f" shape {list(pad_coilpair_ksp.shape)}.",
            )
            print(f"Resize k-space locations by {resize_factors.tolist()}")

        # Group patch groups by number of neighbors
        neighbor_groups = defaultdict(list[PatchGroup])
        for patch_group in self.patch_groups:
            if patch_group.is_empty:  # Do not group emtpy patch groups
                continue
            neighbor_groups[patch_group.num_neighbors].append(patch_group)

        progress_bar = tqdm(
            total=len(self.patch_groups),
            desc="Calibrating patches",
            disable=(self.verbose < 2),
        )
        for num_neighbors, patch_groups in neighbor_groups.items():
            t1 = time.time()
            # Get the shifts of all patch groups as tensor
            # with shape (num_dim, N * N + 2N, len(patch_groups))
            shift_combinations = torch.stack(
                [pg.combined_shifts for pg in patch_groups], dim=-1
            )
            num_neighbor_combs = shift_combinations.shape[1]
            num_patch_groups = len(patch_groups)

            # Ravel the patch dimension to have 2D locations
            shift_combinations = shift_combinations.reshape(self.num_dim, -1)

            # scale shifts
            shift_combinations = shift_combinations * resize_factors[:, None]

            if self.verbose > 3:
                print(
                    "Combined batches of patch groups in ",
                    f"{(time.time() - t1) * 1e3} ms",
                )
                print(
                    "[#Batches, #Neighbors]", f"[{num_patch_groups}, {num_neighbors}]"
                )

            # Bilinear interpolation to get k-space signal of shifts
            t2 = time.time()
            shift_signal = _bilinear_interpolate(
                pad_coilpair_ksp,
                shift_combinations,
            )

            # Reshape to (C, num_neigh_comb, num_patch_groups)
            shift_signal = shift_signal.reshape(
                shift_signal.shape[0], num_neighbor_combs, num_patch_groups
            )
            shift_signal = shift_signal.permute(2, 0, 1)

            if self.verbose > 3:
                print(
                    "Bilinear Interpolation of batches of patch groups in ",
                    f"{(time.time() - t2) * 1e3} ms",
                )
                print(
                    "[#Batches, #Coilpairs, #NeighborCombinations]",
                    list(shift_signal.shape),
                )

            # Build the calibration matrices
            t3 = time.time()
            AhA, AhB = _build_block_matrices(
                coil_pair_ind=coilpair_indices,
                ksp_shift_comb=shift_signal,
            )

            if self.verbose > 3:
                print(f"Build block matrices in {(time.time() - t3) * 1e3} ms")
                print(f"AhA : [#Batches, #Rows, #Cols] : {list(AhA.shape)}")
                print(f"AhB : [#Batches, #Rows, #Cols] : {list(AhB.shape)}")

            weights = solve_hpd_batched(
                AhA=AhA, AhB=AhB, lam=self.tik, symmetrize=False, verbose=self.verbose
            )

            for n_pg in range(num_patch_groups):
                # Save the weights in the patch group
                patch_groups[n_pg].weights = weights[n_pg]

            if self.verbose > 3:
                print(f"Calibrated patch groups in {(time.time() - t1) * 1e3} ms")
                print(
                    "Weights : [#Batches, #CoilPairs, #Neighbors] :",
                    list(weights.shape),
                )
                print("")

            progress_bar.update(num_patch_groups)


class PatchGroup:
    """A group of patches that share the same shift pattern."""

    def __init__(
        self,
        patches: Union[list["FilledPatch"], list["EmptyPatch"]],
        verbose: int = 0,
    ):
        """
        Parameters
        ----------
        patches : Union[list[FilledPatch], list[EmptyPatch]]
            List of patches in the group.
            All patches must have the same shift pattern.
        verbose : int, optional
            Verbosity level for debugging. Default is 0 (no output). Maximum is 5.
        """
        if patches == []:
            raise ValueError("PatchGroup must contain at least one patch.")

        num_neighbors_set = {patch.num_neighbors for patch in patches}
        if len(num_neighbors_set) > 1:
            raise ValueError(
                f"All patches in a group must have the same number of neighbors, "
                f"but got counts: {num_neighbors_set}"
            )

        self.patches = patches

    @property
    def num_patches(self) -> int:
        """Return the number of patches in the group."""
        return len(self.patches)

    @property
    def num_neighbors(self) -> int:
        """Return the number of neighbors in each patch."""
        return self.patches[0].num_neighbors

    @property
    def num_dim(self) -> int:
        """Return the number of dimensions in the patches."""
        return self.patches[0].num_dim

    @property
    def is_empty(self) -> bool:
        return isinstance(self.patches[0], EmptyPatch)

    @property
    def int_shift_pattern(self) -> torch.Tensor:
        """Return the integer shift pattern of the patches."""
        if isinstance(self.patches[0], EmptyPatch):
            raise ValueError("Cannot get shift pattern for an empty patch group.")
        else:
            return self.patches[0].get_shift_pattern(return_int=True)

    @property
    def center_indices(self) -> torch.Tensor:
        """Return the center indices of the patches
        as a tensor of shape (1, num_patches)."""
        return torch.stack([patch.center_ind for patch in self.patches], dim=-1)

    @property
    def neighbor_indices(self) -> torch.Tensor:
        """Return the neighbor indices of the patches
        as a tensor of shape (num_neighbors, num_patches)."""
        if isinstance(self.patches[0], EmptyPatch):
            raise ValueError("Cannot get neighbor indices for an empty patch group.")
        else:
            return torch.stack([patch.neighbor_inds for patch in self.patches], dim=-1)

    @property
    def neighbor_shifts(self) -> torch.Tensor:
        """Return the neighbor shifts of the patch
        group as a tensor of shape (num_dim, num_neighbors)
        """
        if isinstance(self.patches[0], EmptyPatch):
            raise ValueError("Cannot get neighbor shifts for an empty patch group.")
        else:
            return self.patches[0].neighbor_shifts

    @property
    def combined_shifts(self) -> torch.Tensor:
        """Return the combined shifts of the patch group
        as a tensor of shape (D, N*N + 2*N).
        Note: As all patches in the group have the same shift pattern,
        only the first patch shift pattern is returned
        """
        if isinstance(self.patches[0], EmptyPatch):
            raise ValueError("Cannot get neighbor shifts for an empty patch group.")
        else:
            return self.patches[0].combined_shifts

    @property
    def weights(self) -> torch.Tensor:
        """Return the weights of the patch group as a tensor of shape
        (num_cha * num_neighbors, num_cha)"""
        if isinstance(self.patches[0], EmptyPatch):
            raise ValueError("Cannot get weights for an empty patch group.")
        else:
            return self.patches[0].weights

    @weights.setter
    def weights(self, value: torch.Tensor) -> None:
        """Set the weights of the patch group."""
        if isinstance(self.patches[0], EmptyPatch):
            raise ValueError("Cannot set weights for an empty patch group.")
        else:
            self.patches[0].weights = value

    @classmethod
    def create_patchgroups(
        cls,
        patches: list[Union["FilledPatch", "EmptyPatch"]],
        shift_tolerance: float = 1e-6,
        verbose: int = 0,
    ) -> list["PatchGroup"]:
        """Create a list patch groups from a list of different patches."""
        start_time = time.time()

        patch_groups = []

        # Split patches into filled and empty patches
        empty_patches = [p for p in patches if isinstance(p, EmptyPatch)]
        filled_patches = [p for p in patches if isinstance(p, FilledPatch)]

        # Empty patches are always their own group
        if empty_patches:
            patch_groups.append(cls(empty_patches))

        # 1. Group by the number of neighbors
        neighbour_groups: defaultdict[int, list[FilledPatch]] = defaultdict(list)

        for patch in filled_patches:
            neighbour_groups[patch.num_neighbors].append(patch)

        # Sort groups by number of neighbors
        sorted_group_keys = sorted(neighbour_groups.keys())

        # Save debug information as how many indivial patches where created
        debug_info = [[nn, len(pl), 0] for nn, pl in neighbour_groups.items()]

        # 2. Create PatchGroups for neighbor groups with same shift pattern
        for enum, num_neighbors in enumerate(sorted_group_keys):
            patch_list = neighbour_groups[num_neighbors]

            # Create matrix with integer shifts of all pattern
            shift_matrix = torch.stack(
                [patch.get_shift_pattern(tol=shift_tolerance) for patch in patch_list],
                dim=0,
            )
            shift_matrix = shift_matrix.reshape(shift_matrix.shape[0], -1)

            # Find unique shift patterns
            _, group_indices = torch.unique(shift_matrix, dim=0, return_inverse=True)

            # Create PatchGroups for each unique shift pattern
            shift_groups = defaultdict(list)
            for patch, idx in zip(patch_list, group_indices):
                shift_groups[idx.item()].append(patch)

            debug_info[enum][2] = len(shift_groups)

            patch_groups.extend([cls(group) for group in shift_groups.values()])

        end_time = time.time()

        if verbose > 2:
            print(
                f"Created {len(patch_groups)} patch groups",
                f"in {(end_time - start_time) * 1e3} ms",
            )
        if verbose > 3:
            print(
                "Patch group constellations ",
                "#Neighbors, #IndividualPatches/#Patches):",
            )
            print([f"{n[0]}: {n[1]}/{n[2]}" for n in debug_info])

        return patch_groups


class PatchBase:
    """Base class for patches in non-cartesian GRAPPA."""

    def __init__(
        self,
        ktraj: torch.Tensor,
        center_ind: Union[int, torch.Tensor],
        do_sift: bool = True,
        verbose: int = 0,
        device: Optional[Union[torch.device, str]] = None,
    ):
        self.device = device if device is not None else ktraj.device

        self.ktraj = ktraj
        self.do_sift = do_sift
        self.verbose = verbose
        self.num_dim = ktraj.shape[0] - 1  # Last dim is sampling mask

        if isinstance(center_ind, int):
            self.center_ind = torch.tensor([center_ind], device=ktraj.device)
        elif isinstance(center_ind, torch.Tensor):
            if center_ind.ndim == 0:
                self.center_ind = torch.tensor([center_ind], device=ktraj.device)
            elif center_ind.ndim > 1:
                raise TypeError("center_ind must be a int or a 1D tensor.")
            else:
                self.center_ind = center_ind
        else:
            raise TypeError("center_ind must be an int or a 1D tensor.")

    @property
    def center_loc(self) -> torch.Tensor:
        """Return the location of the center sample."""
        return self.ktraj[:-1, self.center_ind]

    @property
    def num_neighbors(self) -> int:
        """Return the number of neighbors."""
        raise NotImplementedError

    @property
    def neighbor_inds(self) -> torch.Tensor:
        """Return the indices of the neighbors."""
        raise NotImplementedError

    @property
    def neighbor_locs(self) -> torch.Tensor:
        """Return the locations of the neighbors."""
        raise NotImplementedError

    @property
    def neighbor_shifts(self) -> torch.Tensor:
        """Return the shifts to the neighbors."""
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
        ktraj: torch.Tensor,
        center_ind: Union[int, torch.Tensor],
        neighbor_inds: torch.Tensor,
        device: Optional[Union[torch.device, str]] = None,
        verbose: int = 0,
        do_sift: bool = True,
    ):
        """
        Parameters
        ----------
        ktraj : torch.Tensor, shape (D+1, N)
            The trajectory in k-space with D spatial dimensions and N samples.
            The last dimension is a sampling mask with 1 for sampled locations.
        center_ind : Union[int, torch.Tensor]
            Index of the unsampled point in the center of the patch.
        neighbor_inds : torch.Tensor, shape (N,)
            Indices of the sampled neighbors in k-space.
        do_sift : bool, optional
            Sift neighbors to remove duplicates.
        verbose : int, optional
            Verbosity level for debugging. Default is 0 (no output). Maximum is 5.
        """
        super().__init__(
            ktraj=ktraj,
            center_ind=center_ind,
            do_sift=do_sift,
            verbose=verbose,
            device=device,
        )

        self.weights = None  # To be filled during calibration

        # Sort neighbors by integer distance
        dist = ktraj[:-1, neighbor_inds] - self.center_loc
        dist = dist.to(torch.int32)
        sorted_order = distance_sort(dist)
        neighbor_inds = neighbor_inds[sorted_order]

        # Perfome sifting
        if do_sift:
            self.sift_mask = sift_mask(
                ktraj[:-1, neighbor_inds] - self.center_loc
                # dist[self.neighbor_inds]
            )
        else:
            self.sift_mask = torch.ones(
                neighbor_inds.shape[0],
                dtype=torch.bool,
                device=ktraj.device,
            )

        self._neighbor_inds = neighbor_inds

    @property
    def neighbor_inds(self) -> torch.Tensor:
        """Return the indices of the neighbors."""
        return self._neighbor_inds[self.sift_mask]

    @property
    def neighbor_locs(self) -> torch.Tensor:
        """Return the locations of the neighbors."""
        return self.ktraj[:-1, self.neighbor_inds]

    @property
    def neighbor_shifts(self) -> torch.Tensor:
        """Return the distances to the neighbors."""
        return self.neighbor_locs - self.center_loc

    @property
    def num_neighbors(self) -> int:
        """Return the number of neighbors."""
        return self.neighbor_inds.shape[0]

    @property
    def combined_shifts(self) -> torch.Tensor:
        """Return the combined shifts of the
        patch as a tensor of shape (D, N*N + 2*N)."""
        combined_shifts = (
            self.neighbor_shifts[:, :, None] - self.neighbor_shifts[:, None, :]
        ).reshape(self.num_dim, -1)

        # Add negative and positive shifts
        combined_shifts = torch.cat(
            [combined_shifts, -self.neighbor_shifts, self.neighbor_shifts], dim=-1
        )

        return combined_shifts

    def get_shift_pattern(
        self,
        return_int: bool = False,
        tol: float = 1e-6,
    ) -> torch.Tensor:
        """Return the integer shift pattern (D, N)."""
        if return_int:
            return torch.round(self.neighbor_shifts).to(torch.int32)
        else:
            return torch.round(self.neighbor_shifts / tol) * tol

    @classmethod
    def from_indices(
        cls,
        ktraj: torch.Tensor,
        patch_indices: list[torch.Tensor],
        device: Optional[Union[torch.device, str]] = None,
        verbose: int = 0,
        do_sift: bool = True,
    ) -> list["FilledPatch"]:
        """Return a list of FilledPatch objects from a list
        of center and neighbor indices in ktraj.
        Parameters
        ----------
        ktraj : torch.Tensor, shape (D+1, N)
            The trajectory in k-space with D spatial dimensions and N samples.
            The last dimension is a sampling mask with 1 for sampled locations.
        patch_indices : list[torch.Tensor]
            List of tensors with center and neighbor indices in ktraj.
            Each tensor should have shape (N,) where N is the number of neighbors.
            The first element is the index of the unsampled point
            in the center of the patch, and the other elements are
            the indices of the sampled neighbors.
        device : Optional[Union[torch.device, str]], optional
            Device to perform computations on
            (default is None, which uses the current device).
        verbose : int, optional
            Verbosity level for debugging. Default is 0 (no output). Maximum is 5.
        do_sift : bool, optional
            Sift neighbors to remove duplicates (default is True).
        """
        patches = []

        start_time = time.time()
        for indices in patch_indices:
            if indices.shape[0] < 2:
                raise ValueError(
                    "Patch indices must contain at least one center and one neighbor."
                )

            cls_instance = cls(
                ktraj=ktraj,
                center_ind=indices[0],
                neighbor_inds=indices[1:],
                device=device if device is not None else ktraj.device,
                verbose=verbose,
                do_sift=do_sift,
            )

            patches.append(cls_instance)

        end_time = time.time()

        if verbose > 2:
            print(
                f"Created {len(patches)} filled patches",
                f" in {(end_time - start_time) * 1e3} ms.",
            )

        return patches


class EmptyPatch(PatchBase):
    """Patch with no neighbors."""

    def __init__(
        self,
        ktraj: torch.Tensor,
        center_ind: Union[int, torch.Tensor],
        device: Optional[Union[torch.device, str]] = None,
        verbose: int = 0,
    ):
        """
        Parameters
        ----------
        ktraj : torch.Tensor, shape (D+1, N)
            The trajectory in k-space with D spatial dimensions and N samples.
            The last dimension is a sampling mask with 1 for sampled locations.
        center_ind : Union[int, torch.Tensor]
            Index of the unsampled point in the center of the patch.
        device : Optional[Union[torch.device, str]], optional
            Device to perform computations on
            (default is None, which uses the current device).
        verbose : int, optional
            Verbosity level for debugging. Default is 0 (no output). Maximum is 5.
        """
        super().__init__(
            ktraj=ktraj,
            center_ind=center_ind,
            do_sift=False,
            verbose=verbose,
            device=device,
        )

    @property
    def num_neighbors(self) -> int:
        """Return the number of neighbors."""
        return 0

    @classmethod
    def from_indices(
        cls,
        ktraj: torch.Tensor,
        patch_indices: list[torch.Tensor],
        device: Optional[Union[torch.device, str]] = None,
        verbose: int = 0,
    ) -> list["EmptyPatch"]:
        """Return a list of EmptyPatch objects from a list
        of center indices in ktraj.
        Parameters
        ----------
        ktraj : torch.Tensor, shape (D+1, N)
            The trajectory in k-space with D spatial dimensions and N samples.
            The last dimension is a sampling mask with 1 for sampled locations.
        patch_indices : list[torch.Tensor]
            List of tensors with center indices in ktraj.
            Each tensor should have shape (1,) as it contains only the index of the
            center location.
        device : Optional[Union[torch.device, str]], optional
            Device to perform computations on
            (default is None, which uses the current device).
        verbose : int, optional
            Verbosity level for debugging. Default is 0 (no output). Maximum is 5.
        """
        patches = []

        for indices in patch_indices:
            if indices.shape[0] != 1:
                raise ValueError(
                    "Patch indices must contain only a center index for EmptyPatch."
                )
            cls_instance = cls(
                ktraj=ktraj,
                center_ind=indices[0],
                device=device if device is not None else ktraj.device,
                verbose=verbose,
            )

            patches.append(cls_instance)

        return patches


def sift_mask(k: torch.Tensor) -> torch.Tensor:
    """
    Return a mask to sift locations.

    Parameters
    ----------
    k : torch.Tensor, shape (D, N)
        Location samples with D dimensions and N samples.
        Must be in [cycles/fov] units.

    Returns : torch.Tensor, shape (N,)
        A boolean mask where True indicates unique locations.

    Notes
    -----
    This function rounds the input tensor to whole numbers and then finds unique
    locations, returning a mask that indicates which locations are unique.
    """

    # Get unique distances and their indices
    # Remember that torch.unique is a shit function and does not work
    # as numpy unique
    # This is from https://stackoverflow.com/questions/72001505/how-to-get-unique-elements-and-their-firstly-appeared-indices-of-a-pytorch-tenso
    # and I do not know why, but it works the way it should.

    # Round to whole numbers
    k = torch.round(k).to(torch.int32)
    num_dim, num_samples = k.shape

    _, idx, counts = torch.unique(
        k,
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
        num_samples,
        dtype=torch.bool,
        device=k.device,
    )
    sift_mask[unique_indices] = True

    return sift_mask


def distance_sort(d: torch.Tensor) -> torch.Tensor:
    """Sort the order of given distances in lexographic order.

    Parameters
    ----------
    d : torch.Tensor, (D, N)
        Distances in D dimensions and N samples.

    Returns
    -------
    torch.Tensor
        Sorted indices of the distances.
    """
    num_dim, num_samples = d.shape
    keys = [d[dim, :] for dim in reversed(range(num_dim))]
    sorted_order = lexsort(keys)

    return sorted_order


def create_patches(
    ktraj: torch.Tensor,
    kernel_size: torch.Tensor,
    p_norm: float = torch.inf,
    verbose: int = 0,
    do_sift: bool = True,
) -> list[Union[FilledPatch, EmptyPatch]]:
    """_summary_

    Parameters
    ----------
    ktraj : torch.Tensor, shape (D+1, N)
        The trajectory in k-space with D spatial dimensions and N samples.
        The last dimension is a sampling mask with 1 for sampled locations.
        Trajectory must be scaled in cycles/FOV units.
    kernel_size : torch.Tensor
        The size of the kernel in cycles/FOV units.
    p_norm : float, optional
        p_norm of the KDTree search of neighbours inside kernel_size,
        by default torch.inf
    verbose : int, optional
        Verbosity level between 0 and 5, by default 0

    Returns
    -------
    list[Union[FilledPatch, EmptyPatch]]
        List of patches found in KDTree search.
        Each patch is either a FilledPatch with
        neighbors or an EmptyPatch with no neighbors
        and just the unsampled center index.
    """
    patch_indices = _get_patch_indices(
        ktraj=ktraj,
        kernel_size=kernel_size,
        p_norm=p_norm,
        verbose=verbose,
    )

    if verbose > 2:
        print("------------------------------------------------")
        print(f"Create patches with sifitng set to {do_sift}.")

    start_time = time.time()

    empty_patch_list = EmptyPatch.from_indices(
        ktraj=ktraj,
        patch_indices=[ind for ind in patch_indices if ind.shape[0] == 1],
        device=ktraj.device,
        verbose=verbose,
    )

    filled_patch_list = FilledPatch.from_indices(
        ktraj=ktraj,
        patch_indices=[ind for ind in patch_indices if ind.shape[0] > 1],
        do_sift=do_sift,
        device=ktraj.device,
        verbose=verbose,
    )

    end_time = time.time()

    if verbose > 2:
        print(
            f"Created {len(empty_patch_list)} empty patches."
            f"Created {len(filled_patch_list)} filled patches."
            f"Patch creation completed in {(end_time - start_time) * 1e3:.3f} ms."
        )

    if verbose > 3 and do_sift:
        list_nneigh = [patch.num_neighbors for patch in filled_patch_list]
        list_nneigh.extend([patch.num_neighbors for patch in empty_patch_list])

        neigh_counts = Counter(list_nneigh)
        sorted_counts = sorted(neigh_counts.items())

        print("Patch constellations after sifting (#Neighbors, #Patches):")
        print(sorted_counts)

    # Combine empty and filled patches
    return empty_patch_list + filled_patch_list


def _get_patch_indices(
    ktraj: torch.Tensor,
    kernel_size: torch.Tensor,
    p_norm: float = torch.inf,
    verbose: int = 0,
) -> list[torch.Tensor]:
    """
    Get indices of sampled neighbors for each unsampled point in k-space using
    KDTree search.
    For each list of indices, the first element is the index of the unsampled
    point while the other indices are the indices of the sampled neighbors.

    Parameters
    ----------
    ktraj : torch.Tensor, shape (D+1, N)
        The trajectory in k-space with D spatial dimensions and N samples.
        The last dimension is a sampling mask with 1 for sampled locations.
    kernel_size : torch.Tensor, shape (D,)
        The size of the kernel in cycles/FOV units.
    p_norm : float, optional
        The p-norm to use for distance calculation (default is infinity norm).
    threads : int, optional
        Number of threads to use for KDTree search
        (default is -1, which uses all available threads).
    verbose : int, optional
        Verbosity level for debugging. Default is 0 (no output). Maximum is 5.
    """
    # Start timer
    start_time = time.time()

    if verbose > 2:
        print(
            "------------------------------------------------\n",
            "Perform KDTree search of patch constellations.",
            sep="",
        )
    # Get a mask for unsampled and sampled locations
    sample_mask = ktraj[-1, :] == 1
    inv_sample_mask = ~sample_mask

    ind_sampled = torch.nonzero(sample_mask).squeeze()
    ind_unsampled = torch.nonzero(inv_sample_mask).squeeze()

    # Scale the trajectory units of kernel size
    norm_factor = kernel_size / 2.0
    ktraj_scaled = ktraj[:-1, :] / norm_factor[:, None]

    if verbose > 3:
        print(f"Neighbor search distance: \u00b1{norm_factor.tolist()}")
        print(f"Sampled points: {len(ind_sampled)}")
        print(f"Unsampled points: {len(ind_unsampled)}")

    kdtree_sampled = KDTree(ktraj_scaled[:, sample_mask].cpu().numpy().T)
    kdtree_unsampled = KDTree(ktraj_scaled[:, inv_sample_mask].cpu().numpy().T)

    radius = 1.0 + 1e-6  # Buffer for floating point precision

    neighbor_inds = kdtree_unsampled.query_ball_tree(kdtree_sampled, r=radius, p=p_norm)

    def comb(center_ind, neighbor_inds):
        return torch.cat((center_ind.unsqueeze(0), neighbor_inds), dim=0)

    inds_c = [
        comb(ind_unsampled[i], ind_sampled[n]) for i, n in enumerate(neighbor_inds)
    ]

    if verbose > 3:
        print("Patch constellations before sifting (#Neighbors, #Patches):")
        counts = Counter(len(inds) - 1 for inds in inds_c)
        sorted_counts = sorted(counts.items())  # sort by (length-1) ascending
        print(sorted_counts)

    stop_time = time.time() - start_time
    if verbose > 2:
        print(f"Found {len(inds_c)} patches.")
        print(f"KDTree search completed in {(1e3 * stop_time):.3f} ms. \n")

    return inds_c


def _bilinear_interpolate(
    ksp: torch.Tensor,
    locs: torch.Tensor,
) -> torch.Tensor:
    """_summary_

    Parameters
    ----------
    ksp : torch.Tensor, shape (C, *spatial_dims)
        _description_
    locs : torch.Tensor, shape (D, N)
        _description_

    Returns
    -------
    torch.Tensor, C, N
        Interpolated k-space data at the given locations.

    """
    num_comb, *spatial_dims = ksp.shape
    # Locs must be in range [-1, 1]
    spatial_dims = torch.tensor(spatial_dims, dtype=torch.float32, device=ksp.device)
    locs = locs.clone() / (spatial_dims[:, None] / 2)

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


def _build_block_matrices(
    coil_pair_ind: torch.Tensor,
    ksp_shift_comb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """_summary_

    Parameters
    ----------
    coil_pair_ind : torch.Tensor, shape (C*(C+1)/2, 2)
        _description_
    ksp_shift_comb : torch.Tensor, shape (M, C*(C+1)/2, N*N + 2*N)
        _description_

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        _description_
    """
    num_cha = int(coil_pair_ind.max() + 1)
    num_neigh = round(math.sqrt(ksp_shift_comb.shape[2] + 1)) - 1
    num_neigh = int(num_neigh)
    num_groups = ksp_shift_comb.shape[0]
    N = num_cha * num_neigh

    AhA = torch.zeros(
        (num_groups, N, N), dtype=torch.complex64, device=ksp_shift_comb.device
    )
    AhB = torch.zeros(
        (num_groups, N, num_cha), dtype=torch.complex64, device=ksp_shift_comb.device
    )

    # Fill AhA and Ahb matrices
    for enum, (cha_i, cha_j) in enumerate(coil_pair_ind.T):
        start_1 = cha_i * num_neigh
        end_1 = (cha_i + 1) * num_neigh

        start_2 = cha_j * num_neigh
        end_2 = (cha_j + 1) * num_neigh

        for ng in range(num_groups):
            block = ksp_shift_comb[ng, enum, : -2 * num_neigh]
            block = block.reshape(num_neigh, num_neigh).T
            vec_1 = ksp_shift_comb[ng, enum, -2 * num_neigh : -num_neigh]
            vec_2 = ksp_shift_comb[ng, enum, -num_neigh:]

            AhA[ng, start_1:end_1, start_2:end_2] = block
            AhB[ng, start_1:end_1, cha_j] = vec_1

            if cha_i != cha_j:
                AhA[ng, start_2:end_2, start_1:end_1] = torch.conj(block.T)
                AhB[ng, start_2:end_2, cha_i] = torch.conj(vec_2)

    return AhA, AhB


@torch.no_grad()  # remove if you need grads
def solve_hpd_batched(
    AhA: torch.Tensor,  # (M, N, N)
    AhB: torch.Tensor,  # (M, N, C)
    lam: float = 0.0,  # Tikhonov (λ >= 0). Use >0 to guarantee PD.
    symmetrize: bool = True,  # enforce exact Hermitian structure
    chol_retries=(1e-8, 1e-6, 1e-4),  # extra diagonal loads if initial Cholesky fails
    use_qr_fallback: bool = True,  # QR least-squares fallback if everything fails
    verbose: int = 0,  # verbosity level between 0 and 5
) -> torch.Tensor:
    """
    Solves (AhA) X = AhB for *batched* inputs.
    Returns X with shape (M, N, C).
    """
    # --- shape & dtype checks ---
    if AhA.ndim != 3 or AhB.ndim != 3:
        raise ValueError(
            f"Expected AhA (M,N,N) & AhB (M,N,C); got {AhA.shape}, {AhB.shape}"
        )
    M, N, N2 = AhA.shape
    if N2 != N or AhB.shape[0] != M or AhB.shape[1] != N:
        raise ValueError(f"Shape mismatch: AhA {AhA.shape}, AhB {AhB.shape}")
    if AhA.device != AhB.device or AhA.dtype != AhB.dtype:
        raise ValueError("AhA and AhB must share device and dtype")
    if AhA.dtype not in (torch.complex64, torch.complex128):
        raise TypeError("Use complex dtype (complex64 recommended)")

    A = AhA.contiguous()
    B = AhB.contiguous()

    # Make exactly Hermitian if desired (cheap O(N^2) projection)
    if symmetrize:
        A = 0.5 * (A + A.mH)
        if verbose > 4:
            print("Symmetrized AhA matrix to ensure exact Hermitian structure.")

    # Add λI (broadcast over batch)
    if lam != 0.0:
        Eye = torch.eye(N, dtype=A.dtype, device=A.device)
        A = A + lam * Eye
    if verbose > 4:
        print(f"Added Tikhonov regularization with λ={lam}.")

    # --- Fast path: Cholesky on the whole batch ---
    L, info = torch.linalg.cholesky_ex(A, upper=False)  # info: (M,)
    X = torch.empty_like(B)

    ok = info == 0
    t0 = time.time()
    if ok.any():
        X[ok] = torch.cholesky_solve(B[ok], L[ok], upper=False)
    if verbose > 3:
        print(f"Cholesky succeeded for {ok.sum()} out of {M} batches.")
        print(f"Cholesky solve time: {(time.time() - t0) * 1e3:.2f} ms")

    remaining = ~ok
    if remaining.any():
        # Retry failing batches with extra diagonal loading
        Eye = torch.eye(N, dtype=A.dtype, device=A.device)  # ensure defined
        idx = remaining.nonzero(as_tuple=False).squeeze(-1)
        t1 = time.time()
        if verbose > 4:
            print(
                f"Cholesky failed for {idx.numel()} batches, retrying with extra loads."
            )
        for extra in chol_retries:
            if idx.numel() == 0:
                break
            A_try = A[idx] + extra * Eye
            L_try, info_try = torch.linalg.cholesky_ex(A_try, upper=False)
            ok2 = info_try == 0
            if ok2.any():
                good_idx = idx[ok2]
                X[good_idx] = torch.cholesky_solve(B[good_idx], L_try[ok2], upper=False)
                # keep only the still-failing indices
                idx = idx[~ok2]
        if verbose > 4:
            print(
                f"Cholesky retries took {(time.time() - t1) * 1e3:.2f} ms, "
                f"still failing {idx.numel()} batches."
            )

        # Final fallbacks for any still-failing systems
        if idx.numel() > 0:
            if verbose > 4:
                print(
                    "Falling back to least-squares for ",
                    f"{idx.numel()} remaining batches.",
                )
            try:
                X[idx] = torch.linalg.solve(A[idx], B[idx])
                idx = idx.new_empty(0)  # all done
            except RuntimeError:
                pass  # fall back further below if requested

            if idx.numel() > 0 and use_qr_fallback:
                # Least-squares via QR (works for singular/non-HPD cases)
                X[idx] = torch.linalg.lstsq(A[idx], B[idx], driver="gels").solution

    return X
