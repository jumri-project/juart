import math
from collections import Counter, defaultdict
from typing import Optional, Union

import torch
from scipy.spatial import KDTree
from torch.nn.functional import grid_sample
from torch_geometric.utils import lexsort
from tqdm.auto import tqdm

from ..conopt.functional.fourier import (
    fourier_transform_adjoint,
    fourier_transform_forward,
)
from ..utils import Timer, resize, verbose_print


class NonCartesianGrappa:
    """Non-Cartesian GRAPPA reconstruction class."""

    def __init__(
        self,
        ktraj: torch.Tensor,
        calib_signal: Optional[torch.Tensor] = None,
        kernel_size: Union[torch.Tensor, list, tuple] = (5, 5),
        do_sift: bool = True,
        tik: float = 0.0,
        p_norm: float = torch.inf,
        shift_tolerance: float = 1e-6,
        verbose: int = 0,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """Initialize the Non-Cartesian GRAPPA reconstruction.
        The calibration is performed during initialization.

        Parameters
        ----------
        ktraj : torch.Tensor, shape (D+1, N)
            The trajectory in k-space with D spatial dimensions and N samples
            in units of cycles/FOV.
            The last dimension is a sampling mask with 1 for sampled locations.
        calib_signal : torch.Tensor, shape (C, spatial_dims)
            Autocalibration signal with C coils and D spatial dimensions.
        kernel_size : tensor, list, or tuple, shape (D,)
            Kernel size in cycles/FOV units.
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
            Device Pytorch will run the computations on, by default None
        """

        self.pad_out = 128
        self.device = torch.device(device) if device is not None else ktraj.device
        self.ktraj = ktraj
        self.num_dim = ktraj.shape[0] - 1
        self.shift_tol = shift_tolerance
        self.do_sift = do_sift
        self.verbose = verbose
        self.tik = tik
        self.p_norm = p_norm

        if isinstance(kernel_size, (list, tuple)):
            self.kernel_size = torch.tensor(
                kernel_size, dtype=torch.float32, device=self.device
            )
        elif isinstance(kernel_size, torch.Tensor):
            self.kernel_size = kernel_size.to(device=self.device, dtype=torch.float32)
        else:
            raise TypeError("kernel_size must be a list, tuple, or torch.Tensor.")

        if calib_signal.ndim - 1 != self.num_dim:
            raise ValueError(
                f"Dimension mismatch between trajectory ({self.num_dim}D) "
                f"and calibration ({calib_signal.ndim - 1}D)."
            )

        self.calib_signal = calib_signal

        self.num_cha = self.calib_signal.shape[0]

        if self.kernel_size.shape[0] != self.num_dim:
            raise ValueError(
                f"Kernel size must have {self.num_dim} dimensions, "
                f"but got {self.kernel_size.shape[0]}."
            )

        num_sampled_locs = int(ktraj[-1, :].sum())
        num_unsampled_locs = ktraj.shape[1] - num_sampled_locs
        verbose_print(
            self.verbose,
            1,
            "------------------------------------------------\n"
            "Non-Cartesian GRAPPA reconstruction initialised. \n"
            f"Kernel size: {self.kernel_size.tolist()} \n"
            f"Number of dimensions: {self.num_dim} \n"
            f"Device: {self.device} \n"
            f"Shift tolerance: {self.shift_tol} \n"
            f"Number of locations in k-space: {ktraj.shape[1]} \n"
            f"Number of sampled locations in k-space: {num_sampled_locs} \n"
            f"Number of unsampled locations in k-space: {num_unsampled_locs}",
        )

        # Create patches from ktraj
        self.patches = _create_patches(
            ktraj=ktraj,
            kernel_size=self.kernel_size,
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
    def num_patches(self) -> int:
        """Return the number of sampled trajectory locations."""
        return len(self.patches)

    @property
    def weights(self) -> torch.Tensor:
        """Return the weights of all patch groups as a tensor of shape
        (num_patch_groups, C*(C+1)/2, num_neighbors)."""
        weights = []
        for pg in self.patch_groups:
            weights.append(pg.weights)
        return torch.stack(weights, dim=0)

    @property
    def coilpair_indices(self) -> torch.Tensor:
        """Return the indices channel pairs as a tensor of shape (2, C*(C+1)/2)."""
        ind_ij = torch.triu_indices(self.num_cha, self.num_cha, device=self.device)
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
            [(old_size[i] - 1) / (new_size[i] - 1) for i in range(self.num_dim)],
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

    @property
    def organised_neighbor_groups(self) -> dict[int, list["PatchGroup"]]:
        """Return a dictionary of patch groups organized
        by number of neighbors of the patch groups."""

        # Create ordered dictionary
        neighbor_groups = defaultdict(list)

        # FIll
        for patch_group in self.patch_groups:
            if patch_group.is_empty:  # Do not group emtpy patch groups
                continue
            neighbor_groups[patch_group.num_neighbors].append(patch_group)

        return neighbor_groups

    @torch.inference_mode()
    def calibrate(self):
        """Calibrate the patch weights."""
        verbose_print(
            self.verbose,
            1,
            "------------------------------------------------ \nPerform calibration.",
        )

        # Save some arrays for computation efficiency
        with Timer(
            name="Preparation",
            text="Prepared calibration in {:.2f} ms.",
            current_level=self.verbose,
            trigger_level=3,
        ):
            coilpair_indices = self.coilpair_indices
            pad_coilpair_ksp = self.padded_coilpair_ksp
            resize_factors = self.pad_resize_factor

        # Group patch groups by number of neighbors to performe batched ops
        neighbor_groups = self.organised_neighbor_groups

        verbose_print(
            self.verbose,
            5,
            (
                f"Found {coilpair_indices.shape[1]} coil pairs. "
                f"Found {len(neighbor_groups)} neighbor-count groups. "
                f"Padded coilpair ACS shape: {list(pad_coilpair_ksp.shape)}."
            ),
        )

        # Create progress bar
        total_patch_groups = sum(len(pgs) for pgs in neighbor_groups.values())
        progress_bar = tqdm(
            total=total_patch_groups,
            desc="Calibrating patches",
            disable=(self.verbose < 1),
        )

        # Calibrate batched patch groups
        for _, patch_groups in neighbor_groups.items():
            weights = _calculate_patchgroups_weights(
                patch_groups=patch_groups,
                ksp=pad_coilpair_ksp,
                cp_ind=coilpair_indices,
                resize_factors=resize_factors,
                tik=self.tik,
                verbose=self.verbose,
            )

            for n_pg, w in enumerate(weights):
                # Save the weights in the patch group
                patch_groups[n_pg].weights = w

            progress_bar.update(len(patch_groups))

    def apply(
        self,
        ksp: torch.Tensor,
        inplace_fill: bool = False,
        verbose: int = 0,
    ) -> torch.Tensor:
        """Apply the calibrated patch weights to the k-space data.

        Parameters
        ----------
        ksp : torch.Tensor, complex, shape (C, N)
            The k-space data with missing signal samples set to zero
            for C coil channels and N samples.
        inplace_fill : bool, optional
            If True, fill the missing samples in k-space inplace.
            If False, return a new tensor with the filled samples.
        verbose : int, optional
            Verbosity level for debugging. Default is 0 (no output). Maximum is 5.

        Returns
        -------
        torch.Tensor
            The k-space data with filled missing samples.
        """
        if inplace_fill:
            ksp_out = ksp
        else:
            ksp_out = ksp.clone()

        num_cha, num_samp = ksp_out.shape
        if num_cha != self.num_cha:
            raise ValueError(
                f"Channel mismatch: k-space has {num_cha} channels, "
                f"calibration has {self.num_cha}."
            )

        if num_samp != self.ktraj.shape[1]:
            raise ValueError(
                f"Sample mismatch: k-space has {num_samp} samples, "
                f"trajectory has {self.ktraj.shape[1]}."
            )

        progress_bar = tqdm(
            total=len(self.patches),
            desc="Applying patch weights",
            disable=(verbose < 2),
        )
        for patch_group in self.patch_groups:
            if patch_group.is_empty:
                progress_bar.update(patch_group.num_patches)
                continue

            # Get neighbor k-space data
            ksp_neigh = ksp_out[:, patch_group.neighbor_indices]

            # This is faster than a batches matmul
            ksp_fill = torch.einsum(
                "ij,ik->kj",
                ksp_neigh.reshape(-1, patch_group.num_patches),
                patch_group.weights,
            )

            # Fill missing kspace data
            center_indices = patch_group.center_indices[0]
            ksp_out[:, center_indices] = ksp_fill

            progress_bar.update(patch_group.num_patches)

        return ksp_out


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
        with Timer(
            name="Create groups of similar patches",
            text="Created groups in {:0.4f} ms",
            current_level=verbose,
            trigger_level=3,
        ):
            patch_groups = []

            # Split patches into filled and empty patches
            empty_patches = [p for p in patches if isinstance(p, EmptyPatch)]
            filled_patches = [p for p in patches if isinstance(p, FilledPatch)]

            # Empty patches are always their own group
            if empty_patches:
                patch_groups.append(cls(empty_patches))

            # 1. Group by the number of neighbors
            neighbour_groups = defaultdict(list)

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
                shift_patterns = [
                    patch.get_shift_pattern(tol=shift_tolerance) for patch in patch_list
                ]
                shift_matrix = torch.stack(shift_patterns, dim=0)

                # Flatten the shift patterns
                shift_matrix = shift_matrix.reshape(shift_matrix.shape[0], -1)

                # Find unique shift patterns
                _, group_indices = torch.unique(
                    shift_matrix, dim=0, return_inverse=True
                )

                # Create PatchGroups for each unique shift pattern
                shift_groups = defaultdict(list)
                for patch, idx in zip(patch_list, group_indices):
                    shift_groups[idx.item()].append(patch)

                debug_info[enum][2] = len(shift_groups)

                patch_groups.extend([cls(group) for group in shift_groups.values()])

            verbose_print(
                verbose,
                1,
                f"Created {len(patch_groups)} patch groups "
                f"from {len(filled_patches)} filled patches "
                f"and {len(empty_patches)} empty patches.",
            )
            verbose_print(
                verbose,
                5,
                (
                    "Patch group constellations \
                    (#Neighbors, #IndividualPatches/#Patches): \n"
                    + " ".join([f"({n[0]},{n[2]}/{n[1]})" for n in debug_info])
                ),
            )

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
        self.device = torch.device(device) if device is not None else ktraj.device

        self.ktraj = ktraj
        self.do_sift = do_sift
        self.verbose = verbose
        self.num_dim = ktraj.shape[0] - 1  # Last dim is sampling mask

        if isinstance(center_ind, int):
            self.center_ind = torch.tensor([center_ind], device=self.device)
        elif isinstance(center_ind, torch.Tensor):
            if center_ind.ndim == 0:
                self.center_ind = torch.tensor([center_ind], device=self.device)
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
        sorted_order = _distance_sort(dist)
        neighbor_inds = neighbor_inds[sorted_order]

        # Perfome sifting
        if do_sift:
            self.sift_mask = _sift_mask(
                ktraj[:-1, neighbor_inds] - self.center_loc
                # dist[self.neighbor_inds]
            )
        else:
            self.sift_mask = torch.ones(
                neighbor_inds.shape[0],
                dtype=torch.bool,
                device=self.device,
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
        """Return the shift pattern as shape (D, N)
        rounded to interger numbers or to some tolerance."""
        if return_int:
            return torch.round(self.neighbor_shifts).to(torch.int32)
        else:
            return torch.round(self.neighbor_shifts / tol) * tol

    @classmethod
    def from_indices(
        cls,
        ktraj: torch.Tensor,
        patch_indices: list[torch.Tensor],
        do_sift: bool = True,
        device: Optional[Union[torch.device, str]] = None,
        verbose: int = 0,
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
        do_sift : bool, optional
            Sift neighbors to remove duplicates (default is True).
        device : Optional[Union[torch.device, str]], optional
            Device to perform computations on
            (default is None, which uses the current device).
        verbose : int, optional
            Verbosity level for debugging. Default is 0 (no output). Maximum is 5.
        """
        patches = []
        for indices in patch_indices:
            if indices.shape[0] < 2:
                raise ValueError(
                    "Patch indices must contain at least",
                    "one center and one neighbor.",
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


def _sift_mask(k: torch.Tensor) -> torch.Tensor:
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


def _distance_sort(d: torch.Tensor) -> torch.Tensor:
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
    num_dim, _ = d.shape
    keys = [d[dim, :] for dim in reversed(range(num_dim))]
    sorted_order = lexsort(keys)

    return sorted_order


@torch.inference_mode()
def _create_patches(
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

    verbose_print(
        verbose,
        1,
        "-------------------------------------------------\n"
        f"Create patches with sifting set to {do_sift}.",
    )

    with Timer(
        name="Create Patches:",
        text="Created patches in {:.2f} ms.",
        current_level=verbose,
        trigger_level=3,
    ):
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

    verbose_print(
        verbose,
        1,
        f"Created {len(empty_patch_list)} empty patches. \n"
        f"Created {len(filled_patch_list)} filled patches. \n",
    )

    if verbose >= 4 and do_sift:
        list_nneigh = [patch.num_neighbors for patch in filled_patch_list]
        list_nneigh.extend([patch.num_neighbors for patch in empty_patch_list])

        neigh_counts = Counter(list_nneigh)
        sorted_counts = sorted(neigh_counts.items())

        verbose_print(
            verbose,
            4,
            "Patch constellations before sifting "
            f"(#Neighbors, #Patches): \n {sorted_counts}",
        )

    # Combine empty and filled patches
    return empty_patch_list + filled_patch_list


@torch.inference_mode()
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
    verbose_print(
        verbose,
        1,
        "\n".join(
            [
                "-------------------------------------------------",
                "Perform KDTree search.",
            ]
        ),
    )
    with Timer(
        name="KDTree search",
        text="KDTree search completed in {:.2f} ms.",
        current_level=verbose,
        trigger_level=2,
    ):
        # Get a mask for unsampled and sampled locations
        sample_mask = ktraj[-1, :] == 1
        inv_sample_mask = ~sample_mask

        ind_sampled = torch.nonzero(sample_mask).squeeze()
        ind_unsampled = torch.nonzero(inv_sample_mask).squeeze()

        # Scale the trajectory units of kernel size
        norm_factor = kernel_size / 2.0
        ktraj_scaled = ktraj[:-1, :] / norm_factor[:, None]

        verbose_print(
            verbose,
            3,
            "\n".join(
                [
                    f"Neighbor search distance: \u00b1{norm_factor.tolist()}",
                    f"Sampled points: {len(ind_sampled)}",
                    f"Unsampled points: {len(ind_unsampled)}",
                ]
            ),
        )
        # Create KDTree for sampled and unsampled locations
        kdtree_sampled = KDTree(ktraj_scaled[:, sample_mask].cpu().numpy().T)
        kdtree_unsampled = KDTree(ktraj_scaled[:, inv_sample_mask].cpu().numpy().T)

        radius = 1.0 + 1e-6  # Buffer for floating point precision

        neighbor_inds = kdtree_unsampled.query_ball_tree(
            kdtree_sampled, r=radius, p=p_norm
        )

        def comb(center_ind, neighbor_inds):
            return torch.cat((center_ind.unsqueeze(0), neighbor_inds), dim=0)

        inds_c = [
            comb(ind_unsampled[i], ind_sampled[n]) for i, n in enumerate(neighbor_inds)
        ]

        verbose_print(verbose, 2, f"Found {len(inds_c)} patches.")

        counts = Counter(len(inds) - 1 for inds in inds_c)
        sorted_counts = sorted(counts.items())  # sort by (length-1) ascending
        verbose_print(
            verbose,
            4,
            "Patch constellations before sifting (#Neighbors, #Patches): \n%s",
            sorted_counts,
        )

    return inds_c


def _calculate_patchgroups_weights(
    patch_groups: list[PatchGroup],
    ksp: torch.Tensor,
    cp_ind: torch.Tensor,
    resize_factors: torch.Tensor,
    tik: float = 0,
    verbose: int = 0,
):
    """Calculate the weights of a batch of patch groups
    with the same number of neighbors.

    Parameters
    ----------
    patch_groups : list[PatchGroup]
        List of patch groups to calibrate.
        The patch groups must have the same number of neighbors.
    ksp : torch.Tensor, shape (CP, *spatial_dimensions), complex
        The k-space data with CP coil pairs and *spatial_dimensions.
    cp_ind : torch.Tensor
        Coil pair indices with shape (CP, 2).
        CP is the number of coil pairs.
    resize_factors : torch.Tensor
        Resize factors for each dimension because ksp is zero-filled to a different fov.
    tik : float, optional
        Tikhonov regularisation parameter, by default 0
    verbose : int, optional
        Verbosity level, by default 0

    Returns
    -------
    torch.Tensor, shape (M, N*CP, CP)
        weights for M batches, N neighbors and C coil pairs.
    """
    with Timer(
        name="Batch shift combinations",
        text="Created Batches of shift combinations in {:.2f} ms.",
        current_level=verbose,
        trigger_level=5,
    ):
        # Get the shifts of all patch groups as tensor
        # with shape (num_dim, N * N + 2N, len(patch_groups))
        shift_combinations = torch.stack(
            [pg.combined_shifts for pg in patch_groups], dim=-1
        )
        num_neighbors = patch_groups[0].num_neighbors
        num_dim = patch_groups[0].num_dim
        num_neighbor_combs = shift_combinations.shape[1]
        num_patch_groups = len(patch_groups)

        # Ravel the patch dimension to have 2D locations
        shift_combinations = shift_combinations.reshape(num_dim, -1)

        # scale shifts
        shift_combinations = shift_combinations / resize_factors[:, None]

    verbose_print(
        verbose, 5, "#Batches, #Neighbors: [%d, %d]", num_patch_groups, num_neighbors
    )

    with Timer(
        name="Bilinear Interpolation",
        text="Bilinear Interpolation patch groups in {:.2f} ms",
        current_level=verbose,
        trigger_level=5,
    ):
        shift_signal = _bilinear_interpolate(
            ksp,
            shift_combinations,
        )

        # Reshape to (C, num_neigh_comb, num_patch_groups)
        shift_signal = shift_signal.reshape(
            shift_signal.shape[0], num_neighbor_combs, num_patch_groups
        )
        shift_signal = shift_signal.permute(2, 0, 1)

    verbose_print(
        verbose,
        5,
        "[#Batches, #Coilpairs, #NeighborCombinations]: %s",
        list(shift_signal.shape),
    )

    # Build the calibration matrices
    with Timer(
        name="Build block matrices",
        text="Built block matrices in {:.2f} ms",
        current_level=verbose,
        trigger_level=5,
    ):
        AhA, AhB = _build_block_matrices(
            coil_pair_ind=cp_ind,
            ksp_shift_comb=shift_signal.contiguous(),
        )

    verbose_print(
        verbose,
        5,
        "\n".join(
            [
                f"AhA : [#Batches, #Rows, #Cols]: {list(AhA.shape)}",
                f"AhB : [#Batches, #Rows, #Cols]: {list(AhB.shape)}",
            ]
        ),
    )

    # Solve the linear system
    with Timer(
        name="Solve linear system",
        text="Solved linear system in {:.2f} ms",
        current_level=verbose,
        trigger_level=5,
    ):
        weights = _solve_hpd_batched(
            AhA=AhA, AhB=AhB, tik=tik, symmetrize=False, verbose=verbose
        )

    verbose_print(
        verbose,
        5,
        "Weights : [#Batches, #CoilPairs, #Neighbors] : %s",
        list(weights.shape),
    )

    return weights


@torch.inference_mode()
def _bilinear_interpolate(
    ksp: torch.Tensor,
    locs: torch.Tensor,
) -> torch.Tensor:
    """Interpolate k-space data at given locations using bilinear interpolation.

    Parameters
    ----------
    ksp : torch.Tensor, shape (C, *spatial_dims)
        The k-space data with C channels and 2 or 3spatial dimensions.
    locs : torch.Tensor, shape (D, N)
        Locations in pixel offsets from the ksp tensor center,
        where N is the number of locations.

    Returns
    -------
    torch.Tensor, shape (C, N)
        Interpolated k-space data at the given locations.
    """
    D, N = locs.shape
    C, *spatial_dims = ksp.shape

    # Check if the number of dimensions is valid
    if D != len(spatial_dims):
        raise ValueError(
            f"Number of dimensions in locs ({D}) "
            f"does not match number of spatial dimensions in ksp ({len(spatial_dims)})."
        )

    # Make ksp real-valued for grid_sample and put complex dimension in channel dim
    ksp = torch.view_as_real(ksp)
    ksp = ksp.moveaxis(-1, 1).reshape(-1, *spatial_dims)

    # Normalize locations
    spatial_dims_ten = torch.tensor(
        spatial_dims, dtype=torch.float32, device=ksp.device
    )
    locs = locs + spatial_dims_ten[:, None] / 2
    locs = 2 * locs / (spatial_dims_ten[:, None] - 1) - 1

    # Reorder locs to match grid_sample format
    locs = locs.flip(0).T

    # Shape grid for grid_sample
    if D == 2:
        locs = locs.view(1, 1, N, 2)
        ksp = ksp.unsqueeze(0)
        out = grid_sample(ksp, locs, mode="bilinear", align_corners=True)
        out = out.squeeze(0).reshape(C, 2, -1).movedim(1, -1)
    else:  # S == 3
        locs = locs.view(1, 1, 1, N, 3)
        ksp = ksp.unsqueeze(0)
        out = grid_sample(ksp, locs, mode="bilinear", align_corners=True)
        out = out.squeeze(0).reshape(C, 2, -1).movedim(1, -1)

    return torch.view_as_complex(out.contiguous())


@torch.inference_mode()
def _build_block_matrices(
    coil_pair_ind: torch.Tensor,
    ksp_shift_comb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build block matrices AhA and AhB for the linear system
    based on the coil pair indices and k-space shift combinations.

    Parameters
    ----------
    coil_pair_ind : torch.Tensor, shape (2, P)
        Indices of coil pairs, where P = C*(C+1)/2.
        Each column contains indices of a coil pair (ci, cj).
    ksp_shift_comb : torch.Tensor, shape (M, P, N*N + 2*N)
        Signal data from neighbor shifts for each coil pair for M batches.
    """
    device = ksp_shift_comb.device
    dtype = torch.complex64

    # --- derive sizes ---
    num_cha = int(coil_pair_ind.max().item() + 1)
    num_neigh = int(round(math.sqrt(ksp_shift_comb.shape[2] + 1)) - 1)
    M, P, _ = ksp_shift_comb.shape
    N = num_cha * num_neigh

    # --- outputs ---
    AhA = torch.zeros((M, N, N), dtype=dtype, device=device)
    AhB = torch.zeros((M, N, num_cha), dtype=dtype, device=device)

    # Helpful constants
    nn = num_neigh
    nn2 = nn * nn

    def row(c):
        return slice(c * nn, (c + 1) * nn)  # range of rows/cols for a coil

    # --- main loop over coil pairs (vectorized over batches) ---
    # For each pair, we:
    #  - take block (M, nn, nn), transpose last two dims to match your original,
    #  - take vec1 (M, nn), vec2 (M, nn),
    #  - place them into AhA and AhB.
    for p, (ci, cj) in enumerate(coil_pair_ind.T.tolist()):
        r1, r2 = row(ci), row(cj)

        chunk = ksp_shift_comb[:, p, :]
        blocks = chunk[:, :nn2].reshape(M, nn, nn).transpose(1, 2)
        vec1 = chunk[:, nn2 : nn2 + nn]
        vec2 = chunk[:, nn2 + nn :]

        # Place A[i,j] block and B rows
        AhA[:, r1, r2] = blocks
        AhB[:, r1, cj] = vec1

        # If off-diagonal, also place the conjugate-symmetric counterparts
        if ci != cj:
            AhA[:, r2, r1] = blocks.transpose(1, 2).conj()
            AhB[:, r2, ci] = vec2.conj()

    return AhA, AhB


@torch.no_grad()
def _solve_hpd_batched(
    AhA: torch.Tensor,
    AhB: torch.Tensor,
    tik: float = 0.0,
    symmetrize: bool = True,
    chol_retries=(1e-8, 1e-6, 1e-4),
    use_qr_fallback: bool = True,
    verbose: int = 0,
) -> torch.Tensor:
    """
    Solves the batched linear system (AhA + λ I) X = AhB for X.
    AhA is assumed to be Hermitian positive definite (HPD) in each batch,
    but may be ill-conditioned or slightly non-Hermitian due to numerical errors.
    Tikhonov regularization with λ >= 0 can be used to improve conditioning.
    If Cholesky fails, retries with extra diagonal loading are attempted.
    If still failing, a least-squares solution via QR is used as a fallback.

    Parameters
    ----------
    AhA : torch.Tensor, shape (M, N, N)
        Batched Hermitian positive definite matrices with M batches.
    AhB : torch.Tensor, shape (M, N, C)
        Batched right-hand side matrices.
    tik : float, optional
        Tikhonov regularization parameter (default is 0.0).
    symmetrize : bool, optional
        Whether to enforce exact Hermitian structure on AhA (default is True).
    chol_retries : tuple, optional
        Extra diagonal loading values to try
        if Cholesky fails (default is (1e-8, 1e-6, 1e-4)).
    use_qr_fallback : bool, optional
        Whether to use QR least-squares fallback if all else fails (default is True).
    verbose : int, optional
        Verbosity level for debugging. Default is 0 (no output). Maximum is 5.

    Returns
    -------
    torch.Tensor, shape (M, N, C)
        Solution matrices for each batch.
    """
    # Validate input shape
    _validate_hpd_inputs(AhA, AhB)

    M, N, _ = AhA.shape

    A = AhA.contiguous()
    B = AhB.contiguous()

    # Make exactly Hermitian if desired (cheap O(N^2) projection)
    if symmetrize:
        A = 0.5 * (A + A.mH)
        verbose_print(
            verbose, 5, "Symmetrized AhA matrix to ensure exact Hermitian structure."
        )

    # Add λI (broadcast over batch)
    Eye = torch.eye(N, dtype=A.dtype, device=A.device)
    A = A + tik * Eye
    verbose_print(verbose, 5, f"Added Tikhonov regularization with λ={tik}.")

    # --- Fast path: Cholesky on the whole batch ---
    with Timer(
        name="Cholesky decomposition",
        text="Cholesky decomposition took {:.2f} ms.",
        current_level=verbose,
        trigger_level=5,
    ):
        L, info = torch.linalg.cholesky_ex(A, upper=False)  # info: (M,)
        X = torch.empty_like(B)

    ok = info == 0
    verbose_print(verbose, 5, f"Cholesky succeeded for {ok.sum()} out of {M} batches.")
    if ok.any():
        with Timer(
            name="Cholesky solve",
            text="Cholesky solve took {:.2f} ms.",
            current_level=verbose,
            trigger_level=5,
        ):
            X[ok] = torch.cholesky_solve(B[ok], L[ok], upper=False)

    remaining = ~ok
    if remaining.any():
        # Retry failing batches with extra diagonal loading
        Eye = torch.eye(N, dtype=A.dtype, device=A.device)  # ensure defined
        idx = remaining.nonzero(as_tuple=False).squeeze(-1)
        verbose_print(
            verbose,
            5,
            f"Cholesky failed for {idx.numel()} batches, retrying with extra loads.",
        )

        with Timer(
            name="Cholesky retries",
            text="Cholesky retries took {:.2f} ms.",
            current_level=verbose,
            trigger_level=5,
        ):
            for extra in chol_retries:
                if idx.numel() == 0:
                    break
                A_try = A[idx] + extra * Eye
                L_try, info_try = torch.linalg.cholesky_ex(A_try, upper=False)
                ok2 = info_try == 0
                if ok2.any():
                    good_idx = idx[ok2]
                    X[good_idx] = torch.cholesky_solve(
                        B[good_idx], L_try[ok2], upper=False
                    )
                    # keep only the still-failing indices
                    idx = idx[~ok2]

        verbose_print(
            verbose,
            5,
            f"Cholesky retries succeeded for {remaining.sum() - idx.numel()} batches.",
        )

        # Final fallbacks for any still-failing systems
        if idx.numel() > 0:
            try:
                verbose_print(
                    verbose,
                    5,
                    f"Falling back to direct solve for {idx.numel()} batches.",
                )
                with Timer(
                    name="Direct solve fallback",
                    text="Direct solve fallback took {:.2f} ms.",
                    current_level=verbose,
                ):
                    # Direct solve (slower, but works for non-HPD too)
                    X[idx] = torch.linalg.solve(A[idx], B[idx])
                idx = idx.new_empty(0)  # all done
            except RuntimeError:
                pass  # fall back further below if requested

            if idx.numel() > 0 and use_qr_fallback:
                verbose_print(
                    verbose,
                    5,
                    f"Falling back to least-squares via QR for {idx.numel()} batches.",
                )
                with Timer(
                    name="QR least-squares fallback",
                    text="QR least-squares fallback took {:.2f} ms.",
                    current_level=verbose,
                    trigger_level=5,
                ):
                    # Least-squares via QR (works for singular/non-HPD cases)
                    X[idx] = torch.linalg.lstsq(A[idx], B[idx], driver="gels").solution

    return X


def _validate_hpd_inputs(AhA, AhB):
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
        raise ValueError("AhA must be complex64 or complex128")
