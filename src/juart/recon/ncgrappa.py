from collections import defaultdict
from typing import Optional, Union

import torch
from scipy.spatial import KDTree
from torch_geometric.utils import lexsort


class NonCartGrappa:
    def __init__(
        self,
        ktraj: torch.Tensor,
        calib_signal: torch.Tensor,
        img_size: Union[tuple[int, ...], list[int], int],
        kernel_size: Union[tuple[int, ...], list[int], int] = 5,
        device: Optional[torch.device] = None,
        p_norm: float = torch.inf,
        threads: int = -1,
        # img_size: Optional[Union[Tuple[int], torch.Tensor, list[int]]] = None,+
    ):
        """Non cartesian GRAPPA reconstruction.
        This class is used to perform non-cartesian GRAPPA reconstruction
        on k-space data.


        Parameters
        ----------
        ktraj : torch.Tensor, shape (D+1, N)
            _description_
        calib_signal : torch.Tensor, shape (C, R, P1, P2)
            _description_
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
        """
        if device is None:
            device = calib_signal.device
        else:
            ktraj = ktraj.to(device)
            calib_signal = calib_signal.to(device)

        self.threads = threads
        self.num_dim = ktraj.shape[0] - 1

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

        self.ktraj = ktraj
        self.ktraj[: self.num_dim, :] = (
            self.ktraj[: self.num_dim, :] * self.img_size[:, None]
        )  # scale to cycle/fov

        # Normalize acs kernel by sum of squares
        self.calib_signal = calib_signal / torch.sqrt(
            torch.sum(calib_signal**2, dim=0, keepdim=True)
        )

        # Define limitation of the kernel shape
        self.p_norm = p_norm
        self.p_mask = _norm_ball_mask(kernel_size, p=self.p_norm)

        # Get indices of neighbors for each unsampled point in k-space
        self.patches = NCG_Patch.from_ktraj(
            self.ktraj_sampled,
            self.ktraj_unsampled,
            self.kernel_size,
            p_norm=self.p_norm,
            threads=self.threads,
        )

    @property
    def sample_mask(self) -> torch.Tensor:
        """Mask of sampled points in k-space."""
        return self.ktraj[-1, :].bool()

    @property
    def ktraj_sampled(self) -> torch.Tensor:
        """Sampled k-space trajectory."""
        return self.ktraj[: self.num_dim, self.sample_mask]

    @property
    def ktraj_unsampled(self) -> torch.Tensor:
        """Unsampled k-space trajectory."""
        return self.ktraj[: self.num_dim, ~self.sample_mask]


class NCG_PatchGroup:
    """A group of patches that share the same shift pattern."""

    def __init__(self, patches: list["NCG_Patch"], tol: float = 1e-6):
        if not patches:
            raise ValueError("Cannot create a patch group from an empty list.")
        # Check that all patches have the same number of neighbors
        num_neighbors_set = {patch.num_neighbors for patch in patches}
        if len(num_neighbors_set) > 1:
            raise ValueError(
                f"All patches in a group must have the same number of neighbors, "
                f"but got counts: {num_neighbors_set}"
            )
        self.patches = patches
        self.kernel = None  # To be filled during calibration

    @property
    def shift_pattern(self) -> torch.Tensor:
        """Return the integer shift pattern (D, N), or None if neighbor_dist is None."""
        return self.patches[0].get_int_shift_pattern()

    @property
    def num_dim(self) -> int:
        """Number of dimensions in the patches."""
        return self.patches[0].num_dim

    @property
    def num_patches(self) -> int:
        return len(self.patches)

    @property
    def num_neighbors(self) -> int:
        return self.patches[0].num_neighbors

    @classmethod
    def create_patch_groups(
        cls, patches: list["NCG_Patch"], tol: float = 1e-6
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

        # Step 1: Group by number of neighbors
        neighbor_buckets: defaultdict[int, list[NCG_Patch]] = defaultdict(list)
        for patch in patches:
            neighbor_buckets[patch.num_neighbors].append(patch)

        # Step 2: For each bucket, subgroup by distance pattern of neighbors
        for _, patch_list in neighbor_buckets.items():
            # Init matrix with all shifts of all patches
            shift_matrix = torch.zeros(
                (len(patch_list), patch_list[0].num_dim * patch_list[0].num_neighbors),
                dtype=torch.float32,
            )
            for i, patch in enumerate(patch_list):
                shift_matrix[i] = patch.get_round_shift_pattern(tol=tol).flatten()

            # Find unique patches
            unique_rows, inverse_indices = torch.unique(
                shift_matrix, dim=0, return_inverse=True
            )

            # Create groups based on unique shift patterns and the number of neighbors
            grouped_dict = defaultdict(list)
            for patch, group_idx in zip(patch_list, inverse_indices):
                grouped_dict[group_idx.item()].append(patch)

            patch_groups.extend([cls(group) for group in grouped_dict.values()])

        return patch_groups


class NCG_Patch:
    """
    Patch of a constellation for an unsampled point in the center of the patch and
    its sampled neighbors.
    """

    def __init__(
        self,
        center_ind: int,
        center_loc: torch.Tensor,
        neighbor_inds: Optional[list[int]] = None,
        neighbor_locs: Optional[torch.Tensor] = None,
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
            Indices of the sampled neighbors in k-space.
        neighbor_locs : torch.Tensor, shape (D, N)
            Locations of the sampled neighbors in k-space.
        do_sift : bool, optional
            Sift neighbors to remove duplicates.

        """
        self.center_ind = center_ind
        self.center_loc = center_loc

        if self.center_loc.dim() != 2 or self.center_loc.shape[1] != 1:
            self.center_loc = self.center_loc[:, None]  # Ensure shape (D, 1)

        self.num_dim = center_loc.shape[0]

        # Ensure neighbors are initialized as empty structures if not provided
        if neighbor_inds is None or neighbor_locs is None:
            neighbor_locs = torch.empty(
                self.num_dim, 0, dtype=center_loc.dtype, device=center_loc.device
            )
            neighbor_inds = []

        self.neighbor_locs = neighbor_locs
        self.neighbor_inds = neighbor_inds

        self.neighbor_dist = neighbor_locs - self.center_loc

        if self.num_neighbors > 0:
            if do_sift:
                self.sift_neighbors()  # Remove duplicates
            self.sort_neighbors()  # Sort by distance

    def sift_neighbors(self):
        """Remove neighbors with similar relative distance to the center point.
        The distances are rounded to integer values before."""

        if self.num_neighbors == 0:
            return None

        # Get unique distances and their indices
        _, unique_indices = torch.unique(
            self.get_int_shift_pattern(),
            dim=1,
            return_inverse=True,
        )

        # Sift is doing some nonsense (return [0, 0] for two neighbors with the same
        # distance)
        # So we need to filter out the duplicates manually
        unique_indices = torch.unique(unique_indices)

        # Keep only the unique ones
        self.neighbor_locs = self.neighbor_locs[:, unique_indices]
        self.neighbor_inds = [self.neighbor_inds[i] for i in unique_indices.tolist()]
        self.neighbor_dist = self.neighbor_locs - self.center_loc

    def sort_neighbors(self) -> None:
        """Sort neighbors by distance to the center point.
        The distance is rounded to integer values and sorted lexicographically."""
        if self.num_neighbors == 0:
            return None

        neighbor_dist_int = self.get_int_shift_pattern()

        # Sort the neighbors by rounded distance to the center point
        keys = [neighbor_dist_int[dim, :] for dim in reversed(range(self.num_dim))]
        sorted_indices = lexsort(
            keys
        )  # sort by last dimension first (that's why we reversed the order)

        # Apply sorting
        self.neighbor_locs = self.neighbor_locs[:, sorted_indices]
        self.neighbor_inds = [self.neighbor_inds[i] for i in sorted_indices]
        self.neighbor_dist = self.neighbor_dist[:, sorted_indices]

    def get_int_shift_pattern(self) -> torch.Tensor:
        """Return the integer shift pattern (D, N)."""
        if self.num_neighbors == 0:
            return torch.empty((self.num_dim, 0), dtype=torch.int32)
        return torch.round(self.neighbor_dist).int()

    def get_round_shift_pattern(self, tol: float = 1e-6) -> torch.Tensor:
        """Return the rounded shift pattern (D, N)."""
        if self.num_neighbors == 0:
            return torch.empty((self.num_dim, 0), dtype=torch.float32)
        return torch.round(self.neighbor_dist / tol) * tol

    @property
    def num_neighbors(self) -> int:
        """Number of neighbors in the patch."""
        return self.neighbor_locs.shape[1]

    @classmethod
    def from_indices(
        cls,
        inds_c: list[int],
        ktraj_sampled: torch.Tensor,
        ktraj_unsampled: torch.Tensor,
        do_sift: bool = True,
    ) -> "NCG_Patch":
        """Create patches from indices of neighbors."""
        if len(inds_c) == 1:
            instance = cls(
                center_ind=inds_c[0],
                center_loc=ktraj_unsampled[:, inds_c[0]],
                do_sift=do_sift,
            )
        else:
            instance = cls(
                center_ind=inds_c[0],
                center_loc=ktraj_unsampled[:, inds_c[0]],
                neighbor_inds=inds_c[1:],
                neighbor_locs=ktraj_sampled[:, inds_c[1:]],
                do_sift=do_sift,
            )
        return instance

    @classmethod
    def from_indices_list(
        cls,
        inds_c_list: list[list[int]],
        ktraj_sampled: torch.Tensor,
        ktraj_unsampled: torch.Tensor,
        do_sift: bool = True,
    ) -> list["NCG_Patch"]:
        """Create a list of NCG_Patch instances from a list of neighbor indices."""
        return [
            cls.from_indices(inds_c, ktraj_sampled, ktraj_unsampled, do_sift=do_sift)
            for inds_c in inds_c_list
        ]

    @classmethod
    def from_ktraj(
        cls,
        ktraj_sampled: torch.Tensor,
        ktraj_unsampled: torch.Tensor,
        kernel_size: torch.Tensor,
        p_norm: float = torch.inf,
        threads: int = -1,
        do_sift: bool = True,
    ) -> list["NCG_Patch"]:
        """Create patches from k-space trajectory using a KDTree search."""
        inds_c_list = _get_neighbor_indices(
            traj_sampled=ktraj_sampled,
            traj_unsampled=ktraj_unsampled,
            kernel_size=kernel_size,
            p_norm=p_norm,
            threads=threads,
        )
        return cls.from_indices_list(
            inds_c_list, ktraj_sampled, ktraj_unsampled, do_sift=do_sift
        )


def _get_neighbor_indices(
    traj_sampled: torch.Tensor,
    traj_unsampled: torch.Tensor,
    kernel_size: torch.Tensor,
    p_norm: float = torch.inf,
    threads: int = -1,
) -> list[list[int]]:
    """
    Get indices of neighbors for each unsampled point in k-space using KDTree search.
    """
    norm_factor = kernel_size / 2.0
    traj_sampled = traj_sampled / norm_factor[:, None]
    traj_unsampled = traj_unsampled / norm_factor[:, None]

    kdtree = KDTree(traj_sampled.cpu().numpy().T)
    radius = 1.0 + 1e-6  # Buffer for floating point precision

    neighbors = kdtree.query_ball_point(
        traj_unsampled.cpu().numpy().T, r=radius, p=p_norm, workers=threads
    )

    inds_c = [[i] + n for i, n in enumerate(neighbors)]
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
