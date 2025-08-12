import time
from typing import Optional, Union

import torch
from scipy.spatial import KDTree
from torch_geometric.utils import lexsort
from tqdm import tqdm


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
        if isinstance(center_ind, int):
            self.center_ind = torch.tensor(center_ind, device=ktraj.device)
        elif isinstance(center_ind, torch.Tensor):
            self.center_ind = center_ind
        else:
            raise TypeError("center_ind must be an int or a torch.Tensor.")

        self.ktraj = ktraj
        self.do_sift = do_sift
        self.verbose = verbose
        self.device = device if device is not None else ktraj.device
        self.num_dim = ktraj.shape[0] - 1  # Last dim is sampling mask

    @property
    def center_loc(self) -> torch.Tensor:
        """Return the location of the center sample."""
        return self.ktraj[:-1, self.center_ind]

    @property
    def num_neighbors(self) -> int:
        """Return the number of neighbors."""
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

        progress_bar = tqdm(
            total=len(patch_indices),
            desc="Creating patches with neighbors.",
            disable=(verbose < 2),
        )
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

            progress_bar.update(1)

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


def _get_patch_indices(
    ktraj: torch.Tensor,
    kernel_size: torch.Tensor,
    p_norm: float = torch.inf,
    threads: int = -1,
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
        print(f"Sampled points shape: {len(ind_sampled)}")
        print(f"Unsampled points shape: {len(ind_unsampled)}")

    kdtree_sampled = KDTree(ktraj_scaled[:, sample_mask].cpu().numpy().T)
    kdtree_unsampled = KDTree(ktraj_scaled[:, inv_sample_mask].cpu().numpy().T)

    radius = 1.0 + 1e-6  # Buffer for floating point precision

    neighbor_inds = kdtree_unsampled.query_ball_tree(kdtree_sampled, r=radius, p=p_norm)

    def comb(center_ind, neighbor_inds):
        return torch.cat((center_ind.unsqueeze(0), neighbor_inds), dim=0)

    inds_c = [
        comb(ind_unsampled[i], ind_sampled[n]) for i, n in enumerate(neighbor_inds)
    ]

    stop_time = time.time() - start_time
    if verbose > 2:
        print(f"Found {len(inds_c)} patches.")
        print(f"KDTree search completed in {(1e3 * stop_time):.3f} ms. \n")

    return inds_c
