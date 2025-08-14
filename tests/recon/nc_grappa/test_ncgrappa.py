import sys
from typing import Optional

import pytest
import torch

from juart.recon import ncgrappa_2

torch.manual_seed(2)  # Set a fixed seed for reproducibility


def filled_and_empty_patch_locs(
    num_filled_patches: int = 5,
    num_empty_patches: int = 5,
    num_dim: int = 2,
    num_neighbors: int = 10,
    kernel_size: int = 7,
    k_max: float = 100.0,
    device: Optional[torch.device] = None,
) -> list[torch.Tensor]:
    """Create k-space location data for multiple patches.
    Returns a list of (num_filled_patches+num_empty_patches)
    patch locations of shape (num_dim+1, num_neighbors)"""

    if device is None:
        device = torch.device("cpu")
    else:
        device = torch.device(device)

    patches = []
    centers = []

    while len(patches) < (num_filled_patches + num_empty_patches):
        # Generate a candidate center location
        candidate_center = (
            torch.rand(num_dim, 1, device=device) - 0.5
        ) * k_max - kernel_size

        candidate_center = torch.clamp(
            candidate_center,
            min=-(k_max - kernel_size),
            max=(k_max - kernel_size),
        )

        # Check distance to all existing centers
        if centers:
            dists = torch.stack(
                [torch.norm(candidate_center.squeeze() - c.squeeze()) for c in centers]
            )
            if torch.any(dists < kernel_size):
                continue

        # Generate patch with this center
        candidate_center = torch.cat((candidate_center, torch.zeros(1, 1)), dim=0)

        if len(patches) < num_filled_patches:
            neibhor_locs = (
                torch.rand(num_dim, num_neighbors, device=device) - 0.5
            ) * kernel_size

            neibhor_locs = torch.cat(
                (neibhor_locs, torch.ones(1, neibhor_locs.shape[1])), dim=0
            )

            neibhor_locs[:-1, :] = neibhor_locs[:-1, :] + candidate_center[:-1, :]

            locs = torch.cat((candidate_center, neibhor_locs), dim=1)
        else:
            # Create an empty patch with no neighbors
            locs = candidate_center.clone()

        patches.append(locs)
        centers.append(candidate_center[:-1])

    return patches


class KtrajTestPatches:
    def __init__(
        self,
        num_dim: int = 2,
        num_neighbors: int = 10,
        num_filled_patches: int = 5,
        num_empty_patches: int = 5,
        kernel_size: int = 7,
        k_max: float = 100.0,
        device: Optional[torch.device] = None,
    ):
        self.num_dim = num_dim
        self.num_neighbors = num_neighbors
        self.num_filled_patches = num_filled_patches
        self.num_empty_patches = num_empty_patches
        self.k_max = k_max
        self.kernel_size = kernel_size
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        if self.kernel_size * self.num_patches > 2 * self.k_max:
            raise ValueError(
                "Center locations are to near to each other."
                "Reduce kernel_size or number of patches or increase k_max."
            )

        # Create a list of patch locations
        self.patch_locs = filled_and_empty_patch_locs(
            num_filled_patches=self.num_filled_patches,
            num_empty_patches=self.num_empty_patches,
            num_dim=self.num_dim,
            num_neighbors=self.num_neighbors,
            kernel_size=self.kernel_size,
            device=self.device,
        )

        for i, locs in enumerate(self.patch_locs):
            # Mark empty patches and patch number
            if locs.shape[1] == 1:  # Empty patch
                patch_mark = i * torch.ones((2, 1), device=device)
                patch_mark[0] = 0
                locs = torch.cat((locs, patch_mark), dim=0)
            else:
                patch_mark = i * torch.ones((2, locs.shape[1]), device=device)
                patch_mark[0] = 1
                locs = torch.cat((locs, patch_mark), dim=0)
            self.patch_locs[i] = locs

        k = torch.cat(self.patch_locs, dim=1)
        self.ktraj_order = torch.randperm(k.shape[1], device=self.device)
        self.flat_patch_locs = k[:, self.ktraj_order]

        self.patch_indices = []
        for _, locs in enumerate(self.patch_locs):
            n_patch = int(locs[-1, 0])
            patch_indices = self.get_patch_indices(n_patch)
            self.patch_indices.append(patch_indices)

    @property
    def num_patches(self):
        return self.num_filled_patches + self.num_empty_patches

    def get_patch_instances(
        self,
    ) -> list[ncgrappa_2.FilledPatch | ncgrappa_2.EmptyPatch]:
        """Get the patch instances for all patches."""
        patch_instances = []
        for locs, indices in zip(self.patch_locs, self.patch_indices):
            if locs.shape[1] == 1:  # Empty patch
                patch = ncgrappa_2.EmptyPatch(
                    self.flat_patch_locs[: (self.num_dim + 1), :],  # not use markings
                    center_ind=indices[0],
                    device=self.device,
                )
                patch_instances.append(patch)
            else:
                patch = ncgrappa_2.FilledPatch(
                    self.flat_patch_locs[: (self.num_dim + 1), :],  # not use markings
                    center_ind=indices[0],
                    neighbor_inds=indices[1:],
                    device=self.device,
                )
                patch_instances.append(patch)
        return patch_instances

    def get_patch_numbers(self, loc: torch.Tensor) -> torch.Tensor:
        """Get the N patch numbers for a given k-space
        location array of shape (num_dim, N)."""
        if loc.shape[0] != self.num_dim:
            raise ValueError(
                f"Location must have {self.num_dim} dimensions, "
                f"but got {loc.shape[0]} dimensions."
            )

        if loc.ndim == 1:
            loc = loc.unsqueeze(1)
        elif loc.ndim > 2:
            raise ValueError("Location must be a 1D or 2D tensor.")

        ktraj = self.flat_patch_locs[: self.num_dim, :]

        matching_indices = torch.isin(loc, ktraj)

        return ktraj[-1, matching_indices]  # Last row contains patch numbers

    def get_patch_indices(self, patch_number: int) -> torch.Tensor:
        """Get the indices of a specific patch number."""
        if patch_number < 0 or patch_number >= self.num_patches:
            raise ValueError("Patch number out of range.")

        indices = torch.where(self.flat_patch_locs[-1, :] == patch_number)[0]

        return indices


def single_patch_locs(
    num_dim: int = 2,
    num_neighbors: int = 10,
    kernel_size: int = 7,
    k_max: float = 100.0,
) -> torch.Tensor:
    """Create k-space location data for a patch
    with one center and several neighbors.
    Shape is (num_dim+1, num_neighbors) with
    the last axis of the first dim being the sampling mask.
    """
    center_loc = (torch.rand(num_dim, 1) - 0.5) * k_max

    center_loc = torch.cat((center_loc, torch.zeros(1, center_loc.shape[1])), dim=0)

    neibhor_locs = (torch.rand(num_dim, num_neighbors) - 0.5) * kernel_size
    neibhor_locs = torch.cat(
        (neibhor_locs, torch.ones(1, neibhor_locs.shape[1])), dim=0
    )

    neibhor_locs[:-1, :] = neibhor_locs[:-1, :] + center_loc[:-1, :]

    locs = torch.cat((center_loc, neibhor_locs), dim=1)

    return locs


def empty_patch_locs(
    num_dim: int = 2,
    k_max: float = 100.0,
) -> torch.Tensor:
    """Create k-space location data for an empty patch"""
    center_loc = (torch.rand(num_dim, 1) - 0.5) * k_max
    center_loc = torch.cat((center_loc, torch.zeros(1, center_loc.shape[1])), dim=0)
    return center_loc


def multiple_patch_locs(
    num_patches: int = 10,
    num_dim: int = 2,
    num_neighbors: int = 10,
    kernel_size: int = 7,
    k_max: float = 100.0,
) -> torch.Tensor:
    """Create k-space location data for multiple patches.
    Shape of the output is (num_dim+1, num_neighbors, num_patches)"""

    patches = []
    centers = []

    while len(patches) < num_patches:
        # Generate a candidate center location
        candidate_center = (torch.rand(num_dim, 1) - 0.5) * k_max - kernel_size
        candidate_center = torch.clamp(
            candidate_center,
            min=-(k_max - kernel_size),
            max=(k_max - kernel_size),
        )

        # Check distance to all existing centers
        if centers:
            dists = torch.stack(
                [torch.norm(candidate_center.squeeze() - c.squeeze()) for c in centers]
            )
            if torch.any(dists < kernel_size):
                continue
        # Generate patch with this center

        candidate_center = torch.cat((candidate_center, torch.zeros(1, 1)), dim=0)

        neibhor_locs = (torch.rand(num_dim, num_neighbors) - 0.5) * kernel_size
        neibhor_locs = torch.cat(
            (neibhor_locs, torch.ones(1, neibhor_locs.shape[1])), dim=0
        )

        neibhor_locs[:-1, :] = neibhor_locs[:-1, :] + candidate_center[:-1, :]

        locs = torch.cat((candidate_center, neibhor_locs), dim=1)

        patches.append(locs)
        centers.append(candidate_center[:-1])

    locs = torch.stack(patches, dim=-1)

    return locs


### Start of test functions
# fmt : off
@pytest.mark.parametrize(
    "d, expected",
    [
        (torch.tensor([[0.2, 0.1, 0.3], [0.5, 0.4, 0.6]]), torch.tensor([1, 0, 2])),
        (
            torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 4.0, 4.0]]),
            torch.tensor([0, 1, 2, 3]),
        ),
        (
            torch.tensor(
                [
                    [1.0, 0.0, 1.0, 0.0, 1.0],
                    [2.0, 5.0, 1.0, 5.0, 1.0],
                    [9.0, 0.0, 3.0, 2.0, 1.0],
                ]
            ),
            torch.tensor([1, 3, 4, 2, 0]),
        ),
        (
            torch.tensor(
                [[5.0, 5.0, 5.0, 5.0], [2.0, 1.0, 1.0, 3.0], [2.0, 9.0, 3.0, 0.0]]
            ),
            torch.tensor([2, 1, 0, 3]),
        ),
    ],
)  # fmt : on
def test_distance_sort(d, expected):
    assert torch.equal(ncgrappa_2.distance_sort(d), expected)


# fmt : off
@pytest.mark.parametrize(
    "k,idx",
    [
        (torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 2.0]]), [0, 2, 3]),
        (torch.tensor([[0.49, 0.51, 0.49], [1.49, 1.51, 1.49]]), [0, 1]),
        (
            torch.tensor(
                [
                    [1.0, 0.0, 1.0, 0.0, 1.0],
                    [2.0, 5.0, 1.0, 5.0, 1.0],
                    [9.0, 0.0, 3.0, 0.0, 1.0],
                ]
            ),
            [0, 1, 2, 4],
        ),
        (torch.tensor([[2.0, 1.0, 2.0, 1.0], [0.0, 0.0, 0.0, 0.0]]), [0, 1]),
    ],
)  # fmt : on
def test_sift_mask_compact(k, idx):
    exp = torch.zeros(k.shape[1], dtype=torch.bool)
    exp[idx] = True
    assert torch.equal(ncgrappa_2.sift_mask(k), exp)


@pytest.mark.parametrize(
    "k, k_neigh_expect, sift_mask_expect, do_sift",
    [
        (
            torch.tensor(
                [
                    [0.6, 3.2, -2.3, -1.1, 1.0, 2.0, -3.0],
                    [1.4, 0.0, 3.1, -2, -1.2, -1.5, 2.0],
                    [0, 1, 1, 1, 1, 1, 1],
                ]
            ),
            torch.tensor(
                [[-3.0, -2.3, -1.1, 1.0, 2.0, 3.2], [2.0, 3.1, -2.0, -1.2, -1.5, 0.0]]
            ),
            torch.tensor([True, True, True, True, True, True]),
            False,
        ),
        (
            torch.tensor(
                [
                    [0.6, 3.2, -2.3, -1.1, 1.0, 1.001, 2.0, -3.0],
                    [1.4, 0.0, 3.1, -2, -1.2, -1.202, -1.5, 2.0],
                    [0, 1, 1, 1, 1, 1, 1, 1],
                ]
            ),
            torch.tensor(
                [[-3.0, -2.3, -1.1, 1.0, 2.0, 3.2], [2.0, 3.1, -2.0, -1.2, -1.5, 0.0]]
            ),
            torch.tensor([True, True, True, True, False, True, True]),
            True,
        ),
    ],
)
def test_FilledPatch_init(k, k_neigh_expect, sift_mask_expect, do_sift):
    """Test initialization of FilledPatch."""
    ktraj = torch.rand(3, 500)
    ktraj[-1, :] = 2.0  # Last row is the sampling mask

    ktraj = torch.cat((ktraj, k), dim=1)

    # Shuffle ktraj along the second dimension (axis=1)
    perm = torch.randperm(ktraj.shape[1])
    ktraj = ktraj[:, perm]

    center_ind = torch.where(ktraj[-1] == 0)[0]
    neighbor_inds = torch.where(ktraj[-1] == 1)[0]

    k_shift_exp = k_neigh_expect - ktraj[:-1, center_ind]

    patch = ncgrappa_2.FilledPatch(
        ktraj=ktraj,
        center_ind=center_ind,
        neighbor_inds=neighbor_inds,
        do_sift=do_sift,
    )

    assert patch.ktraj.data_ptr() == ktraj.data_ptr()

    assert patch.center_ind == center_ind
    assert patch.center_ind.shape == (1,)

    # Check not regarding order
    assert set(neighbor_inds.tolist()) == set(patch._neighbor_inds.tolist())
    assert patch.center_loc.ndim == 2
    assert patch.center_loc.shape == (ktraj.shape[0] - 1, 1)
    assert torch.equal(patch.center_loc, ktraj[:-1, center_ind])

    assert torch.equal(sift_mask_expect, patch.sift_mask)
    assert sift_mask_expect.sum() == patch.num_neighbors

    assert torch.equal(patch.neighbor_shifts, k_shift_exp)

    # Check with order
    assert torch.equal(patch.neighbor_locs, k_neigh_expect)

    assert patch.center_loc.device == ktraj.device
    assert patch.neighbor_locs.device == ktraj.device
    assert patch.neighbor_inds.device == ktraj.device
    assert patch.sift_mask.device == ktraj.device
    assert patch.sift_mask.sum() == k_neigh_expect.shape[1]


def test_EmptyPatch_init():
    """Test initialization of EmptyPatch."""
    kpatch = empty_patch_locs(num_dim=2, k_max=100.0)
    ktraj = (torch.rand(3, 500) - 0.5) * 100
    ktraj[-1, :] = 2.0

    ktraj = torch.cat((ktraj, kpatch), dim=1)

    # Shuffle ktraj along the second dimension (axis=1)
    perm = torch.randperm(ktraj.shape[1])
    ktraj = ktraj[:, perm]

    center_ind_exp = torch.where(ktraj[-1] == 0)[0]

    patch = ncgrappa_2.EmptyPatch(
        ktraj=ktraj,
        center_ind=center_ind_exp,
    )

    assert patch.ktraj.data_ptr() == ktraj.data_ptr()
    assert patch.center_ind == center_ind_exp
    assert patch.center_ind.shape == (1,)
    assert patch.center_loc.ndim == 2
    assert patch.center_loc.shape == (ktraj.shape[0] - 1, 1)


def test_EmptyPatch_from_indices():
    """Test initialization of EmptyPatch from indices."""
    kpatch = empty_patch_locs(num_dim=2, k_max=100.0)
    ktraj = (torch.rand(3, 500) - 0.5) * 100
    ktraj[-1, :] = 2.0

    ktraj = torch.cat((ktraj, kpatch), dim=1)

    # Shuffle ktraj along the second dimension (axis=1)
    perm = torch.randperm(ktraj.shape[1])
    ktraj = ktraj[:, perm]

    center_ind_exp = torch.where(ktraj[-1] == 0)[0]

    patch = ncgrappa_2.EmptyPatch.from_indices(
        ktraj=ktraj,
        patch_indices=[center_ind_exp],
    )

    assert patch[0].ktraj.data_ptr() == ktraj.data_ptr()
    assert torch.equal(patch[0].center_ind, center_ind_exp)


def test_EmptyPatch_init_notempty():
    kpatch = single_patch_locs(num_dim=2, num_neighbors=10, kernel_size=7, k_max=100.0)
    ktraj = (torch.rand(3, 500) - 0.5) * 100
    ktraj[-1, :] = 2.0

    ktraj = torch.cat((ktraj, kpatch), dim=1)

    # Shuffle ktraj along the second dimension (axis=1)
    perm = torch.randperm(ktraj.shape[1])
    ktraj = ktraj[:, perm]

    patch_indices = torch.where(ktraj[-1] != 2)[0]

    with pytest.raises(ValueError):
        ncgrappa_2.EmptyPatch.from_indices(
            ktraj=ktraj,
            patch_indices=[patch_indices],
        )


def test_get_patch_indices():
    """Test the get_patch_indices function."""

    num_patches = 10
    num_dim = 2
    num_neighbors = 10
    kernel_size = 7

    ktraj = multiple_patch_locs(
        num_patches=num_patches,
        num_dim=num_dim,
        num_neighbors=num_neighbors,
        kernel_size=kernel_size,
    )

    # Add a new axis in the first dimension of ktraj
    patch_ind = torch.zeros((1, *ktraj.shape[1:]), device=ktraj.device)
    for i in range(ktraj.shape[-1]):
        patch_ind[0, :, i] = i

    ktraj = torch.cat((ktraj, patch_ind), dim=0)
    ktraj_flat = ktraj.reshape(ktraj.shape[0], -1)

    patch_inds = ncgrappa_2._get_patch_indices(
        ktraj=ktraj_flat[:-1, :],
        kernel_size=torch.tensor(
            [
                kernel_size,
            ]
            * num_dim,
            device=ktraj.device,
        ),
    )

    assert len(patch_inds) == num_patches

    for _, patch in enumerate(patch_inds):
        center_ind = patch[0]
        neighbor_inds = patch[1:]

        # Center must be unsampled neighbours must be sampled
        assert ktraj_flat[-2, center_ind] == 0.0
        assert torch.all(ktraj_flat[-2, neighbor_inds] == 1.0)

        # Check that all indices in patch correspond
        # to the same patch index in the last row
        patch_indices = ktraj_flat[-1, patch]
        assert torch.all(patch_indices == patch_indices[0])


def test_get_patch_indices_emtpy():
    """Test the get_patch_indices function with an ktraj containing empty patches."""
    num_patches = 10
    num_dim = 2
    num_neighbors = 10
    kernel_size = 7

    patch_loc_list = filled_and_empty_patch_locs(
        num_filled_patches=num_patches - 1,
        num_empty_patches=1,
        num_dim=num_dim,
        num_neighbors=num_neighbors,
        kernel_size=kernel_size,
    )

    # Add a new axis in the first dimension of ktraj
    for i, patch in enumerate(patch_loc_list):
        if patch.shape[1] == 1:  # Empty patch
            patch_ind = i * torch.ones((1, 1), device=patch.device)
        else:
            patch_ind = i * torch.ones((1, patch.shape[1]), device=patch.device)
        patch_loc_list[i] = torch.cat((patch, patch_ind), dim=0)

    ktraj_flat = torch.concat(patch_loc_list, dim=1)

    patch_inds = ncgrappa_2._get_patch_indices(
        ktraj=ktraj_flat[:-1, :],
        kernel_size=torch.tensor(
            [
                kernel_size,
            ]
            * num_dim,
            device=ktraj_flat.device,
        ),
    )

    assert len(patch_inds) == num_patches

    for _, patch in enumerate(patch_inds):
        # Check if the patch is empty
        if patch.shape[0] == 1:
            # Empty patch, only center index
            assert ktraj_flat[-2, patch[0]] == 0
        else:
            center_ind = patch[0]
            neighbor_inds = patch[1:]

            # Center must be unsampled neighbours must be sampled
            assert ktraj_flat[-2, center_ind] == 0.0
            assert torch.all(ktraj_flat[-2, neighbor_inds] == 1.0)

            # Check that all indices in patch correspond
            # to the same patch index in the last row
            patch_indices = ktraj_flat[-1, patch]
            assert torch.all(patch_indices == patch_indices[0])


def test_group_patches_init():
    """Test the grouping of patches."""
    num_filled_patches = 10
    num_emtpy_patches = 5
    num_dim = 2
    num_neighbors = 10
    kernel_size = 7

    testpatches = KtrajTestPatches(
        num_dim=num_dim,
        num_neighbors=num_neighbors,
        kernel_size=kernel_size,
        num_filled_patches=num_filled_patches,
        num_empty_patches=num_emtpy_patches,
        device=torch.device("cpu"),
    )

    patch_instances = testpatches.get_patch_instances()

    grouped_patches = ncgrappa_2.PatchGroup.create_patchgroups(patch_instances)

    assert (
        len(grouped_patches) == num_filled_patches + 1
    )  # +1 for the empty patch group
    assert len(grouped_patches[0].patches) == num_emtpy_patches


def show_locs():
    """Visualize the k-space locations for a single patch and multiple patches."""
    import matplotlib.pyplot as plt

    # Single patch locations
    single_patch = single_patch_locs(
        num_dim=2, num_neighbors=10, kernel_size=7, k_max=100.0
    )

    multiple_patch = multiple_patch_locs(
        num_patches=10, num_dim=2, num_neighbors=10, kernel_size=7, k_max=100.0
    )

    plt.figure()
    plt.scatter(
        single_patch[0, :], single_patch[1, :], label="Single Patch", color="blue"
    )

    plt.scatter(
        single_patch[0, single_patch[-1] == 0],
        single_patch[1, single_patch[-1] == 0],
        label="Single Patch (Unsampled)",
        color="red",
    )

    plt.title("Single Patch Locations")
    plt.show()

    plt.figure()
    for i_patch in range(multiple_patch.shape[-1]):
        plt.scatter(
            multiple_patch[0, :, i_patch],
            multiple_patch[1, :, i_patch],
        )

    plt.title("Multiple Patch Locations")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "visual":
        show_locs()
    else:
        pytest.main([__file__])


# def _get_ktraj_patches_2D(
#     img_size: tuple[int, int],
#     kernel_size: tuple[int, int],
#     num_patches: int,
# ) -> torch.Tensor:
#     """Create a k-space trajectory from patches in 2D."""
#     # Create unsampled locations
#     x = np.linspace(-img_size[0]/2, img_size[0]/2, num_patches//2)
#     y = np.linspace(-img_size[1]/2, img_size[1]/2, num_patches - x.shape[0])
#     xx, yy = np.meshgrid(x, y, indexing='ij')
#     grid_locations = np.stack([xx.ravel(), yy.ravel()], axis=0)

#     neighbor_locations = []
#     for loc in grid_locations.T:
#         num_neighbors = np.random.randint(0, 5)  # Random number of neighbors
#         neighbor_distances = np.random.uniform(
#             -kernel_size[0] // 2,
#             kernel_size[0] // 2,
#             size=(2, num_neighbors)
#         )
#         neighbor_locations.append(loc[:, None] + neighbor_distances)
#     neighbor_locations = np.concatenate(neighbor_locations, axis=1)

#     grid_locations = torch.from_numpy(grid_locations)
#     grid_locations = torch.concatenate(
#         (
#             grid_locations,
#             torch.zeros((1, grid_locations.shape[1]), dtype=torch.float32),
#         ),
#         dim=0,
#     )

#     neighbor_locations = torch.from_numpy(neighbor_locations)
#     neighbor_locations = torch.concatenate(
#         (
#             neighbor_locations,
#             torch.ones(
#                 (1, neighbor_locations.shape[1]),
#                 dtype=torch.float32,
#             ),
#         ),
#         dim=0,
#     )

#     # Concatenate only torch tensors
#     ktraj = torch.cat((grid_locations, neighbor_locations), dim=1)

#     # Shuffle the k-space trajectory
#     ktraj = ktraj[:, torch.randperm(ktraj.shape[1])]

#     return ktraj


# def _get_rand_ktraj(n_samples=5000, n_dims=2, unsampled_ratio=0.3):
#     """Generate a random k-space trajectory with
#     some unsampled points and included sample dim (n_dims+1)."""
#     ktraj = torch.rand(n_dims, n_samples, dtype=torch.float32) - 0.5
#     sample_mask = torch.ones(n_samples, dtype=torch.float32)
#     unsampled_ind = torch.randperm(n_samples)[: int(n_samples * unsampled_ratio)]
#     sample_mask[unsampled_ind] = 0

#     ktraj = torch.cat((ktraj, sample_mask[None, :]), dim=0)
#     return ktraj


# def _shepp_logan_img_2D(fov=(0.2, 0.2), matrix=(96, 96)):
#     """Generate a Shepp-Logan phantom image in 2D with shape (8, 96, 96, 1)."""
#     sl = SheppLogan(fov=fov, matrix=matrix)
#     sl.add_coil()
#     img = sl.get_object()
#     img = img[..., 0]
#     return img

# def _shepp_logan_img_3D(fov=(0.2, 0.2, 0.2), matrix=(96, 96, 96)):
#     """Generate a Shepp-Logan phantom image in 2D with shape (8, 96, 96, 96)."""
#     sl = SheppLogan(fov=fov, matrix=matrix)
#     sl.add_coil()
#     img = sl.get_object()
#     img = img[..., 0]
#     return img

# def _shepp_logan_calib_2D(fov=(0.2, 0.2), img_size=(96, 96), calib_size=(96, 20)):
#     # Create a Shepp-Logan phantom
#     sl_phantom = SheppLogan(fov=fov, matrix=img_size)
#     sl_phantom.add_coil()
#     img = sl_phantom.get_object()
#     img = img[..., 0, 0]  # Get first echo and partition

#     kcalib = torch.fft.ifftshift(torch.fft.fft2(img, dim=(-1, -2, -3)))
#     start = img_size[1] // 2 - calib_size[1] // 2
#     end = -(img_size[1] // 2 - calib_size[1] // 2)
#     kcalib = kcalib[:, :, start:end]

#     return kcalib

# def test_ncgrappa_init_with_ints():
#     img_size = 96
#     kernel_size = 6
#     ktraj = _get_rand_ktraj(n_samples=100, n_dims=2)
#     calib_signal = _shepp_logan_img_2D(matrix=(img_size, 20))[..., 0]
#     ncg = NonCartGrappa(ktraj, calib_signal, img_size, kernel_size)
#     assert torch.equal(
#         ncg.kernel_size, torch.tensor([kernel_size, kernel_size], dtype=torch.int32)
#     )
#     assert torch.equal(
#         ncg.img_size, torch.tensor([img_size, img_size], dtype=torch.int32)
#     )
#     assert ncg.num_dims == 2


# def test_ncgrappa_init_with_tuple():
#     img_size = (96, 96)
#     kernel_size = (6, 6)
#     ktraj = _get_rand_ktraj(n_samples=50, n_dims=2)
#     calib_signal = _shepp_logan_img_2D(matrix=(96, 20))[..., 0]
#     ncg = NonCartGrappa(ktraj, calib_signal, img_size, kernel_size)
#     assert torch.equal(ncg.kernel_size, torch.tensor(kernel_size, dtype=torch.int32))
#     assert torch.equal(ncg.img_size, torch.tensor(img_size, dtype=torch.int32))
#     assert ncg.num_dims == 2


# def test_ncgrappa_init_with_list():
#     img_size = [96, 96]
#     kernel_size = [7, 7]
#     ktraj = _get_rand_ktraj(n_samples=30, n_dims=2)
#     calib_signal = _shepp_logan_img_2D(matrix=(96, 20))[..., 0]
#     ncg = NonCartGrappa(ktraj, calib_signal, img_size, kernel_size)
#     assert torch.equal(ncg.kernel_size, torch.tensor(kernel_size, dtype=torch.int32))
#     assert torch.equal(ncg.img_size, torch.tensor(img_size, dtype=torch.int32))
#     assert ncg.num_dims == 2


# def test_ncgrappa_init_wrong_kernel_size_length():
#     img_size = (96, 96)
#     kernel_size = (8, 8, 8)
#     ktraj = _get_rand_ktraj(n_samples=20, n_dims=2)
#     calib_signal = _shepp_logan_img_3D(matrix=(96, 20, 20))[..., 0]
#     with pytest.raises(AssertionError):
#         NonCartGrappa(ktraj, calib_signal, img_size, kernel_size)


# def test_ncgrappa_init_wrong_img_size_length():
#     img_size = (96, 96, 96)
#     kernel_size = (6, 6)
#     ktraj = _get_rand_ktraj(n_samples=20, n_dims=2)
#     calib_signal = _shepp_logan_img_3D(matrix=(96, 20, 20))[..., 0]
#     with pytest.raises(AssertionError):
#         NonCartGrappa(ktraj, calib_signal, img_size, kernel_size)


# def test_ncgrappa_init_with_device():
#     img_size = 96
#     kernel_size = 6
#     ktraj = _get_rand_ktraj(n_samples=10, n_dims=2).to("cpu")
#     calib_signal = _shepp_logan_img_3D(matrix=(96, 20, 20))[..., 0].to("cpu")
#     ncg = NonCartGrappa(
#         ktraj, calib_signal, img_size, kernel_size, device=torch.device("cpu")
#     )
#     assert ncg.kernel_size.device.type == "cpu"
#     assert ncg.img_size.device.type == "cpu"


# def test_ncgrappa_seperated_ktraj():
#     img_size = (96, 96)
#     kernel_size = (6, 6)
#     ktraj = _get_ktraj_patches_2D(
#         img_size=img_size, kernel_size=kernel_size, num_patches=60
#     )
#     calib_signal = _shepp_logan_calib_2D(
#         fov=(0.2, 0.2), img_size=img_size, calib_size=(96, 20)
#     )
#     ncg = NonCartGrappa(ktraj, calib_signal, img_size, kernel_size)

#     out_ktraj_sampled = ncg.ktraj_sampled
#     out_ktraj_unsampled = ncg.ktraj_unsampled

#     exp_ktraj_sampled = ktraj[:-1, ktraj[-1, :].bool()]
#     exp_ktraj_unsampled = ktraj[:-1, ~ktraj[-1, :].bool()]

#     assert torch.equal(out_ktraj_sampled, exp_ktraj_sampled)
#     assert torch.equal(out_ktraj_unsampled, exp_ktraj_unsampled)


# def test_ncgrappa_constellations():
#     img_size = (96, 96)
#     kernel_size = (6, 6)

#     ktraj = _get_ktraj_patches_2D(img_size, kernel_size, 60)

#     calib_signal = _shepp_logan_calib_2D(
#         fov=(0.2, 0.2), img_size=img_size, calib_size=(96, 20)
#     )

#     ncg = NonCartGrappa(ktraj, calib_signal, img_size, kernel_size)

#     assert len(ncg.patches) == ncg.ktraj_unsampled.shape[1]


# def visual_inspection():
#     """Visual inspection of the NonCartGrappa class."""

#     import matplotlib.pyplot as plt

#     ktraj = _get_rand_ktraj(n_samples=100, n_dims=2)
#     calib_signal = _shepp_logan_img_2D(matrix=(96, 20))[..., 0]
#     ncg = NonCartGrappa(ktraj, calib_signal, img_size=96, kernel_size=6)

#     sample_mask = ktraj[-1, :].bool()
#     ktraj_samples = ktraj[:2, sample_mask]
#     ktraj_unsampled = ktraj[:2, ~sample_mask]

#     plt.figure()
#     plt.plot(ktraj_samples[0], ktraj_samples[1], "ro", label="Sampled")
#     plt.plot(ktraj_unsampled[0], ktraj_unsampled[1], "bo", label="Unsampled")
#     plt.xlabel("k_x")
#     plt.ylabel("k_y")
#     plt.legend()
#     plt.show()

#     print(ncg.inds_c)


# if __name__ == "__main__":
#     if len(sys.argv) > 1 and sys.argv[1] == "visual":
#         visual_inspection()
#     else:
#         pytest.main([__file__])
#         pytest.main([__file__])
