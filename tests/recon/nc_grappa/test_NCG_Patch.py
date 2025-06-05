import pytest
import torch

from juart.recon.ncgrappa import NCG_Patch, NCG_PatchGroup


class TestNCGPatch:
    def test_basic_init(self):
        # Test for 2D patch
        center = torch.zeros((2, 1), dtype=torch.float32)
        neighbors = torch.tensor([[0.6, -2.4], [1.2, 3.3]])
        patch = NCG_Patch(
            center_ind=0,
            center_loc=center,
            neighbor_inds=[1, 2],
            neighbor_locs=neighbors,
            do_sift=False,
        )
        assert patch.num_neighbors == 2
        assert patch.neighbor_dist.shape == (2, 2)
        assert patch.num_dim == 2

        # Test for 3D patch
        center = torch.zeros((3, 1), dtype=torch.float32)
        neighbors = torch.tensor([[0.6, -2.4], [1.2, 3.3], [0.5, 1.5]])
        patch = NCG_Patch(
            center_ind=0,
            center_loc=center,
            neighbor_inds=[1, 2],
            neighbor_locs=neighbors,
            do_sift=False,
        )
        assert patch.num_neighbors == 2
        assert patch.neighbor_dist.shape == (3, 2)
        assert patch.num_dim == 3

    def test_from_indices(self):
        sampled = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        unsampled = torch.zeros((2, 3), dtype=torch.float32)
        inds_c = [0, 0]
        patch = NCG_Patch.from_indices(inds_c, sampled, unsampled)

        assert patch.center_loc.shape == (2, 1)
        assert patch.num_neighbors == 1
        assert patch.neighbor_dist.shape == (2, 1)
        assert patch.num_dim == 2
        assert torch.equal(patch.neighbor_locs, sampled[:, 0][:, None])

    def test_no_neighbor_init(self):
        center = torch.zeros((2, 1), dtype=torch.float32)
        patch = NCG_Patch(
            center_ind=0,
            center_loc=center,
            do_sift=False,
        )
        assert patch.num_neighbors == 0
        assert patch.neighbor_dist.shape == (2, 0)
        assert patch.num_dim == 2

    def test_sifting_duplicates(self):
        # Test for 2D patch with duplicate neighbors
        center = torch.zeros((2, 1), dtype=torch.float32)
        neighbors = torch.tensor([[0.6, 0.7], [0.6, 0.7]])
        patch = NCG_Patch(
            center_ind=0,
            center_loc=center,
            neighbor_inds=[1, 2],
            neighbor_locs=neighbors,
            do_sift=True,
        )

        assert patch.num_neighbors == 1
        assert patch.neighbor_locs.shape == (2, 1)
        assert torch.equal(patch.neighbor_locs, torch.tensor([[0.6], [0.6]]))

        # Test for 3D patch with duplicate neighbors
        center = torch.zeros((3, 1), dtype=torch.float32)
        neighbors = torch.tensor([[0.6, 0.7], [0.6, 0.7], [0.8, 0.8]])
        patch = NCG_Patch(
            center_ind=0,
            center_loc=center,
            neighbor_inds=[1, 2],
            neighbor_locs=neighbors,
            do_sift=True,
        )

        exp_neighbor_locs = torch.tensor([[0.6], [0.6], [0.8]])

        assert patch.num_neighbors == 1
        assert patch.neighbor_locs.shape == (3, 1)
        assert torch.equal(patch.neighbor_locs, exp_neighbor_locs)

    def test_sorting(self):
        center = torch.zeros((2, 1), dtype=torch.float32)
        neighbors = torch.tensor(
            [
                [2.0, 1.0],
                [0.0, 0.0],
            ]
        )
        patch = NCG_Patch(
            center_ind=0,
            center_loc=center,
            neighbor_inds=[1, 2],
            neighbor_locs=neighbors,
            do_sift=False,
        )
        expect_neighbor_locs = torch.tensor(
            [
                [1.0, 2.0],
                [0.0, 0.0],
            ]
        )
        expect_neighbor_inds = [2, 1]
        expect_neighbor_dist = torch.tensor(
            [
                [1.0, 2.0],
                [0.0, 0.0],
            ]
        )

        assert torch.equal(patch.neighbor_locs, expect_neighbor_locs)
        assert patch.neighbor_inds == expect_neighbor_inds
        assert torch.equal(patch.neighbor_dist, expect_neighbor_dist)


class TestNCGPatchGroup:
    def test_group_init(self):
        center = torch.zeros(2, dtype=torch.float32)
        neighbors = torch.tensor(
            [
                [0.6, -2.4, 3.5, -4.8, 2.1, -1.7, 4.9],
                [1.2, 3.3, -3.7, 0.0, 4.5, -5.0, 2.8],
            ]
        )

        expect_shift_pattern = torch.tensor(
            [[-5, -2, -2, 1, 2, 4, 5], [0, -5, 3, 1, 4, -4, 3]]
        )

        num_neighbors = neighbors.shape[1]
        num_patches = 10
        patch_dislocs = (torch.randn(2, num_patches) - 0.5) * 5

        patches = []
        for disloc in patch_dislocs.T:
            patch = NCG_Patch(
                center_ind=0,
                center_loc=center + disloc,
                neighbor_inds=torch.randint(0, 100, (num_neighbors,)).tolist(),
                neighbor_locs=neighbors + disloc[:, None],
                do_sift=False,
            )
            patches.append(patch)

        group = NCG_PatchGroup(patches=patches)

        assert group.num_patches == num_patches
        assert group.num_neighbors == num_neighbors
        assert group.num_dim == 2
        assert torch.equal(group.int_shift_pattern, expect_shift_pattern)


if __name__ == "__main__":
    pytest.main([__file__])
