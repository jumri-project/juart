import pytest
import torch

from juart.conopt.tfs.fourier import nonuniform_transfer_function


class TestNonuniformTransferFunction:
    grid_size = 32
    ktraj_grid = torch.stack(
        torch.meshgrid(
            torch.arange(grid_size) / grid_size - 0.5,
            torch.arange(grid_size) / grid_size - 0.5,
            torch.arange(grid_size) / grid_size - 0.5,
            indexing="ij",
        ),
        dim=-1,
    )

    def test_no_add_axes(self):
        # Test with a simple case

        k = self.ktraj_grid.view(3, -1)  # Reshape to (D, N)
        output_shape = (1, self.grid_size, self.grid_size, self.grid_size)
        oversampling = 2
        eps = 1e-6

        expected_shape = (
            1,
            self.grid_size * oversampling,
            self.grid_size * oversampling,
            self.grid_size * oversampling,
        )

        transfer_function = nonuniform_transfer_function(
            k, output_shape, oversampling, eps
        )

        assert transfer_function.shape == expected_shape, "Output shape mismatch"

    def test_add_shared_axes(self):
        k = self.ktraj_grid.view(3, -1)

        # Add axes
        k = k[..., None, None]
        k = k.expand(-1, -1, 2, 3)

        output_shape = (1, self.grid_size, self.grid_size, self.grid_size, 2, 3)
        oversampling = 2
        eps = 1e-6

        expected_shape = (
            1,
            self.grid_size * oversampling,
            self.grid_size * oversampling,
            self.grid_size * oversampling,
            2,
            3,
        )

        transfer_function = nonuniform_transfer_function(
            k, output_shape, oversampling, eps
        )

        assert transfer_function.shape == expected_shape, "Output shape mismatch"

    def test_add_excl_axes(self):
        k = self.ktraj_grid.view(3, -1)

        # Add axes
        k = k[..., None, None]
        k = k.expand(-1, -1, 2, 3)

        output_shape = (1, self.grid_size, self.grid_size, self.grid_size, 2, 3, 4, 5)
        oversampling = 2
        eps = 1e-6

        expected_shape = (
            1,
            self.grid_size * oversampling,
            self.grid_size * oversampling,
            self.grid_size * oversampling,
            2,
            3,
            4,
            5,
        )

        transfer_function = nonuniform_transfer_function(
            k, output_shape, oversampling, eps
        )

        assert transfer_function.shape == expected_shape, "Output shape mismatch"


if __name__ == "__main__":
    pytest.main([__file__])
