import pytest
import torch

from juart.conopt.functional.fourier import (
    nonuniform_fourier_transform_adjoint,
    nonuniform_fourier_transform_forward,
)


class TestFourierTransformAdjoint:
    num_channels = 8

    @pytest.mark.parametrize(
        "dim, shape, expected_shape",
        [
            (1, (20,), (num_channels, 20, 1, 1)),
            (2, (20, 20), (num_channels, 20, 20, 1)),
            (3, (20, 20, 20), (num_channels, 20, 20, 20)),
        ],
    )
    def test_fourier_transform_adjoint_same_nodes(self, dim, shape, expected_shape):
        k = torch.rand(dim, 100, dtype=torch.float32) - 0.5
        x = torch.rand(self.num_channels, 100, dtype=torch.complex64)
        n_modes = shape

        result = nonuniform_fourier_transform_adjoint(k=k, x=x, n_modes=n_modes)

        assert result.shape == expected_shape
        assert result.dtype == torch.complex64

    @pytest.mark.parametrize(
        "k_shape, nodes, expected_shape",
        [
            ((1, 1000, 5, 3, 2), (20,), (num_channels, 20, 1, 1, 5, 3, 2)),
            ((2, 1000, 5, 3, 2), (20, 20), (num_channels, 20, 20, 1, 5, 3, 2)),
            ((3, 1000, 5, 3, 2), (20, 20, 20), (num_channels, 20, 20, 20, 5, 3, 2)),
        ],
    )
    def test_fourier_transform_adjoint_add_axes(self, k_shape, nodes, expected_shape):
        k = torch.rand(*k_shape, dtype=torch.float32) - 0.5
        x = torch.rand(self.num_channels, *k_shape[1:], dtype=torch.complex64)
        n_modes = nodes

        result = nonuniform_fourier_transform_adjoint(k=k, x=x, n_modes=n_modes)

        assert result.shape == expected_shape
        assert result.dtype == torch.complex64

    @pytest.mark.parametrize(
        "k_shape, x_shape, nodes, expected_shape",
        [
            (
                (1, 1000, 5, 3),
                (num_channels, 1000, 5, 3, 4, 2),
                (20,),
                (num_channels, 20, 1, 1, 5, 3, 4, 2),
            ),
            (
                (2, 1000, 5, 3),
                (num_channels, 1000, 5, 3, 4, 2),
                (20, 20),
                (num_channels, 20, 20, 1, 5, 3, 4, 2),
            ),
            (
                (3, 1000, 5, 3),
                (num_channels, 1000, 5, 3, 4, 2),
                (20, 20, 20),
                (num_channels, 20, 20, 20, 5, 3, 4, 2),
            ),
        ],
    )
    def test_fourier_transform_adjoint_more_x_than_k(
        self, k_shape, x_shape, nodes, expected_shape
    ):
        k = torch.rand(*k_shape, dtype=torch.float32) - 0.5
        x = torch.rand(*x_shape, dtype=torch.complex64)
        n_modes = nodes

        result = nonuniform_fourier_transform_adjoint(k=k, x=x, n_modes=n_modes)

        assert result.shape == expected_shape
        assert result.dtype == torch.complex64

    @pytest.mark.parametrize(
        "dim, n_nodes, expect_shape",
        [
            (1, (20, 20), (20, 20, 1)),
            (2, (20, 20, 20), (20, 20, 20)),
            (2, (20,), (20, 1, 1)),
            (3, (20,), (20, 1, 1)),
        ],
    )
    def test_fourier_transform_adjoint_dim_not_nodes(self, dim, n_nodes, expect_shape):
        """Test for when the number of dimensions of
        k is greater than the number of dimensions of n_modes.
        Should raise a warning and add additional nodes.
        """
        k = torch.rand(dim, 100, dtype=torch.float32) - 0.5
        x = torch.rand(100, dtype=torch.complex64)

        with pytest.raises(ValueError):
            nonuniform_fourier_transform_adjoint(k=k, x=x, n_modes=n_nodes)

    # TODO: Add test for signal with channel and additional axes


class TestFourierTransformForward:
    num_channels = 8
    num_columns = 1000

    # Test for no additional axes, only channels, read, phase1, phase2
    @pytest.mark.parametrize(
        "k_shape, x_shape, expected_shape",
        [
            ((1, num_columns), (num_channels, 20, 1, 1), (num_channels, num_columns)),
            ((2, num_columns), (num_channels, 20, 20, 1), (num_channels, num_columns)),
            ((3, num_columns), (num_channels, 20, 20, 20), (num_channels, num_columns)),
        ],
    )
    def test_fourier_transform_forward_noadd(self, k_shape, x_shape, expected_shape):
        k = torch.rand(k_shape, dtype=torch.float32) - 0.5
        x = torch.rand(x_shape, dtype=torch.complex64)

        result = nonuniform_fourier_transform_forward(k=k, x=x)

        assert result.shape == expected_shape
        assert result.dtype == torch.complex64

    # Test for additional axes
    @pytest.mark.parametrize(
        "k_shape, x_shape, expected_shape",
        [
            (
                (1, num_columns, 4, 2, 1),
                (num_channels, 20, 1, 1, 4, 2, 1),
                (num_channels, num_columns, 4, 2, 1),
            ),
            (
                (2, num_columns, 4, 2, 1),
                (num_channels, 20, 20, 1, 4, 2, 1),
                (num_channels, num_columns, 4, 2, 1),
            ),
            (
                (3, num_columns, 4, 2, 1),
                (num_channels, 20, 20, 20, 4, 2, 1),
                (num_channels, num_columns, 4, 2, 1),
            ),
        ],
    )
    def test_fourier_transform_forward_add_axes(self, k_shape, x_shape, expected_shape):
        k = torch.rand(k_shape, dtype=torch.float32) - 0.5
        x = torch.rand(x_shape, dtype=torch.complex64)

        result = nonuniform_fourier_transform_forward(k=k, x=x)

        assert result.shape == expected_shape
        assert result.dtype == torch.complex64


if __name__ == "__main__":
    pytest.main([__file__])
