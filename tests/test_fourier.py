import pytest
import torch

from juart.conopt.functional.fourier import (
    nonuniform_fourier_transform_adjoint,
    nonuniform_fourier_transform_forward,
)


class TestFourierTransformAdjoint:
    @pytest.mark.parametrize(
        "dim, shape, expected_shape",
        [
            (1, (20,), (1, 20, 1, 1)),
            (2, (20, 20), (1, 20, 20, 1)),
            (3, (20, 20, 20), (1, 20, 20, 20)),
        ],
    )
    def test_fourier_transform_adjoint_same_nodes(self, dim, shape, expected_shape):
        k = torch.rand(dim, 100, dtype=torch.float32) - 0.5
        x = torch.rand(100, dtype=torch.complex64)
        n_modes = shape

        result = nonuniform_fourier_transform_adjoint(k=k, x=x, n_modes=n_modes)

        assert result.shape == expected_shape
        assert result.dtype == torch.complex64

    @pytest.mark.parametrize(
        "k_shape, channels, nodes, expected_shape",
        [
            ((1, 1000, 5, 3, 2), 5, (20,), (5, 20, 1, 1, 5, 3, 2)),
            ((2, 1000, 5, 3, 2), 5, (20, 20), (5, 20, 20, 1, 5, 3, 2)),
            ((3, 1000, 5, 3, 2), 5, (20, 20, 20), (5, 20, 20, 20, 5, 3, 2)),
        ],
    )
    def test_fourier_transform_adjoint_add_axes(
        self, k_shape, channels, nodes, expected_shape
    ):
        k = torch.rand(*k_shape, dtype=torch.float32) - 0.5
        x = torch.rand(channels, *k_shape[1:], dtype=torch.complex64)
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
    # Test for no additional axes, only channels, read, phase1, phase2
    @pytest.mark.parametrize(
        "dim, num_col, shape, expected_shape",
        [
            (1, 10000, (5, 20, 1, 1), (5, 10000)),
            (2, 10000, (5, 20, 20, 1), (5, 10000)),
            (3, 10000, (5, 20, 20, 20), (5, 10000)),
        ],
    )
    def test_fourier_transform_forward_noadd(self, dim, num_col, shape, expected_shape):
        k = torch.rand(dim, num_col, dtype=torch.float32) - 0.5
        x = torch.rand(shape, dtype=torch.complex64)

        result = nonuniform_fourier_transform_forward(k=k, x=x)

        assert result.shape == expected_shape
        assert result.dtype == torch.complex64

    # Test for additional axes
    @pytest.mark.parametrize(
        "dim, num_col, shape, expected_shape",
        [
            (1, 10000, (5, 20, 1, 1, 4, 2, 1), (5, 10000, 4, 2, 1)),
            (2, 10000, (5, 20, 20, 1, 4, 2, 1), (5, 10000, 4, 2, 1)),
            (3, 10000, (5, 20, 20, 20, 4, 2, 1), (5, 10000, 4, 2, 1)),
        ],
    )
    def test_fourier_transform_forward_add(self, dim, num_col, shape, expected_shape):
        k = torch.rand(dim, num_col, *shape[4:], dtype=torch.float32) - 0.5
        x = torch.rand(shape, dtype=torch.complex64)

        result = nonuniform_fourier_transform_forward(k=k, x=x)

        assert result.shape == expected_shape
        assert result.dtype == torch.complex64


if __name__ == "__main__":
    pytest.main([__file__])
