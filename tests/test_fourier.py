import warnings

import pytest
import torch

from jail.conopt.functional.fourier import (
    nonuniform_fourier_transform_adjoint,
    nonuniform_fourier_transform_forward,
)


class TestFourierTransformAdjoint:
    @pytest.mark.parametrize(
        "dim, shape, expected_shape",
        [
            (1, (20,), (20, 1, 1)),
            (2, (20, 20), (20, 20, 1)),
            (3, (20, 20, 20), (20, 20, 20)),
        ],
    )
    def test_fourier_transform_adjoint_same_nodes(self, dim, shape, expected_shape):
        k = torch.rand(dim, 100, dtype=torch.float32) * 2 - 1
        x = torch.rand(100, dtype=torch.complex64)
        n_modes = shape

        result = nonuniform_fourier_transform_adjoint(k=k, x=x, n_modes=n_modes)

        assert result.shape == expected_shape
        assert result.dtype == torch.complex64

    @pytest.mark.parametrize(
        "dim, n_nodes, expect_shape",
        [
            (2, (20,), (20, 1, 1)),
            (3, (20,), (20, 1, 1)),
        ],
    )
    def test_fourier_transform_adjoint_k_gr_nodes(self, dim, n_nodes, expect_shape):
        """Test for when the number of dimensions of
        k is greater than the number of dimensions of n_modes.
        Should raise a warning and add additional nodes.
        """
        k = torch.rand(dim, 100, dtype=torch.float32) * 2 - 1
        x = torch.rand(100, dtype=torch.complex64)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = nonuniform_fourier_transform_adjoint(k=k, x=x, n_modes=n_nodes)
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert "Adding additional dimensions to n_modes..." in str(w[-1].message)

            assert result.shape == expect_shape
            assert result.dtype == torch.complex64

    @pytest.mark.parametrize(
        "dim, n_nodes, expect_shape",
        [
            (1, (20, 20), (20, 20, 1)),
            (2, (20, 20, 20), (20, 20, 20)),
        ],
    )
    def test_fourier_transform_adjoint_k_le_nodes(self, dim, n_nodes, expect_shape):
        """Test for when the number of dimensions of
        k is less than the number of dimensions of n_modes.
        Should raise a warning and add additional dimensions to k.
        """
        k = torch.rand(dim, 100, dtype=torch.float32) * 2 - 1
        x = torch.rand(100, dtype=torch.complex64)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = nonuniform_fourier_transform_adjoint(k=k, x=x, n_modes=n_nodes)
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert "Adding additional dimensions with zeros to k ..." in str(
                w[-1].message
            )

            assert result.shape == expect_shape
            assert result.dtype == torch.complex64

    # TODO: Add test for signal with channel and additional axes


class TestFourierTransformForward:
    @pytest.mark.parametrize(
        "dim, num_col, shape, expected_shape",
        [
            (1, 10000, (1, 20, 1, 1), (1, 1, 10000, 1)),
            (2, 10000, (1, 20, 20, 1), (1, 2, 10000, 1)),
            (3, 10000, (1, 20, 20, 20), (1, 3, 10000, 1)),
        ],
    )
    def test_fourier_transform_forward(self, dim, num_col, shape, expected_shape):
        k = torch.rand(dim, num_col, dtype=torch.float32) - 0.5
        x = torch.rand(shape, dtype=torch.complex64)

        result = nonuniform_fourier_transform_forward(k=k, x=x)

        assert result.shape == expected_shape
        assert result.dtype == torch.complex64


if __name__ == "__main__":
    pytest.main([__file__])
