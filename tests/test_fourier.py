import warnings

import pytest
import torch

from jail.conopt.functional.fourier import (
    nonuniform_fourier_transform_adjoint,
)


class TestFourierTransformForward:
    @pytest.mark.parametrize("D", [1, 2, 3])
    def test_fourier_transform_forward_same_nodes(self, D):
        k = torch.rand(D, 100, dtype=torch.float32) * 2 - 1
        x = torch.rand(100, dtype=torch.complex64)
        n_modes = (20,) * D

        result = nonuniform_fourier_transform_adjoint(k=k, x=x, n_modes=n_modes)

        assert result.shape == n_modes
        assert result.dtype == torch.complex64

    @pytest.mark.parametrize(
        "dim, n_nodes, expect_shape",
        [
            (2, (20,), (20,)),
            (3, (20,), (20,)),
        ],
    )
    def test_fourier_transform_forward_k_gr_nodes(self, dim, n_nodes, expect_shape):
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
            (1, (20, 20), (20, 20)),
            (2, (20, 20, 20), (20, 20, 20)),
        ],
    )
    def test_fourier_transform_forward_k_le_nodes(self, dim, n_nodes, expect_shape):
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


if __name__ == "__main__":
    pytest.main([__file__])
