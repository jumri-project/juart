import pytest
import torch

from juart.recon.sense import cgsense


@pytest.mark.parametrize(
    "input_ktraj_shape, input_ksp_shape, coilsens_shape, expected_shape",
    [
        # Simple 2D case
        ((2, 100, 1), (10, 100, 1), (10, 64, 64, 1, 1), (1, 64, 64, 1, 1)),
        # Simple 3D case
        ((3, 100, 1), (10, 100, 1), (10, 32, 32, 32, 1), (1, 32, 32, 32, 1)),
        # 2D case multicontrast
        ((2, 100, 1, 3), (10, 100, 1, 3), (10, 64, 64, 1, 1), (1, 64, 64, 1, 1, 3)),
        # 3D case multicontrast
        ((3, 100, 1, 3), (10, 100, 1, 3), (10, 32, 32, 32, 1), (1, 32, 32, 32, 1, 3)),
    ],
)
def test_cgnufft_shapes(
    input_ktraj_shape, input_ksp_shape, coilsens_shape, expected_shape
):
    # Test parameters

    ktraj = torch.rand(input_ktraj_shape, dtype=torch.float32) - 0.5
    ksp = torch.rand(input_ksp_shape, dtype=torch.complex64)
    coilsens = torch.rand(coilsens_shape, dtype=torch.complex64)

    # Run cgsense
    output = cgsense(ksp=ksp, ktraj=ktraj, coilsens=coilsens, maxiter=20, l2_reg=0.0)

    # Check output shape
    assert output.shape == expected_shape


if __name__ == "__main__":
    pytest.main([__file__])
