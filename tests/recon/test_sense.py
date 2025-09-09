import pytest
import torch

from juart.recon.sense import cgsense


@pytest.mark.parametrize(
    "img_size, input_ktraj_shape, input_ksp_shape, expected_shape",
    [
        # Simple 2D case
        ((64, 64, 1), (2, 100), (10, 100), (1, 64, 64, 1)),
        # Simple 3D case
        ((32, 32, 32), (3, 100), (10, 100), (1, 32, 32, 32)),
        # 2D case multicontrast
        ((64, 64, 1), (2, 100, 3), (10, 100, 3), (1, 64, 64, 1, 3)),
        # 2D case multicontrast
        ((32, 32, 32), (3, 100, 3), (10, 100, 3), (1, 32, 32, 32, 3)),
    ],
)
def test_cgnufft_shapes(img_size, input_ktraj_shape, input_ksp_shape, expected_shape):
    # Test parameters

    ktraj = torch.rand(input_ktraj_shape, dtype=torch.float32) - 0.5
    ksp = torch.rand(input_ksp_shape, dtype=torch.complex64)
    num_cha = ksp.shape[0]
    coilsens = torch.rand((num_cha, *img_size), dtype=torch.complex64)

    # Run cgsense
    output = cgsense(
        ksp=ksp, ktraj=ktraj, coil_sensitivities=coilsens, maxiter=20, l2_reg=0.0
    )

    # Check output shape
    assert output.shape == expected_shape


if __name__ == "__main__":
    pytest.main([__file__])
