import pytest
import torch

from juart.recon.cgnufft import cgnufft


@pytest.mark.parametrize(
    "img_size, input_ktraj_shape, input_ksp_shape, expected_shape",
    [
        ((64, 64, 1), (2, 100), (10, 100), (10, 64, 64, 1)),  # Simple 2D case
        ((32, 32, 32), (3, 100), (10, 100), (10, 32, 32, 32)),  # Simple 3D case
        (
            (64, 64, 1),
            (2, 100, 3),
            (10, 100, 3),
            (10, 64, 64, 1, 3),
        ),  # 2D case multicontrast
        (
            (32, 32, 32),
            (3, 100, 3),
            (10, 100, 3),
            (10, 32, 32, 32, 3),
        ),  # 2D case multicontrast
    ],
)
def test_cgnufft_shapes(img_size, input_ktraj_shape, input_ksp_shape, expected_shape):
    # Test parameters

    ktraj = torch.rand(input_ktraj_shape, dtype=torch.float32) - 0.5
    ksp = torch.rand(input_ksp_shape, dtype=torch.complex64)

    # Run cgnufft
    output = cgnufft(ksp, ktraj, img_size)

    # Check output shape
    assert output.shape == expected_shape


if __name__ == "__main__":
    pytest.main([__file__])
