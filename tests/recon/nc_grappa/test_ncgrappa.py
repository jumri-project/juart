import sys

import numpy as np
import pytest
import torch

from juart.ellipsoid_phantoms.ellipsoids import SheppLogan
from juart.recon.ncgrappa import NonCartGrappa


def _get_ktraj_patches_2D(
    img_size: tuple[int, int],
    kernel_size: tuple[int, int],
    num_patches: int,
) -> torch.Tensor:
    """Create a k-space trajectory from patches in 2D."""
    # Create unsampled locations
    x = np.linspace(-img_size[0] / 2, img_size[0] / 2, num_patches // 2)
    y = np.linspace(-img_size[1] / 2, img_size[1] / 2, num_patches - x.shape[0])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    grid_locations = np.stack([xx.ravel(), yy.ravel()], axis=0)

    neighbor_locations = []
    for loc in grid_locations.T:
        num_neighbors = np.random.randint(0, 5)  # Random number of neighbors
        neighbor_distances = np.random.uniform(
            -kernel_size[0] // 2,
            kernel_size[0] // 2,
            size=(2, num_neighbors),
        )
        neighbor_locations.append(loc[:, None] + neighbor_distances)
    neighbor_locations = np.concatenate(neighbor_locations, axis=1)

    grid_locations = torch.from_numpy(grid_locations)
    grid_locations = torch.concatenate(
        (
            grid_locations,
            torch.zeros((1, grid_locations.shape[1]), dtype=torch.float32),
        ),
        dim=0,
    )

    neighbor_locations = torch.from_numpy(neighbor_locations)
    neighbor_locations = torch.concatenate(
        (
            neighbor_locations,
            torch.ones((1, neighbor_locations.shape[1]), dtype=torch.float32),
        ),
        dim=0,
    )

    # Concatenate only torch tensors
    ktraj = torch.cat((grid_locations, neighbor_locations), dim=1)

    # Shuffle the k-space trajectory
    ktraj = ktraj[:, torch.randperm(ktraj.shape[1])]

    return ktraj


def _get_rand_ktraj(n_samples=5000, n_dims=2, unsampled_ratio=0.3):
    """
    Generate a random k-space trajectory with some unsampled points and included
    sample dim (n_dims+1).
    """
    ktraj = torch.rand(n_dims, n_samples, dtype=torch.float32) - 0.5
    sample_mask = torch.ones(n_samples, dtype=torch.float32)
    unsampled_ind = torch.randperm(n_samples)[: int(n_samples * unsampled_ratio)]
    sample_mask[unsampled_ind] = 0

    ktraj = torch.cat((ktraj, sample_mask[None, :]), dim=0)
    return ktraj


def _shepp_logan_img_2D(fov=(0.2, 0.2), matrix=(96, 96)):
    """Generate a Shepp-Logan phantom image in 2D with shape (8, 96, 96, 1)."""
    sl = SheppLogan(fov=fov, matrix=matrix)
    sl.add_coil()
    img = sl.get_object()
    img = img[..., 0]
    return img


def _shepp_logan_img_3D(fov=(0.2, 0.2, 0.2), matrix=(96, 96, 96)):
    """Generate a Shepp-Logan phantom image in 2D with shape (8, 96, 96, 96)."""
    sl = SheppLogan(fov=fov, matrix=matrix)
    sl.add_coil()
    img = sl.get_object()
    img = img[..., 0]
    return img


def _shepp_logan_calib_2D(fov=(0.2, 0.2), img_size=(96, 96), calib_size=(96, 20)):
    # Create a Shepp-Logan phantom
    sl_phantom = SheppLogan(fov=fov, matrix=img_size)
    sl_phantom.add_coil()
    img = sl_phantom.get_object()
    img = img[..., 0, 0]  # Get first echo and partition

    kcalib = torch.fft.ifftshift(torch.fft.fft2(img, dim=(-1, -2, -3)))
    start = img_size[1] // 2 - calib_size[1] // 2
    end = -(img_size[1] // 2 - calib_size[1] // 2)
    kcalib = kcalib[:, :, start:end]

    return kcalib


def test_ncgrappa_init_with_ints():
    img_size = 96
    kernel_size = 6
    ktraj = _get_rand_ktraj(n_samples=100, n_dims=2)
    calib_signal = _shepp_logan_img_2D(matrix=(img_size, 20))[..., 0]
    ncg = NonCartGrappa(ktraj, calib_signal, img_size, kernel_size)
    assert torch.equal(
        ncg.kernel_size, torch.tensor([kernel_size, kernel_size], dtype=torch.int32)
    )
    assert torch.equal(
        ncg.img_size, torch.tensor([img_size, img_size], dtype=torch.int32)
    )
    assert ncg.num_dim == 2


def test_ncgrappa_init_with_tuple():
    img_size = (96, 96)
    kernel_size = (6, 6)
    ktraj = _get_rand_ktraj(n_samples=50, n_dims=2)
    calib_signal = _shepp_logan_img_2D(matrix=(96, 20))[..., 0]
    ncg = NonCartGrappa(ktraj, calib_signal, img_size, kernel_size)
    assert torch.equal(ncg.kernel_size, torch.tensor(kernel_size, dtype=torch.int32))
    assert torch.equal(ncg.img_size, torch.tensor(img_size, dtype=torch.int32))
    assert ncg.num_dim == 2


def test_ncgrappa_init_with_list():
    img_size = [96, 96]
    kernel_size = [7, 7]
    ktraj = _get_rand_ktraj(n_samples=30, n_dims=2)
    calib_signal = _shepp_logan_img_2D(matrix=(96, 20))[..., 0]
    ncg = NonCartGrappa(ktraj, calib_signal, img_size, kernel_size)
    assert torch.equal(ncg.kernel_size, torch.tensor(kernel_size, dtype=torch.int32))
    assert torch.equal(ncg.img_size, torch.tensor(img_size, dtype=torch.int32))
    assert ncg.num_dim == 2


def test_ncgrappa_init_wrong_kernel_size_length():
    img_size = (96, 96)
    kernel_size = (8, 8, 8)
    ktraj = _get_rand_ktraj(n_samples=20, n_dims=2)
    calib_signal = _shepp_logan_img_3D(matrix=(96, 20, 20))[..., 0]
    with pytest.raises(AssertionError):
        NonCartGrappa(ktraj, calib_signal, img_size, kernel_size)


def test_ncgrappa_init_wrong_img_size_length():
    img_size = (96, 96, 96)
    kernel_size = (6, 6)
    ktraj = _get_rand_ktraj(n_samples=20, n_dims=2)
    calib_signal = _shepp_logan_img_3D(matrix=(96, 20, 20))[..., 0]
    with pytest.raises(AssertionError):
        NonCartGrappa(ktraj, calib_signal, img_size, kernel_size)


def test_ncgrappa_init_with_device():
    img_size = 96
    kernel_size = 6
    ktraj = _get_rand_ktraj(n_samples=10, n_dims=2).to("cpu")
    calib_signal = _shepp_logan_img_3D(matrix=(96, 20, 20))[..., 0].to("cpu")
    ncg = NonCartGrappa(
        ktraj, calib_signal, img_size, kernel_size, device=torch.device("cpu")
    )
    assert ncg.kernel_size.device.type == "cpu"
    assert ncg.img_size.device.type == "cpu"


def test_ncgrappa_seperated_ktraj():
    img_size = (96, 96)
    kernel_size = (6, 6)
    ktraj = _get_ktraj_patches_2D(
        img_size=img_size,
        kernel_size=kernel_size,
        num_patches=60,
    )
    calib_signal = _shepp_logan_calib_2D(
        fov=(0.2, 0.2),
        img_size=img_size,
        calib_size=(96, 20),
    )
    ncg = NonCartGrappa(ktraj, calib_signal, img_size, kernel_size)

    out_ktraj_sampled = ncg.ktraj_sampled
    out_ktraj_unsampled = ncg.ktraj_unsampled

    exp_ktraj_sampled = ktraj[:-1, ktraj[-1, :].bool()]
    exp_ktraj_unsampled = ktraj[:-1, ~ktraj[-1, :].bool()]

    assert torch.equal(out_ktraj_sampled, exp_ktraj_sampled)
    assert torch.equal(out_ktraj_unsampled, exp_ktraj_unsampled)


def test_ncgrappa_constellations():
    img_size = (96, 96)
    kernel_size = (6, 6)

    ktraj = _get_ktraj_patches_2D(img_size, kernel_size, 60)

    calib_signal = _shepp_logan_calib_2D(
        fov=(0.2, 0.2),
        img_size=img_size,
        calib_size=(96, 20),
    )

    ncg = NonCartGrappa(ktraj, calib_signal, img_size, kernel_size)

    assert len(ncg.patches) == ncg.ktraj_unsampled.shape[1], (
        "Number of constellations should match number of unsampled points"
    )


def visual_inspection():
    """Visual inspection of the NonCartGrappa class."""

    import matplotlib.pyplot as plt

    ktraj = _get_rand_ktraj(n_samples=100, n_dims=2)
    calib_signal = _shepp_logan_img_2D(matrix=(96, 20))[..., 0]
    ncg = NonCartGrappa(ktraj, calib_signal, img_size=96, kernel_size=6)

    sample_mask = ktraj[-1, :].bool()
    ktraj_samples = ktraj[:2, sample_mask]
    ktraj_unsampled = ktraj[:2, ~sample_mask]

    plt.figure()
    plt.plot(ktraj_samples[0], ktraj_samples[1], "ro", label="Sampled")
    plt.plot(ktraj_unsampled[0], ktraj_unsampled[1], "bo", label="Unsampled")
    plt.xlabel("k_x")
    plt.ylabel("k_y")
    plt.legend()
    plt.show()

    print(ncg.inds_c)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "visual":
        visual_inspection()
    else:
        pytest.main([__file__])
