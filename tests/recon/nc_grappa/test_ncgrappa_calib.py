import pytest
import torch

from juart.recon.ncgrappa import NCG_Patch, NCG_PatchGroup


class TestNCGPatch_calibration:
    def test_patch_group_calibrate(self):
        # Create a fake calibration signal (C, R, P1, P2)
        C, R, P1, P2 = 2, 4, 4, 4
        calib_signal = torch.randn(C, R, P1, P2, dtype=torch.complex64)

        # Create a simple patch group (2D, 2 neighbors)
        center_loc = torch.zeros((3, 1))
        neighbor_locs = torch.tensor(
            [
                [1.0, 2.0],
                [0.0, 0.0],
                [-1.0, -2.0],
            ]
        )
        patch = NCG_Patch(
            center_ind=0,
            center_loc=center_loc,
            neighbor_inds=[1, 2],
            neighbor_locs=neighbor_locs,
            do_sift=False,
        )
        group = NCG_PatchGroup([patch])

        # Calibrate
        group.calibrate(calib_signal, tik=1e-3)

        # Check that kernel is set and has expected shape
        assert group.kernel is not None
        assert torch.is_tensor(group.kernel)
        assert group.kernel.shape[0] == group.num_neighbors * C

        # Optionally: check values are finite
        assert torch.isfinite(group.kernel).all()


if __name__ == "__main__":
    pytest.main([__file__])
