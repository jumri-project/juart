import torch

from ..sampling.radial import radial_trajectory_3d
from ..sampling.ring import correct_kspace_trajectory, estimate_gradient_delay


class KSpaceTrajectory:
    def __init__(self, NLin, NCol, NSet, NEco):
        self.ktraj_shape = (2, NLin, NCol, 1, 1, NSet, NEco)
        self.ktraj_nom = torch.zeros(self.ktraj_shape)

        for ti in range(NSet):
            for te in range(NEco):
                Kx, Ky = radial_trajectory_3d(
                    ti,
                    te,
                    0,
                    NSet,
                    NEco,
                    1,
                    NCol,
                    NLin,
                    version=2,
                    phi0=torch.pi / 2,
                    readoutDownsampling=False,
                )
                self.ktraj_nom[0, :, :, 0, 0, ti, te] = Kx.reshape(NLin, NCol)
                self.ktraj_nom[1, :, :, 0, 0, ti, te] = Ky.reshape(NLin, NCol)

    def estimate_gradient_delay(self, kdata, Npad=100, beta=2, NDiff=4):
        _, NLin, NCol, _, _, NSet, NEco = self.ktraj_shape

        NPairs = NLin - NDiff

        S_corr = estimate_gradient_delay(
            torch.sum(kdata, axis=4, keepdims=True),
            self.ktraj_nom,
            (NLin, NCol, 1, 1, NSet, NEco),
            Npad=Npad,
            beta=beta,
            nDiff=NDiff,
            nPairs=NPairs,
            readoutDownsampling=False,
        )

        return S_corr

    def correct_kspace_trajectory(self, S_corr):
        _, NLin, NCol, _, _, NSet, NEco = self.ktraj_shape

        ktraj_corr = correct_kspace_trajectory(
            self.ktraj_nom, (NLin, NCol, 1, 1, NSet, NEco), S_corr
        )

        return ktraj_corr
