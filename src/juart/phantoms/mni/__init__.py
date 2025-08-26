import os

import nibabel as nib
import numpy as np
import pandas as pd

from .. import zpad
from ..shimming.SphericalHarmonics import getSphericalHarmonics
from ..shimming.VoxelCoordinates import getVoxelCoordinates

path = os.path.split(__file__)[0]


def tissue_map(B0):
    """
    Susceptibility values taken from

    Marques et al. 2021

    QSM reconstruction challenge 2.0: A realistic in silico head
    phantom for MRI data simulation and evaluation of susceptibility
    mapping proced

    We assume skin, glial matter and meat to have susceptibilities comparable
    to muscle (0.000, 0.000) and white matter (-0.030).
    """

    parameter_type = ("M0", "T1", "T2", "T2s", "CS", "X", "Label")

    tissue_type = (
        "Air",
        "CSF",
        "Gray Matter",
        "White Matter",
        "Fat",
        "Muscle",
        "Skin",
        "Skull",
        "Glial Matter",
        "Meat",
    )

    if B0 == 7.0:
        data = (
            (0, 1, 0.83, 0.7, 1, 1, 1, 0, 0.86, 0.77),
            (0, 4391, 2065, 1284, 650, 2250, 2569, 0, 2065, 1284),
            (0, 825, 63, 58, 28, 29, 160, 0, 63, 58),
            (0, 120, 30, 26, 20, 15, 20, 0, 30, 26),
            (0, 0, 0, 0, 220, 0, 0, 0, 0, 0),
            (9.2, 0.019, 0.020, -0.030, 0.019, 0.000, 0.000, -2.10, -0.030, 0.000),
            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        )

        tissue_map = pd.DataFrame(
            data=np.array(data).T, index=tissue_type, columns=parameter_type
        )

    elif B0 == 3.0:
        data = (
            (0, 1, 0.83, 0.7, 1, 1, 1, 0, 0.86, 0.77),
            (0, 4391, 1615, 911, 450, 1750, 2569, 0, 1615, 911),
            (0, 2000, 93, 61, 40, 37, 270, 0, 99, 50),
            (0, 158, 55, 48, 40, 30, 50, 0, 55, 48),
            (0, 0, 0, 0, 220, 0, 0, 0, 0, 0),
            (9.2, 0.019, 0.020, -0.030, 0.019, 0.000, 0.000, -2.10, -0.030, 0.000),
            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        )

        tissue_map = pd.DataFrame(
            data=np.array(data).T, index=tissue_type, columns=parameter_type
        )

    elif B0 == 1.5:
        data = (
            (0, 1, 0.83, 0.7, 1, 1, 1, 0, 0.86, 0.77),
            (0, 2569, 833, 500, 350, 900, 2569, 0, 833, 500),
            (0, 329, 83, 70, 70, 47, 329, 0, 83, 70),
            (0, 158, 69, 61, 58, 30, 58, 0, 69, 61),
            (0, 0, 0, 0, 220, 0, 0, 0, 0, 0),
            (9.2, 0.019, 0.020, -0.030, 0.019, 0.000, 0.000, -2.10, -0.030, 0.000),
            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        )

        tissue_map = pd.DataFrame(
            data=np.array(data).T, index=tissue_type, columns=parameter_type
        )

    else:
        raise NotImplementedError(
            "Parameters only defined for field strenghts of 1.5T, 3.0T and 7.0T"
        )

    return tissue_map


def label_map():
    label_map = nib.load(os.path.join(path, "MNIbrain_corr.nii")).get_fdata()
    label_map = np.rot90(label_map)
    label_map = zpad(label_map, (256, 256, 256))

    return label_map


def parameter_maps(label_map, tissue_map):
    M0 = np.zeros(label_map.shape)
    for label, value in zip(tissue_map["Label"], tissue_map["M0"]):
        M0[label_map == label] = value

    T1 = np.zeros(label_map.shape)
    for label, value in zip(tissue_map["Label"], tissue_map["T1"]):
        T1[label_map == label] = value

    T2 = np.zeros(label_map.shape)
    for label, value in zip(tissue_map["Label"], tissue_map["T2"]):
        T2[label_map == label] = value

    T2s = np.zeros(label_map.shape)
    for label, value in zip(tissue_map["Label"], tissue_map["T2s"]):
        T2s[label_map == label] = value

    CS = np.zeros(label_map.shape)
    for label, value in zip(tissue_map["Label"], tissue_map["CS"]):
        CS[label_map == label] = value

    X = np.zeros(label_map.shape)
    for label, value in zip(tissue_map["Label"], tissue_map["X"]):
        X[label_map == label] = value

    return M0, T1, T2, T2s, CS, X


class BrainPhantom5D(object):
    def __init__(
        self, B0=1.5, B1=5, TR=30, FM_scaling=1, CS_scaling=0, B0_shimming=True
    ):
        print("Constructing Numerical Brain Phantom ...")

        tm = tissue_map(B0)
        lm = label_map()

        self.lm = lm

        # Load parameter maps
        M0, T1, T2, T2s, CS, X = parameter_maps(lm, tm)

        self.X = X

        FM = dipole_convolution(X, B0)

        self.FM = FM

        # Center frequency
        if B0_shimming:
            FM = b0_shimming(FM, self.lm > 0)

            self.FM = FM

        B1_rad = np.deg2rad(B1)

        with np.errstate(all="ignore"):
            R1 = 1 / T1
            R1[~np.isfinite(R1)] = 0

            R2s = 1 / T2s
            R2s[~np.isfinite(R2s)] = 0

        self.M0 = M0
        self.R1 = R1

        # Steady-state magnetization
        self.MSS = (
            M0
            * np.sin(B1_rad)
            * (1 - np.exp(-TR * R1))
            / (1 - np.cos(B1_rad) * np.exp(-TR * R1))
        )

        # Effective longitudinal relaxation rate
        self.R1s = R1 - np.log(np.cos(B1_rad)) / TR

        # Complex effective transverse relaxation rate with off-resonance map
        # SC: Field map scaling
        self.R2s = R2s + 1j * (FM_scaling * FM + CS_scaling * CS)

    def signal(self, TP, TE, TS, IE, slice_z=None):
        """
        TP: Inversion time points
        TE: Echo time points
        TS: Time between two inversion pulses
        IE: Inversion efficiency
        """

        if slice_z is None:
            slice_z = slice(0, self.M0.shape[2])

        TP = TP[None, None, None, :, None]
        TE = TE[None, None, None, None, :]
        MSS = self.MSS[:, :, slice_z, None, None]
        R1s = self.R1s[:, :, slice_z, None, None]
        R2s = self.R2s[:, :, slice_z, None, None]

        # Compute multi-echo inversion recovery curve
        s = (
            MSS
            * (1 - (1 + IE) / (1 + IE * np.exp(-TS * R1s)) * np.exp(-TP * R1s))
            * np.exp(-TE * R2s)
        )

        return s


def dipole_convolution(X, B0, gamma=42576000):
    """
    X: magnetic susceptibility
    B0: static magnetic field [T]
    gamma: gyromagnetic Ratio    [MHz/T]
    """

    omega = 2 * np.pi * gamma * B0  # Larmor frequency      [1/s]
    omega = omega / 1000  # Convert from [1/s] to [1/ms]

    N = np.shape(X)

    # Zero-pad to power of 2
    M = (2 ** np.ceil(np.log2(N))).astype(int)
    tmp = np.zeros(M)
    tmp[0 : N[0], 0 : N[1], 0 : N[2]] = X
    X = tmp
    X = X * 1e-6  # Susceptibility given in ppm

    # Differential operator in frequency domain
    Nx, Ny, Nz = np.shape(X)

    fx, fy, fz = [np.linspace(0, 0.5, n // 2 + 1) for n in [Nx, Ny, Nz]]
    fx, fy, fz = [np.concatenate((f, -f[-2:0:-1])) for f in [fx, fy, fz]]

    K = np.array(np.meshgrid(fx, fy, fz))  # [Kx, Ky, Kz]

    with np.errstate(all="ignore"):
        Dk = 1 / 3 - (K[-1, ...] ** 2 / np.sum(K**2, axis=0))
        Dk[0, 0, 0] = 1 / 3

    Dk[~np.isfinite(Dk)] = 0

    # Magnetic field distortion [T]
    B = np.real(np.fft.ifftn(Dk * np.fft.fftn(X, norm="ortho"), norm="ortho"))

    # Remove zero-padding
    B = B[0 : N[0], 0 : N[1], 0 : N[2]]

    FM = B * omega

    return FM


def b0_shimming(FM, mask, order=2):
    ScanParameters = dict()
    ScanParameters["rotationInfo"] = dict()
    ScanParameters["stackNormal"] = dict()
    ScanParameters["sliceGap"] = 0
    ScanParameters["fieldOfView"] = np.array((256, 256, 256))
    ScanParameters["voxelSize"] = np.array((1, 1, 1))
    ScanParameters["stackCenter"] = np.array((0, 0, 0))
    ScanParameters["rotationInfo"]["rotationMatrix"] = np.eye(3)
    ScanParameters["stackNormal"]["orientation"] = "transversal"
    ScanParameters["matrixSize"] = np.array((256, 256, 256))

    voxels, xyz = getVoxelCoordinates(ScanParameters)
    idealHarmonics = getSphericalHarmonics(xyz, order)
    idealHarmonics = idealHarmonics.reshape((256, 256, 256, -1))

    b = FM[mask]
    A = idealHarmonics[mask, :]
    Ap = np.linalg.pinv(A)
    # unconstrained pseudo-inverse solution
    x_lsq = Ap @ b

    A_full = idealHarmonics.reshape((-1, idealHarmonics.shape[-1]))
    FM = FM - (A_full @ x_lsq).reshape((256, 256, 256))

    return FM
