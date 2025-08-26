# -*- coding: utf-8 -*-


import numpy as np

from phantoms.jemris import jemris_parmaps


class BrainPhantom1D(object):

    def __init__(self, t, relaxmap=1, fieldmap=1):

        print('Constructing Numerical Brain Phantom ...')

        # Load JEMRIS parameter maps
        M0, _, _, R2s, FM = jemris_parmaps()

        # Complex effective transverse relaxation rate with off-resonance map
        R2s = R2s * relaxmap + 1j * FM * fieldmap

        # Mono-exponential MEGE signal equation
        s = M0[:, :, None] * np.exp(-t[None, None, :] * R2s[..., None])

        self.M0 = M0
        self.R2s = R2s

        self.s = s


class BrainPhantom2D(object):

    def __init__(self, TP, TE):

        print('Constructing Numerical Brain Phantom ...')

        # Load JEMRIS parameter maps
        M0, R1, _, R2s, FM = jemris_parmaps()

        # Complex effective transverse relaxation rate with off-resonance map
        R2s = R2s + 1j * FM

        self.M0 = M0
        self.R1 = R1
        self.R2s = R2s

        TP = TP[None, None, :, None]
        TE = TE[None, None, None, :]
        M0 = M0[:, :, None, None]
        R1 = R1[:, :, None, None]
        R2s = R2s[:, :, None, None]

        # Compute multi-echo inversion recovery curve
        s = M0 * (1 - 2 * np.exp(-TP * R1)) * np.exp(-TE * R2s)

        self.s = s


class MyelinBrainPhantom(object):

    def __init__(self, t):

        print('Constructing Numerical Myelin Brain Phantom ...')

        # ---------------------------------------------------------------------
        # Load JEMRIS parameter maps
        M0, _, _, R2s, FM = jemris_parmaps()

        # Complex effective transverse relaxation rate with off-resonance map
        R2s = R2s + 1j * FM
        # ---------------------------------------------------------------------

        # ---------------------------------------------------------------------
        # Myelinated white matter
        self.MASK_WM = np.zeros(M0.shape, dtype=np.bool)
        self.MASK_WM[(0.0163 < np.real(R2s)) & (np.real(R2s) < 0.0165)] = True

        M0_WM = np.zeros(M0.shape, dtype=np.complex)
        M0_WM[self.MASK_WM] = 0.2

        R2s_WM = np.zeros(M0.shape, dtype=np.complex)
        R2s_WM[self.MASK_WM] = 1 / 15 + 1j * FM[self.MASK_WM]
        # ---------------------------------------------------------------------

        # ---------------------------------------------------------------------
        # Remove this fraction from the IE pool
        M0[self.MASK_WM] = 0.73 - 0.2
#        M0 = M0 - M0_WM
        # ---------------------------------------------------------------------

        # ---------------------------------------------------------------------
        # Mono-exponential MEGE signal equation
        s = M0[:, :, None] * \
            np.exp(-t[None, None, :] * R2s[..., None])

        s = s + M0_WM[:, :, None] * \
            np.exp(-t[None, None, :] * R2s_WM[..., None])
        # ---------------------------------------------------------------------

        self.M0 = np.concatenate((M0, M0_WM), axis=1)
        self.R2s = np.concatenate((R2s, R2s_WM), axis=1)

        self.s = s


class CompartmentPhantom(object):

    def __init__(self, t):

        print('Constructing Numerical Multi-Compartment Phantom ...')

        xv, yv = np.meshgrid(np.linspace(-.5, .5, 256),
                             np.linspace(-.5, .5, 256))

        mask = np.zeros((256, 256))

        mask[(xv**2 + yv**2) < 0.3**2] = 1

        for i in range(2, 5):
            xc = xv[128, 128] + 0.15 * np.sin(2 * np.pi * (i - 1) / 3)
            yc = yv[128, 128] - 0.15 * np.cos(2 * np.pi * (i - 1) / 3)
            mask[((xv - xc)**2 + (yv - yc)**2) < 0.1**2] = i

        M0 = np.zeros((3, 256, 256), dtype=np.complex)
        R2s = np.zeros((3, 256, 256), dtype=np.complex)

        M0[0, np.in1d(mask, [1, 2, 3, 4]).reshape((256, 256))] = 1
        R2s[0, np.in1d(mask, [1, 2, 3, 4]).reshape((256, 256))] = 1 / 50

        M0[1, np.in1d(mask, [2, 4]).reshape((256, 256))] = 0.2
        R2s[1, np.in1d(mask, [2, 4]).reshape((256, 256))] = 1 / 15

        M0[2, np.in1d(mask, [3, 4]).reshape((256, 256))] = 0.2
        R2s[2, np.in1d(mask, [3, 4]).reshape((256, 256))] = 1 / 15 + 0.2j

        s = np.sum(M0[..., None] * np.exp(-R2s[..., None] * t[None, ...]),
                   axis=0)

        self.M0 = np.concatenate(M0, axis=1)
        self.R2s = np.concatenate(R2s, axis=1)

        self.mask = mask
        self.s = s
