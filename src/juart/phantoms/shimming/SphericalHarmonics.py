#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 16:28:00 2020

@author: mschwerter
"""

import math

import numpy as np
from scipy import special as sp


def getSphericalHarmonics(xyz, maxOrder):
    r = np.sqrt(xyz[0, :] ** 2 + xyz[1, :] ** 2 + xyz[2, :] ** 2)
    theta = np.arccos(xyz[2, :] / r)
    phi = np.arctan2(xyz[1, :], xyz[0, :])

    rtp = np.zeros(xyz.shape)
    rtp[0, :] = r
    rtp[1, :] = theta
    rtp[2, :] = phi

    numPoints = xyz.shape[1]
    numHarmonics = (maxOrder + 1) ** 2
    sphericalHarmonics = np.zeros((numHarmonics, numPoints))

    sphericalHarmonics[0, :] = 1.0

    for idx in range(maxOrder):
        n = idx + 1
        nthOrderHarmonics = calculateHarmonics(n, rtp)

        idxLo = n**2
        idxHi = idxLo + 2 * n + 1

        sphericalHarmonics[idxLo:idxHi, :] = nthOrderHarmonics

    sphericalHarmonics = sphericalHarmonics.T

    return sphericalHarmonics


def calculateHarmonics(n, rtp):
    u = np.cos(rtp[1, :])
    P = legendre(n, u)
    rn = np.power(rtp[0, :], n)

    Fdo = 1
    m = 0
    Fdo1 = Fdo * math.factorial(n - m) / math.factorial(n + m)
    # Pdo = P[m, :]
    Pdo = P[m]

    numTerms = 2 * n + 1
    numPoints = rtp.shape[1]
    harmonics = np.zeros([numTerms, numPoints])

    harmonics[0, :] = Fdo1 * rn * Pdo

    for idx in range(n):
        m = idx + 1

        angularc = np.cos(m * rtp[2, :])
        angulars = np.sin(m * rtp[2, :])

        Fdo = Fdo * 2 * m
        Fdo1 = Fdo * math.factorial(n - m) / math.factorial(n + m)

        harmonics[2 * m - 1, :] = Fdo1 * rn * P[m] * angularc
        harmonics[2 * m - 0, :] = Fdo1 * rn * P[m] * angulars
        # harmonics[2*m-1, :] = Fdo1 * rn * P[m, :] * angularc
        # harmonics[2*m-0, :] = Fdo1 * rn * P[m, :] * angulars

    return harmonics


def legendre(n, X):
    res = []
    for m in range(n + 1):
        res.append(sp.lpmv(m, n, X))
    return res


# def legendre_old(N, X):
#     matrixReturn = np.zeros((N+1, X.shape[0]))
#     for i in enumerate(X):
#         currValues = sp.lpmn(N, N, i[1])
#         matrixReturn[:, i[0]] = np.array([j[N] for j in currValues[0]])
#     return matrixReturn
