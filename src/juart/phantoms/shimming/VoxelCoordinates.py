#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 13:55:08 2020

@author: mschwerter
"""

import numpy as np


def getVoxelCoordinates(scanParameters):

    sliceGap = scanParameters['sliceGap']
    fieldOfView = scanParameters['fieldOfView']
    voxelSize = scanParameters['voxelSize']
    stackCenter = scanParameters['stackCenter']
    rotationMatrix = scanParameters['rotationInfo']['rotationMatrix']
    stackOrientation = scanParameters['stackNormal']['orientation']
    matrixSize = scanParameters['matrixSize'][0:3]

    coordsDelta = voxelSize + [0, 0, sliceGap]
    coordsStart = (-fieldOfView+voxelSize)/2
    coordsStop = (fieldOfView-voxelSize)/2 + coordsDelta

    xvec = np.arange(coordsStart[0], coordsStop[0], coordsDelta[0])
    yvec = np.arange(coordsStart[1], coordsStop[1], coordsDelta[1])
    zvec = np.arange(coordsStart[2], coordsStop[2], coordsDelta[2])

    coordinateArrays = getMeshgrid3D(xvec, yvec, zvec)

    X = coordinateArrays[0]
    Y = coordinateArrays[1]
    Z = coordinateArrays[2]

    xyz = np.vstack((X.reshape(1, X.size, order='F'),
                     Y.reshape(1, Y.size, order='F'),
                     Z.reshape(1, Z.size, order='F')))

    voxelCoordinates = rotationMatrix @ xyz + stackCenter[:, np.newaxis]

    # Above OK
    X = np.reshape(voxelCoordinates[0, :], matrixSize, order='F')
    Y = np.reshape(voxelCoordinates[1, :], matrixSize, order='F')
    Z = np.reshape(voxelCoordinates[2, :], matrixSize, order='F')

    dX1 = X[-1, 0, 0] - X[0, 0, 0]
    dX2 = X[0, -1, 0] - X[0, 0, 0]
    dX3 = X[0, 0, -1] - X[0, 0, 0]
    slopeX = np.array((dX1, dX2, dX3)) / matrixSize

    dY1 = Y[-1, 0, 0] - Y[0, 0, 0]
    dY2 = Y[0, -1, 0] - Y[0, 0, 0]
    dY3 = Y[0, 0, -1] - Y[0, 0, 0]
    slopeY = np.array((dY1, dY2, dY3)) / matrixSize

    dZ1 = Z[-1, 0, 0] - Z[0, 0, 0]
    dZ2 = Z[0, -1, 0] - Z[0, 0, 0]
    dZ3 = Z[0, 0, -1] - Z[0, 0, 0]
    slopeZ = np.array((dZ1, dZ2, dZ3)) / matrixSize

    if stackOrientation == 'transversal':

        if np.abs(slopeX[0]) > np.abs(slopeY[0]):
            X = np.transpose(X, (1, 0, 2))
            slopeX = slopeX[[1, 0, 2]]
            Y = np.transpose(Y, (1, 0, 2))
            slopeY = slopeY[[1, 0, 2]]
            Z = np.transpose(Z, (1, 0, 2))
            slopeZ = slopeZ[[1, 0, 2]]

        if slopeX[1] < 0:
            X = np.flip(X, 1)
            Y = np.flip(Y, 1)
            Z = np.flip(Z, 1)

        if slopeY[0] < 0:
            X = np.flip(X, 0)
            Y = np.flip(Y, 0)
            Z = np.flip(Z, 0)

        if slopeZ[2] < 0:
            X = np.flip(X, 2)
            Y = np.flip(Y, 2)
            Z = np.flip(Z, 2)

    elif stackOrientation == 'sagittal':
        print('Not implemented yet.')

    elif stackOrientation == 'coronal':
        print('Not implemented yet.')

    xyz.fill(0)
    xyz[0, :] = X.ravel(order='F')
    xyz[1, :] = Y.ravel(order='F')
    xyz[2, :] = Z.ravel(order='F')

    voxels = {'x': X, 'y': Y, 'z': Z}

    return voxels, xyz


def getMeshgrid3D(*args):
    arrayDimensions = len(args)
    arraySize = np.asarray([len(coordinateVector) for coordinateVector in args])
    meshgrids = []

    for idx in range(arrayDimensions):
        x = args[idx]
        s = np.ones(arrayDimensions).astype(int)
        s[idx] = len(x)
        x = np.reshape(x, s, order="F")
        s = arraySize.copy()
        s[idx] = 1

        bla = np.tile(x, s)

        meshgrids.append(bla)

    return meshgrids
