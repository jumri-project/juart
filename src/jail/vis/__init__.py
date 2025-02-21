# -*- coding: utf-8 -*-
"""
Markus Zimmermann (m.zimmermann@fz-juelich.de) 2017
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


class MultiPlotter(object):
    def __init__(
        self,
        data,
        idx,
        axis=0,
        title=None,
        fig=None,
        ax=None,
        drawnow=False,
        vmin=None,
        vmax=None,
        cmap=None,
        cbar=True,
        norm=None,
        cbar_label="",
        cbar_tick="right",
        orientation="vertical",
        cbar_size="5%",
        cbar_labelsize=12,
        cbar_labelpad=None,
        cbar_nticks=4,
        cbar_pad=0.0,
        alpha=1,
        interpolation="antialiased",
        facecolor="white",
        figsize=None,
    ):
        self.rows, self.cols = idx[0], idx[1]

        if isinstance(axis, int):
            self.axis = [axis]
        elif isinstance(axis, list) or isinstance(axis, tuple):
            if all(isinstance(axis, int) for axis in axis):
                self.axis = list(axis)

        self.drawnow = drawnow

        if isinstance(fig, int):
            plt.close(fig)
            fig = plt.figure(fig, figsize=figsize, facecolor=facecolor)

        if (fig is None) and (ax is None):
            fig = plt.gcf()
            fig.clf()

        if ax is None:
            fig.clf()
            ax = plt.subplot(1, 1, 1)

        self.fig = fig
        self.ax = ax

        self.im = self.ax.imshow(
            self.prepare(data),
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            norm=norm,
            alpha=alpha,
            interpolation=interpolation,
        )
        plt.xticks([])
        plt.yticks([])

        if isinstance(title, str):
            plt.title(title)

        if cbar:
            colorbar(
                self.ax,
                self.im,
                tick=cbar_tick,
                label=cbar_label,
                orientation=orientation,
                size=cbar_size,
                labelsize=cbar_labelsize,
                nticks=cbar_nticks,
                pad=cbar_pad,
                labelpad=cbar_labelpad,
            )

        if self.drawnow:
            self.draw_and_flush()

    def prepare(self, data):
        data = np.array(data)

        data = np.moveaxis(data, self.axis, list(range(-len(self.axis), 0)))

        data = data.reshape(data.shape[0], data.shape[1], -1)

        data = np.split(data, data.shape[-1], axis=-1)

        # Remove singleton dimensions created by split
        data = [d.squeeze(axis=-1) for d in data]

        while len(data) < self.rows * self.cols:
            # Add eps to avoid issues with LogNorm.
            data.append(np.zeros(data[0].shape, dtype=data[0].dtype) + 1e-16)

        assert len(data) == self.rows * self.cols

        hstack = [
            np.hstack(data[i : i + self.cols]) for i in range(0, len(data), self.cols)
        ]
        data = np.vstack(hstack)

        assert np.ndim(data) == 2

        return data

    def update(self, data):
        self.im.set_data(self.prepare(data))

        if self.drawnow:
            self.draw_and_flush()

    def draw_and_flush(self):
        self.fig.canvas.draw()

        try:
            self.fig.canvas.flush_events()
        except BaseException:
            pass


class ParmapPlotter(object):
    def __init__(self, maps, fig=None, vmin=None, vmax=None):
        if isinstance(fig, int):
            fig = plt.figure(fig)

        elif fig is None:
            fig = plt.gcf()

        fig.clf()

        self.fig = fig

        title = ["Proton Density", "Initial Phase", "Relaxivity", "Fieldmap"]
        #    vmin = [0, -np.pi, 0, -np.pi]
        #    vmax = [2, np.pi, .05, np.pi]
        if vmin is None:
            vmin = [0, -np.pi, 0, -0.5]
        if vmax is None:
            vmax = [2, np.pi, 0.05, 0.5]

        ticks = ["left", "right", "left", "right"]

        self.ax = list()
        self.im = list()

        for i in range(4):
            self.ax.append(plt.subplot(2, 2, i + 1))
            self.im.append(
                self.ax[i].imshow(maps[i, ...], cmap="gray", vmin=vmin[i], vmax=vmax[i])
            )

            plt.xticks([])
            plt.yticks([])
            plt.title(title[i])

            colorbar(self.ax[i], self.im[i], ticks[i])

        self.draw_and_flush()

    def draw_and_flush(self):
        self.fig.canvas.draw()

        try:
            self.fig.canvas.flush_events()
        except BaseException:
            pass

    def update(self, maps):
        for i in range(4):
            self.im[i].set_data(maps[i, ...])

        self.draw_and_flush()


def colorbar(
    ax,
    im,
    tick="right",
    label="",
    orientation="vertical",
    size="5%",
    labelsize=12,
    nticks=4,
    pad=0.0,
    labelpad=None,
):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(tick, size=size, pad=pad)

    if label is not None:
        # Workaround to keep subplots of same size
        cb = plt.colorbar(mappable=im, cax=cax, orientation=orientation)
        cb.ax.tick_params(labelsize=labelsize)
        cb.set_label(label=label, fontsize=labelsize, labelpad=labelpad)
        if nticks is not None:
            from matplotlib import ticker

            tick_locator = ticker.MaxNLocator(nbins=nticks)
            cb.locator = tick_locator
            cb.update_ticks()

        if tick in ["left", "right"]:
            cb.ax.yaxis.set_ticks_position(tick)
    else:
        cax.axis("Off")
