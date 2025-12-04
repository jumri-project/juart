import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interactive, widgets



class InteractiveFunctionPlotter:
    def __init__(
        self,
        x: list,
        y: list,
        start_intslider: float = None,
        end_intslider: float = None,
        step_intslider: float = None,
        title: str = None,
        x_label: str = None,
        y_label: str = None,
        x_lim: list[float] = None,
        y_lim: list[float] = None,
        figsize: list[float] = None,
    ):
        fig, ax = plt.subplots(1, 1)
        self.x = x
        self.y = y
        self.start_intslider = start_intslider
        self.end_intslider = end_intslider
        self.step_intslider = step_intslider
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.figsize = figsize

        fig.figure(figsize=figsize)
        ax.set_title(title)
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)

        if x_lim:
            ax.set_xlim(x_lim)

        if y_lim:
            ax.set_ylim(y_lim)

        self.interactive = interactive(
            self.show,
            z=widgets.IntSlider(
                min=1,
                max=self.data[0].shape[2],
                value=self.data[0].shape[2] // 2,
                description=description,
            ),
        )

    def show(self, z):

        for i, data in enumerate(self.data):
            self.ims[i].set_data(data[:, :, z - 1])

        self.fig.canvas.flush_events()

class InteractiveMultiPlotter3D:
    def __init__(
        self,
        data: list,
        layout: list = [1, 1],
        vmin: int = None,
        vmax: int = None,
        title: list = None,
        cmap: str = "gray",
        description: str = "Dimension 3:",
        activate_colorbar: bool = True,
        show_axis: bool = True,
        compact_plotting: bool = False
    ):
        self.data = data
        self.vmin = vmin
        self.vmax = vmax
        self.title = title
        self.cmap = cmap
        self.layout = layout

        self.ims = list()
        if layout[0] * layout[1] < len(data):
            self.fig, self.ax = plt.subplots(1, len(data))

        elif layout[0] * layout[1] >= len(data):
            self.fig, self.ax = plt.subplots(layout[0],layout[1])

        if isinstance(self.ax, np.ndarray):
            self.ax = self.ax.flatten()

        else:
            self.ax = [self.ax]

        if compact_plotting:
            plt.subplots_adjust(left=0, right=1, bottom=0.15, top=1, wspace=0, hspace=0)

        for i, ax in enumerate(self.ax):

            img = ax.imshow(self.data[i][:, :, 0], vmin=self.vmin, vmax=self.vmax, cmap=self.cmap)
            self.ims.append(img)

            if self.title != None:
                ax.set_title(self.title[i])

            if not show_axis:
                ax.axis('off')

            if activate_colorbar:
                self.fig.colorbar(self.ax.imshow)

        self.interactive = interactive(
            self.show,
            z=widgets.IntSlider(
                min=1,
                max=self.data[0].shape[2],
                value=self.data[0].shape[2] // 2,
                description=description,
            ),
        )

    def show(self, z):

        for i, data in enumerate(self.data):
            self.ims[i].set_data(data[:, :, z - 1])

        self.fig.canvas.flush_events()



class InteractiveFigure3D:
    def __init__(
        self,
        data,
        vmin=None,
        vmax=None,
        title=None,
        figure=None,
        axes=(0, 1, 2),
        cmap=None,
        description="Dimension 3",
    ):
        self.img = data.transpose(axes)
        self.vmin = vmin
        self.vmax = vmax
        self.title = title

        # let 'inf' and 'nan' appear as the highest value
        # self.img[~np.isfinite(self.img)] = np.max(self.img[np.isfinite(self.img)])

        self.fig = plt.figure(figure, figsize=(3, 3))
        plt.title(self.title)
        # self.ax = self.fig.add_subplot(1, 1, 1)
        self.imshow = plt.imshow(
            self.img[:, :, 0], vmin=self.vmin, vmax=self.vmax, cmap=cmap
        )
        # self.ax.axis("off")
        self.fig.colorbar(self.imshow)

        self.interactive = interactive(
            self.show,
            z=widgets.IntSlider(
                min=1,
                max=self.img.shape[2],
                value=self.img.shape[2] // 2,
                description=description,
            ),
        )

    def show(self, z):
        self.imshow.set_data(self.img[:, :, z - 1])
        self.fig.canvas.flush_events()


class InteractiveFigure4D:
    def __init__(
        self,
        data,
        vmin=None,
        vmax=None,
        title=None,
        figure=None,
        axes=(0, 1, 2, 3),
        cmap=None,
        description=("Dimension 3", "Dimension 4"),
    ):
        self.img = data.transpose(axes)
        self.vmin = vmin
        self.vmax = vmax
        self.title = title

        # let 'inf' and 'nan' appear as the highest value
        # self.img[~np.isfinite(self.img)] = np.max(self.img[np.isfinite(self.img)])

        self.fig = plt.figure(figure, figsize=(3, 3))
        plt.title(self.title)
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.imshow = plt.imshow(
            self.img[:, :, 0, 0], vmin=self.vmin, vmax=self.vmax, cmap=cmap
        )
        self.ax.axis("off")
        self.fig.colorbar(self.imshow)

        self.interactive = interactive(
            self.show,
            z=widgets.IntSlider(
                min=1,
                max=self.img.shape[2],
                value=self.img.shape[2] // 2,
                description=description[0],
            ),
            t=widgets.IntSlider(
                min=1,
                max=self.img.shape[3],
                value=self.img.shape[3] // 2,
                description=description[1],
            ),
        )

    def show(self, z, t):
        self.imshow.set_data(self.img[:, :, z - 1, t - 1])
        self.fig.canvas.flush_events()


class InteractiveFigure5D:
    def __init__(
        self,
        data,
        vmin=None,
        vmax=None,
        title=None,
        figure=None,
        axes=(0, 1, 2, 3, 4),
        cmap=None,
        description=("Dimension 3", "Dimension 4", "Dimension 5"),
    ):
        self.img = data.transpose(axes)
        self.vmin = vmin
        self.vmax = vmax
        self.title = title

        # let 'inf' and 'nan' appear as the highest value
        # self.img[~np.isfinite(self.img)] = np.max(self.img[np.isfinite(self.img)])

        self.fig = plt.figure(figure, figsize=(3, 3))
        plt.title(self.title)
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.imshow = plt.imshow(
            self.img[:, :, 0, 0, 0], vmin=self.vmin, vmax=self.vmax, cmap=cmap
        )
        self.ax.axis("off")
        self.fig.colorbar(self.imshow)

        self.interactive = interactive(
            self.show,
            x=widgets.IntSlider(
                min=1,
                max=self.img.shape[2],
                value=self.img.shape[2] // 2,
                description=description[0],
            ),
            y=widgets.IntSlider(
                min=1,
                max=self.img.shape[3],
                value=self.img.shape[3] // 2,
                description=description[1],
            ),
            z=widgets.IntSlider(
                min=1,
                max=self.img.shape[4],
                value=self.img.shape[4] // 2,
                description=description[2],
            ),
        )

    def show(self, x, y, z):
        self.imshow.set_data(self.img[:, :, x - 1, y - 1, z - 2])
        self.fig.canvas.flush_events()
