import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interactive, widgets


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
    ):
        self.data = data
        self.vmin = vmin
        self.vmax = vmax
        self.title = title
        self.cmap = cmap
        self.layout = layout

        if layout[0] * layout[1] < len(data):
            self.fig, self.ax = plt.subplots(1, len(data))

        if layout[0] * layout[1] >= len(data):
            self.fig, self.ax = plt.subplots(2, 3)

        if type(self.ax) is np.ndarray:
            ims = list()

            if type(self.ax[0]) != np.ndarray:
                for i, ax in enumerate(self.ax):
                    ax.imshow = self.ax[i].imshow(
                        self.data[i][:, :, 0],
                        vmin=self.vmin,
                        vmax=self.vmax,
                        cmap=self.cmap,
                    )
                    ax.set_title(self.title[i])
                    ims.append(ax.imshow)

            else:
                i = 0
                rows, cols = layout
                for row in range(0, rows, 1):
                    for col, ax in enumerate(self.ax[row]):
                        if len(data) > i:
                            ax.imshow = self.ax[row][col].imshow(
                                self.data[i][:, :, 0],
                                vmin=self.vmin,
                                vmax=self.vmax,
                                cmap=self.cmap,
                            )
                            ax.set_title(self.title[i])
                            ims.append(ax.imshow)

                        i += 1

            if activate_colorbar:
                self.fig.colorbar(
                    ims[-1], ax=self.ax.ravel().tolist(), location="right"
                )

        else:
            self.ax.imshow = plt.imshow(
                self.data[0][:, :, 0], vmin=self.vmin, vmax=self.vmax, cmap=self.cmap
            )
            self.ax.set_title(self.title[0])

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
        if type(self.ax) is np.ndarray:
            if type(self.ax[0]) is not np.ndarray:
                for i, ax in enumerate(self.ax):
                    ax.imshow.set_data(self.data[i][:, :, z - 1])

            else:
                i = 0
                rows, cols = self.layout

                for row in range(0, rows, 1):
                    for col, ax in enumerate(self.ax[row]):
                        if len(self.data) > i:
                            ax.imshow.set_data(self.data[i][:, :, z - 1])

                        i += 1

            self.fig.canvas.flush_events()

        else:
            self.ax.imshow.set_data(self.data[0][:, :, z - 1])
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
