import matplotlib.pyplot as plt
from ipywidgets import interactive, widgets
import numpy as np

class InteractiveMultiPlotter3D:
    def __init__(
        self,
        data: list,
        vmin: int = None,
        vmax: int = None,
        title: str = None,
        cmap: str = "gray",
        description = "Dimension 3:"
    ):
        self.data = data
        self.vmin = vmin
        self.vmax = vmax
        self.title = title
        self.cmap = cmap
        self.description = description

        self.fig, self.ax = plt.subplots(1, len(data))
        plt.title(self.title)

        i = 0
        if type(self.ax) is np.ndarray:
            for ax in self.ax:
                print(ax)
                ax.imshow = plt.imshow(
                    self.data[i][:, :, 0], vmin=self.vmin, vmax=self.vmax, cmap=self.cmap
                )
                i += 1
        else:
            self.ax.imshow = plt.imshow(
                self.data[i][:, :, 0], vmin=self.vmin, vmax=self.vmax, cmap=self.cmap
            )

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

        if type (self.ax) is np.ndarray:
            for i, ax in enumerate(self.ax):
    
                ax.imshow.set_data(self.data[i][:, :, z - 1])
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
        #self.ax = self.fig.add_subplot(1, 1, 1)
        self.imshow = plt.imshow(
            self.img[:, :, 0], vmin=self.vmin, vmax=self.vmax, cmap=cmap
        )
        #self.ax.axis("off")
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
