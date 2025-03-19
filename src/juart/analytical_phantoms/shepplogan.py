import time  # noqa: I001
from typing import Tuple, Union

import numpy as np

from . import coils as cs
from . import imagespace as im
from . import kspace as ks
from . import ellipsoids as pa
from . import utils as ut


class SheppLogan:
    """Class of a Shepp Logan MRI phantom.

    References
    ----------
        ..[1] Gach, H. Michael, Costin Tanase, und Fernando Boada.
        „2D & 3D Shepp-Logan Phantom Standards for MRI“.
        2008. IEEE. https://doi.org/10.1109/ICSEng.2008.15.
    """

    def __init__(
        self,
        fov: Union[float, np.ndarray, list],
        ndim=None,
        t2star: bool = True,
        b0: float = 3.0,
        gamma: float = 42.576e6,
        blood_clot: bool = False,
        homogeneous: bool = True,
    ):
        fov, ndim = ut._checkdim(fov, ndim, "fov")

        self.fov = fov
        self.ndim = ndim
        self.t2star = t2star
        self.b0 = b0
        self.gamma = gamma
        self.blood_clot = blood_clot
        self.homogeneous = homogeneous
        self.Coil = None

    def addCoil(
        self,
        res: Union[int, np.ndarray, list, tuple],
        radius: Union[float, None] = None,
        n_channel_ring: Union[int, None] = 8,
        z_pos: Union[float, np.ndarray, list, tuple] = 0,
        phi0: Union[float, np.ndarray, list, tuple] = 0,
        z_orientation: Union[float, np.ndarray, list, tuple] = 0,
        adjust_ellipse: bool = False,
    ):
        """Add a cyclic head coil to the phantom to add coil sensitivities.

        Parameters
        ----------
        res : Union[int, np.ndarray, list, tuple]
            Grid size / resolution of the coil sensitivity maps.
        radius : float or None
            Radius of the coil [m].
            If None, the FOV of the phantom will be taken as the radius.
        n_channel_ring : int, optional
            Number of channels on one ring of the head coil
            equaly placed along the circumfence, by default 8 for 2d and 4 for 3d
        z_pos : Union[float, np.ndarray, list, tuple], optional
            Positions of the single channel array rings along the z axis, by default 0
        phi0 : Union[float, np.ndarray, list, tuple], optional
            Start angle of position of the first coil in the ring, by default 0
        z_orientation : Union[float, np.ndarray, list, tuple], optional
            _description_, by default 0
        adjust_ellipse : bool
            Adjust coil ring to the Shepp-Logan phantom shape, by default False
        """
        if radius is None:
            radius = self.fov[0]

        res, _ = ut._checkdim(res, self.ndim, "grid size")

        if n_channel_ring is None:
            if res.shape[0] == 2:
                n_channel_ring = 8
            elif res.shape[0] == 3:
                n_channel_ring = 4

        self.Coil = cs.Coil(
            coil_radius=radius,
            n_channel_ring=n_channel_ring,
            z_pos=z_pos,
            phi0=phi0,
            z_orientation=z_orientation,
        )

        # adjust for ellipse phantom
        if adjust_ellipse:
            for coil in self.Coil.coils:
                phi = np.arctan2(coil.r_cent[1], coil.r_cent[0])

                # adjust radius to shepp logan shape
                a = pa.getParams(fov=self.fov).iloc[0].axis_a
                b = pa.getParams(fov=self.fov).iloc[0].axis_b
                R = self.fov[0] / 1.5 + np.sqrt(
                    (a * np.cos(phi)) ** 2 + (b * np.sin(phi)) ** 2
                )
                coil.r_cent = np.array(
                    [R * np.cos(phi), R * np.sin(phi), coil.r_cent[2]]
                )

                # rebuild coil
                coil.build_coil_elements()

        # create locations on a cartesian grid, where coil sensitivities are defined
        loc_img = ut.createCartImgGridLocation(res, self.fov, format="vec")

        # use biggest ellipse of SL phantom as support region of coil sensitivities
        support = im.ellipsemask(loc_img, pa.getParams(fov=self.fov).iloc[0])

        _, _ = self.Coil.getCoilSensCoeff(
            r=loc_img, fov=self.fov, support=support, save=True, verbose=1
        )

    def getKspaceSignal(
        self,
        k: np.ndarray,
        t: np.ndarray,
        tr: float = 4,
        flip: float = 90,
        snr_db: float = 40,
        coil_sens=False,
        verbose=0,
    ) -> np.ndarray:
        """Calulate kspace signal of the Shepp Logan
        phantom for every k_i at time t_i.

        Parameters
        ----------
        k : np.ndarray
            Trajectory k-space locations of shape (Nsamples, ndim) [1/m].
        t : np.ndarray
            Time points of the trajectory of shape (Nsamples) [s].
        tr : float
            Repitition time [s].
        flip : float
            Magnetisation flip angle [degree]
        snr_db : float
            Signal to Noise ratio [dB].
            Following Nishimura "Principles of Magnetic Resonance Imaging"
            :math:'SNR_{dB} = 20 \log(SNR)'
        coil_sens : bool
            Add coil sensitivities to signal.
        verbose : int
            Verbose

        Returns
        -------
        signal : np.ndarray, complex
            MR signal of the Shepp Logan phantom at the given k-space locations.
        """  # noqa: W605

        Nsamples, Ndim = k.shape
        Nchannel = self.Coil.n_channels if coil_sens else 1

        if coil_sens:
            s_sens = self.Coil.coeff_sig
            k_sens = self.Coil.coeff_k

        else:  # empty coil sensitivities
            k_sens = np.zeros((1, Ndim))
            s_sens = np.ones((1, Nchannel), dtype=np.complex64)

        if Ndim != self.ndim:
            raise ValueError(
                f"Dimensions of given kspace ({Ndim}) location do not match"
                f"with phantom dimensions ({self.ndim})"
            )

        params = pa.getParams(
            fov=self.fov,
            ndim=self.ndim,
            b0=self.b0,
            gamma=self.gamma,
            blood_clot=self.blood_clot,
            homogeneous=self.homogeneous,
        )

        n_ellipsoids = params.shape[0]

        if Ndim == 3:
            geo_columns = [
                "center_x",
                "center_y",
                "center_z",
                "axis_a",
                "axis_b",
                "axis_c",
                "angle",
            ]
        if Ndim == 2:
            geo_columns = ["center_x", "center_y", "axis_a", "axis_b", "angle"]

        geo_params = params[geo_columns].values

        output = np.zeros((Nsamples, Nchannel), dtype=np.complex64)

        start = time.time()

        for n in range(Nchannel):
            if verbose == 1:
                print(
                    f"Simulating Shepp-Logan signal for channel {n + 1} of {Nchannel}."
                )
            for i in range(n_ellipsoids):
                # fourier transform of the ellips itself
                # geometry = ks.ellipseFT(k, geo_params[i])
                geometry = ks.ellipseFT_sens(k, geo_params[i], k_sens, s_sens[:, n])

                # signal decay during time evolution
                decay = ut.decay(t, params.iloc[i], flip, tr, self.t2star)

                output[:, n] += geometry * decay.astype(np.complex64)

        # add noise
        output = ks.add_noise(output, k, snr_db)

        end = time.time()
        if verbose == 1:
            print(
                f"Finished signal simulation of SheppLogan \
                after {end - start} wallseconds."
            )
        return output

    def getImgspace(
        self, res: Union[int, np.ndarray], te: float, tr: float, flip: float
    ) -> np.ndarray:
        """Calulate image space signal of the Shepp Logan phantom.

        Parameters
        ----------
        Res : int, np.ndarray
            Resolution of image.
        te : float
            Echotime TE [s].
        tr : float
            Repitition TR time [s].
        flip : float
            Magnetisation flip angle [degree]

        Returns
        -------
        signal : np.ndarray
            MR signal of the Shepp Logan phantom in image space.
        """

        res, ndim = ut._checkdim(res, self.ndim, "resolution")

        if ndim != self.ndim:
            raise ValueError(
                f"Dimension of Res ({ndim}) does not"
                f"math with dimension of phanom ({self.ndim})"
            )

        # get image locations
        loc = ut.createCartImgGridLocation(res, self.fov, format="vec")

        params = pa.getParams(
            fov=self.fov,
            ndim=self.ndim,
            b0=self.b0,
            gamma=self.gamma,
            blood_clot=self.blood_clot,
            homogeneous=self.homogeneous,
        )

        img = np.zeros(loc.shape[0])

        # fill image with the ellipsoids
        for i in range(params.shape[0]):
            mask = im.ellipsemask(loc, params.iloc[i])

            decay = ut.decay(te, params.iloc[i], flip, tr, self.t2star)

            img[mask] += decay

        return img.reshape(*res)

    def getImgMaps(
        self, res: Union[int, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns PD, T2(s), T1 maps form phantom.

        Parameters
        ----------
        res : int or float or np.ndarray or list
            Resolution of image.

        Returns
        -------
        PD, T2, T1 : tuple(np.ndarray)
            Quantitative Maps of the Shepp Logan phantom in image space.
        """
        res, ndim = ut._checkdim(res, self.ndim, "resolution")

        if ndim != self.ndim:
            raise ValueError(
                f"Dimension of Res ({ndim}) does not math with dimension of"
                f"phanom ({len(self.ndim)})"
            )

        # get image locations
        loc = ut.createCartImgGridLocation(res, self.fov, format="vec")

        params = pa.getParams(
            fov=self.fov,
            ndim=self.ndim,
            b0=self.b0,
            gamma=self.gamma,
            blood_clot=self.blood_clot,
            homogeneous=self.homogeneous,
        )

        PD, T2, T1 = np.zeros((3, loc.shape[0]))

        # fill maps of the ellipsoids
        for i in range(params.shape[0]):
            mask = im.ellipsemask(loc, params.iloc[i])

            p_dens = params.iloc[i]["spin"]
            transv = params.iloc[i]["t2s"] if self.t2star else params.iloc[i]["t2"]
            long = params.iloc[i]["t1"]

            # portions have to be subtracted not only for the pd but t2 and t1
            if np.sign(p_dens) < 0:
                transv *= -1
                long *= -1

            PD[mask] += p_dens
            T2[mask] += transv
            T1[mask] += long

        PD = PD.reshape(*res)
        T2 = T2.reshape(*res)
        T1 = T1.reshape(*res)

        return PD, T2, T1

    def getSupportRegion(self, res: Union[int, np.ndarray]) -> np.ndarray:
        """Returns support region of the phantom.
        Is 1 where phantom is defined and 0 elsewhere.

        Parameters
        ----------
        res : int or float or np.ndarray or list
            Resolution of image.

        Returns
        -------
        SR : np.ndarray, int
            Support region of Phantom.
        """
        res, ndim = ut._checkdim(res, self.ndim, "resolution")

        if ndim != self.ndim:
            raise ValueError(
                f"Dimension of Res ({ndim}) does not match with dimension of"
                f"phantom ({len(self.ndim)})"
            )

        # get image locations
        loc = ut.createCartImgGridLocation(res, self.fov, format="vec")

        params = pa.getParams(
            fov=self.fov,
            ndim=self.ndim,
            b0=self.b0,
            gamma=self.gamma,
            blood_clot=self.blood_clot,
            homogeneous=self.homogeneous,
        )

        # reshape SR (*res) shape
        SR = im.ellipsemask(loc, params.iloc[0]).reshape(res)

        return SR
