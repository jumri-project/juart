from copy import deepcopy
from types import SimpleNamespace
from typing import Tuple, Union

import numpy as np
from sympy import rf
import pypulseq as pp
from pypulseq import Opts, Sequence
from pypulseq.make_adiabatic_pulse import make_adiabatic_pulse


class InversionKernel_pTX:
    """
    Represents an inversion pulse sequence using an adiabatic RF pulse
    and gradient spoilers for MRI imaging.
    """

    def __init__(
        self,
        fov: Tuple[int, int, int],
        matrix_size: Tuple[int, int, int],
        axes: SimpleNamespace,
        num_tx: int = 8,
        Coil_shim_array: np.ndarray = None,
        adiabatic_pulse_type: str = "hypsec_n",
        adiabatic_pulse_beta: float = 400.0,
        adiabatic_pulse_duration: float = 10.24e-3,
        adiabatic_pulse_mu: float = 9.8,
        adiabatic_pulse_order: float = 4.0,
        adiabatic_pulse_overdrive: float = 2.0,
        adiabatic_pulse_dwell: float = 1e-5,
        gradient_spoiler_duration: float = 5e-3,
        system: Union[Opts, None] = None,
    ) -> None:
        """
        Initializes the InversionKernel class.

        Parameters
        ----------
        fov : Tuple[int, int, int]
            Field of view in millimeters (x, y, z).
        matrix_size : Tuple[int, int, int]
            Matrix size (x, y, z).
        axes : SimpleNamespace
            Axis mapping for the encoding directions.
        num_tx : int
            Number of transmit channels for pTx.
        Coil_shim_array : np.ndarray
            Array containing coil shim values for each transmit channel.
        adiabatic_pulse_type : str
            Type of adiabatic pulse to use.
        adiabatic_pulse_beta : float
            AM waveform parameter.
        adiabatic_pulse_duration : float
            Duration of the adiabatic pulse in seconds.
        adiabatic_pulse_mu : float
            A constant, determines amplitude of frequency sweep.
        adiabatic_pulse_order : float
            Order of the hyperbolic secant pulse.
        adiabatic_pulse_overdrive : float
            Overdrive factor.
        adiabatic_pulse_dwell : float
            Pulse dwell time in seconds.
        gradient_spoiler_duration : float
            Duration of the gradient spoiler in seconds.
        system : Opts, default=Opts()
            System limits.
        """

        # Create an adiabatic inversion pulse
        self.rf180 = make_adiabatic_pulse(
            pulse_type=adiabatic_pulse_type,
            beta=adiabatic_pulse_beta,
            duration=adiabatic_pulse_duration,
            mu=adiabatic_pulse_mu,
            order=adiabatic_pulse_order,
            overdrive=adiabatic_pulse_overdrive,
            dwell=adiabatic_pulse_dwell,
            system=system,
        )

        # Copy the non-pTx RF pulse
        self.rf180.signal = self.rf180.signal.astype(
            np.complex128
        )  # Ensure RF signal is complex for pTx manipulation
        self.rf180_ptx = deepcopy(self.rf180)
        # repeat signal for each channel
        self.rf180_ptx.signal = np.tile(self.rf180.signal, (num_tx))
        iSamples_per_channel = self.rf180.t.shape[0]
        for i in range(num_tx):
            self.rf180_ptx.signal[
                i * iSamples_per_channel : (i + 1) * iSamples_per_channel
            ] *= Coil_shim_array[i, 0] * np.exp(1j * Coil_shim_array[i, 1])
        # And repeat the time vector with the number of Tx channels
        self.rf180_ptx.t = np.tile(self.rf180.t, (num_tx))

        # Spoiling with 4x cycles per voxel
        gradient_spoiler_area = 4 * np.max(np.array(matrix_size) / np.array(fov))

        # Create gradient spoilers along x, y, and z axes
        self.gradient_spoiler_x = pp.make_trapezoid(
            channel="x",
            area=gradient_spoiler_area,
            duration=gradient_spoiler_duration,
            system=system,
        )
        self.gradient_spoiler_y = pp.make_trapezoid(
            channel="y",
            area=gradient_spoiler_area,
            duration=gradient_spoiler_duration,
            system=system,
        )
        self.gradient_spoiler_z = pp.make_trapezoid(
            channel="z",
            area=gradient_spoiler_area,
            duration=gradient_spoiler_duration,
            system=system,
        )

        # Calculate total time of the RF pulse and gradient spoilers
        self.total_time = pp.calc_duration(self.rf180) + pp.calc_duration(
            self.gradient_spoiler_x, self.gradient_spoiler_y, self.gradient_spoiler_z
        )

        # (TODO) Timing calculation is currently not correct
        # Post-time
        self.post_time = (
            pp.calc_duration(self.rf180)
            + pp.calc_rf_center(self.rf180)[0]
            + self.rf180.delay
        )

    def prep(
        self,
        seq: Sequence,
    ) -> None:
        """
        Prepares the inversion kernel by registering gradient and RF events
        with the given sequence.

        Parameters:
        - seq (Sequence): The pypulseq Sequence object for registering events.
        """

        # Register gradient events
        result = seq.register_grad_event(self.gradient_spoiler_x)
        self.gradient_spoiler_x.id = result if isinstance(result, int) else result[0]

        result = seq.register_grad_event(self.gradient_spoiler_y)
        self.gradient_spoiler_y.id = result if isinstance(result, int) else result[0]

        result = seq.register_grad_event(self.gradient_spoiler_z)
        self.gradient_spoiler_z.id = result if isinstance(result, int) else result[0]

        # Register RF event
        self.rf180.id, self.rf180.shape_IDs = seq.register_rf_event(self.rf180)

    def run(
        self,
        seq: Sequence,
    ) -> None:
        """
        Executes the inversion kernel by adding the RF pulse and gradient spoilers
        to the sequence.

        Parameters:
        - seq (Sequence): The pypulseq Sequence object for adding blocks.
        """

        # Add RF pulse block
        seq.add_block(self.rf180_ptx)

        # Add gradient spoiler blocks
        seq.add_block(
            self.gradient_spoiler_x, self.gradient_spoiler_y, self.gradient_spoiler_z
        )
