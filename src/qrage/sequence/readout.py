from types import SimpleNamespace
from typing import Tuple, Union

import numpy as np
import pypulseq as pp
from pypulseq import Opts, Sequence

from . import rasterize, set_grad
from .sampling import AngularTrajectory


class ReadoutKernel:
    """
    A class representing the readout kernel for MRI acquisition, including
    RF pulses, multiple gradient echoes and gradient spoilers.
    """

    def __init__(
        self,
        fov: Tuple[int, int, int],
        matrix_size: Tuple[int, int, int],
        axes: SimpleNamespace,
        num_sets: int = 19,
        num_echoes: int = 9,
        rf_pulse_deg: float = 5.0,
        # rf_pulse_duration: float = 100e-6,
        rf_pulse_duration: float = 2e-3,
        # rf_pulse_duration: float = 4e-3,
        rf_apodization: float = 0.5,
        # rf_time_bw_product: float = 4.0,
        rf_time_bw_product: float = 10,
        # rf_spoiling_increment: float = 117.0,
        rf_spoiling_increment: float = 50.0,
        readout_bandwidth: float = 390.625,
        readout_oversampling: float = 1.0,
        # readout_spoiling: float = 3.0,
        readout_spoiling: float = 2.0,
        # readout_spoiling: float = 1.5,
        debug: bool = False,
        system: Union[Opts, None] = None,
    ) -> None:
        """
        Initialize the ReadoutKernel.

        Parameters:
        ----------
        fov : Tuple[int, int, int]
            Field of view in millimeters (x, y, z).
        matrix_size : Tuple[int, int, int]
            Matrix size (x, y, z).
        axes : SimpleNamespace
            Axis mapping for the encoding directions.
        num_sets : int
            The number of sets in the acquisition.
        num_echoes : int
            The number of echoes per acquisition.
        rf_pulse_deg : float
            Flip angle in degrees.
        rf_pulse_duration : float
            RF pulse duration in seconds.
        rf_spoiling_increment : float
            RF spoiling increment in degrees.
        readout_bandwidth : float
            Readout bandwidth in kHz.
        readout_oversampling : float
            Oversampling factor.
        readout_spoiling : floatl
            Spoiling factor.
        debug : bool
            If True, enables debug mode.
        system : Opts, default=Opts()
            System limits.
        """
        voxel_size = np.array(fov) / np.array(matrix_size)

        self.trajectory = AngularTrajectory(num_sets, num_echoes)
        self.num_sets = num_sets
        self.num_echoes = num_echoes
        self.system = system
        self.axes = axes
        self.debug = debug
        self.rf_spoiling_increment = rf_spoiling_increment

        # Partition encoding steps -- control reordering
        self.partition_encoding_steps = np.linspace(
            -1, +1, num=matrix_size[axes.n3], endpoint=False
        )

        # ---
        # RF pulse and slab-selection gradient
        # ---

        # Create alpha-degree RF pulse and gradients
        (self.rf, self.gradient_slice_select, self.gradient_slice_rephaser) = (
            pp.make_sinc_pulse(
                flip_angle=np.radians(rf_pulse_deg),
                duration=rf_pulse_duration,
                slice_thickness=fov[axes.n3],
                apodization=rf_apodization,
                time_bw_product=rf_time_bw_product,
                return_gz=True,
                system=system,
                use="excitation",
            )
        )

        # ---
        # Partition encoding gradient
        # including rewinder for slab-selection gradient
        # ---

        self.gradient_partition_encoding_area = -0.5 / voxel_size[axes.n3]
        self.gradient_partition_encoding = pp.make_trapezoid(
            channel=axes.d3,
            area=(
                self.gradient_partition_encoding_area
                + self.gradient_slice_rephaser.area
            ),
            system=system,
        )

        # ---
        # ADC
        # ---

        readout_num_samples = matrix_size[axes.n1] * readout_oversampling
        readout_duration = rasterize(
            1 / readout_bandwidth, system.block_duration_raster
        )
        self.adc = pp.make_adc(
            num_samples=readout_num_samples,
            duration=readout_duration,
            delay=0,
            system=system,
        )

        # ---
        # Readout gradient
        # ---

        gradient_readout_area = 1 / voxel_size[axes.n1]
        # gradient_flat_time = pp.calc_duration(self.adc)
        gradient_flat_time = rasterize(
            pp.calc_duration(self.adc) + system.adc_dead_time, system.grad_raster_time
        )
        self.gradient_readout = pp.make_trapezoid(
            channel=axes.d1,
            amplitude=gradient_readout_area / readout_duration,
            flat_time=gradient_flat_time,
            system=system,
        )

        # Split the readout gradient into rise, flat, and fall
        (
            self.gradient_readout_rise,
            self.gradient_readout_flat,
            self.gradient_readout_fall,
        ) = pp.split_gradient(grad=self.gradient_readout, system=system)

        self.gradient_readout_rise.delay = 0
        self.gradient_readout_flat.delay = 0
        self.gradient_readout_fall.delay = 0

        # ---
        # Readout dephaser gradient
        # ---

        gradient_readout_dephaser_area = -self.gradient_readout.amplitude * (
            self.gradient_readout.flat_time / 2 + self.gradient_readout.rise_time / 2
        )
        self.gradient_readout_dephaser = pp.make_trapezoid(
            channel=axes.d1,
            area=gradient_readout_dephaser_area,
            system=system,
        )

        self.gradient_readout_rise.delay = pp.calc_duration(
            self.gradient_readout_dephaser
        )
        self.gradient_readout_dephaser = pp.add_gradients(
            grads=[self.gradient_readout_dephaser, self.gradient_readout_rise],
            system=system,
        )

        (self.gradient_readout_dephaser, self.gradient_partition_encoding) = pp.align(
            right=[
                self.gradient_readout_dephaser,
                self.gradient_partition_encoding,
            ],
        )

        # ---
        # Blip gradient
        # ---

        gradient_blip_area = (
            self.gradient_readout.area
            / 2
            * np.sqrt(2 + 2 * np.cos(self.trajectory.increment_echo))
        )
        self.gradient_blip = pp.make_trapezoid(
            channel=axes.d1,
            area=gradient_blip_area,
            system=system,
        )

        self.gradient_readout_fall_0 = set_grad(
            grad=self.gradient_readout_fall,
            area=self.gradient_readout_fall.area,
            channel=axes.d1,
        )

        self.gradient_readout_rise_1 = set_grad(
            grad=self.gradient_readout_rise,
            area=self.gradient_readout_rise.area,
            channel=axes.d1,
        )

        self.gradient_readout_fall_0.delay = 0
        self.gradient_blip.delay = pp.calc_duration(self.gradient_readout_fall_0)
        self.gradient_readout_rise_1.delay = pp.calc_duration(self.gradient_blip)

        # ---
        # Spoiler gradient
        # ---

        self.gradient_readout_spoiler_area = (
            readout_spoiling * self.gradient_readout.area
        )
        self.gradient_x_readout_spoiler = pp.make_trapezoid(
            channel=axes.d1,
            area=self.gradient_readout_spoiler_area,
            system=system,
        )
        self.gradient_y_readout_spoiler = pp.make_trapezoid(
            channel=axes.d2,
            area=self.gradient_readout_spoiler_area,
            system=system,
        )
        self.gradient_z_readout_spoiler = pp.make_trapezoid(
            channel=axes.d3,
            area=(
                self.gradient_readout_spoiler_area
                - self.gradient_partition_encoding_area
            ),
            system=system,
        )

        self.gradient_x_readout_spoiler.delay = pp.calc_duration(
            self.gradient_readout_fall_0
        )
        self.gradient_y_readout_spoiler.delay = pp.calc_duration(
            self.gradient_readout_fall_0
        )

        # ---
        # Timing
        # ---

        # (TODO) Timing calculation is currently not correct

        # self.total_time = (
        #     pp.calc_duration(
        #         self.rf,
        #         self.gradient_slice_select,
        #     )
        #     + pp.calc_duration(
        #         self.gradient_partition_encoding,
        #         self.gradient_readout_dephaser,
        #     )
        #     + self.num_echoes * pp.calc_duration(
        #         self.adc,
        #         self.gradient_readout_flat,
        #     )
        #     + (self.num_echoes - 1) * pp.calc_duration(
        #         self.gradient_blip,
        #     )
        #     + pp.calc_duration(
        #         self.gradient_x_readout_spoiler,
        #         self.gradient_y_readout_spoiler,
        #         self.gradient_z_readout_spoiler,
        #     )
        # )
        #
        # # Time before the readout process starts
        # self.pre_time = self.rf.delay + pp.calc_rf_center(self.rf)[0]

    def prep(self, seq: Sequence) -> None:
        """
        Prepares the kernel by registering the RF and gradient events with the
        sequence.

        Parameters:
        ----------
        seq : Sequence
            The pypulseq Sequence object to register the events.
        """
        result = seq.register_grad_event(self.gradient_slice_select)
        self.gradient_slice_select.id = result if isinstance(result, int) else result[0]

        # Phase of the RF object will change, therefore we only pre-register the shapes
        _, self.rf.shape_IDs = seq.register_rf_event(self.rf)

    def reset(self, seq: Sequence) -> None:
        """
        Resets the RF spoiling phase and increment.

        Parameters:
        ----------
        seq : Sequence
            The sequence object (unused in the current implementation).
        """
        self.rf_increment = 0
        self.rf_phase = 0

    def run(
        self, seq: Sequence, index_phase: int, index_partition: int, index_set: int
    ) -> None:
        """
        Executes the readout kernel by adding the RF pulses and gradient blocks
        to the sequence.

        Parameters:
        ----------
        seq : Sequence
            Sequence object.
        index_phase : int
            The phase encoding index.
        index_partition : int
            The partition encoding index.
        index_set : int
            The set index.
        """
        if self.debug:
            print(f"Phase index: {index_phase}, Partition index: {index_partition}")

        self.rf.phase_offset = np.radians(self.rf_phase)
        self.adc.phase_offset = np.radians(self.rf_phase)

        self.rf_increment = np.mod(
            self.rf_increment + self.rf_spoiling_increment,
            360.0,
        )
        self.rf_phase = np.mod(
            self.rf_phase + self.rf_increment,
            360.0,
        )

        angle = self.trajectory.angle(index_phase, index_set, 0)

        # Log gradient updates
        self.gradient_x_logs = list()
        self.gradient_y_logs = list()

        gradient_x_readout_dephaser = set_grad(
            grad=self.gradient_readout_dephaser,
            area=self.gradient_readout_dephaser.area * np.cos(angle),
            channel=self.axes.d1,
        )
        gradient_y_readout_dephaser = set_grad(
            grad=self.gradient_readout_dephaser,
            area=self.gradient_readout_dephaser.area * np.sin(angle),
            channel=self.axes.d2,
        )

        gradient_partition_encoding_dephaser = set_grad(
            grad=self.gradient_partition_encoding,
            area=-self.partition_encoding_steps[index_partition]
            * self.gradient_partition_encoding_area
            + self.gradient_slice_rephaser.area,
            channel=self.axes.d3,
        )
        gradient_partition_encoding_rephaser = set_grad(
            grad=self.gradient_z_readout_spoiler,
            area=self.partition_encoding_steps[index_partition]
            * self.gradient_partition_encoding_area
            + self.gradient_readout_spoiler_area,
            channel=self.axes.d3,
        )

        self.gradient_x_logs.append(gradient_x_readout_dephaser)
        self.gradient_y_logs.append(gradient_y_readout_dephaser)

        if not self.debug:
            seq.add_block(
                self.rf,
                self.gradient_slice_select,
            )
            seq.add_block(
                gradient_x_readout_dephaser,
                gradient_y_readout_dephaser,
                gradient_partition_encoding_dephaser,
                pp.make_label(type="SET", label="LIN", value=index_phase),
                pp.make_label(type="SET", label="PAR", value=index_partition),
                pp.make_label(type="SET", label="SET", value=index_set),
            )

        # Iterate through echoes
        for index_echo in range(self.num_echoes):
            angle_0 = self.trajectory.angle(index_phase, index_set, index_echo)
            angle_1 = self.trajectory.angle(index_phase, index_set, index_echo + 1)
            alpha_blip = (angle_1 + angle_0) / 2

            if self.debug:
                print(f"Set index: {index_set}, Echo index: {index_echo}")
                print(f"Angle: {np.rad2deg(np.mod(angle_0, 2 * np.pi))}")

            gradient_x_readout_flat = set_grad(
                grad=self.gradient_readout_flat,
                area=self.gradient_readout_flat.area * np.cos(angle_0),
                channel=self.axes.d1,
            )
            gradient_y_readout_flat = set_grad(
                grad=self.gradient_readout_flat,
                area=self.gradient_readout_flat.area * np.sin(angle_0),
                channel=self.axes.d2,
            )

            self.gradient_x_logs.append(gradient_x_readout_flat)
            self.gradient_y_logs.append(gradient_y_readout_flat)

            if not self.debug:
                seq.add_block(
                    gradient_x_readout_flat,
                    gradient_y_readout_flat,
                    self.adc,
                    pp.make_label(type="SET", label="ECO", value=index_echo),
                )

            if index_echo < (self.num_echoes - 1):
                # Handle blips and rephasing
                gradient_x_readout_fall_0 = set_grad(
                    grad=self.gradient_readout_fall_0,
                    area=self.gradient_readout_fall_0.area * np.cos(angle_0),
                    channel=self.axes.d1,
                )
                gradient_y_readout_fall_0 = set_grad(
                    grad=self.gradient_readout_fall_0,
                    area=self.gradient_readout_fall_0.area * np.sin(angle_0),
                    channel=self.axes.d2,
                )

                gradient_x_blip = set_grad(
                    grad=self.gradient_blip,
                    area=self.gradient_blip.area * np.cos(alpha_blip),
                    channel=self.axes.d1,
                )
                gradient_y_blip = set_grad(
                    grad=self.gradient_blip,
                    area=self.gradient_blip.area * np.sin(alpha_blip),
                    channel=self.axes.d2,
                )

                gradient_x_readout_rise_1 = set_grad(
                    grad=self.gradient_readout_rise_1,
                    area=self.gradient_readout_rise_1.area * np.cos(angle_1),
                    channel=self.axes.d1,
                )
                gradient_y_readout_rise_1 = set_grad(
                    grad=self.gradient_readout_rise_1,
                    area=self.gradient_readout_rise_1.area * np.sin(angle_1),
                    channel=self.axes.d2,
                )

                gradient_x_blip = pp.add_gradients(
                    grads=[
                        gradient_x_readout_fall_0,
                        gradient_x_blip,
                        gradient_x_readout_rise_1,
                    ],
                    system=self.system,
                )
                gradient_y_blip = pp.add_gradients(
                    grads=[
                        gradient_y_readout_fall_0,
                        gradient_y_blip,
                        gradient_y_readout_rise_1,
                    ],
                    system=self.system,
                )

                self.gradient_x_logs.append(gradient_x_blip)
                self.gradient_y_logs.append(gradient_y_blip)

                if not self.debug:
                    seq.add_block(
                        gradient_x_blip,
                        gradient_y_blip,
                    )

            else:
                gradient_x_readout_fall = set_grad(
                    grad=self.gradient_readout_fall,
                    area=self.gradient_readout_fall.area * np.cos(angle_0),
                    channel=self.axes.d1,
                )
                gradient_y_readout_fall = set_grad(
                    grad=self.gradient_readout_fall,
                    area=self.gradient_readout_fall.area * np.sin(angle_0),
                    channel=self.axes.d2,
                )

                gradient_x_readout_spoiler = pp.add_gradients(
                    grads=[self.gradient_x_readout_spoiler, gradient_x_readout_fall],
                    system=self.system,
                )
                gradient_y_readout_spoiler = pp.add_gradients(
                    grads=[self.gradient_y_readout_spoiler, gradient_y_readout_fall],
                    system=self.system,
                )

                self.gradient_x_logs.append(gradient_x_readout_spoiler)
                self.gradient_y_logs.append(gradient_y_readout_spoiler)

                if not self.debug:
                    seq.add_block(
                        gradient_x_readout_spoiler,
                        gradient_y_readout_spoiler,
                        gradient_partition_encoding_rephaser,
                    )
